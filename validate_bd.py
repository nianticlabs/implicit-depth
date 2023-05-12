import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import options
from experiment_modules.bd_model import BDModel
from modules.layers import sigmoid_custom
from utils.binary_metrics_utils import PlaneEvaluator
from utils.dataset_utils import get_dataset
from utils.generic_utils import to_gpu
from utils.metrics_utils import ResultsAverager


def main(opts):
    # get dataset
    dataset_class, scans = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(
        opts.output_base_path, opts.name, opts.dataset, opts.frame_tuple_type
    )

    if opts.binary_inference_width is not None:
        results_path = os.path.join(results_path, str(opts.binary_inference_width))

    # save predictions
    pred_output_dir = os.path.join(results_path, "predictions_planes")
    Path(pred_output_dir).mkdir(parents=True, exist_ok=True)

    # set up directory for saving scores
    scores_output_dir = os.path.join(
        results_path, "depth_scores" if opts.binary_eval_depth else "iou_scores"
    )
    Path(scores_output_dir).mkdir(parents=True, exist_ok=True)

    # Set up model. Note that we're not passing in opts as an argument, although
    # we could. We're being pretty stubborn with using the options the model had
    # used when training, saved internally as part of hparams in the checkpoint.
    # You can change this at inference by passing in 'opts=opts,' but there
    # be dragons if you're not careful.
    model = BDModel.load_from_checkpoint(opts.load_weights_from_checkpoint, args=None)
    if opts.fast_cost_volume:
        model.cost_volume = model.cost_volume.to_fast()

    model.run_opts.bd_sigmoid_multiplier = opts.bd_sigmoid_multiplier
    model = model.cuda().eval()

    # setting up overall result averagers
    all_frame_metrics = ResultsAverager(opts.name, f"frame metrics")

    depths_for_printing = [1.5 + x * 0.5 for x in range(8)]
    thresholds = np.linspace(0.1, 0.9, 17)
    evaluator = PlaneEvaluator(thresholds=thresholds)

    with torch.inference_mode():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # loop over scans
        for scan in tqdm(scans):
            # set up dataset with current scan
            dataset = dataset_class(
                opts.dataset_path,
                split=opts.split,
                mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                limit_to_scan_id=scan,
                include_full_res_depth=True,
                tuple_info_file_location=opts.tuple_info_file_location,
                num_images_in_tuple=None,
                shuffle_tuple=opts.shuffle_tuple,
                include_high_res_color=(
                    (opts.fuse_color and opts.run_fusion) or opts.dump_depth_visualization
                ),
                include_full_depth_K=True,
                skip_frames=opts.skip_frames,
                image_width=opts.image_width,
                image_height=opts.image_height,
                pass_frame_id=True,
                get_bd_info=True,
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opts.batch_size,
                shuffle=False,
                num_workers=opts.num_workers,
                drop_last=False,
            )

            evaluator.initialise_for_new_scene()

            # initialize scene averager
            scene_frame_metrics = ResultsAverager(opts.name, f"scene {scan} metrics")

            for batch_ind, batch in enumerate(tqdm(dataloader)):
                # get data, move to GPU
                cur_data, src_data = batch
                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

                depth_gt_b1hw = cur_data["full_res_depth_b1hw"]

                rendered_depth_bdhw = cur_data["rendered_depth"]
                depth_planes_1D11 = torch.tensor(depths_for_printing).reshape(1, -1, 1, 1).cuda()
                rendered_depth_bDhw = (
                    torch.ones_like(rendered_depth_bdhw[:, 0:1]) * depth_planes_1D11
                )
                cur_data["rendered_depth"] = rendered_depth_bDhw

                # run to get output, also measure time
                start_time.record()
                # use unbatched (looping) matching encoder image forward passes
                # for numerically stable testing. If opts.fast_cost_volume, then
                # batch.
                outputs = model(
                    "test",
                    cur_data,
                    src_data,
                    unbatched_matching_encoder_forward=(not opts.fast_cost_volume),
                    return_mask=True,
                    infer_depth=opts.binary_eval_depth,
                    infer_res=None
                    if opts.binary_inference_width is None
                    else [opts.binary_inference_height, opts.binary_inference_width],
                )
                end_time.record()
                torch.cuda.synchronize()

                elapsed_model_time = start_time.elapsed_time(end_time)

                outputs["pred_0"] = sigmoid_custom(outputs["pred_0"], multiplier=1.0)

                upsampled_pred_bdhw = F.interpolate(
                    outputs["pred_0"],
                    size=(depth_gt_b1hw.shape[-2], depth_gt_b1hw.shape[-1]),
                    mode="nearest" if opts.render_eval else "bilinear",
                )

                upsampled_query_bdhw = F.interpolate(
                    cur_data["rendered_depth"],
                    size=(depth_gt_b1hw.shape[-2], depth_gt_b1hw.shape[-1]),
                    mode="nearest",
                )

                # inf max depth matches DVMVS metrics, using minimum of 0.5m
                thresh_to_check = 0.5 if opts.binary_eval_depth else 0.0
                valid_mask_b = cur_data["full_res_depth_b1hw"] > thresh_to_check

                # Check if there any valid gt points in this sample
                if (valid_mask_b).any():
                    # compute metrics
                    metrics_b_dict = evaluator.compute_batch_scores(
                        query_depth_bdhw=upsampled_query_bdhw,
                        gt_depth_b1hw=depth_gt_b1hw,
                        prediction_bdhw=upsampled_pred_bdhw,
                        is_rendering=opts.render_eval,
                        depth_planes=depths_for_printing,
                    )

                    # go over batch and get metrics frame by frame to update
                    # the averagers
                    for element_index in range(depth_gt_b1hw.shape[0]):
                        if (~valid_mask_b[element_index]).all():
                            # ignore if no valid gt exists
                            continue

                        element_metrics = {}
                        for key in list(metrics_b_dict.keys()):
                            element_metrics[key] = metrics_b_dict[key][element_index]

                        # get per frame time in the batch
                        element_metrics["model_time"] = elapsed_model_time / depth_gt_b1hw.shape[0]

                        # both this scene and all frame averagers
                        scene_frame_metrics.update_results(element_metrics)
                        all_frame_metrics.update_results(element_metrics)

        # compute and print final average
        print("\nFinal metrics:")
        print("\n")
        all_frame_metrics.compute_final_average(ignore_nans=True)
        all_frame_metrics.pretty_print_metric_table(
            print_running_metrics=False, depths=depths_for_printing, thresholds=thresholds
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    # don't need grad for test.
    torch.set_grad_enabled(False)

    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)
