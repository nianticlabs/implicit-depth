import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import options
from experiment_modules.bd_model import BDModel
from modules.layers import sigmoid_custom
from utils.binary_metrics_utils import (
    PlaneEvaluator,
    TemporalEvaluator,
    Thresholder,
    get_boundary_mask,
    get_surface_mask,
)
from utils.dataset_utils import get_dataset
from utils.generic_utils import cache_model_outputs, to_gpu
from utils.metrics_utils import ResultsAverager, compute_depth_metrics_batched
from utils.visualization_utils import colormap_image


def main(opts):
    # get dataset
    dataset_class, scans = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(
        opts.output_base_path, opts.name, opts.dataset, opts.frame_tuple_type
    )

    # save predictions
    pred_output_dir = os.path.join(results_path, "predictions_planes")
    Path(pred_output_dir).mkdir(parents=True, exist_ok=True)

    # set up directories for caching depths
    if opts.cache_depths:
        # path where we cache depth maps
        depth_output_dir = os.path.join(results_path, "depths")

        Path(depth_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Caching depths.".center(80, "#"))
        print(f"Output directory:\n{depth_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    if opts.dump_depth_visualization:
        viz_output_folder_name = "quick_viz"
        viz_output_dir = os.path.join(results_path, "viz", viz_output_folder_name)

        Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Saving quick viz.".center(80, "#"))
        print(f"Output directory:\n{viz_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    # set up directory for saving scores
    scores_output_dir = os.path.join(
        results_path, "depth_scores" if opts.binary_eval_depth else "iou_scores"
    )
    Path(scores_output_dir).mkdir(parents=True, exist_ok=True)

    # save predictions
    pred_output_dir = os.path.join(results_path, "predictions_planes")
    Path(pred_output_dir).mkdir(parents=True, exist_ok=True)

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
    all_scene_metrics = ResultsAverager(opts.name, f"scene metrics")

    # set up thresholder
    thresholder = Thresholder(
        planes=torch.linspace(1.5, 5.0, 8).float().cuda(),
        thresholds=torch.tensor([0.5, 0.400, 0.3000, 0.3000, 0.3000, 0.3000, 0.300, 0.300])
        .float()
        .cuda(),
    )

    if opts.use_validation_thresholds:
        print(f"using thresholds: {thresholder.thresholds}")
    else:
        thresholder = None
    model.thresholder = thresholder  # for depth eval

    if not opts.binary_eval_depth:
        depths_for_printing = [1.5 + x * 0.5 for x in range(8)]
        evaluator = PlaneEvaluator()

    if opts.temporal_eval:
        print(f"temporal_eval requested; using subset of scans and forcing batch_size to be 1")
        temporal_evaluator = TemporalEvaluator()
        depths_for_printing = [-1]
        opts.batch_size = 1
        eval_length = opts.eval_length
        eval_frame_multiplier = opts.eval_frame_multiplier
        warmup = opts.warmup

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
                include_high_res_color=opts.dump_depth_visualization,
                include_full_depth_K=True,
                skip_frames=opts.skip_frames,
                image_width=opts.image_width,
                image_height=opts.image_height,
                pass_frame_id=True,
                get_bd_info=True,
            )

            if opts.temporal_eval:
                dataset.frame_tuples = dataset.frame_tuples[: eval_length * eval_frame_multiplier]

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opts.batch_size,
                shuffle=False,
                num_workers=opts.num_workers,
                drop_last=False,
            )

            # initialize scene averager
            scene_frame_metrics = ResultsAverager(opts.name, f"scene {scan} metrics")

            if opts.temporal_eval:
                gt_mesh_path = dataset.get_gt_mesh_path(opts.dataset_path, opts.split, scan)
                temporal_evaluator.initialise_new_scene(gt_mesh_path=gt_mesh_path)

            prev_pred = None
            prev_cam_T_world = None
            eval_frame_count = 0
            for batch_ind, batch in enumerate(tqdm(dataloader)):
                # get data, move to GPU
                cur_data, src_data = batch
                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string", "dataset_name"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string", "dataset_name"])

                depth_gt_b1hw = cur_data["full_res_depth_b1hw"]

                if opts.temporal_eval:
                    if batch_ind % eval_length == 0:
                        temporal_evaluator.initialise_new_plane(
                            depth_gt_b1hw, cur_data["world_T_cam_b44"]
                        )
                        eval_frame_count = 0
                    rendered_depth = temporal_evaluator.rasterizer(
                        cur_data["cam_T_world_b44"], cur_data["K_s0_b44"]
                    )
                    cur_data["rendered_depth"] = rendered_depth
                    cur_data["prior_prediction"] = prev_pred
                    cur_data["prior_cam_T_world"] = prev_cam_T_world

                # get surface mask
                surface_mask_bdhw = get_surface_mask(
                    cur_data["depth_b1hw"], cur_data["rendered_depth"]
                )

                # get boundary mask
                boundary_mask_bdhw = get_boundary_mask(
                    cur_data["depth_b1hw"], cur_data["rendered_depth"]
                )

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
                    infer_res=None,
                )
                end_time.record()
                torch.cuda.synchronize()

                elapsed_model_time = start_time.elapsed_time(end_time)

                if opts.temporal_eval:
                    prev_cam_T_world = cur_data["cam_T_world_b44"]
                    prev_pred = sigmoid_custom(outputs["pred_0"], multiplier=1.0)
                    # edges are unreliable because of the black rectification border -> so remove them for next frame
                    # inference
                    temporal_evaluator.mask_prediction_edges(prediction=prev_pred)

                eval_frame_count += 1
                if opts.temporal_eval and eval_frame_count < warmup + 1:
                    continue

                outputs["pred_0"] = sigmoid_custom(
                    outputs["pred_0"], multiplier=opts.bd_sigmoid_multiplier
                )

                if opts.temporal_eval:
                    pred = outputs["pred_0"].clone()
                    temporal_evaluator.update_vertex_predictions(
                        pred, cur_data["cam_T_world_b44"], cur_data["K_s0_b44"]
                    )

                    if batch_ind % (eval_length - 1) == 0:
                        temporal_evaluator.compute_vertex_occlusion_changes()

                upsampled_pred_bdhw = F.interpolate(
                    outputs["pred_0"],
                    size=(depth_gt_b1hw.shape[-2], depth_gt_b1hw.shape[-1]),
                    mode="nearest" if opts.temporal_eval else "bilinear",
                )

                upsampled_query_bdhw = F.interpolate(
                    cur_data["rendered_depth"],
                    size=(depth_gt_b1hw.shape[-2], depth_gt_b1hw.shape[-1]),
                    mode="nearest",
                )

                boundary_query_bdhw = cur_data["rendered_depth"].clone()
                boundary_query_bdhw[~boundary_mask_bdhw.bool()] = -1
                boundary_query_bdhw = F.interpolate(
                    boundary_query_bdhw,
                    size=(depth_gt_b1hw.shape[-2], depth_gt_b1hw.shape[-1]),
                    mode="nearest",
                )

                surface_query_bdhw = cur_data["rendered_depth"].clone()
                surface_query_bdhw[~surface_mask_bdhw.bool()] = -1
                surface_query_bdhw = F.interpolate(
                    surface_query_bdhw,
                    size=(depth_gt_b1hw.shape[-2], depth_gt_b1hw.shape[-1]),
                    mode="nearest",
                )

                if opts.binary_eval_depth:
                    upsampled_pred_bdhw = F.interpolate(
                        outputs["search_depths"],
                        size=(depth_gt_b1hw.shape[-2], depth_gt_b1hw.shape[-1]),
                        mode="nearest",
                    )

                # inf max depth matches DVMVS metrics, using minimum of 0.5m
                thresh_to_check = 0.5 if opts.binary_eval_depth else 0.0
                valid_mask_b = cur_data["full_res_depth_b1hw"] > thresh_to_check

                # Check if there are any valid gt points in this sample
                if (valid_mask_b).any():
                    if opts.binary_eval_depth:
                        metrics_b_dict = compute_depth_metrics_batched(
                            depth_gt_b1hw.flatten(start_dim=1).float(),
                            upsampled_pred_bdhw.flatten(start_dim=1).float(),
                            valid_mask_b.flatten(start_dim=1),
                            mult_a=False,
                        )
                    else:
                        # compute metrics
                        metrics_b_dict = evaluator.compute_batch_scores_test(
                            query_depth_bdhw=upsampled_query_bdhw,
                            gt_depth_b1hw=depth_gt_b1hw,
                            prediction_bdhw=upsampled_pred_bdhw,
                            is_rendering=opts.temporal_eval,
                            thresholder=thresholder,
                        )

                        # surfaces evaluation
                        metrics_b_dict.update(
                            evaluator.compute_batch_scores_test(
                                query_depth_bdhw=surface_query_bdhw,
                                gt_depth_b1hw=depth_gt_b1hw,
                                prediction_bdhw=upsampled_pred_bdhw,
                                is_rendering=opts.temporal_eval,
                                tag="surface",
                                thresholder=thresholder,
                            )
                        )

                        # surfaces evaluation
                        metrics_b_dict.update(
                            evaluator.compute_batch_scores_test(
                                query_depth_bdhw=boundary_query_bdhw,
                                gt_depth_b1hw=depth_gt_b1hw,
                                prediction_bdhw=upsampled_pred_bdhw,
                                is_rendering=opts.temporal_eval,
                                tag="boundary",
                                thresholder=thresholder,
                            )
                        )

                    # go over batch and get metrics frame by frame to update
                    # the averagers
                    for element_index in range(depth_gt_b1hw.shape[0]):
                        if (~valid_mask_b[element_index]).all():
                            # ignore if no valid gt exists
                            continue

                        element_metrics = {}
                        for key in list(metrics_b_dict.keys()):
                            if isinstance(metrics_b_dict[key], torch.Tensor):
                                element_metrics[key] = metrics_b_dict[key][element_index].cpu()
                            else:
                                element_metrics[key] = metrics_b_dict[key][element_index]

                        # get per frame time in the batch
                        element_metrics["model_time"] = elapsed_model_time / depth_gt_b1hw.shape[0]

                        # both this scene and all frame averagers
                        scene_frame_metrics.update_results(element_metrics)
                        all_frame_metrics.update_results(element_metrics)

                if opts.dump_depth_visualization:
                    if not opts.binary_eval_depth:
                        raise Exception(
                            f"You can't dump depth predictions ",
                            f"if you aren't predicting a proper depth map.",
                        )

                    valid_mask_b = cur_data["full_res_depth_b1hw"] > 0.5

                    if valid_mask_b.sum() == 0:
                        batch_vmin = 0.0
                        batch_vmax = 5.0
                    else:
                        batch_vmin = cur_data["full_res_depth_b1hw"][valid_mask_b].min()
                        batch_vmax = cur_data["full_res_depth_b1hw"][valid_mask_b].max()

                    output_path = os.path.join(viz_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    for elem_ind in range(upsampled_pred_bdhw.shape[0]):
                        if "frame_id_string" in cur_data:
                            frame_id = cur_data["frame_id_string"][elem_ind]
                        else:
                            frame_id = (batch_ind * opts.batch_size) + elem_ind
                            frame_id = f"{str(frame_id):6d}"

                        # check for valid depths from dataloader
                        if valid_mask_b[elem_ind].sum() == 0:
                            sample_vmax = 0.0
                            sample_vmin = 0.0
                            print(frame_id)
                        else:
                            # these will be the same when the depth map is all ones.
                            sample_vmax = cur_data["full_res_depth_b1hw"][elem_ind][
                                valid_mask_b[elem_ind]
                            ].max()
                            sample_vmin = cur_data["full_res_depth_b1hw"][elem_ind][
                                valid_mask_b[elem_ind]
                            ].min()

                        # if no meaningful gt depth in dataloader, don't viz gt and
                        # set vmin/max to default
                        if sample_vmax != sample_vmin:
                            full_res_depth_1hw = cur_data["full_res_depth_b1hw"][elem_ind]

                            full_res_depth_3hw = colormap_image(
                                full_res_depth_1hw, vmin=batch_vmin, vmax=batch_vmax
                            )

                            full_res_depth_hw3 = np.uint8(
                                full_res_depth_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255
                            )
                            Image.fromarray(full_res_depth_hw3).save(
                                os.path.join(output_path, f"{frame_id}_gt_depth.png")
                            )

                        depth_3hw = colormap_image(
                            upsampled_pred_bdhw[elem_ind], vmin=batch_vmin, vmax=batch_vmax
                        )
                        pil_image = Image.fromarray(
                            np.uint8(depth_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
                        )

                        pil_image.save(os.path.join(output_path, f"{frame_id}_pred_depth.png"))

                ########################## Cache Depths ########################
                if opts.cache_depths:
                    output_path = os.path.join(depth_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    cache_model_outputs(
                        output_path,
                        outputs,
                        cur_data,
                        src_data,
                        batch_ind,
                        opts.batch_size,
                        predictions_to_save=[
                            "depth_pred_s0_b1hw",
                            "pred_0",
                            "pred_1",
                            "pred_2",
                            "pred_3",
                            "search_depths",
                            "rendered_texture",
                            "rendered_depth",
                        ],
                    )

            # compute a clean average
            scene_frame_metrics.compute_final_average(ignore_nans=True)

            # one scene counts as a complete unit of metrics
            all_scene_metrics.update_results(scene_frame_metrics.final_metrics)

            # print running metrics.
            scene_frame_metrics.output_json(
                os.path.join(scores_output_dir, f"{scan.replace('/', '_')}_metrics.json")
            )
            torch.cuda.empty_cache()

        # compute and print final average
        print("\nFinal metrics:")
        all_scene_metrics.compute_final_average(ignore_nans=True)
        all_scene_metrics.output_json(
            os.path.join(scores_output_dir, f"all_scene_avg_metrics_{opts.split}.json")
        )

        print("\n")
        all_frame_metrics.compute_final_average(ignore_nans=True)
        if opts.temporal_eval:
            temporal_d = -1
            total_diffs_key = f"total_diffs_d_{temporal_d:.1f}"
            temporal_key = f"temporal_score_d_{temporal_d:.1f}"

            all_frame_metrics.final_metrics[total_diffs_key] = temporal_evaluator.total_diffs
            all_frame_metrics.final_metrics[temporal_key] = temporal_evaluator.total_diffs / (
                (eval_length - warmup) * eval_frame_multiplier * len(scans)
            )

            all_frame_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                metric_name="total_diffs",
                depths=[temporal_d],
                single_iou=True,
            )

            all_frame_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                metric_name="temporal_score",
                depths=[temporal_d],
                single_iou=True,
            )

        if opts.binary_eval_depth:
            all_frame_metrics.print_sheets_friendly(print_running_metrics=False)
        else:
            all_frame_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                depths=depths_for_printing,
                single_iou=thresholder is not None,
            )

            all_frame_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                metric_name="surface_iou",
                depths=depths_for_printing,
                single_iou=thresholder is not None,
            )

            all_frame_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                metric_name="boundary_iou",
                depths=depths_for_printing,
                single_iou=thresholder is not None,
            )
        all_frame_metrics.output_json(
            os.path.join(scores_output_dir, f"all_frame_avg_metrics_{opts.split}.json")
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
