import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import modules.cost_volume as cost_volume
import options
from experiment_modules.depth_model import DepthModel
from utils.binary_metrics_utils import (
    PlaneEvaluator,
    TemporalEvaluator,
    get_boundary_mask,
    get_surface_mask,
)
from utils.dataset_utils import get_dataset
from utils.generic_utils import cache_model_outputs, to_gpu
from utils.metrics_utils import ResultsAverager, compute_depth_metrics_batched
from utils.visualization_utils import quick_viz_export


def main(opts):
    # get dataset
    dataset_class, scans = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(
        opts.output_base_path, opts.name, opts.dataset, opts.frame_tuple_type
    )

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

    # set up directories for quick depth visualizations
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
        results_path, "iou_scores" if opts.regression_plane_eval else "scores"
    )
    Path(scores_output_dir).mkdir(parents=True, exist_ok=True)

    # Set up model. Note that we're not passing in opts as an argument, although
    # we could. We're being pretty stubborn with using the options the model had
    # used when training, saved internally as part of hparams in the checkpoint.
    # You can change this at inference by passing in 'opts=opts,' but there
    # be dragons if you're not careful.

    model = DepthModel.load_from_checkpoint(opts.load_weights_from_checkpoint, args=None)
    if opts.fast_cost_volume and isinstance(model.cost_volume, cost_volume.FeatureVolumeManager):
        model.cost_volume = model.cost_volume.to_fast()

    model = model.cuda().eval()

    all_frame_metrics = ResultsAverager(opts.name, f"frame metrics")
    all_scene_metrics = ResultsAverager(opts.name, f"scene metrics")

    if opts.regression_plane_eval:
        depths_for_printing = [1.5 + x * 0.5 for x in range(8)]
        plane_evaluator = PlaneEvaluator()

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
                get_bd_info=opts.regression_plane_eval,
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

            eval_frame_count = 0
            for batch_ind, batch in enumerate(tqdm(dataloader)):
                # get data, move to GPU
                cur_data, src_data = batch

                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string", "dataset_name"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string", "dataset_name"])

                depth_gt = cur_data["full_res_depth_b1hw"]

                if opts.temporal_eval:
                    if batch_ind % eval_length == 0:
                        temporal_evaluator.initialise_new_plane(
                            depth_gt, cur_data["world_T_cam_b44"]
                        )
                        eval_frame_count = 0
                    rendered_depth = temporal_evaluator.rasterizer(
                        cur_data["cam_T_world_b44"], cur_data["K_s0_b44"]
                    )
                    cur_data["rendered_depth"] = rendered_depth

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
                )
                end_time.record()
                torch.cuda.synchronize()

                elapsed_model_time = start_time.elapsed_time(end_time)

                eval_frame_count += 1
                if opts.temporal_eval and eval_frame_count < warmup + 1:
                    continue

                if opts.temporal_eval:
                    pred = (
                        cur_data["rendered_depth"] < outputs["depth_pred_s0_b1hw"]
                    ).float() + 0.1
                    temporal_evaluator.update_vertex_predictions(
                        pred, cur_data["cam_T_world_b44"], cur_data["K_s0_b44"]
                    )

                    if batch_ind % (eval_length - 1) == 0:
                        temporal_evaluator.compute_vertex_occlusion_changes()

                upsampled_depth_pred_b1hw = F.interpolate(
                    outputs["depth_pred_s0_b1hw"],
                    size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                    mode="nearest" if opts.temporal_eval else "bilinear",
                )

                # inf max depth matches DVMVS metrics, using minimum of 0.5m
                thresh_to_check = 0.0 if opts.regression_plane_eval else 0.5
                valid_mask_b = cur_data["full_res_depth_b1hw"] > thresh_to_check

                # Check if there any valid gt points in this sample
                if (valid_mask_b).any():
                    # compute metrics
                    if opts.regression_plane_eval:
                        # get surface mask
                        surface_mask_bdhw = get_surface_mask(
                            cur_data["depth_b1hw"], cur_data["rendered_depth"]
                        )

                        # get boundary mask
                        boundary_mask_bdhw = get_boundary_mask(
                            cur_data["depth_b1hw"], cur_data["rendered_depth"]
                        )

                        upsampled_query_bdhw = F.interpolate(
                            cur_data["rendered_depth"],
                            size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                            mode="nearest",
                        )

                        boundary_query_bdhw = cur_data["rendered_depth"].clone()
                        boundary_query_bdhw[~boundary_mask_bdhw.bool()] = -1
                        boundary_query_bdhw = F.interpolate(
                            boundary_query_bdhw,
                            size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                            mode="nearest",
                        )

                        surface_query_bdhw = cur_data["rendered_depth"].clone()
                        surface_query_bdhw[~surface_mask_bdhw.bool()] = -1
                        surface_query_bdhw = F.interpolate(
                            surface_query_bdhw,
                            size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                            mode="nearest",
                        )

                        metrics_b_dict = plane_evaluator.compute_regressed_depth_batch_scores(
                            query_depth_bdhw=upsampled_query_bdhw,
                            gt_depth_b1hw=depth_gt,
                            prediction_b1hw=upsampled_depth_pred_b1hw,
                            is_rendering=opts.temporal_eval,
                        )
                        # surfaces evaluation
                        metrics_b_dict.update(
                            plane_evaluator.compute_regressed_depth_batch_scores(
                                query_depth_bdhw=surface_query_bdhw,
                                gt_depth_b1hw=depth_gt,
                                prediction_b1hw=upsampled_depth_pred_b1hw,
                                is_rendering=opts.temporal_eval,
                                tag="surface",
                            )
                        )

                        # surfaces evaluation
                        metrics_b_dict.update(
                            plane_evaluator.compute_regressed_depth_batch_scores(
                                query_depth_bdhw=boundary_query_bdhw,
                                gt_depth_b1hw=depth_gt,
                                prediction_b1hw=upsampled_depth_pred_b1hw,
                                is_rendering=opts.temporal_eval,
                                tag="boundary",
                            )
                        )
                    else:
                        metrics_b_dict = compute_depth_metrics_batched(
                            depth_gt.flatten(start_dim=1).float(),
                            upsampled_depth_pred_b1hw.flatten(start_dim=1).float(),
                            valid_mask_b.flatten(start_dim=1),
                            mult_a=True,
                        )

                    # go over batch and get metrics frame by frame to update
                    # the averagers
                    for element_index in range(depth_gt.shape[0]):
                        if (~valid_mask_b[element_index]).all():
                            # ignore if no valid gt exists
                            continue

                        element_metrics = {}
                        for key in list(metrics_b_dict.keys()):
                            element_metrics[key] = metrics_b_dict[key][element_index]

                        # get per frame time in the batch
                        element_metrics["model_time"] = elapsed_model_time / depth_gt.shape[0]

                        # both this scene and all frame averagers
                        scene_frame_metrics.update_results(element_metrics)
                        all_frame_metrics.update_results(element_metrics)

                ########################### Quick Viz ##########################
                if opts.dump_depth_visualization:
                    # make a dir for this scan
                    output_path = os.path.join(viz_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    quick_viz_export(
                        output_path,
                        outputs,
                        cur_data,
                        batch_ind,
                        valid_mask_b,
                        opts.batch_size,
                        save_depth_overlay=True,
                    )
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
                        predictions_to_save=["depth_pred_s0_b1hw", "rendered_depth"]
                        if opts.skinny_cache_dump
                        else None,
                    )

            # compute a clean average
            scene_frame_metrics.compute_final_average(ignore_nans=True)

            # one scene counts as a complete unit of metrics
            all_scene_metrics.update_results(scene_frame_metrics.final_metrics)

            # print running metrics.
            # print("\nScene metrics:")
            # scene_frame_metrics.print_sheets_friendly(include_metrics_names=True)
            scene_frame_metrics.output_json(
                os.path.join(scores_output_dir, f"{scan.replace('/', '_')}_metrics.json")
            )

            torch.cuda.empty_cache()

            # print running metrics.
            print("\nRunning frame metrics:")

            if opts.regression_plane_eval:
                all_frame_metrics.compute_final_average(ignore_nans=True)
                all_frame_metrics.pretty_print_metric_table(
                    print_running_metrics=False, single_iou=True, depths=depths_for_printing
                )
            else:
                all_frame_metrics.print_sheets_friendly(
                    include_metrics_names=False,
                    print_running_metrics=True,
                )

        # compute and print final average
        print("\nFinal metrics:\n")

        print("Scene metrics:")
        all_scene_metrics.compute_final_average(ignore_nans=True)

        if opts.regression_plane_eval:
            all_scene_metrics.pretty_print_metric_table(
                print_running_metrics=False, single_iou=True, depths=depths_for_printing
            )
            all_scene_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                metric_name="iou_neg",
                single_iou=True,
                depths=depths_for_printing,
            )
            all_scene_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                metric_name="iou_pos",
                single_iou=True,
                depths=depths_for_printing,
            )

        else:
            all_scene_metrics.print_sheets_friendly(
                include_metrics_names=True, print_running_metrics=False
            )
        all_scene_metrics.output_json(
            os.path.join(scores_output_dir, f"all_scene_avg_metrics_{opts.split}.json")
        )

        print("\n\n\n\n\nFrame metrics:")
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

        if opts.regression_plane_eval:
            all_frame_metrics.pretty_print_metric_table(
                print_running_metrics=False, single_iou=True, depths=depths_for_printing
            )

            all_frame_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                metric_name="surface_iou",
                single_iou=True,
                depths=depths_for_printing,
            )

            all_frame_metrics.pretty_print_metric_table(
                print_running_metrics=False,
                metric_name="boundary_iou",
                single_iou=True,
                depths=depths_for_printing,
            )
        else:
            all_frame_metrics.print_sheets_friendly(
                include_metrics_names=True, print_running_metrics=False
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
