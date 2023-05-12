from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm import tqdm

import modules.cost_volume as cost_volume
import options
from experiment_modules.bd_model import BDModel
from modules.layers import sigmoid_custom
from utils.dataset_utils import get_dataset
from utils.generic_utils import to_gpu


@torch.inference_mode()
def main(opts):
    opts.batch_size = 1  # for this script we force batch size to be 1.

    # If opts.rendered_depth_map_load_dir is None, we don't load in pre-rendered
    # depth maps, but instead set the depth map as a plane fixed at 2.0m from the camera.
    # This is useful for visualising results on sequences without having to render an asset.
    if opts.rendered_depth_map_load_dir is None:
        asset_name = "plane_2.0"
    else:
        rendered_depth_map_load_dir = Path(opts.rendered_depth_map_load_dir)
        asset_name = "render"

    # get dataset
    dataset_class, scans = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    # path where results for this model, dataset, and tuple type are.
    pred_output_dir = Path(opts.output_base_path)
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {pred_output_dir}")
    print(f"Tuple file location: {opts.tuple_info_file_location}")
    print(f"Dataset: {opts.dataset}")
    print(f"Split: {opts.dataset_scan_split_file}")
    print(f"Scan id: {opts.single_debug_scan_id}")
    print(f"Path: {opts.dataset_path}")

    # Set up model. Note that we're not passing in opts as an argument, although
    # we could. We're being pretty stubborn with using the options the model had
    # used when training, saved internally as part of hparams in the checkpoint.
    # You can change this at inference by passing in 'opts=opts,' but there
    # be dragons if you're not careful.
    model = BDModel.load_from_checkpoint(opts.load_weights_from_checkpoint, args=None)

    if opts.fast_cost_volume and isinstance(model.cost_volume, cost_volume.FeatureVolumeManager):
        model.cost_volume = model.cost_volume.to_fast()

    model = model.cuda().eval()

    if model.run_opts.use_prior and opts.use_prior:
        print("########################################")
        print(f"using prior model with real priors; forcing batch_size to be 1")
        print("########################################")
        opts.batch_size = 1

    # loop over scans
    for scan in tqdm(scans, desc="Looping over scans"):
        scan_name = Path(scan).name

        print("Making implicit depth predictions for ", scan_name)

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
            include_high_res_color=False,
            include_full_depth_K=True,
            skip_frames=opts.skip_frames,
            image_width=opts.image_width,
            image_height=opts.image_height,
            pass_frame_id=True,
            get_bd_info=True,
        )

        if len(dataset) == 0:
            raise ValueError(f"Found 0 frames for dataset {scan_name} – please check your paths!")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=opts.num_workers,
            drop_last=False,
        )

        (pred_output_dir / asset_name / scan_name).mkdir(exist_ok=True, parents=True)

        if model.run_opts.use_prior and opts.use_prior:
            prev_pred = None
            prev_cam_T_world = None

        for batch_ind, (cur_data, src_data) in enumerate(tqdm(dataloader, desc=scan_name)):
            if opts.max_frames is not None and batch_ind >= opts.max_frames:
                print(f"Stopping early, as max_frames is {opts.max_frames}")
                break

            assert len(cur_data["frame_id_string"]) == 1, "Expect batch size to be 1 for infrence!"
            frame_idx = int(cur_data["frame_id_string"][0])

            if asset_name == "render":
                # load a rendered depth from disk
                rendered_depth = np.load(rendered_depth_map_load_dir / f"frame_{frame_idx:05d}.npy")
                rendered_depth_bchw = torch.Tensor(rendered_depth)[None, None, ...]
                # adding _padded per discussion with team
                rendered_depth_bchw_padded = F.max_pool2d(rendered_depth_bchw, 7, 1, 3).clone()
                rendered_depth_bchw[rendered_depth_bchw == 0] = rendered_depth_bchw_padded[
                    rendered_depth_bchw == 0
                ]
                # resize to the model input shape
                _, _, h, w = cur_data["rendered_depth"].shape
                cur_data["rendered_depth"] = resize(
                    rendered_depth_bchw, size=(h, w), interpolation=InterpolationMode.NEAREST
                )
            elif asset_name == "plane_2.0":
                # Make the input rendered depth a plane fixed at 2m from the camera
                cur_data["rendered_depth"] = cur_data["rendered_depth"][:, 0:1] * 0.0 + 2.0
            else:
                raise ValueError(f"Unknown asset name {asset_name}")

            # move data to GPU
            cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
            src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

            if model.run_opts.use_prior and opts.use_prior:
                cur_data["prior_prediction"] = prev_pred
                cur_data["prior_cam_T_world"] = prev_cam_T_world

            # use unbatched (looping) matching encoder image forward passes
            # for numerically stable testing. If opts.fast_cost_volume, then batch.
            outputs = model(
                "test",
                cur_data,
                src_data,
                unbatched_matching_encoder_forward=not opts.fast_cost_volume,
                return_mask=True,
                infer_depth=False,
                infer_res=None,
            )

            if model.run_opts.use_prior and opts.use_prior:
                prev_pred = sigmoid_custom(outputs["pred_0"], multiplier=1.0)
                prev_cam_T_world = cur_data["cam_T_world_b44"]

            pred_bdhw = sigmoid_custom(outputs["pred_0"], multiplier=1.0)
            pred_dhw = pred_bdhw.squeeze(0).detach().cpu().numpy().astype(np.float32)

            np.save(pred_output_dir / asset_name / scan_name / f"{frame_idx:05d}.npy", pred_dhw[0])

        del dataloader
        del dataset


if __name__ == "__main__":
    # don't need grad for test.
    torch.set_grad_enabled(False)

    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    # option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)
