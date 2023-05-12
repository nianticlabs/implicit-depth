import argparse
import pickle
import subprocess as sp
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .vdr_sequence import VDRSequence, pad_image_fname

VIDEO_MP4_NAME = "composited.mp4"
DEPTH_ALPHA_BAND_SIZE = 0.2  # in metres
FADE_IN_FRAMES = 45


def get_mask(predicted, virtual, soft: bool):
    if soft:
        mask = (1 / DEPTH_ALPHA_BAND_SIZE) * (predicted - virtual + DEPTH_ALPHA_BAND_SIZE / 2)
        return np.clip(mask, 0.0, 1.0)
    else:
        return (predicted > virtual).astype(np.float32)


def determine_method(
    predicted_masks_dir: Optional[Path], predicted_depths_dir: Optional[Path]
) -> str:
    if predicted_depths_dir is not None and predicted_masks_dir is not None:
        raise ValueError(
            "Expected either --predicted-depths-dir or --predicted-masks-dir to be given,"
            " or neither, but not both."
        )

    if predicted_depths_dir is not None:
        return "predicted_depth"
    elif predicted_masks_dir is not None:
        return "mask"
    else:
        return "lidar"


def composite(
    vdr_dir: Path,
    output_dir: Path,
    save_img_extension: str = ".jpg",
    fadein: bool = False,
    use_depth_banding=True,  # if set, uses alpha value for depth methods
    predicted_depths_dir: Optional[Path] = None,
    predicted_masks_dir: Optional[Path] = None,
    virtual_depth: Optional[float] = None,
    rendered_rgb_dir: Optional[Path] = None,
    limit_frames: Optional[int] = None,
) -> None:
    print(f"Running on {vdr_dir}")

    output_dir.mkdir(exist_ok=True, parents=True)
    sequence = VDRSequence(vdr_dir)

    method = determine_method(
        predicted_depths_dir=predicted_depths_dir, predicted_masks_dir=predicted_masks_dir
    )
    print(f"Compositing using {method}")

    for frame_idx, frame in enumerate(tqdm(sequence.frames)):
        if frame_idx == 0:
            # skip the 0th frame as some methods don't make predictions for this
            continue

        if limit_frames is not None and frame_idx >= limit_frames:
            print(f"Stopping early as limit_frames is {limit_frames}")
            break

        w, h = frame["resolution"]
        im = sequence.load_rgb_from_frame(frame) / 255.0

        padded_image_name = pad_image_fname(frame["image"])

        if rendered_rgb_dir is not None:
            virtual_im_path = (rendered_rgb_dir / padded_image_name).with_suffix(".png")
            virtual_rgba = np.array(Image.open(virtual_im_path)).astype(np.float32) / 255.0
            virtual_rgb = virtual_rgba[:, :, :3]
            valid_virtual_pixels = virtual_rgba[:, :, 3]
        else:
            virtual_rgb = np.zeros((h, w, 3))
            virtual_rgb[:, :, 0] = 0.30
            virtual_rgb[:, :, 1] = 0.9
            virtual_rgb[:, :, 2] = 0.78
            valid_virtual_pixels = np.ones_like(virtual_rgb[:, :, 0])

        if fadein and frame_idx < FADE_IN_FRAMES:
            fade_amount = frame_idx / FADE_IN_FRAMES
            valid_virtual_pixels *= fade_amount

        if method == "mask":
            # load matte directly
            raw_matte = np.load(
                (predicted_masks_dir / padded_image_name.lstrip("frame_")).with_suffix(".npy")
            )
            matte = cv2.resize(raw_matte, (w, h), cv2.INTER_LINEAR)
            matte = 1.0 - matte * valid_virtual_pixels.astype(np.float32)

        elif method in ("predicted_depth", "lidar"):
            if method == "lidar":
                depth = sequence.load_lidar_from_frame(frame)
            else:
                # Regression baselines
                with open(
                    (predicted_depths_dir / frame["image"].lstrip("frame_")).with_suffix(".pickle"),
                    "rb",
                ) as f:
                    depth = pickle.load(f)
                depth = depth["depth_pred_s0_b1hw"][0, 0].cpu().numpy()

            # Resize the depth if needed
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

            if rendered_rgb_dir is not None:
                virtual_depthmap_path = (rendered_rgb_dir / padded_image_name).with_suffix(".npy")
                virtual_depthmap = np.load(virtual_depthmap_path)
                valid_virtual_pixels = (virtual_depthmap > 0).astype(np.float32)
                if fadein and frame_idx < FADE_IN_FRAMES:
                    fade_amount = frame_idx / FADE_IN_FRAMES
                    valid_virtual_pixels *= fade_amount

                matte = get_mask(predicted=depth, virtual=virtual_depthmap, soft=use_depth_banding)
                matte = 1.0 - matte * valid_virtual_pixels
            else:
                virtual_depthmap = np.ones((h, w)) * virtual_depth
                matte = 1.0 - get_mask(
                    predicted=depth, virtual=virtual_depthmap, soft=use_depth_banding
                )

        # composite and save
        matte = matte[:, :, None].astype(np.float32)
        composited = matte * im + (1 - matte) * virtual_rgb

        cv2.imwrite(
            str((output_dir / padded_image_name).with_suffix(save_img_extension)),
            (composited * 255.0).astype(np.uint8)[:, :, ::-1],
        )

    print(f"Saving final video to {output_dir / VIDEO_MP4_NAME}")
    sp.call(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",  # make ffmpeg quiet
            "-pattern_type",
            "glob",
            "-i",
            output_dir / f"*{save_img_extension}",
            output_dir / VIDEO_MP4_NAME,
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Composites an AR asset into a VDR sequence, using some pre-computed depth "
        "or compositing masks."
    )
    parser.add_argument(
        "--predicted-depths-dir",
        required=False,
        default=None,
        help="Directory in which predicted depths should be loaded from",
    )
    parser.add_argument(
        "--predicted-masks-dir",
        required=False,
        default=None,
        help="Directory in which predicted masks should be loaded from",
    )
    parser.add_argument(
        "--vdr-dir",
        required=True,
        help="Path to directory of images of the real scene. "
        "This directory is also assumed to contain a capture.json file containing intrinsics"
        " and extrinsics; see README for a link to an example file.",
    )
    parser.add_argument(
        "--renders-dir",
        required=False,
        default=None,
        help="Path to directory of virtual object renders. "
        "If not provided, a flat plane will be used.",
    )
    parser.add_argument(
        "--out-dir", required=True, help="directory in which to save composited images"
    )
    parser.add_argument(
        "--limit-frames",
        required=False,
        type=int,
        default=None,
        help="Optionally stop processing after this many frames; useful for debugging.",
    )
    args = parser.parse_args()

    composite(
        vdr_dir=Path(args.vdr_dir),
        predicted_depths_dir=Path(args.predicted_depths_dir) if args.predicted_depths_dir else None,
        predicted_masks_dir=Path(args.predicted_masks_dir) if args.predicted_masks_dir else None,
        output_dir=Path(args.out_dir),
        rendered_rgb_dir=Path(args.renders_dir) if args.renders_dir else None,
        virtual_depth=2.0,
        limit_frames=args.limit_frames,
    )


if __name__ == "__main__":
    main()
