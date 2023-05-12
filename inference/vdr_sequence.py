from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# By default we will pad images so that the numbers in the filenames have at least 5 digits
DEFAULT_NUM_PAD_DIGITS = 5


def pad_image_fname(fname: str, num_digits: int = DEFAULT_NUM_PAD_DIGITS) -> str:
    """
    Converts e.g. frame_25.jpg -> frame_0000000025.jpg, so filenames can be sorted
    """
    number = fname.lstrip("frame_").rstrip(".jpg")
    return f"frame_{number.zfill(num_digits)}.jpg"


@dataclass
class Pose:
    orientation: Rotation
    position: np.ndarray

    def __post_init__(self):
        assert self.position.shape == (3,)

    def as_transform(self) -> Transform:
        return Transform(rotation=self.orientation, translation=self.position)


@dataclass
class Transform:
    """
    A transform consisting of a rotation followed by a translation, defined in the active sense
    """

    rotation: Rotation
    translation: np.ndarray

    def __post_init__(self):
        assert self.translation.shape == (3,)

    def as_matrix(self) -> np.ndarray:
        transform_mat = np.zeros((4, 4))
        transform_mat[:3, :3] = self.rotation.as_matrix()
        transform_mat[:3, 3] = self.translation
        transform_mat[3, 3] = 1.0
        return transform_mat

    def invert(self) -> Transform:
        inv_rotation = self.rotation.inv()
        return Transform(rotation=inv_rotation, translation=-inv_rotation.apply(self.translation))


class VDRSequence:
    # This matrix converts from opengl (x right, y up, z back) to cv-style
    # (x right, y down, z forward) coordinates
    M = np.eye(4)
    M[1, 1] = -1
    M[2, 2] = -1

    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path / "capture.json") as f:
            self.capture = json.load(f)

        self.num_random_samples = 2000
        np.random.seed(10)
        self.x_samples_normalised = np.random.rand(self.num_random_samples)
        self.y_samples_normalised = np.random.rand(self.num_random_samples)

    @property
    def frames(self):
        return self.capture["frames"]

    def load_extrinsics_for_frame(self, frame: dict) -> Pose:
        extrinsics = np.asarray(frame["pose4x4"])
        extrinsics = np.reshape(extrinsics, [4, 4])
        extrinsics = np.transpose(extrinsics)
        extrinsics = np.matmul(np.matmul(self.M, extrinsics), self.M)

        camera_position = extrinsics[:3, 3]
        camera_orientation = Rotation.from_matrix(extrinsics[:3, :3])
        return Pose(orientation=camera_orientation, position=camera_position)

    @staticmethod
    def load_intrinsics_from_frame(frame: dict) -> Tuple[np.ndarray, Tuple]:
        intrinsics = np.array(frame["intrinsics"])
        K = np.eye(3)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]
        rgb_hw = frame["resolution"][::-1]
        return K, rgb_hw

    def load_rgb_from_frame(self, frame: dict) -> np.ndarray:
        assert (self.path / frame["image"]).is_file(), self.path / frame["image"]
        return cv2.imread(str(self.path / frame["image"]))[:, :, ::-1]

    def load_lidar_from_frame(self, frame: dict) -> np.ndarray:
        depth_wh = frame["depthResolution"]
        return np.fromfile(self.path / frame["depth"], dtype="float32").reshape(depth_wh[::-1])
