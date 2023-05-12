import os
import struct
import zlib
from contextlib import contextmanager

import cv2
import imageio
import numpy as np
import png
from PIL import Image

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


@contextmanager
def print_array_on_one_line():
    oldoptions = np.get_printoptions()
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(linewidth=np.inf)
    yield
    np.set_printoptions(**oldoptions)


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack("c" * self.color_size_bytes, file_handle.read(self.color_size_bytes))
        )
        self.depth_data = b"".join(
            struct.unpack("c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes))
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def dump_color_to_file(self, compression_type, filepath):
        if compression_type == "jpeg":
            filepath += ".jpg"
        else:
            raise
        f = open(filepath, "wb")
        f.write(self.color_data)
        f.close()

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            self.num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(self.num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)
            self.num_IMU_frames = struct.unpack("Q", f.read(8))[0]

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting", len(self.frames), frame_skip, " depth frames to", output_path)
        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )

            if image_size is not None:
                depth = cv2.resize(
                    depth, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST
                )
                filepath = os.path.join(
                    output_path, f"frame-{f:06d}.depth.{int(image_size[0])}.png"
                )
            else:
                filepath = os.path.join(output_path, f"frame-{f:06d}.depth.png")

            with open(filepath, "wb") as f:  # write 16-bit
                writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
                depth = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(f, depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting", len(self.frames), frame_skip, "color frames to", output_path)
        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f].decompress_color(self.color_compression_type)

            if image_size is not None:
                resized = Image.fromarray(color).resize(
                    (image_size[0], image_size[1]), resample=Image.BILINEAR
                )
                filepath = os.path.join(
                    output_path, f"frame-{f:06d}.color.{int(image_size[0])}.png"
                )
                resized.save(filepath)
            else:
                filepath = os.path.join(output_path, f"frame-{f:06d}.color")
                self.frames[f].dump_color_to_file(self.color_compression_type, filepath)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting", len(self.frames), frame_skip, "camera poses to", output_path)
        for f in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(
                self.frames[f].camera_to_world, os.path.join(output_path, f"frame-{f:06d}.pose.txt")
            )

    def export_intrinsics(self, output_path, scan_name):
        default_intrinsics_path = os.path.join(output_path, "intrinsic")
        if not os.path.exists(default_intrinsics_path):
            os.makedirs(default_intrinsics_path)
        print("exporting camera intrinsics to", default_intrinsics_path)
        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(default_intrinsics_path, "intrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(default_intrinsics_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.intrinsic_depth, os.path.join(default_intrinsics_path, "intrinsic_depth.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(default_intrinsics_path, "extrinsic_depth.txt")
        )
