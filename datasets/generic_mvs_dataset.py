import logging
import os
import random
from multiprocessing.dummy import Value
from pkgutil import get_data
from webbrowser import get

import numpy as np
import PIL.Image as pil
import torch
import torch.nn.functional as F
from kornia.filters import sobel
from torch.utils.data import Dataset
from torchvision import transforms

from utils.generic_utils import imagenet_normalize, read_image_file, readlines
from utils.geometry_utils import pose_distance

logger = logging.getLogger(__name__)


class GenericMVSDataset(Dataset):
    """
    Generic MVS dataset class for SimpleRecon. This class can be used as a base
    for different multi-view datasets.

    It houses the main __getitem__ function that will assemble a tuple of imgaes
    and their data.

    Tuples are read from a tuple file defined as
        tuple_info_file_location/{split}{mv_tuple_file_suffix}

    Each line in the tuple file should contain a scene id and frame ids for each
    frame in the tuple:

        scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    where frame_id_0 is the reference image.

    These will be loaded and stored in self.frame_tuples.

    If no tuple file suffix is provided, the dataset will only allow basic frame
    data loading from the split.

    Datasets that use this base class as a parent should modify base file load
    functions that do not have an implementation below.

    """

    def __init__(
        self,
        dataset_path,
        split,
        mv_tuple_file_suffix,
        tuple_info_file_location=None,
        limit_to_scan_id=None,
        num_images_in_tuple=None,
        image_height=384,
        image_width=512,
        include_high_res_color=False,
        high_res_image_width=640,
        high_res_image_height=480,
        image_depth_ratio=2,
        include_full_res_depth=False,
        include_full_depth_K=False,
        color_transform=transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        shuffle_tuple=False,
        pass_frame_id=False,
        skip_frames=None,
        verbose_init=True,
        native_depth_width=640,
        native_depth_height=480,
        image_resampling_mode=pil.BILINEAR,
        get_bd_info=False,
        num_rays=4096,
        samples_per_ray=64,
        near_surface_sampling=True,
        near_surface_ratio=0.75,
        full_depth_supervision=False,
        near_edge_sampling=False,
        near_edge_ratio=0.4,
        surface_noise_type="additive",
    ):
        """
        Args:
            dataset_path: base path to the dataaset directory.
            split: the dataset split.
            mv_tuple_file_suffix: a suffix for the tuple file's name. The
                tuple filename searched for wil be
                {split}{mv_tuple_file_suffix}.
            tuple_info_file_location: location to search for a tuple file, if
                None provided, will search in the dataset directory under
                'tuples'.
            limit_to_scan_id: limit loaded tuples to one scan's frames.
            num_images_in_tuple: optional integer to limit tuples to this number
                of images.
            image_height, image_width: size images should be loaded at/resized
                to.
            include_high_res_color: should the dataset pass back higher
                resolution images.
            high_res_image_height, high_res_image_width: resolution images
                should be resized if we're passing back higher resolution
                images.
            image_depth_ratio: returned gt depth maps "depth_b1hw" will be of
                size (image_height, image_width)/image_depth_ratio.
            include_full_res_depth: if true will return depth maps from the
                dataset at the highest resolution available.
            color_transform: optional color transform that applies when split is
                "train".
            shuffle_tuple: by default source images will be ordered according to
                overall pose distance to the reference image. When this flag is
                true, source images will be shuffled. Only used for ablation.
            pass_frame_id: if we should return the frame_id as part of the item
                dict
            skip_frames: if not none, will stride the tuple list by this value.
                Useful for only fusing every 'skip_frames' frame when fusing
                depth.
            verbose_init: if True will let the init print details on the
                initialization.
            native_depth_width, native_depth_height: for some datasets, it's
                useful to know what the native depth resolution is in advance.
            image_resampling_mode: resampling method for resizing images.

        """
        super(GenericMVSDataset).__init__()

        self.split = split
        scan_folder = self.get_sub_folder_dir(split)

        self.dataset_path = dataset_path
        self.scenes_path = os.path.join(dataset_path, scan_folder)

        self.mv_tuple_file_suffix = mv_tuple_file_suffix
        self.num_images_in_tuple = num_images_in_tuple
        self.shuffle_tuple = shuffle_tuple

        # default to where the dataset is to look for a tuple file
        if tuple_info_file_location is None:
            tuple_info_file_location = os.path.join(dataset_path, "tuples")

        if mv_tuple_file_suffix is not None:
            # tuple info should be available
            tuple_information_filepath = os.path.join(
                tuple_info_file_location, f"{split}{mv_tuple_file_suffix}"
            )

            # check if this file exists
            assert os.path.exists(tuple_information_filepath), (
                "Tuple file "
                "doesn't exist! Pass none for mv_tuple_file_suffix if you don't"
                " actually need a tuple file, otherwise check your paths."
            )

            # read in those tuples
            self.frame_tuples = readlines(tuple_information_filepath)

            # optionally limit frames to just one scan.
            if limit_to_scan_id is not None:
                self.frame_tuples = [
                    frame_tuple
                    for frame_tuple in self.frame_tuples
                    if limit_to_scan_id == frame_tuple.split(" ")[0]
                ]

            # optionally skip every frame with interval skip_frame
            if skip_frames is not None:
                if verbose_init:
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f" Skipping every {skip_frames} ".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"), "\n")
                self.frame_tuples = self.frame_tuples[::skip_frames]
        else:
            if verbose_init:
                print(f"".center(80, "#"))
                print(
                    f" tuple_information_filepath isn't provided."
                    "Only basic dataloader functions are available. ".center(80, "#")
                )
                print(f"".center(80, "#"), "\n")

        self.color_transform = color_transform

        self.image_width = image_width
        self.image_height = image_height
        self.high_res_image_width = high_res_image_width
        self.high_res_image_height = high_res_image_height

        # size up depth using ratio of RGB to depth
        self.depth_height = self.image_height // image_depth_ratio
        self.depth_width = self.image_width // image_depth_ratio

        self.native_depth_width = native_depth_width
        self.native_depth_height = native_depth_height

        self.include_full_depth_K = include_full_depth_K
        self.include_high_res_color = include_high_res_color
        self.include_full_res_depth = include_full_res_depth

        self.pass_frame_id = pass_frame_id

        self.disable_resize_warning = False
        self.image_resampling_mode = image_resampling_mode

        self.get_bd_info = get_bd_info
        self.full_depth_supervision = full_depth_supervision
        self.near_surface_sampling = near_surface_sampling if near_surface_sampling else 0.0
        self.near_surface_ratio = near_surface_ratio
        self.near_edge_sampling = near_edge_sampling
        self.near_edge_ratio = near_edge_ratio
        self.surface_noise_type = surface_noise_type
        if self.get_bd_info:
            self.num_rays = num_rays
            random_samples_per_ray = int(samples_per_ray * (1 - self.near_surface_ratio))
            surface_samples_per_ray = samples_per_ray - random_samples_per_ray
            self.random_samples_per_ray = random_samples_per_ray
            self.surface_samples_per_ray = surface_samples_per_ray

            # set up for sampling
            if self.full_depth_supervision:
                xs, ys = np.meshgrid(
                    np.arange(self.native_depth_width) + 0.5,
                    np.arange(self.native_depth_height) + 0.5,
                )
            else:
                xs, ys = np.meshgrid(
                    np.arange(self.depth_width) + 0.5, np.arange(self.depth_height) + 0.5
                )

            self.sampling_grid = np.stack((xs, ys), -1)
            self.sampling_grid = torch.tensor(self.sampling_grid).float().reshape(-1, 2)

            self.ray_samples_Nd = (
                torch.linspace(0, 1.0, self.random_samples_per_ray)
                .unsqueeze(0)
                .expand(self.num_rays, -1)
            )

            self.validation_planes = torch.linspace(1.5, 5.0, 8).float().view(8, 1, 1)

    def __len__(self):
        return len(self.frame_tuples)

    @staticmethod
    def get_sub_folder_dir(split):
        """Where scans are for each split."""
        return ""

    def get_valid_frame_path(self, split, scan):
        """returns the filepath of a file that contains valid frame ids for a
        scan."""

        raise NotImplementedError()

    def get_valid_frame_ids(self, split, scan, store_computed=True):
        """Either loads or computes the ids of valid frames in the dataset for
        a scan.

        A valid frame is one that has an existing RGB frame, an existing
        depth file, and existing pose file where the pose isn't inf, -inf,
        or nan.

        Args:
            split: the data split (train/val/test)
            scan: the name of the scan
            store_computed: store the valid_frame file where we'd expect to
            see the file in the scan folder. get_valid_frame_path defines
            where this file is expected to be. If the file can't be saved,
            a warning will be printed and the exception reason printed.

        Returns:
            valid_frames: a list of strings with info on valid frames.
            Each string is a concat of the scan_id and the frame_id.
        """
        raise NotImplementedError()

    def get_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's color file at the dataset's
        configured RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached RGB file at the size
            required, or if that doesn't exist, the full size RGB frame
            from the dataset.

        """
        raise NotImplementedError()

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's higher res color file at the
        dataset's configured high RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached RGB file at the high res
            size required, or if that doesn't exist, the full size RGB frame
            from the dataset.

        """

        raise NotImplementedError()

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the dataset's
        configured depth resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for a precached depth file at the size
            required.

        """
        raise NotImplementedError()

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the native
        resolution in the dataset.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached depth file at the size
            required, or if that doesn't exist, the full size depth frame
            from the dataset.

        """
        raise NotImplementedError()

    def get_pose_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for pose information.

        """
        raise NotImplementedError()

    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame
        within the scan.

        This string is what this dataset uses as a reference to store files
        on disk.
        """
        raise NotImplementedError()

    def get_gt_mesh_path(dataset_path, split, scan_id):
        """
        Returns a path to a gt mesh reconstruction file.
        """
        raise NotImplementedError()

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame. Not needed for ScanNet as images
            share intrinsics across a scene.

        Returns:
            output_dict: A dict with
                - K_s{i}_b44 (intrinsics) and invK_s{i}_b44
                (backprojection) where i in [0,1,2,3,4]. i=0 provides
                intrinsics at the scale for depth_b1hw.
                - K_full_depth_b44 and invK_full_depth_b44 provides
                intrinsics for the maximum available depth resolution.
                Only provided when include_full_res_depth is true.

        """
        raise NotImplementedError()

    def load_target_size_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the resolution the dataset is configured for.

        Internally, if the loaded depth map isn't at the target resolution,
        the depth map will be resized on-the-fly to meet that resolution.

        NOTE: This function will place NaNs where depth maps are invalid.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            depth: depth map at the right resolution. Will contain NaNs
                where depth values are invalid.
            mask: a float validity mask for the depth maps. (1.0 where depth
            is valid).
            mask_b: like mask but boolean.
        """
        raise NotImplementedError()

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the native resolution the dataset provides.

        NOTE: This function will place NaNs where depth maps are invalid.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            full_res_depth: depth map at the right resolution. Will contain
                NaNs where depth values are invalid.
            full_res_mask: a float validity mask for the depth maps. (1.0
            where depth is valid).
            full_res_mask_b: like mask but boolean.
        """
        raise NotImplementedError()

    def load_pose(self, scan_id, frame_id):
        """Loads a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            world_T_cam (numpy array): matrix for transforming from the
                camera to the world (pose).
            cam_T_world (numpy array): matrix for transforming from the
                world to the camera (extrinsics).

        """
        raise NotImplementedError()

    def load_color(self, scan_id, frame_id):
        """Loads a frame's RGB file, resizes it to configured RGB size.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at self.image_height and
            self.image_width resolution.

        """

        color_filepath = self.get_color_filepath(scan_id, frame_id)
        image = read_image_file(
            color_filepath,
            height=self.image_height,
            width=self.image_width,
            resampling_mode=self.image_resampling_mode,
            disable_warning=self.disable_resize_warning,
        )

        return image

    def load_high_res_color(self, scan_id, frame_id):
        """Loads a frame's RGB file at a high resolution as configured.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at
            self.high_res_image_height and self.high_res_image_width
            resolution.

        """

        color_high_res_filepath = self.get_high_res_color_filepath(scan_id, frame_id)
        high_res_color = read_image_file(
            color_high_res_filepath,
            height=self.high_res_image_height,
            width=self.high_res_image_width,
            resampling_mode=self.image_resampling_mode,
            disable_warning=self.disable_resize_warning,
        )

        return high_res_color

    def get_frame(self, scan_id, frame_id, load_depth, get_bd_info=False, flip=False):
        """Retrieves a single frame's worth of information.

        NOTE: Returned depth maps will use NaN for values where the depth
        map is invalid.

        Args:
            scan_id: a string defining the scan this frame belongs to.
            frame_id: an integer id for this frame.
            load_depth: a bool flag for loading depth maps and not dummy
                data
            get_bd_info: return binary depth samples
        Returns:
            output_dict: a dictionary with this frame's information,
            including:
             - image_b3hw: an imagenet normalized RGB tensor of the image,
                resized to [self.image_height, self.image_width].
             - depth_b1hw: groundtruth depth map for this frame tensor,
                resized to [self.depth_height, self.depth_width.]
             - mask_b1hw: valid float mask where 1.0 indicates a valid depth
                value in depth_b1hw.
             - mask_b_b1hw: like mask_b1hw but binary.
             - world_T_cam_b44: transform for transforming points from
                camera to world coordinates. (pose)
             - cam_T_world_b44: transform for transforming points from world
                to camera coordinaetes. (extrinsics)
             - intrinsics: a dictionary with intrinsics at various
                resolutions and their inverses. Includes:
                    - K_s{i}_b44 (intrinsics) and invK_s{i}_b44
                    (backprojection) where i in [0,1,2,3,4]. i=0 provides
                    intrinsics at the scale for depth_b1hw.
                    - K_full_depth_b44 and invK_full_depth_b44 provides
                    intrinsics for the maximum available depth resolution.
                    Only provided when include_full_res_depth is true.
             - frame_id_string: a string that uniquly identifies the frame
                as it is on disk in its filename. Provided when
                pass_frame_id is true.
             - high_res_color_b3hw: an imagenet normalized RGB tensor of the
                image, at 640 (w) by 480 (h) resolution.
                Provided when include_high_res_color is true.
             - full_res_depth_b1hw: highest resolution depth map available.
                Will only be available if include_full_res_depth is true.
                Provided when include_full_res_depth is true.
             - full_res_mask_b1hw: valid float mask where 1.0 indicates a
                valid depth value in full_res_depth_b1hw.
                Provided when include_full_res_depth is true.
             - full_res_mask_b_b1hw: like full_res_mask_b1hw but binary.
             - min_depth: minimum depth in the gt
             - max_depth: maximum depth value in the gt

        """
        # stores output
        output_dict = {}

        # load pose
        world_T_cam, cam_T_world = self.load_pose(scan_id, frame_id)

        if flip:
            T = np.eye(4).astype(world_T_cam.dtype)
            T[0, 0] = -1.0
            world_T_cam = world_T_cam @ T
            cam_T_world = np.linalg.inv(world_T_cam)

        # Load image
        image = self.load_color(scan_id, frame_id)

        # Augment images
        if self.split == "train":
            image = self.color_transform(image)

        if flip:
            image = torch.flip(image, (-1,))

        # Do imagenet normalization
        image = imagenet_normalize(image)

        output_dict.update(
            {
                "image_b3hw": image,
                "world_T_cam_b44": world_T_cam,
                "cam_T_world_b44": cam_T_world,
            }
        )

        # load intrinsics
        intrinsics = self.load_intrinsics(scan_id, frame_id, flip=flip)

        output_dict.update(intrinsics)

        if load_depth:
            # get depth
            depth, mask, mask_b = self.load_target_size_depth_and_mask(scan_id, frame_id)

            if flip:
                depth = torch.flip(depth, (-1,))
                mask = torch.flip(mask, (-1,))
                mask_b = torch.flip(mask_b, (-1,))

            output_dict.update(
                {
                    "depth_b1hw": depth,
                    "mask_b1hw": mask,
                    "mask_b_b1hw": mask_b,
                }
            )

        # Load high res image
        if self.include_high_res_color:
            high_res_color = self.load_high_res_color(scan_id, frame_id)
            high_res_color = imagenet_normalize(high_res_color)

            if flip:
                high_res_color = torch.flip(high_res_color, (-1,))

            output_dict.update(
                {
                    "high_res_color_b3hw": high_res_color,
                }
            )

        if self.include_full_res_depth:
            # get high res depth
            full_res_depth, full_res_mask, full_res_mask_b = self.load_full_res_depth_and_mask(
                scan_id, frame_id
            )

            if flip:
                full_res_depth = torch.flip(full_res_depth, (-1,))
                full_res_mask = torch.flip(full_res_mask, (-1,))
                full_res_mask_b = torch.flip(full_res_mask_b, (-1,))

            output_dict.update(
                {
                    "full_res_depth_b1hw": full_res_depth,
                    "full_res_mask_b1hw": full_res_mask,
                    "full_res_mask_b_b1hw": full_res_mask_b,
                }
            )

        if self.pass_frame_id:
            output_dict["frame_id_string"] = self.get_frame_id_string(frame_id)

        if get_bd_info:
            if self.full_depth_supervision:
                output_dict.update(
                    self.generate_depth_samples(depth_1hw=full_res_depth, mask_1hw=full_res_mask_b)
                )
            else:
                output_dict.update(self.generate_depth_samples(depth_1hw=depth, mask_1hw=mask_b))

        return output_dict

    @staticmethod
    def get_edge_mask(depth_1hw, threshold=0.975, dilate=False):
        # sobel on disparity to care more about closer stuff
        sobel_map_11hw = sobel(1 / depth_1hw.unsqueeze(0))
        # get threshold, ignoring nans
        edge_threshold = torch.quantile(sobel_map_11hw[sobel_map_11hw == sobel_map_11hw], threshold)
        edge_map_11hw = sobel_map_11hw > edge_threshold
        if dilate:
            edge_map_11hw = F.max_pool2d(edge_map_11hw.float(), 5, 1, 2).bool()
        return edge_map_11hw.squeeze(0)

    def generate_depth_samples(self, depth_1hw, mask_1hw):
        output_dict = {}

        if self.split == "train":
            valid_pixel_mask = mask_1hw.ravel()
            if valid_pixel_mask.sum() < self.num_rays:
                min_depth, max_depth = 0.5, 5.0
                sampled_rays = self.sampling_grid[: self.num_rays]
                surface_depths = depth_1hw.ravel()[: self.num_rays]

            else:
                # try only sampling inside min/max depth range
                min_depth = depth_1hw[mask_1hw].min()
                max_depth = depth_1hw[mask_1hw].max()
                valid_samples = self.sampling_grid[valid_pixel_mask]
                if self.near_edge_sampling:
                    non_edge_samples = int(self.num_rays * (1 - self.near_edge_ratio))
                    non_edge_indices = torch.randperm(len(valid_samples))[:non_edge_samples]
                    non_edge_rays = valid_samples[non_edge_indices]
                    non_edge_depths = depth_1hw.ravel()[valid_pixel_mask][non_edge_indices]

                    edge_samples = self.num_rays - non_edge_samples
                    edge_map = self.get_edge_mask(depth_1hw).ravel()
                    edge_grid = self.sampling_grid[edge_map]
                    edge_indices = torch.randperm(len(edge_grid))[:edge_samples]
                    edge_rays = edge_grid[edge_indices]
                    edge_depths = depth_1hw.ravel()[edge_map][edge_indices]

                    sampled_rays = torch.cat((non_edge_rays, edge_rays))
                    surface_depths = torch.cat((non_edge_depths, edge_depths))
                else:
                    sampled_indices = torch.randperm(len(valid_samples))[: self.num_rays]
                    sampled_rays = valid_samples[sampled_indices]
                    surface_depths = depth_1hw.ravel()[valid_pixel_mask][sampled_indices]

            # generate samples along the rays
            ray_samples_Nd = min_depth + self.ray_samples_Nd * (max_depth - min_depth)
            sample_stride = (max_depth - min_depth) / self.random_samples_per_ray
            # add some randomness to the samples along each ray
            sampled_depths = (
                ray_samples_Nd - sample_stride / 2 + torch.rand_like(ray_samples_Nd) * sample_stride
            )

            if self.near_surface_sampling:
                if self.surface_noise_type == "additive":
                    near_surface_depths = torch.randn(
                        (self.num_rays, self.surface_samples_per_ray)
                    ) * 0.05 + surface_depths.unsqueeze(1)
                elif self.surface_noise_type == "multiplicative":
                    near_surface_depths = surface_depths.unsqueeze(1) * (
                        1.0 + torch.randn((self.num_rays, self.surface_samples_per_ray)) * 0.05
                    )
                else:
                    raise ValueError(f"Unrecognized surface noise type {self.surface_noise_type}.")

                sampled_depths = torch.cat((sampled_depths, near_surface_depths), 1)

            output_dict["sampled_depths"] = sampled_depths
            output_dict["sampled_rays"] = sampled_rays

        else:
            rendered_depth = torch.ones(
                (len(self.validation_planes), self.depth_height, self.depth_width)
            ).float()
            rendered_depth *= self.validation_planes
            output_dict["rendered_depth"] = rendered_depth

        return output_dict

    def stack_src_data(self, src_data):
        """Stacks source image data into tensors."""

        tensor_names = src_data[0].keys()
        stacked_src_data = {}
        for tensor_name in tensor_names:
            if "frame_id_string" in tensor_name:
                stacked_src_data[tensor_name] = [t[tensor_name] for t in src_data]
            else:
                stacked_src_data[tensor_name] = np.stack([t[tensor_name] for t in src_data], axis=0)

        return stacked_src_data

    def __getitem__(self, idx):
        """Loads data for all frames for the MVS tuple at index idx.

        Args:
            idx: the index for the elmeent in the dataset.

        Returns:
            cur_data: frame data for the reference frame
            src_data: stacked frame data for each source frame
        """

        flip_threshold = 0.5 if self.split == "train" else 0.0
        flip = torch.rand(1).item() < flip_threshold

        # get the index of the tuple
        scan_id, *frame_ids = self.frame_tuples[idx].split(" ")

        # shuffle tuple order, by default false
        if self.shuffle_tuple:
            first_frame_id = frame_ids[0]
            shuffled_list = frame_ids[1:]
            random.shuffle(shuffled_list)
            frame_ids = [first_frame_id] + shuffled_list

        # the tuple file may have more images in the tuple than what might be
        # requested, so limit the tuple length to num_images_in_tuple
        if self.num_images_in_tuple is not None:
            frame_ids = frame_ids[: self.num_images_in_tuple]

        # assemble the dataset element by getting all data for each frame
        inputs = []
        for frame_ind, frame_id in enumerate(frame_ids):
            inputs += [
                self.get_frame(
                    scan_id,
                    frame_id,
                    load_depth=True,
                    get_bd_info=(self.get_bd_info and frame_ind == 0),
                    flip=flip,
                )
            ]

        # cur_data is the reference frame
        cur_data, *src_data_list = inputs
        # src_data contains data for all source frames
        src_data = self.stack_src_data(src_data_list)

        # now sort all source frames (src_data) according to pose penalty w.r.t
        # to the refernce frame (cur_data)
        if not self.shuffle_tuple:
            # order source images based on pose penalty
            src_world_T_cam = torch.tensor(src_data["world_T_cam_b44"])
            cur_cam_T_world = torch.tensor(cur_data["cam_T_world_b44"])

            # Compute cur_cam_T_src_cam
            cur_cam_T_src_cam = cur_cam_T_world.unsqueeze(0) @ src_world_T_cam

            # get penalties.
            frame_penalty_k, _, _ = pose_distance(cur_cam_T_src_cam)

            # order based on indices
            indices = torch.argsort(frame_penalty_k).tolist()
            src_data_list = [src_data_list[index] for index in indices]

            # stack again
            src_data = self.stack_src_data(src_data_list)

        return cur_data, src_data
