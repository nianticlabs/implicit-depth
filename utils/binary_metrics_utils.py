from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch3d.io import load_ply
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardFlatShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesAtlas,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

from .metrics_utils import compute_depth_metrics_batched


def get_boundary_mask(depth_b1hw, rendered_depth_bdhw):
    mask_b1hw = depth_b1hw != depth_b1hw
    target_bdhw = (rendered_depth_bdhw < depth_b1hw).float()
    edges_bdhw = F.max_pool2d(target_bdhw, 3, 1, 1) - target_bdhw

    edges_bdhw[mask_b1hw.expand(edges_bdhw.shape)] = 0
    dilated_edges_bdhw = F.max_pool2d(edges_bdhw, 7, 1, 3)
    dilated_edges_bdhw[mask_b1hw.expand(edges_bdhw.shape)] = torch.nan
    boundary_mask_bdhw = (dilated_edges_bdhw > 0).float()
    return boundary_mask_bdhw


def get_surface_mask(depth_b1hw, rendered_depth_bdhw, threshold=0.05):
    surface_mask_bdhw = (
        torch.abs(depth_b1hw - rendered_depth_bdhw) / depth_b1hw < threshold
    ).float()
    return surface_mask_bdhw


class Thresholder:
    def __init__(self, planes, thresholds):
        self.bins = torch.zeros_like(planes)
        self.bins[:-1] = (planes[1:] + planes[:-1]) / 2
        self.bins[-1] = 100.0

        self.thresholds = thresholds.cuda()

    def get_thresholds(self, query_depth):
        idxs = torch.bucketize(query_depth, self.bins)
        return self.thresholds[idxs]


class PlaneEvaluator:
    def __init__(self, thresholds=np.linspace(0.3, 0.7, 5)):
        self.thresholds = thresholds

    def compute_batch_scores(
        self,
        query_depth_bdhw,
        gt_depth_b1hw,
        prediction_bdhw,
        is_rendering=False,
        tag=None,
        depth_planes=tuple(1.5 + x * 0.5 for x in range(8)),
    ):
        scores = {}
        valid_mask_bdhw = (gt_depth_b1hw.expand(query_depth_bdhw.shape) > 0) * (
            query_depth_bdhw > 0
        )

        prediction_bdN = prediction_bdhw.clone().flatten(start_dim=2)
        query_depth_bdN = query_depth_bdhw.clone().flatten(start_dim=2)
        gt_depth_bdN = gt_depth_b1hw.expand(query_depth_bdhw.shape).clone().flatten(start_dim=2)

        valid_mask_bdN = valid_mask_bdhw.flatten(start_dim=2)

        target_bdN = (query_depth_bdN < gt_depth_bdN).float()

        target_bdN[~valid_mask_bdN] = torch.nan

        for thresh_ind, threshold in enumerate(self.thresholds):
            pred_thresh_bdN = (prediction_bdN > threshold).float()
            pred_thresh_bdN[~valid_mask_bdN] = torch.nan

            # IoU
            intersection_bd = (pred_thresh_bdN * target_bdN).nansum(dim=2)
            target_count_bd = target_bdN.nansum(dim=2)
            pred_count_bd = pred_thresh_bdN.nansum(dim=2)
            union_bd = target_count_bd + pred_count_bd - intersection_bd
            iou_bd_pos = intersection_bd / union_bd

            intersection_bd = ((1 - pred_thresh_bdN) * (1 - target_bdN)).nansum(dim=2)
            target_count_bd = (1 - target_bdN).nansum(dim=2)
            pred_count_bd = (1 - pred_thresh_bdN).nansum(dim=2)
            union_bd = target_count_bd + pred_count_bd - intersection_bd
            iou_bd_neg = intersection_bd / union_bd

            iou_bd = 2 * (iou_bd_pos * iou_bd_neg) / (iou_bd_pos + iou_bd_neg)

            for depth_ind in range(query_depth_bdhw.shape[1]):
                if is_rendering:
                    depth_plane = -1
                else:
                    depth_plane = depth_planes[depth_ind]

                if tag is None:
                    scores[f"iou_{threshold:.1f}_d_{depth_plane:.1f}"] = iou_bd[:, depth_ind]
                    scores[f"iou_pos_{threshold:.1f}_d_{depth_plane:.1f}"] = iou_bd_pos[
                        :, depth_ind
                    ]
                    scores[f"iou_neg_{threshold:.1f}_d_{depth_plane:.1f}"] = iou_bd_neg[
                        :, depth_ind
                    ]
                else:
                    scores[f"{tag}_iou_{threshold:.1f}_d_{depth_plane:.1f}"] = iou_bd[:, depth_ind]
                    scores[f"{tag}_iou_pos_{threshold:.1f}_d_{depth_plane:.1f}"] = iou_bd_pos[
                        :, depth_ind
                    ]
                    scores[f"{tag}_iou_neg_{threshold:.1f}_d_{depth_plane:.1f}"] = iou_bd_neg[
                        :, depth_ind
                    ]

        return scores

    def compute_batch_scores_test(
        self,
        query_depth_bdhw,
        gt_depth_b1hw,
        prediction_bdhw,
        thresholder,
        is_rendering=False,
        tag=None,
        depth_planes=tuple(1.5 + x * 0.5 for x in range(8)),
    ):
        if thresholder is None:
            return self.compute_batch_scores(
                query_depth_bdhw, gt_depth_b1hw, prediction_bdhw, is_rendering, tag, depth_planes
            )
        scores = {}

        valid_mask_bdhw = (gt_depth_b1hw.expand(query_depth_bdhw.shape) > 0) * (
            query_depth_bdhw > 0
        )

        prediction_bdN = prediction_bdhw.clone().flatten(start_dim=2)
        query_depth_bdN = query_depth_bdhw.clone().flatten(start_dim=2)
        gt_depth_bdN = gt_depth_b1hw.expand(query_depth_bdhw.shape).clone().flatten(start_dim=2)

        valid_mask_bdN = valid_mask_bdhw.flatten(start_dim=2)

        threshold = thresholder.get_thresholds(query_depth_bdN)

        target_bdN = (query_depth_bdN < gt_depth_bdN).float()

        target_bdN[~valid_mask_bdN] = torch.nan

        pred_thresh_bdN = (prediction_bdN > threshold).float()
        pred_thresh_bdN[~valid_mask_bdN] = torch.nan

        # IoU
        intersection_bd = (pred_thresh_bdN * target_bdN).nansum(dim=2)
        target_count_bd = target_bdN.nansum(dim=2)
        pred_count_bd = pred_thresh_bdN.nansum(dim=2)
        union_bd = target_count_bd + pred_count_bd - intersection_bd
        iou_bd_pos = intersection_bd / union_bd

        intersection_bd = ((1 - pred_thresh_bdN) * (1 - target_bdN)).nansum(dim=2)
        target_count_bd = (1 - target_bdN).nansum(dim=2)
        pred_count_bd = (1 - pred_thresh_bdN).nansum(dim=2)
        union_bd = target_count_bd + pred_count_bd - intersection_bd
        iou_bd_neg = intersection_bd / union_bd

        iou_bd = 2 * (iou_bd_pos * iou_bd_neg) / (iou_bd_pos + iou_bd_neg)

        for depth_ind in range(query_depth_bdhw.shape[1]):
            if is_rendering:
                depth_plane = -1
            else:
                depth_plane = depth_planes[depth_ind]

            if tag is None:
                scores[f"iou_d_{depth_plane:.1f}"] = iou_bd[:, depth_ind]
                scores[f"iou_pos_d_{depth_plane:.1f}"] = iou_bd_pos[:, depth_ind]
                scores[f"iou_neg_d_{depth_plane:.1f}"] = iou_bd_neg[:, depth_ind]
            else:
                scores[f"{tag}_iou_d_{depth_plane:.1f}"] = iou_bd[:, depth_ind]
                scores[f"{tag}_iou_pos_d_{depth_plane:.1f}"] = iou_bd_pos[:, depth_ind]
                scores[f"{tag}_iou_neg_d_{depth_plane:.1f}"] = iou_bd_neg[:, depth_ind]

        return scores

    def compute_regressed_depth_batch_scores(
        self, query_depth_bdhw, gt_depth_b1hw, prediction_b1hw, is_rendering=False, tag=None
    ):
        scores = {}

        valid_mask_bdhw = (gt_depth_b1hw.expand(query_depth_bdhw.shape) > 0) * (
            query_depth_bdhw > 0
        )

        prediction_bdN = prediction_b1hw.expand(query_depth_bdhw.shape).clone().flatten(start_dim=2)
        query_depth_bdN = query_depth_bdhw.clone().flatten(start_dim=2)
        gt_depth_bdN = gt_depth_b1hw.expand(query_depth_bdhw.shape).clone().flatten(start_dim=2)

        valid_mask_bdN = valid_mask_bdhw.flatten(start_dim=2)

        target_bdN = (query_depth_bdN < gt_depth_bdN).float()
        target_bdN[~valid_mask_bdN] = torch.nan

        pred_thresh_bdN = (query_depth_bdN < prediction_bdN).float()
        pred_thresh_bdN[~valid_mask_bdN] = torch.nan

        intersection_bd = (pred_thresh_bdN * target_bdN).nansum(dim=2)
        target_count_bd = target_bdN.nansum(dim=2)
        pred_count_bd = pred_thresh_bdN.nansum(dim=2)
        union_bd = target_count_bd + pred_count_bd - intersection_bd
        iou_bd_pos = intersection_bd / union_bd

        intersection_bd = ((1 - pred_thresh_bdN) * (1 - target_bdN)).nansum(dim=2)
        target_count_bd = (1 - target_bdN).nansum(dim=2)
        pred_count_bd = (1 - pred_thresh_bdN).nansum(dim=2)
        union_bd = target_count_bd + pred_count_bd - intersection_bd
        iou_bd_neg = intersection_bd / union_bd

        iou_bd = 2 * (iou_bd_pos * iou_bd_neg) / (iou_bd_pos + iou_bd_neg)

        for depth_ind in range(query_depth_bdhw.shape[1]):
            if is_rendering:
                depth_plane = -1
            else:
                depth_plane = [1.5 + x * 0.5 for x in range(8)][depth_ind]

            if tag is None:
                scores[f"iou_d_{depth_plane:.1f}"] = iou_bd[:, depth_ind]
                scores[f"iou_pos_d_{depth_plane:.1f}"] = iou_bd_pos[:, depth_ind]
                scores[f"iou_neg_d_{depth_plane:.1f}"] = iou_bd_neg[:, depth_ind]
            else:
                scores[f"{tag}_iou_d_{depth_plane:.1f}"] = iou_bd[:, depth_ind]
                scores[f"{tag}_iou_pos_d_{depth_plane:.1f}"] = iou_bd_pos[:, depth_ind]
                scores[f"{tag}_iou_neg_d_{depth_plane:.1f}"] = iou_bd_neg[:, depth_ind]

        return scores


class TemporalEvaluator:
    def __init__(self):
        self.rasterizer = None
        self.total_diffs = 0
        self.total_verts = 0

    def initialise_new_scene(self, gt_mesh_path, height=192, width=256):
        self.rasterizer = Pytorch3DRasterizer(height=height, width=width)
        self.rasterizer.load_gt_mesh(gt_mesh_path=gt_mesh_path)

    def initialise_new_plane(self, depth_gt_b1hw, world_T_cam_b44):
        self.rasterizer.create_plane_from_camera(
            world_T_cam_b44, distance=torch.nanquantile(depth_gt_b1hw, 0.75)
        )
        self.rasterizer.gt_vertex_predictions = []

    @staticmethod
    def mask_prediction_edges(prediction, edge_size=4):
        edge_mask = torch.ones_like(prediction).bool()
        edge_mask[..., edge_size:-edge_size, edge_size:-edge_size] = False
        prediction[edge_mask] = -1.0

    def update_vertex_predictions(self, prediction, cam_T_world_b44, K_b44):
        self.mask_prediction_edges(prediction)
        self.rasterizer.update_gt_vertex_predictions(prediction, cam_T_world_b44, K_b44)

    def compute_vertex_occlusion_changes(self):
        predictions = torch.stack(self.rasterizer.gt_vertex_predictions).float().clone()
        predictions[predictions == -1] = torch.nan
        predictions[predictions > 0.5] = 1
        predictions[predictions < 0.5] = 0
        diffs = torch.abs(predictions[1:] - predictions[:-1])
        self.total_diffs += (torch.nansum(diffs, 0).sum()).detach().cpu().numpy()
        self.total_verts += diffs.shape[1]


class Pytorch3DRasterizer:
    def __init__(self, height, width):
        self.image_size = torch.tensor((height, width)).unsqueeze(0)
        self.raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.mesh = None
        self.faces = None
        self.rasterizer = None
        self.gt_mesh = None
        self.gt_vertex_predictions = []

    def load_gt_mesh(self, gt_mesh_path):
        gt_verts, gt_faces = load_ply(gt_mesh_path)
        self.gt_mesh = Meshes(
            verts=[gt_verts.float()],
            faces=[gt_faces.long()],
        )

    def create_plane_from_camera(self, cam_T_world_b44, distance=2.5):
        xs, ys = np.meshgrid((np.arange(1024) - 512) * 0.025, (np.arange(1024) - 512) * 0.025)
        points_14N = (
            torch.tensor(np.stack((xs, ys, np.ones_like(xs), np.ones_like(xs)), 0))
            .float()
            .view(1, 4, -1)
            .cuda()
        )
        points_14N[:, 2] *= distance
        vertices = torch.matmul(cam_T_world_b44, points_14N)
        if self.faces is None:
            self.faces = []
            H, W = 1024, 1024
            for h in range(H - 1):
                for w in range(W - 1):
                    idx = h * W + w
                    self.faces.append((idx, idx + W + 1, idx + W))
                    self.faces.append((idx, idx + 1, idx + 1 + W))
            self.faces = torch.tensor(self.faces).cuda()

        vertices = vertices[0, :3].T  # N x 3
        self.mesh = Meshes(
            verts=[vertices],
            faces=[self.faces],
        )

    def __call__(self, cam_T_world_b44, K_b44, use_cuda=True):
        if self.mesh is None:
            raise ValueError(f"Mesh has not be initialised for rendering!")
        return self.render_depth(cam_T_world_b44, K_b44, use_cuda)

    def render_depth(self, cam_T_world_b44, K_b44, use_cuda=True, mesh=None):
        R = cam_T_world_b44[:, :3, :3]
        T = cam_T_world_b44[:, :3, 3]
        K = K_b44[:, :3, :3]
        cams = cameras_from_opencv_projection(
            R=R, tvec=T, camera_matrix=K, image_size=self.image_size
        )

        if mesh is None:
            mesh = self.mesh
        if use_cuda:
            mesh = mesh.cuda()
            cams = cams.cuda()

        self.rasterizer = MeshRasterizer(
            cameras=cams,
            raster_settings=self.raster_settings,
        )

        mesh = mesh.extend(len(cams))
        fragments = self.rasterizer(mesh)
        rendered_depth_b1hw = fragments.zbuf[..., 0].unsqueeze(1)
        return rendered_depth_b1hw

    def update_gt_vertex_predictions(self, pred, cam_T_world_b44, K_b44):
        # render gt mesh into the camera for visibility check
        rendered_depth = self.render_depth(cam_T_world_b44, K_b44, mesh=self.gt_mesh)

        # project vertices into the camera
        mesh = self.gt_mesh.cuda()
        verts = mesh.verts_list()[0]

        cam_points_N3 = self.rasterizer.cameras.transform_points_screen(verts)
        pix_locs_11N2 = cam_points_N3[:, :2].unsqueeze(0).unsqueeze(0)
        pix_locs_11N2[..., 0] = (pix_locs_11N2[..., 0] / 256 - 0.5) * 2
        pix_locs_11N2[..., 1] = (pix_locs_11N2[..., 1] / 192 - 0.5) * 2
        depths_N = 1 / cam_points_N3[:, 2]

        sampled_pred = F.grid_sample(pred, pix_locs_11N2, mode="nearest")
        sampled_depth = F.grid_sample(rendered_depth, pix_locs_11N2, mode="nearest")

        # get masks
        valid = (
            (sampled_depth > 0)
            * (depths_N > 0)
            * (torch.abs(depths_N - sampled_depth) < 0.05)
            * (sampled_pred > 0)
        )
        valid = valid.view(-1)
        sampled_pred = sampled_pred.view(-1)
        sampled_pred[~valid] = -1

        self.gt_vertex_predictions.append(sampled_pred)
