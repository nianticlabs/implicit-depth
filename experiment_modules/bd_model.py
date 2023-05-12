import logging

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torch import nn

from modules.cost_volume import (
    CostVolumeManager,
    FeatureVolumeManager,
    ZeroCostVolumeManager,
)
from modules.layers import TensorFormatter
from modules.networks import (
    BDDecoderPP,
    BinaryMLPNetwork,
    CVEncoder,
    FPNMatchingEncoder,
    ResnetMatchingEncoder,
)
from modules.networks_fast import SkipDecoder
from utils.generic_utils import (
    get_edge_mask,
    reverse_imagenet_normalize,
    tensor_B_to_bM,
    tensor_bM_to_B,
)
from utils.geometry_utils import BackprojectDepth, Project3D
from utils.visualization_utils import colormap_image, prepare_image_for_logging

logger = logging.getLogger(__name__)

SCALES = list(range(4))


class BDModel(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.save_hyperparameters()

        self.run_opts = opts

        # iniitalize the encoder for strong image priors
        if "efficientnet" in self.run_opts.image_encoder_name:
            self.encoder = timm.create_model(
                "tf_efficientnetv2_s_in21ft1k", pretrained=True, features_only=True
            )

            self.encoder.num_ch_enc = self.encoder.feature_info.channels()
        elif "resnext101" in self.run_opts.image_encoder_name:
            self.encoder = timm.create_model(
                "resnext101_64x4d", pretrained=True, features_only=True
            )

            self.encoder.num_ch_enc = self.encoder.feature_info.channels()
        elif "seresnextaa101d" in self.run_opts.image_encoder_name:
            self.encoder = timm.create_model(
                "seresnextaa101d_32x8d", pretrained=True, features_only=True
            )

            self.encoder.num_ch_enc = self.encoder.feature_info.channels()

        elif "resnet" in self.run_opts.image_encoder_name:
            self.encoder = timm.create_model("resnet18d", pretrained=True, features_only=True)

            self.encoder.num_ch_enc = self.encoder.feature_info.channels()
        else:
            raise ValueError("Unrecognized option for image encoder type!")

        # iniitalize the first half of the U-Net, encoding the cost volume
        # and image prior image feautres
        if self.run_opts.cv_encoder_type == "multi_scale_encoder":
            self.cost_volume_net = CVEncoder(
                num_ch_cv=self.run_opts.matching_num_depth_bins,
                num_ch_enc=self.encoder.num_ch_enc[self.run_opts.matching_scale :],
                num_ch_outs=[64, 128, 256, 384],
            )
            dec_num_input_ch = (
                self.encoder.num_ch_enc[: self.run_opts.matching_scale]
                + self.cost_volume_net.num_ch_enc
            )
        else:
            raise ValueError("Unrecognized option for cost volume encoder type!")

        # iniitalize the final depth decoder
        if self.run_opts.depth_decoder_name == "unet_pp":
            self.depth_decoder = BDDecoderPP(dec_num_input_ch)
        elif self.run_opts.depth_decoder_name == "skip":
            self.depth_decoder = SkipDecoder(dec_num_input_ch)
        else:
            raise ValueError("Unrecognized option for depth decoder name!")

        self.large_feature_width = self.run_opts.image_width // 2
        self.large_feature_height = self.run_opts.image_height // 2

        self.run_opts.binary_loss_weighting = 1.0

        pos_weight = torch.tensor((self.run_opts.binary_loss_positive_weight,)).float()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.sigmoid = nn.Sigmoid()

        # what type of cost volume are we using?
        if self.run_opts.feature_volume_type == "simple_cost_volume":
            cost_volume_class = CostVolumeManager
        elif self.run_opts.feature_volume_type == "zero_cost_volume":
            cost_volume_class = ZeroCostVolumeManager
        elif self.run_opts.feature_volume_type == "mlp_feature_volume":
            cost_volume_class = FeatureVolumeManager
        else:
            raise ValueError("Unrecognized option for feature volume type!")

        self.cost_volume = cost_volume_class(
            matching_height=self.run_opts.image_height // (2 ** (self.run_opts.matching_scale + 1)),
            matching_width=self.run_opts.image_width // (2 ** (self.run_opts.matching_scale + 1)),
            num_depth_bins=self.run_opts.matching_num_depth_bins,
        )

        # init the matching encoder. resnet is fast and is the default for
        # results in the paper, fpn is more accurate but much slower.
        if "resnet" == self.run_opts.matching_encoder_type:
            self.matching_model = ResnetMatchingEncoder(18, self.run_opts.matching_feature_dims)
        elif "fpn" == self.run_opts.matching_encoder_type:
            self.matching_model = FPNMatchingEncoder()
        else:
            raise ValueError("Unrecognized option for matching encoder type!")

        self.tensor_formatter = TensorFormatter()
        self.binary_mlp = BinaryMLPNetwork(
            self.depth_decoder.num_ch_dec,
            mlp_size=128,
            use_prior=self.run_opts.use_prior,
        )

        if self.run_opts.use_prior:
            # TODO: not actually needed!
            self.backprojector = BackprojectDepth(192, 256).cuda()
            self.projector = Project3D().cuda()

        self.thresholder = None

    def compute_matching_feats(
        self,
        cur_image,
        src_image,
        unbatched_matching_encoder_forward,
    ):
        if unbatched_matching_encoder_forward:
            all_frames_bm3hw = torch.cat([cur_image.unsqueeze(1), src_image], dim=1)
            batch_size, num_views = all_frames_bm3hw.shape[:2]
            all_frames_B3hw = tensor_bM_to_B(all_frames_bm3hw)
            matching_feats = [self.matching_model(f) for f in all_frames_B3hw.split(1, dim=0)]

            matching_feats = torch.cat(matching_feats, dim=0)
            matching_feats = tensor_B_to_bM(
                matching_feats,
                batch_size=batch_size,
                num_views=num_views,
            )

        else:
            # Compute matching features and batch them to reduce variance from
            # batchnorm when training.
            matching_feats = self.tensor_formatter(
                torch.cat([cur_image.unsqueeze(1), src_image], dim=1),
                apply_func=self.matching_model,
            )

        matching_cur_feats = matching_feats[:, 0]
        matching_src_feats = matching_feats[:, 1:].contiguous()

        return matching_cur_feats, matching_src_feats

    def forward(
        self,
        phase,
        cur_data,
        src_data,
        unbatched_matching_encoder_forward=False,
        return_mask=False,
        infer_depth=False,
        infer_res=None,
    ):
        # get all tensors from the batch dictioanries.
        cur_image = cur_data["image_b3hw"]
        src_image = src_data["image_b3hw"]
        src_K = src_data[f"K_s{self.run_opts.matching_scale}_b44"]
        cur_invK = cur_data[f"invK_s{self.run_opts.matching_scale}_b44"]
        src_cam_T_world = src_data["cam_T_world_b44"]
        src_world_T_cam = src_data["world_T_cam_b44"]

        cur_cam_T_world = cur_data["cam_T_world_b44"]
        cur_world_T_cam = cur_data["world_T_cam_b44"]

        with torch.cuda.amp.autocast(False):
            # Compute src_cam_T_cur_cam, a transformation for going from 3D
            # coords in current view coordinate frame to source view coords
            # coordinate frames.
            src_cam_T_cur_cam = src_cam_T_world @ cur_world_T_cam.unsqueeze(1)

            # Compute cur_cam_T_src_cam the opposite of src_cam_T_cur_cam. From
            # source view to current view.
            cur_cam_T_src_cam = cur_cam_T_world.unsqueeze(1) @ src_world_T_cam

        # flip transformation! Figure out if we're flipping. Should be true if
        # we are training and a coin flip says we should.
        flip_threshold = 0.5 if phase == "train" else 0.0
        flip = torch.rand(1).item() < flip_threshold

        if flip:
            # flip all images.
            cur_image = torch.flip(cur_image, (-1,))
            src_image = torch.flip(src_image, (-1,))

        # Compute image features for the current view. Used for a strong image
        # prior.
        cur_feats = self.encoder(cur_image)

        # Compute matching features
        matching_cur_feats, matching_src_feats = self.compute_matching_feats(
            cur_image, src_image, unbatched_matching_encoder_forward
        )

        if flip:
            # now (carefully) flip matching features back for correct MVS.
            matching_cur_feats = torch.flip(matching_cur_feats, (-1,))
            matching_src_feats = torch.flip(matching_src_feats, (-1,))

        # Get min and max depth to the right shape, device and dtype
        min_depth = torch.tensor(self.run_opts.min_matching_depth).type_as(src_K).view(1, 1, 1, 1)
        max_depth = torch.tensor(self.run_opts.max_matching_depth).type_as(src_K).view(1, 1, 1, 1)

        # Compute the cost volume. Should be size bdhw.
        cost_volume, lowest_cost, _, overall_mask_bhw = self.cost_volume(
            cur_feats=matching_cur_feats,
            src_feats=matching_src_feats,
            src_extrinsics=src_cam_T_cur_cam,
            src_poses=cur_cam_T_src_cam,
            src_Ks=src_K,
            cur_invK=cur_invK,
            min_depth=min_depth,
            max_depth=max_depth,
            return_mask=return_mask,
        )

        if flip:
            # OK, we've computed the cost volume, now we need to flip the cost
            # volume to have it aligned with flipped image prior features
            cost_volume = torch.flip(cost_volume, (-1,))

        # Encode the cost volume and current image features
        if self.run_opts.cv_encoder_type == "multi_scale_encoder":
            cost_volume_features = self.cost_volume_net(
                cost_volume,
                cur_feats[self.run_opts.matching_scale :],
            )
            cur_feats = cur_feats[: self.run_opts.matching_scale] + cost_volume_features

        # Decode into depth at multiple resolutions.
        feature_outputs = self.depth_decoder(cur_feats)

        # now flip the feature map back
        if flip:
            for scale in SCALES:
                feature_outputs[f"feature_s{scale}_b1hw"] = torch.flip(
                    feature_outputs[f"feature_s{scale}_b1hw"], (-1,)
                )

        # run the MLP step
        if phase == "train":
            outputs = self.run_mlp_train(cur_data, feature_outputs)
        elif infer_depth:
            # setup for binary search - min, max and initial
            min_bound = torch.ones_like(cur_data["rendered_depth"][:, 0:1]) * 0.5
            max_bound = torch.ones_like(cur_data["rendered_depth"][:, 0:1]) * 8.0
            search_depths = torch.ones_like(cur_data["rendered_depth"][:, 0:1]) * 7.5 / 2.0

            for _ in range(12):
                outputs = self.run_mlp_val(cur_data, feature_outputs, search_depths)
                pred = self.sigmoid(outputs["pred_0"])
                if self.thresholder is not None:
                    thresholds = self.thresholder.get_thresholds(search_depths)
                else:
                    thresholds = 0.5
                visible_mask = pred < thresholds

                max_bound[visible_mask] = search_depths[visible_mask]
                min_bound[~visible_mask] = search_depths[~visible_mask]
                search_depths = (max_bound + min_bound) / 2

            outputs["search_depths"] = search_depths
        else:
            # validation/test MLP step
            outputs = None
            rendered_depth = cur_data["rendered_depth"]
            for idx in range(rendered_depth.shape[1]):
                cur_rend_depth = rendered_depth[:, idx : idx + 1]
                cur_output = self.run_mlp_val(cur_data, feature_outputs, cur_rend_depth)
                if outputs is None:
                    outputs = cur_output
                else:
                    for key, val in outputs.items():
                        outputs[key] = torch.cat((val, cur_output[key]), 1)

        # include argmax likelihood depth estimates from cost volume and
        # overall source view mask.
        outputs["lowest_cost_bhw"] = lowest_cost
        outputs["overall_mask_bhw"] = overall_mask_bhw

        return outputs

    def run_mlp_train(self, inputs, feature_maps):
        gt_depth_b1hw = (
            inputs["full_res_depth_b1hw"]
            if self.run_opts.full_depth_supervision
            else inputs["depth_b1hw"]
        )

        sampled_rays = inputs["sampled_rays"].unsqueeze(2)  # B x N x 1 x 2
        sampled_depths = inputs["sampled_depths"].unsqueeze(1)  # B x 1 x N x S
        num_samples = sampled_depths.shape[-1]

        # convert pixels to -1, 1 for grid sampling
        sampled_rays[..., 0] = (sampled_rays[..., 0] / gt_depth_b1hw.shape[-1] - 0.5) * 2
        sampled_rays[..., 1] = (sampled_rays[..., 1] / gt_depth_b1hw.shape[-2] - 0.5) * 2

        # sample the ground truth
        target_depth = F.grid_sample(
            gt_depth_b1hw,
            sampled_rays,
            mode="bilinear",
            align_corners=False,
        )  # B x 1 x N x 1
        inputs["target_depth"] = target_depth
        inputs["rendered_depth"] = sampled_depths  # store for loss calc

        if self.run_opts.bd_edge_regularision:
            edge_mask = get_edge_mask(gt_depth_b1hw)
            edge_mask = F.grid_sample(
                edge_mask,
                sampled_rays,
                mode="nearest",
                align_corners=False,
            )  # B x 1 x N x 1
            inputs["edge_mask"] = edge_mask

        model_inputs = []
        for scale in SCALES:
            features = feature_maps[f"feature_s{scale}_b1hw"]
            # subsample lower scale inputs
            subsampled_rays = sampled_rays[:, :: (scale + 1)]
            subsampled_depths = sampled_depths[:, :, :: (scale + 1)]
            subsampled_target = target_depth[:, :, :: (scale + 1)]

            # each element is B x C x N x 1
            sampled_feature = F.grid_sample(
                features,
                subsampled_rays,
                mode="bilinear",
                align_corners=False,
            )
            # expand for all ray samples
            sampled_feature = sampled_feature.expand(-1, -1, -1, num_samples)

            # concatenate and reshape for MLP layers
            model_input = torch.cat((subsampled_depths, sampled_feature), 1)  # B x C+1 x N x S

            if self.run_opts.use_prior:
                prior = (subsampled_depths < subsampled_target).float()
                offset = torch.rand_like(prior) * 0.45
                prior[prior == 1] -= offset[prior == 1]
                prior[prior == 0] += offset[prior == 0]

                # randomly set pixels to -1 or reverse the label
                augmentation_prob = torch.rand_like(prior)
                prior[augmentation_prob < 0.5] = 1.0 - prior[augmentation_prob < 0.5]
                prior[augmentation_prob < 0.25] = -1.0
                model_input = torch.cat((model_input, prior), 1)  # B x C+2 x N x S

            model_input = model_input.permute(0, 2, 3, 1)  # B x N x S x C+1
            model_inputs.append(model_input)

        model_outputs = self.binary_mlp(model_inputs)

        # change order of outputs for logging/losses
        model_outputs = {key: val.permute(0, 3, 1, 2) for key, val in model_outputs.items()}

        # for debugging
        model_outputs["model_inputs"] = model_inputs
        model_outputs["feature_maps"] = feature_maps

        return model_outputs

    def sample_prior(
        self, rendered_depth, prior_prediction, cam_to_world, prior_world_to_cam, K, invK
    ):
        batch_size, _, height, width = rendered_depth.shape

        cur_to_prior = torch.matmul(prior_world_to_cam, cam_to_world)
        world_points = self.backprojector(rendered_depth, invK)
        cam_points = self.projector(world_points, K, cur_to_prior)
        pix_locs = cam_points[:, :2].reshape(batch_size, 2, height, width).permute((0, 2, 3, 1))

        pix_locs[..., 0] = (pix_locs[..., 0] / width - 0.5) * 2
        pix_locs[..., 1] = (pix_locs[..., 1] / height - 0.5) * 2
        sampled_prior = F.grid_sample(prior_prediction, pix_locs, mode="nearest")
        mask = (rendered_depth > 0) * (cam_points[:, 2:3].view(batch_size, 1, height, width) > 0)
        sampled_prior[~mask] = -1
        return sampled_prior

    def run_mlp_val(self, inputs, feature_maps, rendered_depth):
        # only running validation at max model scale!
        features = feature_maps[f"feature_s{0}_b1hw"]  # B x C x H x W
        model_inputs = torch.cat((rendered_depth, features), 1)

        if self.run_opts.use_prior:
            if self.backprojector is None:
                # initialise here to avoid error of trying to load model without them
                self.backprojector = BackprojectDepth(192, 256).cuda()
                self.projector = Project3D().cuda()

            if inputs.get("prior_prediction", None) is not None:
                prior_mask = self.sample_prior(
                    inputs["rendered_depth"],
                    inputs["prior_prediction"],
                    inputs["world_T_cam_b44"],
                    inputs["prior_cam_T_world"],
                    inputs["K_s0_b44"],
                    inputs["invK_s0_b44"],
                )
                inputs["prior_mask"] = prior_mask
            else:
                prior_mask = -torch.ones_like(rendered_depth)
            model_inputs = torch.cat((model_inputs, prior_mask), 1)

        # mlp expects a list of inputs
        model_inputs = [model_inputs.permute(0, 2, 3, 1)]
        model_outputs = self.binary_mlp(model_inputs, max_scale_only=True)

        # change order of outputs for logging/losses
        model_outputs = {key: val.permute(0, 3, 1, 2) for key, val in model_outputs.items()}

        if self.run_opts.bd_edge_regularision:
            gt_depth_b1hw = inputs["depth_b1hw"]
            edge_mask = get_edge_mask(gt_depth_b1hw)
            inputs["edge_mask"] = edge_mask

        return model_outputs

    def compute_binary_losses(self, inputs, outputs, phase):
        losses = {}
        total_loss = 0.0
        rendered = inputs["rendered_depth"]
        if phase == "train":
            depth = inputs["target_depth"]
        else:
            depth = inputs["depth_b1hw"]

        target = (rendered < depth).float()
        mask = (depth > 0) * (rendered > 0)

        if mask.sum():
            scales = SCALES if phase == "train" else [0]
            for scale in scales:
                pred = outputs[f"pred_{scale}"]
                # lower scales are subsampled
                scale_target = target[:, :, :: (scale + 1)]
                scale_mask = mask[:, :, :: (scale + 1)]

                binary_loss = self.bce_loss(pred[scale_mask], scale_target[scale_mask]).mean()
                losses[f"binary_loss/{scale}"] = binary_loss

                # regluarisation for sharpness
                if self.run_opts.bd_edge_regularision:
                    reg_mask = (inputs["edge_mask"][:, :, :: (scale + 1)] * scale_mask).bool()
                else:
                    reg_mask = scale_mask
                distance_from_center = 2 * (0.5 - torch.abs(self.sigmoid(pred[reg_mask]) - 0.5))

                reg_loss = distance_from_center.mean()
                losses[f"reg_loss/{scale}"] = reg_loss

                total_loss += binary_loss

                if self.run_opts.bd_regularisation_weight > 0.0:
                    total_loss += reg_loss * self.run_opts.bd_regularisation_weight

        else:
            print("TRIGGERED EDGE CASE")
            total_loss = torch.abs(outputs[f"pred_{0}"] - outputs[f"pred_{0}"]).mean()

        total_loss = total_loss / len(scales)
        losses["binary_loss"] = total_loss
        return losses

    def compute_losses(self, phase, cur_data, outputs):
        losses = self.compute_binary_losses(cur_data, outputs, phase=phase)

        loss = losses["binary_loss"] * self.run_opts.binary_loss_weighting
        losses["loss"] = loss

        return losses

    def compute_iou(self, cur_data, outputs, phase, threshold=0.5):
        iou_dict = {}

        query_depth = cur_data["rendered_depth"]
        if phase == "train":
            gt_depth = cur_data["target_depth"]
        else:
            gt_depth = cur_data["depth_b1hw"]

        # legacy_iou
        target = query_depth < gt_depth
        pred = self.sigmoid(outputs["pred_0"]) > threshold
        mask = gt_depth > 0.5
        target *= mask
        pred *= mask
        intersection = (target * pred).float().sum(dim=(0, 2, 3))
        union = ((target + pred) > 0).float().sum(dim=(0, 2, 3))
        iou_dict["iou"] = torch.nanmean(intersection / union)

        ## new_iou ##
        gt_mask = (gt_depth > 0.0).expand(query_depth.shape)
        target_bdhw = (query_depth < gt_depth).float()
        pred_thresh_bdhw = (self.sigmoid(outputs["pred_0"]) > threshold).float()
        pred_thresh_bdhw[~gt_mask] = torch.nan

        pred_thresh_bdN = pred_thresh_bdhw.flatten(2)
        target_bdN = target_bdhw.flatten(2)

        # pos IOU
        intersection_bd = (pred_thresh_bdN * target_bdN).nansum(dim=2)
        target_count_bd = target_bdN.nansum(dim=2)
        pred_count_bd = pred_thresh_bdN.nansum(dim=2)
        union_bd = target_count_bd + pred_count_bd - intersection_bd
        iou_bd_pos = intersection_bd / union_bd
        # inequal nans in each bin, so nanmean twice. once over depth then batch
        iou_dict["pos_iou"] = torch.nanmean(torch.nanmean(iou_bd_pos, dim=1))

        # neg IOU
        intersection_bd = ((1 - pred_thresh_bdN) * (1 - target_bdN)).nansum(dim=2)
        target_count_bd = (1 - target_bdN).nansum(dim=2)
        pred_count_bd = (1 - pred_thresh_bdN).nansum(dim=2)
        union_bd = target_count_bd + pred_count_bd - intersection_bd
        iou_bd_neg = intersection_bd / union_bd
        # inequal nans in each bin, so nanmean twice. once over depth then batch
        iou_dict["neg_iou"] = torch.nanmean(torch.nanmean(iou_bd_neg, dim=1))

        # harmonic iou
        harmonic_iou_bd = 2 * (iou_bd_pos * iou_bd_neg) / (iou_bd_pos + iou_bd_neg)

        iou_dict["harmonic_iou"] = torch.nanmean(torch.nanmean(harmonic_iou_bd, dim=1))

        return iou_dict

    def log_images(self, inputs, outputs, phase="val"):
        for j in range(min(self.run_opts.batch_size, 4)):
            image = reverse_imagenet_normalize(inputs["image_b3hw"][j])
            self.logger.experiment.add_image(
                "{}/image/{}".format(phase, j), image, self.global_step
            )

            mask_i = inputs["mask_b1hw"][j].float().cpu()
            depth_gt_viz_i, vmin, vmax = colormap_image(
                inputs["depth_b1hw"][j].float().cpu(), mask_i, return_vminvmax=True
            )
            self.logger.experiment.add_image(
                "{}/depth/{}".format(phase, j), depth_gt_viz_i, self.global_step
            )

            lowest_cost = colormap_image(
                outputs["lowest_cost_bhw"][j].unsqueeze(0).float().cpu(), vmin=vmin, vmax=vmax
            )
            self.logger.experiment.add_image(
                "{}/lowest_cost/{}".format(phase, j), lowest_cost, self.global_step
            )

            image = inputs["image_b3hw"][j].cpu()
            depth = inputs["depth_b1hw"][j, 0].cpu()
            rendered = inputs["rendered_depth"][j, 0].cpu()
            mask = ((depth > 0) * (rendered > 0)).float()
            target = (rendered < depth).float() * mask

            pred = self.sigmoid(outputs[f"pred_{0}"][j, 0]).cpu()
            pred_masked = pred * mask

            target_up = F.interpolate(
                target[None, None],
                size=image.shape[-2:],
                mode="bilinear",
            ).squeeze()

            pred_masked_up = F.interpolate(
                pred_masked[None, None],
                size=image.shape[-2:],
                mode="bilinear",
            ).squeeze()

            target_colour = image * (1 - target_up) + 1.0 * target_up
            pred_colour = image * (1 - pred_masked_up) + 1.0 * pred_masked_up

            self.logger.experiment.add_image(
                "{}/binary_input/{}".format(phase, j),
                prepare_image_for_logging(rendered, colormap=True, invert=True),
                self.global_step,
            )
            self.logger.experiment.add_image(
                "{}/target/{}".format(phase, j),
                prepare_image_for_logging(target, normalize=False),
                self.global_step,
            )
            self.logger.experiment.add_image(
                "{}/pred/{}".format(phase, j),
                prepare_image_for_logging(pred, normalize=False),
                self.global_step,
            )
            self.logger.experiment.add_image(
                "{}/pred_masked/{}".format(phase, j),
                prepare_image_for_logging(pred_masked, normalize=False),
                self.global_step,
            )
            self.logger.experiment.add_image(
                "{}/pred_masked_tresh/{}".format(phase, j),
                prepare_image_for_logging((pred_masked > 0.5).float(), normalize=False),
                self.global_step,
            )

            target_colour = reverse_imagenet_normalize(target_colour)
            self.logger.experiment.add_image(
                "{}/colour_target/{}".format(phase, j), target_colour, self.global_step
            )

            pred_colour = reverse_imagenet_normalize(pred_colour)
            self.logger.experiment.add_image(
                "{}/colour_pred/{}".format(phase, j), pred_colour, self.global_step
            )

            if self.run_opts.bd_edge_regularision:
                self.logger.experiment.add_image(
                    "{}/edge_mask/{}".format(phase, j),
                    prepare_image_for_logging(inputs["edge_mask"][j, 0].cpu(), normalize=False),
                    self.global_step,
                )

    def step(self, phase, batch, batch_idx):
        """Takes a training/validation step through the model.

        phase: "train" or "val". "train" will signal this function and
            others log results and use flip augmentation.
        batch: (cur_data, src_data) where cur_data is a dict with data on
            the current (reference) view and src_data is a dict with data on
            source views.
        """
        cur_data, src_data = batch

        # forward pass through the model.
        outputs = self(phase, cur_data, src_data)

        # compute losses
        losses = self.compute_losses(phase, cur_data, outputs)

        is_train = phase == "train"

        # logging and validation
        with torch.inference_mode():
            for loss_name, loss_val in losses.items():
                self.log(
                    f"{phase}/{loss_name}",
                    loss_val,
                    sync_dist=True,
                    on_step=is_train,
                    on_epoch=not is_train,
                )

            if not is_train and batch_idx == 0:
                self.log_images(inputs=cur_data, outputs=outputs, phase=phase)

            iou_dict = self.compute_iou(cur_data, outputs, phase=phase)
            for iou_name in iou_dict:
                self.log(
                    f"{phase}/{iou_name}",
                    iou_dict[iou_name],
                    sync_dist=True,
                    on_step=is_train,
                    on_epoch=not is_train,
                )

        return losses["loss"]

    def training_step(self, batch, batch_idx):
        """Runs a training step."""
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Runs a validation step."""
        return self.step("val", batch, batch_idx)

    def configure_optimizers(self):
        """Configuring optmizers and learning rate schedules.

        By default we use a stepped learning rate schedule with steps at
        70000 and 80000.

        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.run_opts.lr, weight_decay=self.run_opts.wd
        )

        def lr_lambda(step):
            if step < self.run_opts.lr_steps[0]:
                return 1
            elif step < self.run_opts.lr_steps[1]:
                return 0.1
            else:
                return 0.01

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
