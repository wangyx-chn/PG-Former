# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from .input import NestedTensor, inverse_sigmoid

# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
# from .matcher import build_matcher
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss, sigmoid_focal_loss)
from .deformable_transformer import DeformableTransformer
import copy



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, num_queries=50, hidden_dim = 256, num_feature_levels=3):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        assert num_feature_levels>=3 or num_feature_levels==1
        if num_feature_levels==1:
            self.backbone = build_backbone(hidden_dim,return_interm_layers=False)
        else:
            self.backbone = build_backbone(hidden_dim,return_interm_layers=True)
        self.num_queries = num_queries
        self.transformer = DeformableTransformer(d_model=hidden_dim,dropout=0.1,nhead=8,dim_feedforward=2048,
                                       num_encoder_layers=2,num_decoder_layers=2,return_intermediate_dec=False,
                                       num_feature_levels=num_feature_levels)
        # hidden_dim = self.transformer.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        self.point_x_embed = MLP(hidden_dim, hidden_dim, 224, 3)
        self.point_y_embed = MLP(hidden_dim, hidden_dim, 224, 3)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.layers[-1].bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.point_x_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_x_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.point_y_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_y_embed.layers[-1].bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # num_pred = self.transformer.decoder.num_layers
        nn.init.constant_(self.point_x_embed.layers[-1].bias.data[2:], -2.0)
        nn.init.constant_(self.point_y_embed.layers[-1].bias.data[2:], -2.0)
        # self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        # self.point_x_embed = nn.ModuleList([self.point_x_embed for _ in range(num_pred)])
        # self.point_y_embed = nn.ModuleList([self.point_y_embed for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None        
        

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        # outputs_classes = []
        # outputs_coords = []
        # for lvl in range(hs.shape[0]):
        #     if lvl == 0:
        #         reference = init_reference
        #     else:
        #         reference = inter_references[lvl - 1]
        #     reference = inverse_sigmoid(reference)
        #     outputs_class = self.class_embed[lvl](hs[lvl])
        #     tmp = self.point_x_embed[lvl](hs[lvl])
        #     if reference.shape[-1] == 4:
        #         tmp += reference
        #     else:
        #         assert reference.shape[-1] == 2
        #         tmp[..., :2] += reference
        #     outputs_coord = tmp.sigmoid()
        #     outputs_classes.append(outputs_class)
        #     outputs_coords.append(outputs_coord)
        # outputs_class = torch.stack(outputs_classes)
        # outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.class_embed(hs)
        outputs_coord_x = self.point_x_embed(hs)
        outputs_coord_y = self.point_y_embed(hs)
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out = {'pred_logits': outputs_class, 'pred_coords_x': outputs_coord_x, 'pred_coords_y': outputs_coord_y}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



