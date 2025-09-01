# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)
from .input import NestedTensor
from .backbone import build_backbone
from .transformer import Transformer
from .position_encoding import PositionEmbeddingSine1D


class TwoLineAutoRegDETR(nn.Module):
    """
        除了预测polygon的点的坐标，还要预测该点到下一个点的edge向量（同样采用分类）
    """
    def __init__(self, hidden_dim = 256, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = build_backbone(hidden_dim)
        # self.num_queries = num_queries
        self.transformer = Transformer(d_model=hidden_dim,dropout=0.1,nhead=8,dim_feedforward=2048,
                                       num_encoder_layers=6,num_decoder_layers=6,return_intermediate_dec=False,)
        # hidden_dim = self.transformer.d_model
        # self.class_embed = nn.Linear(hidden_dim, 1)
        # self.class_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.point_x_head = MLP(hidden_dim, hidden_dim, 244, 3)
        self.point_y_head = MLP(hidden_dim, hidden_dim, 244, 3)
        self.delta_x_head = MLP(hidden_dim, hidden_dim, 244, 3)
        self.delta_y_head = MLP(hidden_dim, hidden_dim, 244, 3)
        self.query_x_embed = nn.Embedding(244, hidden_dim)
        self.query_y_embed = nn.Embedding(244, hidden_dim)
        self.query_proj = nn.Linear(2, 1)
        self.query_pos_enc = PositionEmbeddingSine1D(hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        
        self.aux_loss = aux_loss

    def forward(self, imgs: NestedTensor, queries: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - imgs.tensor: batched images, of shape [batch_size x 3 x H x W]
               - imgs.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

               - queries: [(startx,starty), (x0,y0), (x1,y1), (xn-1,yn-1)]
               - 如果是224的话我们做244分类，前面10个留给start，后面10个留给end
               - 或者我们就做234分类，start就是end，后面10个

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
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(imgs)

        src, mask = features[-1].decompose()

        # memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
        #                   pos=pos_embed, query_pos=query_embed)
        '''
        Transformer里面是需要这些东西的:
        - src 图像特征的embedding [ Bs C_backbone 7 7 ]
        - src_key_padding_mask 图像的padmask，我们这里应该是全0，因为图像已经对齐了 [ Bs 7 7 ]
        - pos/pos_embed 图像特征的位置编码；上面已经给出来了 [ Bs D_model 7 7 ]
        - tgt 就是坐标的embedding，这里就是query [ Bs Len D_model ]
        - memory 是图像的encoder结果embedding 不在这里输入 由encoder给到decoder
        - memory_key_padding_mask 图像的padmask
        - query_pos 应该给出坐标query的位置编码 [ Bs D_model Len ]

        其实decoder还有东西
        - tgt_mask 就是attn mask，用来屏蔽未来位置，下三角矩阵，这个很重要
        - tgt_key_padding_mask 就是坐标序列的padmask [ Bs Len ]
        '''
        assert mask is not None
        query = queries.tensors
        # print(query.max())
        query_x = self.query_x_embed(query[...,0])
        query_y = self.query_y_embed(query[...,1])
        query = self.query_proj(torch.stack([query_x,query_y],dim=3)).squeeze(3)
        query_mask = queries.mask.bool()
        query_pos = self.query_pos_enc(query.permute(0,2,1),query_mask)
        hs = self.transformer(self.input_proj(src), mask, query, query_mask, pos[-1], query_pos)[0]
        
        # outputs_class = self.class_head(hs)
        outputs_coord_x = self.point_x_head(hs)
        outputs_coord_y = self.point_y_head(hs)
        outputs_delta_x = self.delta_x_head(hs)
        outputs_delta_y = self.delta_y_head(hs)
        out = {'pred_coords_x': outputs_coord_x[-1], 'pred_coords_y': outputs_coord_y[-1],
               'pred_delta_x': outputs_delta_x[-1], 'pred_delta_y': outputs_delta_y[-1]}
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

