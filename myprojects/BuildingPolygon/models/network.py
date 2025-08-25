import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Swin_T_Weights, ViT_B_16_Weights, Swin_B_Weights
from .transformer import VisionEncoder,PolygonDecoder,PolygonDecoderV2,SeqEncoder

class PolyRefiner(nn.Module):
    def __init__(self, img_size, patch_size=8, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        assert self.img_size%self.patch_size==0, 'img size cant match patch size'
        self.img_encoder = VisionEncoder(
                                    img_size=img_size, 
                                    patch_size=self.patch_size, 
                                    in_channels=3, 
                                    embed_dim=256, 
                                    num_heads=8, 
                                    depth=6, 
                                    mlp_ratio=4, 
                                    dropout=0.1)
        self.poly_decoder = PolygonDecoder(
                                    out_channels=256, 
                                    embed_dim=256, 
                                    num_heads=8, 
                                    depth=6, 
                                    mlp_ratio=4, 
                                    dropout=0.1)
        self.norm = nn.BatchNorm1d(256)
        self.reg_head = nn.Linear(256,2)
        # self.cls_head = nn.Linear(256,1)
    
    def forward(self,img,coords,pad_mask):
        # img : [ B C H W ]
        # coords: [ B S 2 ]
        img_embed = self.img_encoder(img) # [ N B D ]

        coords = coords.transpose(1,2) # [ B C S ] for Conv1D
        coords = self.poly_decoder(coords,img_embed,pad_mask) # [ S B C ]
        coords = self.norm(coords.permute(1,2,0))# [ B C S ] 
        reg_out = self.reg_head(coords.permute(0,2,1)) # [ B S C ] 
        # cls_out = self.cls_head(coords.permute(0,2,1))
        # return reg_out, cls_out
        return reg_out
    

class PolyRefinerV2(nn.Module):
    def __init__(self, img_size, patch_size=8, embed_dim=256,num_heads=4,**kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        assert self.img_size%self.patch_size==0, 'img size cant match patch size'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_encoder = SeqEncoder(in_channels=2, 
                                    embed_dim=self.embed_dim, 
                                    num_heads=self.num_heads, 
                                    depth=3, 
                                    mlp_ratio=4, 
                                    dropout=0.3)
        self.img_encoder = VisionEncoder(
                                    img_size=img_size, 
                                    patch_size=self.patch_size, 
                                    in_channels=3, 
                                    embed_dim=self.embed_dim, 
                                    num_heads=self.num_heads, 
                                    depth=6, 
                                    mlp_ratio=4, 
                                    dropout=0.3)
        self.poly_decoder = PolygonDecoderV2(
                                    embed_dim=self.embed_dim, 
                                    num_heads=self.num_heads, 
                                    depth=6, 
                                    mlp_ratio=4, 
                                    dropout=0.3)
        self.norm = nn.BatchNorm1d(self.embed_dim)
        self.reg_head = nn.Linear(self.embed_dim,2)
        # self.cls_head = nn.Linear(256,1)
    
    def forward(self,img,coords,pad_mask):
        # img : [ B C H W ]
        # coords: [ B S 2 ]
        img_embed = self.img_encoder(img) # [ N B D ]

        coords = self.seq_encoder(coords,pad_mask) # 
        coords = self.poly_decoder(coords,img_embed) # [ S B C ]
        coords = self.norm(coords.permute(1,2,0))# [ B C S ] 
        reg_out = self.reg_head(coords.permute(0,2,1)) # [ B S C ] 
        # cls_out = self.cls_head(coords.permute(0,2,1))
        # return reg_out, cls_out
        return reg_out
    
class PolyRefinerV3(nn.Module):
    def __init__(self, img_size, patch_size=8, embed_dim=256,num_heads=4,**kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        assert self.img_size%self.patch_size==0, 'img size cant match patch size'
        self.embed_dim = embed_dim
        self.num_heads = num_heads



        self.seq_encoder = SeqEncoder(in_channels=2, 
                                    embed_dim=1024, 
                                    num_heads=self.num_heads, 
                                    depth=1, 
                                    mlp_ratio=1, 
                                    dropout=0.1)
        self.img_encoder = models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        self.img_encoder = self.img_encoder.features
        self.poly_decoder = PolygonDecoderV2(
                                    embed_dim=1024, 
                                    num_heads=self.num_heads, 
                                    depth=6, 
                                    mlp_ratio=4, 
                                    dropout=0.3)
        self.norm = nn.BatchNorm1d(1024)
        self.reg_head = nn.Linear(1024,2)
        # self.cls_head = nn.Linear(256,1)
    
    def forward(self,img,coords,pad_mask):
        # img : [ B C H W ]
        # coords: [ B S 2 ]
        img_embed = self.img_encoder(img) # [ B 7 7 768 ] base版本是1024 channel
        img_embed = img_embed.flatten(1,2).transpose(0,1) # [ 49 B 768 ]
        coords = self.seq_encoder(coords,pad_mask) # 
        coords = self.poly_decoder(coords,img_embed) # [ S B C ]
        coords = self.norm(coords.permute(1,2,0))# [ B C S ] 
        reg_out = self.reg_head(coords.permute(0,2,1)) # [ B S C ] 
        # cls_out = self.cls_head(coords.permute(0,2,1))
        # return reg_out, cls_out
        return reg_out
    
class PolyGenerator(nn.Module):
    def __init__(self, img_size, patch_size=8, embed_dim=256,num_heads=4,**kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        assert self.img_size%self.patch_size==0, 'img size cant match patch size'
        self.embed_dim = embed_dim
        self.num_heads = num_heads



        # self.seq_encoder = SeqEncoder(in_channels=2, 
        #                             embed_dim=1024, 
        #                             num_heads=self.num_heads, 
        #                             depth=1, 
        #                             mlp_ratio=1, 
        #                             dropout=0.1)
        self.img_encoder = models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        self.img_encoder = self.img_encoder.features
        # self.poly_decoder = PolygonDecoderV2(
        #                             embed_dim=1024, 
        #                             num_heads=self.num_heads, 
        #                             depth=6, 
        #                             mlp_ratio=4, 
        #                             dropout=0.3)
        
        self.token_fc = nn.Linear(49,100)
        # self.norm1 = nn.BatchNorm1d(100)
        # self.norm2 = nn.BatchNorm1d(1024)
        self.seqconvs = nn.Sequential(
                                    nn.Conv1d(1024,256,3,1,1),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Conv1d(256,256,3,1,1),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                        )
        self.reg_head = nn.Linear(256,2)
        # self.actv = nn.ReLU()
        # self.cls_head = nn.Linear(256,1)
    
    def forward(self,img):
        # img : [ B C H W ]
        # coords: [ B S 2 ]
        img_embed = self.img_encoder(img) # [ B 7 7 768 ] base版本是1024 channel
        img_embed = img_embed.flatten(1,2).permute(0,2,1) # [ B C 49]
        # coords = self.seq_encoder(coords,pad_mask) # 
        img_embed = self.token_fc(img_embed)# [B C 100]
        img_embed = self.seqconvs(img_embed)
        # coords = self.norm(coords)# [ B C S ] 
        reg_out = self.reg_head(img_embed.permute(0,2,1)) # [ B S 2 ] 
        # cls_out = self.cls_head(coords.permute(0,2,1))
        # return reg_out, cls_out
        return reg_out