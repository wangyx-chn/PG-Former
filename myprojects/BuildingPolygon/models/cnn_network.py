import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNeXt101_64X4D_Weights, ViT_B_16_Weights, Swin_B_Weights
from .transformer import VisionEncoder,PolygonDecoder,PolygonDecoderV2,SeqEncoder

    
class CNNPolyGenerator(nn.Module):
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
        resnext = models.resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        # self.img_encoder = torch.nn.Sequential(*list(img_encoder.children())[:-1])
        self.backbone = nn.Sequential(resnext.conv1,resnext.bn1,resnext.relu,resnext.maxpool)
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        # self.avgpool = resnext.avgpool
        
        # self.poly_decoder = PolygonDecoderV2(
        #                             embed_dim=1024, 
        #                             num_heads=self.num_heads, 
        #                             depth=6, 
        #                             mlp_ratio=4, 
        #                             dropout=0.3)
        self.fpn = FPN([256,512,1024,2048])
        # self.token_fc = nn.Linear(1024,1)
        self.out_conv = nn.Conv2d(64,1,1,1)
        # self.norm1 = nn.BatchNorm1d(100)
        # self.norm2 = nn.BatchNorm1d(2048)
        self.seqconvs = nn.ModuleList([nn.Sequential(nn.Conv2d(1024,512,3,1,1),nn.BatchNorm2d(512),nn.ReLU()),
                                    nn.Sequential(nn.Conv2d(512,512,1,1),nn.BatchNorm2d(512),nn.ReLU()),
                                    nn.Sequential(nn.Conv2d(512,128,1,1),nn.BatchNorm2d(128),nn.ReLU()),
                                    nn.Sequential(nn.Conv2d(128,64,1,1),nn.BatchNorm2d(64),nn.ReLU()),
                                    ])
        # self.reg_head = nn.Linear(64,100)
        # self.actv = nn.ReLU()
        # self.cls_head = nn.Linear(256,1)
    
    def forward(self, x):
        c1 = self.backbone(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # print([c.shape for c in [c1,c2,c3,c4,c5]])
        fpn_features = self.fpn([c2,c3,c4,c5])
        for i in range(3):
            fpn_features[i] = F.interpolate(fpn_features[i],fpn_features[-1].shape[-2:],mode='bilinear')
        x = torch.concat(fpn_features,dim=1)
        for layer in self.seqconvs:
            x = layer(x)
        x = self.out_conv(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.token_fc(x)
        x = F.interpolate(x,self.img_size,mode='bilinear')
        return x
    # def forward(self,img):
    #     # img : [ B C H W ]
    #     # coords: [ B S 2 ]
    #     img_embed = self.img_encoder(img) # [ B 7 7 768 ] base版本是1024 channel
    #     img_embed = img_embed.flatten(1) # [ B C 49]
    #     # coords = self.seq_encoder(coords,pad_mask) # 
    #     img_embed = self.token_fc(img_embed)# [B C 100]
    #     # img_embed = self.seqconvs(img_embed)
    #     # coords = self.norm(coords)# [ B C S ] 
    #     # reg_out = self.reg_head(img_embed) # [ B S 2 ] 
    #     # cls_out = self.cls_head(coords.permute(0,2,1))
    #     # return reg_out, cls_out
    #     return img_embed
    
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()

        # 对不同层的输入特征进行 1x1 卷积映射
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        # 对融合后的特征进行 3x3 卷积平滑
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, inputs):
        # assert inputs 为多特征图列表
        c2, c3, c4, c5 = inputs

        # Step 1: 对最高层 (C5) 特征图进行 lateral 映射
        p5 = self.lateral_convs[3](c5)  # lateral 映射
        fpn_features = [self.fpn_convs[3](p5)]  # 平滑处理

        # Step 2: 自顶向下逐级融合特征
        p4 = self.lateral_convs[2](c4) + nn.functional.interpolate(p5, scale_factor=2, mode='nearest')
        fpn_features.insert(0, self.fpn_convs[2](p4))

        p3 = self.lateral_convs[1](c3) + nn.functional.interpolate(p4, scale_factor=2, mode='nearest')
        fpn_features.insert(0, self.fpn_convs[1](p3))

        p2 = self.lateral_convs[0](c2) + nn.functional.interpolate(p3, scale_factor=2, mode='nearest')
        fpn_features.insert(0, self.fpn_convs[0](p2))

        return fpn_features  # 返回多尺度特征图列表 (P2, P3, P4, P5)