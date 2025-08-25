import argparse
import csv
import os
import os.path as osp
import shutil
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.models.segmentation import deeplabv3_resnet50

from mmengine.dist import master_only
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner
from mmengine.dataset import DefaultSampler

from models.detr import DETR
from models.detr import NestedTensor
from datasets.dynamic_dataset import RealGTPolyDataset
from utils import DynamicBatchSampler, my_collate_fn
from test import GenIoU as IoU
# from tensorboardX import SummaryWriter


class PolyGenModel(BaseModel):

    def __init__(self,model_args):
        super().__init__()
        self.network = DETR(**model_args)
        # self.reg_loss = nn.MSELoss()
        self.reg_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, imgs, data_samples=None, mode='tensor'):
        # aug = random.uniform(0.5, 1.5)
        # aug = 0.8
        # data_samples['aug'] = aug
        # reg = self.network(imgs,input,pad_mask)
        inputs = NestedTensor(imgs,torch.zeros_like(imgs)[:,0].to(imgs.device))
        outputs = self.network(inputs)
        pred_coords_x = outputs['pred_coords_x']
        pred_coords_y = outputs['pred_coords_y']
        pred_coords = torch.concat([pred_coords_x.unsqueeze(2),pred_coords_y.unsqueeze(2)],dim=2).permute(0,3,1,2)
        pred_logits = outputs['pred_logits']
        
        # valid_mask = 1 - data_samples['real_gt']['pad_mask']
        max_len = data_samples['real_gt']['data'].shape[1]
        target_coords = torch.zeros((pred_coords.shape[0],pred_coords.shape[2],pred_coords.shape[3]),dtype=torch.long).to(pred_coords.device)
        target_coords[:,:max_len] = (data_samples['real_gt']['data']*224).to(torch.long)
        target_cls = torch.zeros_like(pred_logits).to(pred_logits.device)
        target_cls[:,:max_len] = 1-data_samples['real_gt']['pad_mask'][...,:1]
        target_coords = torch.clamp(target_coords,0,223)
        target_coords[target_cls.squeeze()==0]=-100

        # 加入一个引导回归分类的软标签，用一个高斯分布
        mu = target_coords.unsqueeze(1)
        positions = torch.arange(224, dtype=torch.float32, device=mu.device)
        positions = positions.view(1, 224, 1, 1)
        sigma = 2
        gaussian = torch.exp(- (positions - mu) ** 2 / (2 * sigma ** 2)) # B 224 50 2
        soft_loss = self.mse_loss(torch.sigmoid(pred_coords)*(target_coords.unsqueeze(1)!=-100),gaussian*(target_coords.unsqueeze(1)!=-100))
        # target = (data_samples['real_gt']['data'].view(-1,100)-0.5)*2
        reg_loss = self.reg_loss(pred_coords, target_coords) # 缩放到[-1,1]
        cls_loss = self.cls_loss(pred_logits,target_cls)
        loss = reg_loss + cls_loss + soft_loss
        # data_samples['imgs'] = imgs.cpu().numpy()
        # writer.add_scalar("train_loss", loss)
        # writer.add_scalar("reg_loss", reg_loss)
        # writer.add_scalar("cls_loss", cls_loss)
        if mode == 'loss':
            return {'loss': loss, 'reg_loss': reg_loss, 'cls_loss': cls_loss, 'soft_loss': soft_loss}
        elif mode == 'predict':
            return {'pred_coords':pred_coords, 'pred_logits':pred_logits}, data_samples




def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(**norm_cfg)])

    target_transform = transforms.Lambda(
        lambda x: torch.tensor(x, dtype=torch.float32))
    
    img_size = 224
    model_args = dict(num_queries=50)
    
    model = PolyGenModel(model_args)
    # for param in model.network.backbone.parameters():
    #     param.requires_grad = False

    train_set = RealGTPolyDataset(
        root = '/home/guning.wyx/code/mmengine/data/WHUBuilding/polyrefine/dataset_sample50',
        img_dir='image_patch',
        gt_dir='gt_poly',
        # pt_len_path='point_length_real_gt.pkl',
        pt_len_path='point_length_train.pkl',
        img_size=img_size,
        # select_k = 30000,
        transform=transform,
        target_transform=target_transform)

    valid_set = RealGTPolyDataset(
        root = '/home/guning.wyx/code/mmengine/data/WHUBuilding/polyrefine/dataset_sample50_val',
        img_dir='image_patch',
        gt_dir='gt_poly',
        # pt_len_path='point_length_real_gt.pkl',
        pt_len_path='point_length_train.pkl',
        img_size=img_size,
        # select_k = 10000,
        transform=transform,
        target_transform=target_transform)

    train_dataloader = DataLoader(
        batch_size=1,
        dataset=train_set,
        batch_sampler=DynamicBatchSampler(dataset=train_set, sampler=DefaultSampler(train_set), max_token=12800, shuffle=True),
        collate_fn=my_collate_fn)
    
    val_dataloader = DataLoader(
        batch_size=1,
        dataset=valid_set,
        batch_sampler=DynamicBatchSampler(dataset=valid_set, sampler=DefaultSampler(valid_set), max_token=12800, shuffle=True),
        collate_fn=my_collate_fn)

    
    runner = Runner(
        model=model,
        work_dir='./work_dirs/PolyGenDETR_CLS_sample50_layer2_full',
        # load_from='/home/guning.wyx/code/mmengine/work_dirs/PolyGenDETR_CLS_sample50_layer12_full/epoch_48.pth',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(
            type=AmpOptimWrapper, optimizer=dict(lr=0.0001, betas=(0.9, 0.999), type='AdamW',eps=1e-8, weight_decay=0.01),
            clip_grad=dict(max_norm=10,norm_type=2)),
        param_scheduler = [dict(begin=0,by_epoch=False,end=1000,start_factor=0.01,type='LinearLR'),
                            dict(begin=0,by_epoch=True,end=48,gamma=0.1,milestones=[24,36],type='MultiStepLR'),],
        train_cfg=dict(by_epoch=True, max_epochs=48, val_begin=40, val_interval=2),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        test_dataloader=val_dataloader,
        test_cfg=dict(),
        val_evaluator=dict(type=IoU),
        test_evaluator=dict(type=IoU),
        launcher=args.launcher,
        # custom_hooks=[SegVisHook('data/CamVid')],
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1),
                           logger=dict(interval=100, type='LoggerHook')),
        # visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend', save_dir='/home/guning.wyx/code/mmengine/runs/polyv2_s1v2d2')])
    )
    runner.train()
    # runner.test()


if __name__ == '__main__':
    main()
    # writer.close()