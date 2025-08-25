import argparse
import csv
import os
import os.path as osp
import shutil

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

from models.network import PolyRefinerV2
from dataset import BuildPolyDataset
from utils import DynamicBatchSampler, my_collate_fn
from test import IoU
from tensorboardX import SummaryWriter
# writer = SummaryWriter(log_dir="/home/guning.wyx/code/mmengine/runs/polyexp")
def create_palette(csv_filepath):
    color_to_class = {}
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_class[(r, g, b)] = idx
    return color_to_class

class PolyRefineModel(BaseModel):

    def __init__(self,model_args):
        super().__init__()
        self.network = PolyRefinerV2(**model_args)
        self.reg_loss = nn.L1Loss()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, imgs, data_samples=None, mode='tensor'):
        reg = self.network(imgs, data_samples['mask']['data'],data_samples['mask']['pad_mask'])
        weights = 1 - data_samples['mask']['pad_mask']
        target = data_samples['poly_target']['data']-data_samples['mask']['data'].detach()
        reg_loss = self.reg_loss(reg*weights, target*weights)
        # cls_loss = self.cls_loss(cls.squeeze(),data_samples['cls_target']['data'])
        loss = reg_loss
        # writer.add_scalar("train_loss", loss)
        # writer.add_scalar("reg_loss", reg_loss)
        # writer.add_scalar("cls_loss", cls_loss)
        if mode == 'loss':
            return {'loss': loss}
        elif mode == 'predict':
            return {'pred_reg':reg}, data_samples




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
    
    img_size = 128
    model_args = dict(img_size = 128,patch_size=4, embed_dim=256, num_heads=1)
    
    model = PolyRefineModel(model_args)
    

    train_set = BuildPolyDataset(
        root = '/home/guning.wyx/code/mmengine/data/WHUBuilding/polyrefine/equal_sample_dataset',
        img_dir='image_patch',
        mask_dir='mask_poly',
        gt_dir='gt_poly',
        pt_len_path='point_length_train.pkl',
        img_size=img_size,
        transform=transform,
        target_transform=target_transform)

    valid_set = BuildPolyDataset(
        root = '/home/guning.wyx/code/mmengine/data/WHUBuilding/polyrefine/equal_sample_dataset',
        img_dir='image_patch',
        mask_dir='mask_poly',
        gt_dir='gt_poly',
        pt_len_path='point_length_train.pkl',
        img_size=img_size,
        transform=transform,
        target_transform=target_transform)

    train_dataloader = DataLoader(
        batch_size=1,
        dataset=train_set,
        batch_sampler=DynamicBatchSampler(dataset=train_set, sampler=DefaultSampler(train_set), max_token=25600, shuffle=True),
        collate_fn=my_collate_fn)
    
    val_dataloader = DataLoader(
        batch_size=1,
        dataset=train_set,
        batch_sampler=DynamicBatchSampler(dataset=valid_set, sampler=DefaultSampler(valid_set), max_token=12800, shuffle=True),
        collate_fn=my_collate_fn)

    
    runner = Runner(
        model=model,
        work_dir='./work_dirs/PolyRefineModel_v2_s1v2d2',
        load_from='/home/guning.wyx/code/mmengine/work_dirs/PolyRefineModel_v2_absolute_offset/epoch_14.pth',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(
            type=AmpOptimWrapper, optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001)),
        param_scheduler = [dict(begin=0,by_epoch=False,end=500,start_factor=0.33,type='LinearLR'),
                            dict(begin=0,by_epoch=True,end=36,gamma=0.1,milestones=[24,],type='MultiStepLR'),],
        train_cfg=dict(by_epoch=True, max_epochs=36, val_begin=40),
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
    # runner.train()
    runner.test()


if __name__ == '__main__':
    main()
    # writer.close()