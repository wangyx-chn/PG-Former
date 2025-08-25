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

from models.cnn_network import CNNPolyGenerator
from dataset import BuildPolyGtOnlyDataset
from utils import DynamicBatchSampler, my_collate_fn
from test import GenIoU as IoU

a = random

class PolyGenModel(BaseModel):

    def __init__(self,model_args):
        super().__init__()
        self.network = CNNPolyGenerator(**model_args)
        self.reg_loss = nn.MSELoss()
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([100]))
        self.bce_loss = nn.BCELoss()

    def forward(self, imgs, data_samples=None, mode='tensor'):
        # aug = random.uniform(0.5, 1.5)
        # aug = 0.8
        # data_samples['aug'] = aug
        # reg = self.network(imgs,input,pad_mask)
        reg = self.network(imgs).squeeze()
        # weights = 1 - data_samples['mask']['pad_mask']
        # target = (data_samples['poly_target']['data'][:,:10,:].flatten(1)-0.5)*2
        # target = (data_samples['poly_target']['data'][:,0]-0.5)*2
        target = data_samples['edge']
        # reg_loss = self.reg_loss(reg, target) # 缩放到[-1,1]
        # cls_loss = self.cls_loss(cls.squeeze(),data_samples['cls_target']['data'])
        loss = self.cls_loss(reg,target)
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
    
    img_size = 224
    model_args = dict(img_size = 224,patch_size=4, embed_dim=16, num_heads=1)
    
    model = PolyGenModel(model_args)
    

    train_set = BuildPolyGtOnlyDataset(
        root = '/home/guning.wyx/code/mmengine/data/WHUBuilding/polyrefine/dataset_sample50',
        img_dir='image_patch',
        gt_dir='gt_poly',
        pt_len_path='point_length_train.pkl',
        img_size=img_size,
        # select_k=10000,
        transform=transform,
        target_transform=target_transform)

    valid_set = BuildPolyGtOnlyDataset(
        root = '/home/guning.wyx/code/mmengine/data/WHUBuilding/polyrefine/dataset_sample50',
        img_dir='image_patch',
        gt_dir='gt_poly',
        pt_len_path='point_length_train.pkl',
        img_size=img_size,
        select_k=1000,
        transform=transform,
        target_transform=target_transform)

    train_dataloader = DataLoader(
        batch_size=1,
        dataset=train_set,
        batch_sampler=DynamicBatchSampler(dataset=train_set, sampler=DefaultSampler(train_set), max_token=9600, shuffle=True),
        collate_fn=my_collate_fn)
    
    val_dataloader = DataLoader(
        batch_size=1,
        dataset=train_set,
        batch_sampler=DynamicBatchSampler(dataset=valid_set, sampler=DefaultSampler(valid_set), max_token=12800, shuffle=True),
        collate_fn=my_collate_fn)

    
    runner = Runner(
        model=model,
        work_dir='./work_dirs/PolyGenerator_CNNModel_sample50_full_edge',
        # load_from='/home/guning.wyx/code/mmengine/work_dirs/PolyGenerator_CNNModel_sample50_full_edge/epoch_36.pth',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(
            # type=AmpOptimWrapper, optimizer=dict(lr=0.0005, betas=(0.9, 0.999), type='AdamW',eps=1e-8, weight_decay=0.01)),
            type=AmpOptimWrapper, optimizer=dict(lr=0.0001, momentum=0.9, type='SGD', weight_decay=0.0001)),
        param_scheduler = [dict(begin=0,by_epoch=False,end=500,start_factor=0.01,type='LinearLR'),
                            dict(begin=0,by_epoch=True,end=36,gamma=0.1,milestones=[18,27],type='MultiStepLR'),],
        train_cfg=dict(by_epoch=True, max_epochs=36, val_begin=40, val_interval=3),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        test_dataloader=val_dataloader,
        test_cfg=dict(),
        val_evaluator=dict(type=IoU),
        test_evaluator=dict(type=IoU),
        launcher=args.launcher,
        # custom_hooks=[SegVisHook('data/CamVid')],
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=10),
                           logger=dict(interval=100, type='LoggerHook')),
        # visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend', save_dir='/home/guning.wyx/code/mmengine/runs/polyv2_s1v2d2')])
    )
    runner.train()
    # runner.test()


if __name__ == '__main__':
    main()
    # writer.close()