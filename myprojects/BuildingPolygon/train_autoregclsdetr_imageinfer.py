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
from copy import deepcopy
from torch.utils.data import DataLoader
from mmengine.model import BaseModel
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner
from mmengine.dataset import DefaultSampler

from models.autoregdetr import AutoRegDETR, NestedTensor
from datasets.imagelevel_dataset_autoreg import ImagePolyDataset
from utils import DynamicBatchSampler, my_collate_fn
from test_autoreg_imginfer import AutoRegIoU as IoU
# from tensorboardX import SummaryWriter


class PolyGenModel(BaseModel):

    def __init__(self,model_args):
        super().__init__()
        self.network = AutoRegDETR(**model_args)
        # self.reg_loss = nn.MSELoss()
        self.reg_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, imgs, data_samples=None, mode='tensor'):
        # aug = random.uniform(0.5, 1.5)
        # aug = 0.8
        # data_samples['aug'] = aug
        # reg = self.network(imgs,input,pad_mask)
        if mode == 'loss':
            print('This script is only for image infer!!!')
            raise NotImplementedError
        elif mode == 'predict':
            bs = imgs.shape[0]
            inputs = NestedTensor(imgs,torch.zeros_like(imgs)[:,0].to(imgs.device))
            seqlen_dict = {}
            flag_dict = {i:i for i in range(bs)}
            # 定义开始标志符
            pred_coords = torch.zeros([bs,1,2],dtype=torch.long).to(imgs.device)
            query = NestedTensor(pred_coords,torch.zeros_like(pred_coords)[...,0].to(imgs.device))
            coords_x=0
            coords_y=0
            l = 1
            while len(seqlen_dict)<bs:
                # cur_bs = len(inputs.tensors)
                outputs = self.network(inputs,query)
                pred_coords_x = outputs['pred_coords_x']
                pred_coords_y = outputs['pred_coords_y']
                coords_x = torch.argmax(pred_coords_x[:,l-1:l],dim=-1,keepdim=True)
                coords_y = torch.argmax(pred_coords_y[:,l-1:l],dim=-1,keepdim=True)
                pred_coords = torch.cat([pred_coords,torch.cat([coords_x,coords_y],dim=-1)], dim=1)
                l += 1
                end_inds = []
                for i in flag_dict:
                    if (coords_x[i]>240 or coords_y[i]>240 or len((query.tensors)[i])>55):
                        seqlen_dict[i] = l
                        end_inds.append(i)
                for i in end_inds:
                    flag_dict.pop(i)
                
                query = NestedTensor(pred_coords,torch.zeros_like(pred_coords)[...,0].to(imgs.device))

            batch_pred = [pred_coords[i][:seqlen_dict[i]] for i in range(bs)]
            return {'pred_coords':batch_pred}, data_samples




def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('-w','--workdir', type=str, default='/home/guning.wyx/code/mmengine/work_dirs/PolyGenDETR_AutoReg_polygon50_margin01')
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
    model_args = dict(hidden_dim=256)
    
    model = PolyGenModel(model_args)
    # for param in model.network.backbone.parameters():
    #     param.requires_grad = False
    img_dir = '/home/guning.wyx/code/mmdetection/demo/AmapServer/pic_muku'
    detect_path = '/home/guning.wyx/code/mmdetection/demo/AmapServer/pred_results.pkl'
    vis_dir = '/home/guning.wyx/code/mmdetection/demo/AmapServer/vis'
    valid_set = ImagePolyDataset(
        img_dir=img_dir,
        detect_path=detect_path,
        thr=0.5,
        img_size=img_size,
        transform=transform,
        target_transform=target_transform)

    train_dataloader = DataLoader(
        batch_size=128,
        dataset=valid_set,
        shuffle=False,
        collate_fn=my_collate_fn)
    
    val_dataloader = DataLoader(
        batch_size=128,
        dataset=valid_set,
        shuffle=False,
        collate_fn=my_collate_fn)
    
    work_dir=args.workdir
    val_evaluator=dict(type=IoU,work_dir=work_dir,vis_dir=vis_dir)
    
    runner = Runner(
        model=model,
        work_dir=work_dir,
        load_from=osp.join(work_dir,'epoch_48.pth'),
        train_dataloader=train_dataloader,
        optim_wrapper=dict(
            type=AmpOptimWrapper, optimizer=dict(lr=0.0001, betas=(0.9, 0.999), type='AdamW',eps=1e-8, weight_decay=0.01),
            clip_grad=dict(max_norm=10,norm_type=2)),
        param_scheduler = [dict(begin=0,by_epoch=False,end=5000,start_factor=0.01,type='LinearLR'),
                            dict(begin=0,by_epoch=True,end=48,gamma=0.1,milestones=[24,36],type='MultiStepLR'),],
        train_cfg=dict(by_epoch=True, max_epochs=48, val_begin=40, val_interval=2),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        test_dataloader=val_dataloader,
        test_cfg=dict(),
        val_evaluator=val_evaluator,
        test_evaluator=val_evaluator,
        launcher=args.launcher,
        # custom_hooks=[SegVisHook('data/CamVid')],
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1),
                           logger=dict(interval=10, type='LoggerHook')),
        # visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend', save_dir='/home/guning.wyx/code/mmengine/runs/polyv2_s1v2d2')])
    )
    # runner.train()
    runner.test()


if __name__ == '__main__':
    main()
    # writer.close()