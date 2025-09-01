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
from torchvision.datasets import VisionDataset
from torchvision.models.segmentation import deeplabv3_resnet50

from mmengine.dist import master_only
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner
from mmengine.dataset import DefaultSampler

from models.autoregdetr import AutoRegDETR, NestedTensor
from datasets.dynamic_dataset_autoreg import RealGTPolyDataset
from datasets.denoise_dataset_autoreg import RealGTPolyDataset as DenoisePolyDataset
from utils import DynamicBatchSampler, my_collate_fn
from test_autoreg import AutoRegIoU as IoU
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
            inputs = NestedTensor(imgs,torch.zeros_like(imgs)[:,0].to(imgs.device))

            # 处理input
            input_coord = data_samples['real_gt']['data']*223+10
            input_coord = torch.clamp(input_coord,10,234)
            device = input_coord.device
            b, s = input_coord.shape[:2] # [ B S 2 ]
            input_coord = torch.cat([input_coord,torch.zeros([b,1,2]).to(device)], dim=1)
            target_coords = deepcopy(input_coord)
            pad_mask = data_samples['real_gt']['pad_mask'][...,0]
            pad_mask = torch.cat([torch.zeros([b,1]).to(device), pad_mask], dim=1)
            lens = (1-pad_mask).sum(dim=1).to(torch.int)-1
            
            for b,l in enumerate(lens):
                input_coord[b] = torch.concat([torch.Tensor([[0,0]]).to(device),input_coord[b][:-1]],dim=0)
                target_coords[b] = torch.concat([target_coords[b][:l],torch.Tensor([[243,243]]).to(device),target_coords[b][l:-1]],dim=0)
            query = NestedTensor(input_coord.to(torch.int64),pad_mask)
            outputs = self.network(inputs,query)
            pred_coords_x = outputs['pred_coords_x']
            pred_coords_y = outputs['pred_coords_y']
            pred_coords = torch.concat([pred_coords_x.unsqueeze(2),pred_coords_y.unsqueeze(2)],dim=2).permute(0,3,1,2)
            # pred_logits = outputs['pred_logits']
            
            # valid_mask = 1 - data_samples['real_gt']['pad_mask']
            # max_len = data_samples['real_gt']['data'].shape[1]
            # target_coords = torch.zeros((pred_coords.shape[0],pred_coords.shape[2],pred_coords.shape[3]),dtype=torch.long).to(pred_coords.device)
            # target_coords[:,:max_len] = (data_samples['real_gt']['data']*224).to(torch.long)
            # target_cls = torch.zeros_like(pred_logits).to(pred_logits.device)
            # target_cls[:,:max_len] = 1-data_samples['real_gt']['pad_mask'][...,:1]
            # target_coords = torch.clamp(target_coords,0,223)
            target_coords = target_coords.to(torch.int64)
            target_coords[pad_mask==1]=-100

            # 加入一个引导回归分类的软标签，用一个高斯分布
            mu = target_coords.unsqueeze(1)
            positions = torch.arange(244, dtype=torch.float32, device=mu.device)
            positions = positions.view(1, 244, 1, 1)
            sigma = 2
            gaussian = torch.exp(- (positions - mu) ** 2 / (2 * sigma ** 2)) # B 224 50 2
            soft_loss = self.mse_loss(torch.sigmoid(pred_coords)*(target_coords.unsqueeze(1)!=-100),gaussian*(target_coords.unsqueeze(1)!=-100))
            # target = (data_samples['real_gt']['data'].view(-1,100)-0.5)*2
            reg_loss = self.reg_loss(pred_coords, target_coords) # 缩放到[-1,1]
            # cls_loss = self.cls_loss(pred_logits,target_cls)
            loss = reg_loss + soft_loss
            # data_samples['imgs'] = imgs.cpu().numpy()
            # writer.add_scalar("train_loss", loss)
            # writer.add_scalar("reg_loss", reg_loss)
            # writer.add_scalar("cls_loss", cls_loss)
        
            return {'loss': loss, 'reg_loss': reg_loss, 'soft_loss': soft_loss}
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
    parser.add_argument('-t','--trainroot',type=str,default='/home/guning.wyx/code/mmengine/data/WHUBuilding/polygeneration/dataset_polygon50_0.1margin_0.5singlenoise20_train')
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
    model_args = dict(hidden_dim=256)
    
    model = PolyGenModel(model_args)
    # for param in model.network.backbone.parameters():
    #     param.requires_grad = False
    train_root = args.trainroot
    print(train_root)
    # RealGTPolyDataset,DenoisePolyDataset
    train_set = DenoisePolyDataset(
        root = train_root,
        img_dir='image_patch',
        gt_dir='gt_poly',
        pt_len_path='point_length.pkl',
        img_size=img_size,
        # select_k = 30000,
        transform=transform,
        target_transform=target_transform)
    
    val_root = '/home/guning.wyx/code/mmengine/data/WHUBuilding/polygeneration/dataset_polygon50_margin01_val'
    valid_set = RealGTPolyDataset(
        root = val_root,
        img_dir='image_patch',
        gt_dir='gt_poly',
        # pt_len_path='point_length_real_gt.pkl',
        pt_len_path='point_length.pkl',
        img_size=img_size,
        # select_k = 10000,
        transform=transform,
        target_transform=target_transform)

    # train_dataloader = DataLoader(
    #     batch_size=1,
    #     dataset=train_set,
    #     batch_sampler=DynamicBatchSampler(dataset=train_set, sampler=DefaultSampler(train_set), max_token=51200, shuffle=True),
    #     collate_fn=my_collate_fn)

    train_dataloader = DataLoader(
        batch_size=128,
        dataset=train_set,
        shuffle=True,
        collate_fn=my_collate_fn)
    
    # val_dataloader = DataLoader(
    #     batch_size=1,
    #     dataset=valid_set,
    #     batch_sampler=DynamicBatchSampler(dataset=valid_set, sampler=DefaultSampler(valid_set), max_token=51200, shuffle=True),
    #     collate_fn=my_collate_fn)

    val_dataloader = DataLoader(
        batch_size=128,
        dataset=valid_set,
        shuffle=False,
        collate_fn=my_collate_fn)
    # work_dir=f'/home/guning.wyx/code/mmengine/work_dirs/PolyGenDETR_AutoReg_polygon50_0.1margin_1.0noise20'
    work_dir=f'/home/guning.wyx/code/mmengine/work_dirs/PolyGenDETR_AutoReg_Denoise_{train_root.split("/")[-1]}'
    # work_dir=f'/home/guning.wyx/code/mmengine/work_dirs/PolyGenDETR_Deniose_AutoReg_{train_root.split("/")[-1]}'
    val_evaluator=dict(type=IoU,work_dir=work_dir,img_dir=osp.join(val_root,'image_patch'))
    
    runner = Runner(
        model=model,
        work_dir=work_dir,
        # load_from='/home/guning.wyx/code/mmengine/work_dirs/PolyGenDETR_AutoReg_polygon50_margin01/epoch_48.pth',
        # resume=True,
        train_dataloader=train_dataloader,
        optim_wrapper=dict(
            type=AmpOptimWrapper, optimizer=dict(lr=0.0001, betas=(0.9, 0.999), type='AdamW',eps=1e-8, weight_decay=0.01),
            clip_grad=dict(max_norm=10,norm_type=2)),
        param_scheduler = [dict(begin=0,by_epoch=False,end=5000,start_factor=0.01,type='LinearLR'),
                            dict(begin=0,by_epoch=True,end=48,gamma=0.1,milestones=[24,36],type='MultiStepLR'),],
        train_cfg=dict(by_epoch=True, max_epochs=48, val_begin=40, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        test_dataloader=val_dataloader,
        test_cfg=dict(),
        val_evaluator=val_evaluator,
        test_evaluator=val_evaluator,
        launcher=args.launcher,
        # custom_hooks=[SegVisHook('data/CamVid')],
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1),
                           logger=dict(interval=100, type='LoggerHook')),
        # visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend', save_dir='/home/guning.wyx/code/mmengine/runs/polyv2_s1v2d2')])
    )
    runner.train()


if __name__ == '__main__':
    main()
    # writer.close()