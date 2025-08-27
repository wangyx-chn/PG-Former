import os
import os.path as osp
import pickle
import numpy as np
import cv2
import math
import json
from PIL import Image
from torchvision.datasets import VisionDataset
import torch.nn.functional as F
from pycocotools.coco import COCO
import random


class RealGTPolyDataset(VisionDataset):

    def __init__(self,
                 root,
                 img_dir,
                 gt_dir,
                 pt_len_path,
                 img_size=128,
                 select_k = None,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root=root, transform=transform, target_transform=target_transform)
        self.img_dir = osp.join(root,img_dir)
        self.gt_dir = osp.join(root,gt_dir)
        self.img_size = img_size
        self.length = len(os.listdir(self.img_dir))
        self.point_length = []
        # for i in range(self.length):
        #     self.point_length.append(len(np.load(osp.join(self.mask_dir,f'{i}.npy'))))
        with open(osp.join(root,pt_len_path),'rb') as f:
            self.point_length = pickle.load(f)

        self.sorted_length = np.sort(self.point_length)
        self.sorted_indices = np.argsort(self.point_length)
        with open(osp.join(root,'noise_record.json'),'r') as f:
            noise = json.load(f)
        
        if select_k:
            random.seed(2)
            self.sorted_indices = random.choices(self.sorted_indices,k=select_k)
            self.sorted_length = [self.sorted_length[i] for i in self.sorted_indices]
            self.length = len(self.sorted_length)
        print('denoise')
        self.denoise_ids = []
        for i in range(self.length):
            if noise==[]:
                self.denoise_ids.append(i)
                continue
            if i==noise[0]['id']:
                noise.pop(0)
            else:
                self.denoise_ids.append(i)
        self.length = len(self.denoise_ids)
        print('finished')
        pass

    def __len__(self):
        return self.length
    
    def __getitem__(self, id):
        index = self.denoise_ids[id]
        img_path = os.path.join(self.img_dir, f'{index}.png')
        
        # gt_path = os.path.join(self.gt_dir, f'{index}_sampled_gt.npy') #for train
        # cls_path = os.path.join(self.gt_dir, f'{index}_iscorner_for_sampled.npy')
        real_gt_path = os.path.join(self.gt_dir, f'{index}_real_gt.npy') # for eval

        img = Image.open(img_path).convert('RGB')
        ori_shape = img.size
        # gt = np.load(gt_path)
        # cls_gt = np.load(cls_path)
        real_gt = np.load(real_gt_path)
        # img = np.array(img)
        # img = np.zeros_like(img)
        # img = cv2.fillPoly(img,[(real_gt*np.array(ori_shape)).astype(np.int32).reshape(-1,1,2)],color=(255,255,255))


        if self.transform is not None:
            img = self.transform(img)
        img = F.interpolate(img.unsqueeze(0),size=self.img_size,mode='bilinear').squeeze(0)
        

        if self.target_transform is not None:
            # gt = self.target_transform(gt)
            # cls_gt = self.target_transform(cls_gt)
            real_gt = self.target_transform(real_gt)
        meta = dict(img_path=img_path, ori_shape=ori_shape)
        data_samples = dict(real_gt=real_gt,meta=meta)
        return img, data_samples
    
    def dynamic_batch_sample(self,max_tokens=500):
        batch_inds = []
        i = self.length - 1
        print('Sampling dynamic batches ...')
        while i>=-1:
            single_token = self.sorted_length[i]
            # num_ind = max_tokens // single_token
            num_ind = int(math.sqrt(max_tokens / single_token))
            if i-num_ind>=-1:
                batch_inds.append(self.sorted_indices[i:max(-1,i-num_ind):-1])
            else:
                batch_inds.append(self.sorted_indices[0:i+1][::-1])
            i = i - num_ind
        print('finished')
        return batch_inds


        
        