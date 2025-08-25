import os
import os.path as osp
import pickle
import numpy as np
import cv2
import math
from PIL import Image
from torchvision.datasets import VisionDataset
import torch.nn.functional as F
from pycocotools.coco import COCO
import random

class BuildPolyDataset(VisionDataset):

    def __init__(self,
                 root,
                 img_dir,
                 mask_dir,
                 gt_dir,
                 pt_len_path,
                 img_size=128,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root=root, transform=transform, target_transform=target_transform)
        self.img_dir = osp.join(root,img_dir)
        self.mask_dir = osp.join(root,mask_dir)
        self.gt_dir = osp.join(root,gt_dir)
        self.img_size = img_size
        self.length = len(os.listdir(self.mask_dir))
        self.point_length = []
        # for i in range(self.length):
        #     self.point_length.append(len(np.load(osp.join(self.mask_dir,f'{i}.npy'))))
        with open(osp.join(root,pt_len_path),'rb') as f:
            self.point_length = pickle.load(f)

        # debug for test 1w
        random.seed(1)
        self.point_length = random.choices(self.point_length,k=2000)
        self.length = len(self.point_length)

        self.sorted_length = np.sort(self.point_length)
        self.sorted_indices = np.argsort(self.point_length)
        
        pass

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f'{index}.png')
        mask_path = os.path.join(self.mask_dir, f'{index}.npy')
        
        gt_path = os.path.join(self.gt_dir, f'{index}_targets.npy') #for train
        cls_path = os.path.join(self.gt_dir, f'{index}_cls_targets.npy')
        real_gt_path = os.path.join(self.gt_dir, f'{index}_real_gt.npy') # for eval

        img = Image.open(img_path).convert('RGB')
        ori_shape = img.size
        mask = np.load(mask_path)
        gt = np.load(gt_path)
        cls_gt = np.load(cls_path)
        real_gt = np.load(real_gt_path)

        if self.transform is not None:
            img = self.transform(img)
        img = F.interpolate(img.unsqueeze(0),size=self.img_size,mode='bilinear').squeeze(0)
        # Convert the RGB values to class indices
        # mask = np.array(mask)
        # mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
        # labels = np.zeros_like(mask, dtype=np.int64)
        # for color, class_index in self.color_to_class.items():
        #     rgb = color[0] * 65536 + color[1] * 256 + color[2]
        #     labels[mask == rgb] = class_index

        if self.target_transform is not None:
            gt = self.target_transform(gt)
            cls_gt = self.target_transform(cls_gt)
            real_gt = self.target_transform(real_gt)
            mask = self.target_transform(mask)
        meta = dict(img_path=img_path, mask_path=mask_path, ori_shape=ori_shape)
        data_samples = dict(
            mask = mask, poly_target=gt, cls_target=cls_gt,real_gt=real_gt,meta=meta)
        return img, data_samples
    
    def dynamic_batch_sample(self,max_tokens=500):
        batch_inds = []
        i = self.length - 1
        print('Sampling dynamic batches ...')
        while i>=-1:
            single_token = self.sorted_length[i]
            num_ind = max_tokens // single_token
            if i-num_ind>=-1:
                batch_inds.append(self.sorted_indices[i:max(-1,i-num_ind):-1])
            else:
                batch_inds.append(self.sorted_indices[0:i+1][::-1])
            i = i - num_ind
        print('finished')
        return batch_inds

class BuildPolyGtOnlyDataset(VisionDataset):

    def __init__(self,
                 root,
                 img_dir,
                 gt_dir,
                 pt_len_path,
                 select_k=None,
                 img_size=128,
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

        # debug for test 1w
        if select_k:
            # random.seed(2)
            self.sorted_indices = random.choices(self.sorted_indices,k=select_k)
            self.sorted_length = [self.sorted_length[i] for i in self.sorted_indices]
            self.length = len(self.sorted_length)
        
        pass

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f'{index}.png')
        
        gt_path = os.path.join(self.gt_dir, f'{index}_sampled_gt.npy') #for train
        cls_path = os.path.join(self.gt_dir, f'{index}_iscorner_for_sampled.npy')
        real_gt_path = os.path.join(self.gt_dir, f'{index}_real_gt.npy') # for eval

        img = Image.open(img_path).convert('RGB')
        ori_shape = img.size
        gt = np.load(gt_path)
        cls_gt = np.load(cls_path)
        real_gt = np.load(real_gt_path)

        edge = np.zeros((self.img_size,self.img_size))
        edge = cv2.polylines(edge,[(real_gt*np.array((self.img_size,self.img_size))).astype(np.int32).reshape(-1,1,2)],color=1,thickness=5,isClosed=True)
        

        if self.transform is not None:
            img = self.transform(img)
        img = F.interpolate(img.unsqueeze(0),size=self.img_size,mode='bilinear').squeeze(0)
        

        if self.target_transform is not None:
            gt = self.target_transform(gt)
            cls_gt = self.target_transform(cls_gt)
            real_gt = self.target_transform(real_gt)
            edge = self.target_transform(edge)
        meta = dict(img_path=img_path, ori_shape=ori_shape)
        data_samples = dict(
            poly_target=gt, cls_target=cls_gt,real_gt=real_gt,edge=edge,meta=meta)
        return img, data_samples
    
    def dynamic_batch_sample(self,max_tokens=500):
        batch_inds = []
        i = self.length - 1
        print('Sampling dynamic batches ...')
        while i>-1:
            single_token = self.sorted_length[i]
            num_ind = max_tokens // single_token
            if i<=num_ind-1:
                batch_inds.append(self.sorted_indices[0:i+1][::-1])
                break
            else:
                batch_inds.append(self.sorted_indices[i:i-num_ind:-1])
            i = i - num_ind
        print('finished')
        return batch_inds

    
class BuildingFastCOCO(VisionDataset):

    def __init__(self,
                 root,
                 list_txt,
                 img_folder,
                 anno_folder,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        self.img_folder = img_folder
        self.list_txt = list_txt
        self.anno_folder = anno_folder
        
        