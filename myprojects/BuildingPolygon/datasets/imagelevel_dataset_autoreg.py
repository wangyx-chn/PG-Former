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


class ImagePolyDataset(VisionDataset):

    def __init__(self,
                 img_dir,
                 detect_path,
                 img_size,
                 thr,
                 select_k = None,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root=img_dir, transform=transform, target_transform=target_transform)
        self.img_dir = osp.join(img_dir)
        self.detect_path = osp.join(detect_path)
        self.img_size = img_size
        self.thr = thr

        # for i in range(self.length):
        #     self.point_length.append(len(np.load(osp.join(self.mask_dir,f'{i}.npy'))))
        with open(osp.join(self.detect_path),'rb') as f:
            self.dt_results = pickle.load(f)
        
        self.all_results = []
        for img in self.dt_results:
            results = img['pred_instances']
            n = len(results['scores'])
            for i in range(n):
                if results['scores'][i]>self.thr:
                    res = {'img_path':osp.join(img_dir,osp.basename(img['img_path'])),'bbox':results['bboxes'][i],'score':results['scores'][i]}
                    self.all_results.append(res)
            
        self.cur_img_path = None

    def __len__(self):
        return len(self.all_results)
    
    def __getitem__(self, index):
        res = self.all_results[index]
        img_path = res['img_path']
        if img_path != self.cur_img_path:
            self.cur_img = Image.open(img_path).convert('RGB')
            self.cur_img_path = img_path
        
        bbox = res['bbox']
        img_patch = self.cur_img.crop(bbox.tolist())
        ori_shape = img_patch.size

        if self.transform is not None:
            img = self.transform(img_patch)
        img = F.interpolate(img.unsqueeze(0),size=self.img_size,mode='bilinear').squeeze(0)
        

        data_samples = dict(img_path=img_path, bbox=bbox, ori_shape=ori_shape)
        return img, data_samples



        
        