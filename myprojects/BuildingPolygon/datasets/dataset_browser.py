'''
用于数据集的可视化
'''

import os
import os.path as osp
import cv2
import numpy as np
import random
import shutil
import json

def browse_polydataset(img_dir,gt_dir,save_dir,vis_num=None,vis_inds=None):
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # 不给定 vis num 则可视化整个数据集，否则采样可视化
    inds = [name.split('.')[0] for name in os.listdir(img_dir)]
    if vis_inds:
        inds = vis_inds
    if vis_num:
        inds = random.choices(inds, k=vis_num)
    
    # 先取真值点、图像
    for i in inds:
        gt_poly = np.load(osp.join(gt_dir,f'{i}_real_gt.npy'))
        img = cv2.imread(osp.join(img_dir,f'{i}.png'), flags=-1)
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)
        size = np.array(img.shape[:2])
        
        #画gt点
        img2 = np.copy(img)
        gt_poly = (gt_poly*size).astype(np.int32)
        img2 = cv2.polylines(img2,[gt_poly.reshape(-1,1,2)],isClosed=True,color=(0,0,255),thickness=2,lineType=cv2.LINE_AA)
        for pt in gt_poly:
            img2 = cv2.circle(img2, (pt[0],pt[1]), radius=3, color=(255,0,0), thickness=-1)

        # 合并、保存图像
        vis_img = np.concatenate([img,img2],axis=1)
        cv2.imwrite(osp.join(save_dir,f'{i}.png'), vis_img)

if __name__=='__main__':
    dataset_dir = '/home/guning.wyx/code/mmengine/data/WHUBuilding/polygeneration/dataset_polygon50_0.1margin_1.0noise20_train'
    img_dir = osp.join(dataset_dir, 'image_patch')
    gt_dir = osp.join(dataset_dir, 'gt_poly')
    save_dir = osp.join(dataset_dir, 'vis_dataset')
    browse_polydataset(img_dir,gt_dir,save_dir,vis_num=500)