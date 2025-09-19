import os
import os.path as osp
import pickle
import numpy as np
import json
import cv2
import shutil
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from utils import square_bbox, ensure_ccw
from tqdm import tqdm
import random

def polygon_dataset_generation(cfg):
    '''
        在确定好进行自回归生成之后，这里只需要生成image patch和gt polygon
    '''

    # step 1: 创建好dataset dir，包含image patch文件夹、输出polygon文件夹
    save_dir = cfg.save_dir
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    
    img_patch_dir = osp.join(save_dir,'image_patch')
    os.makedirs(img_patch_dir, exist_ok=True)
    gt_poly_dir = osp.join(save_dir,'gt_poly')
    os.makedirs(gt_poly_dir, exist_ok=True)    

    # step 2: 读取pkl文件与coco文件，遍历所有image 
    gt_coco = COCO(cfg.coco_path)
    
    # 对数据计数，作为文件名
    data_count = 0

    # 记录一下每个实例加入的噪声
    noise_record = []

    # 要还原到原始的图像所以需要有一个映射
    img_crop_map = {}

    # 为了能够实现动态batchsize需要记录每个实例的点数量
    point_length = []

    for img_id in tqdm(gt_coco.getImgIds(), desc="process results", unit="img"):

        # gt信息提取
        img_info = gt_coco.loadImgs(img_id)
        gts = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=img_id))
        gt_bboxes = [gt['bbox'] for gt in gts]
        gt_polygons = [gt['segmentation'] for gt in gts]
        width = img_info[0]['width']
        height = img_info[0]['height']
        crop_margin = cfg.crop_margin

        #读图像用于后续裁剪
        img_name = gt_coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(osp.join(cfg.img_dir,img_name))

        patch_info = []

        for gi, gt_polygon in enumerate(gt_polygons):
            # 选裁剪图像的bbox
            crop_bbox = square_bbox(gt_bboxes[gi])
            x0, y0, w, h = [int(c) for c in crop_bbox]

            # margin是裁剪的边距，可以支持float(相对margin)和int(绝对margin)
            if crop_margin<1:
                margin = int(crop_margin*w)
            else:
                margin = crop_margin

            # 归一化(顺便把gt也简化一下)
            gt_polygon = np.array(gt_polygons[gi][0]).astype(np.float32).reshape(-1,1,2)
            epsilon = 0
            while gt_polygon.shape[0]>cfg.max_points:
                epsilon+=1
                gt_polygon = cv2.approxPolyDP(gt_polygon, epsilon, True)
            
            
            
            # 噪声生成
            w_noise = 0
            h_noise = 0
            if getattr(cfg,'noise_paras',None):
                if np.random.rand()<noise_paras[0]:
                    # w_noise = np.random.randint(-noise_paras[1],noise_paras[1])
                    # h_noise = np.random.randint(-noise_paras[1],noise_paras[1])
                    if noise_paras[1]>1:
                        noise = noise_paras[1]
                    else:
                        noise = int(w*noise_paras[1])
                    w_noise = np.random.randint(-noise,noise)
                    h_noise = np.random.randint(-noise,noise)
                    # x0 = x0+w_noise
                    # y0 = y0+h_noise
                    noise_record.append({'id':data_count,'w_noise':w_noise,'h_noise':h_noise})

            margin = max(margin,abs(w_noise),abs(h_noise))
            gt_polygon = (gt_polygon.reshape(-1,2) + np.array([[w_noise,h_noise]]) - np.array([x0,y0]) + margin) / (np.array([w,h])+2*margin)
            # pad是图像需要的pad尺寸
            pad = max([0, -x0+margin, -y0+margin, x0+w+margin-width, y0+h+margin-height])
            
            

            # 对GT插值并生成 corner分类的gt
            gt_polygon = ensure_ccw(gt_polygon)
            # pred_polygon = ensure_ccw(pred_polygon)

            # 选择起点
            distances = np.linalg.norm(gt_polygon, axis=1)
            closest_index = np.argmin(distances)
            gt_polygon = np.concatenate([gt_polygon[closest_index:],gt_polygon[0:closest_index]],axis=0)
                
            # 如果crop的时候出去了就用0进行pad
            # 5个像素是避免边界溢出留的余量，和margin没关系
            img_patch = np.pad(img, pad_width=((pad+5, pad+5), (pad+5, pad+5), (0, 0)), mode='constant', constant_values=0)
            img_patch = img_patch[y0+pad+5-margin:y0+pad+5+h+margin,x0+pad+5-margin:x0+pad+5+w+margin,:]

            # 保存
            np.save(osp.join(gt_poly_dir,f'{data_count}_real_gt.npy'),gt_polygon)
            cv2.imwrite(osp.join(img_patch_dir,f'{data_count}.png'), img_patch) 
            patch_info.append({'bbox':[x0, y0, w, h], 'id':data_count})
            point_length.append(len(gt_polygon))
            data_count+=1
        
        img_crop_map[img_name] = patch_info
    print(f'data count: {data_count}')
    print('dumping the crop map json')
    with open(osp.join(save_dir,'crop_map.json'),'w') as f:
        json.dump(img_crop_map, f, indent=4)
    print('dumping the point length pkl')  
    with open(osp.join(save_dir,'point_length.pkl'),'wb') as f:
        pickle.dump(point_length, f)
    print('dumping the noise_record json')  
    with open(osp.join(save_dir,'noise_record.json'),'w') as f:
        json.dump(noise_record, f, indent=4)

class Config:
    def __init__(self,**kwargs):
        self.coco_path = kwargs['coco_path']
        self.img_dir = kwargs['img_dir']
        self.save_dir = kwargs['save_dir']
        self.max_points = kwargs['max_points']
        self.crop_margin = kwargs['crop_margin']
        self.noise_paras = kwargs['noise_paras']

if __name__=='__main__':
    
    split = "train" # ['train','validation']
    max_points = 50
    crop_margin = 0.1
    noise_paras = [0.8,20]
    coco_path = f'/home/guning.wyx/code/mmdetection/data/WHUBuilding/annotation/{split}.json'
    img_dir = f'/home/guning.wyx/code/mmdetection/data/WHUBuilding/{split}'

    save_dir = '/home/guning.wyx/code/mmdetection/data/WHUBuilding/polygeneration/'
    save_dir = save_dir + f'dataset_polygon{max_points}_{crop_margin}margin_{noise_paras[0]}noise{noise_paras[1]}_{split}'

    cfg = Config(coco_path=coco_path,
                 save_dir=save_dir,
                 img_dir=img_dir,
                 max_points=max_points,
                 crop_margin=crop_margin,
                 noise_paras=noise_paras)

    polygon_dataset_generation(cfg)

