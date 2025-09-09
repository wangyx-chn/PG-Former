import os
import os.path as osp
import shutil
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import cv2
from mmengine.evaluator import BaseMetric

class AutoRegIoU(BaseMetric):
    def __init__(self, work_dir, collect_device = 'cpu', prefix = None, collect_dir = None):
        super().__init__(collect_device, prefix, collect_dir)
        self.work_dir = work_dir
        self.cur_img_path = None

    def process(self, data_batch, data_samples):
        data_batch, data_samples = data_samples
        ori_shape = data_samples['ori_shape']
        bboxes = data_samples['bbox']
        pred_coords = [(res[1:-1].cpu().numpy()-10)/224.0 for res in data_batch['pred_coords']]
        pred_poly = [(res * np.array(ori_shape[i])+bboxes[i][:2].cpu().numpy()).astype(np.int32) for i,res in enumerate(pred_coords)]
        # test_vis
        vis_dir = osp.join(self.work_dir,'image_infer')
        # if osp.exists(vis_dir):
        #     shutil.rmtree(vis_dir)
        os.makedirs(vis_dir,exist_ok=True)
        

        for b in range(len(pred_poly)):
            img_path = data_samples['img_path'][b]

            if self.cur_img_path!=img_path:
                
                if self.cur_img_path:
                    img_name = osp.basename(self.cur_img_path)
                    img_name = osp.splitext(img_name)[0] + '.jpg'
                    cv2.imwrite(osp.join(vis_dir,img_name),self.cur_img)

                self.cur_img_path=img_path
                self.cur_img = cv2.imread(osp.join(self.cur_img_path))
                
            single_poly = pred_poly[b]
            self.cur_img = cv2.polylines(self.cur_img,[single_poly.reshape(-1,1,2)],color=(0,0,255),isClosed=True,thickness=2)
            

    def compute_metrics(self, results):
        return dict(acc=0)