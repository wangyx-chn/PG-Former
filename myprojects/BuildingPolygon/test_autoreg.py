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
    def __init__(self, work_dir, img_dir, collect_device = 'cpu', prefix = None, collect_dir = None):
        super().__init__(collect_device, prefix, collect_dir)
        self.work_dir = work_dir
        self.img_dir = img_dir

    def process(self, data_batch, data_samples):
        data_batch, data_samples = data_samples
        ori_shape = data_samples['meta']['ori_shape']
        # imgs = data_samples['imgs']
        # input_poly = (((data_samples['mask']['data']-0.5)*aug+0.5).cpu().numpy() * np.array([ori_shape]).reshape(-1,1,2)).astype(np.int32)
        # pred_coords = torch.argmax(data_batch['pred_coords'],dim=1).cpu().numpy()/224.0
        # pred_logits = torch.sigmoid(data_batch['pred_logits']).cpu().numpy() 
        pred_coords = (data_batch['pred_coords'][:,1:-1].cpu().numpy()-10)/224.0
        pred_poly = (pred_coords * np.array([ori_shape]).reshape(-1,1,2)).astype(np.int32)
        # pred_edge = [F.interpolate(mask.unsqueeze(0).unsqueeze(0),ori_shape[i]).squeeze().cpu().numpy() for i,mask in enumerate(torch.sigmoid(data_batch['pred_reg']))]
        # pred_poly = pred_poly+input_poly
        # pred_cls = (data_batch['pred_cls'].squeeze()>0.5).cpu().numpy()
        real_gt = [((gt[:int((1-data_samples['real_gt']['pad_mask'][i]).sum().item()/2)].cpu().numpy() * np.array([ori_shape[i]])).reshape(-1,1,2)).astype(np.int32) for i,gt in enumerate(data_samples['real_gt']['data'])]

        # test_vis
        vis_dir = osp.join(self.work_dir,'test_val')
        img_dir = self.img_dir
        if osp.exists(vis_dir):
            shutil.rmtree(vis_dir)
        os.makedirs(vis_dir,exist_ok=True)
        

        for b in range(len(real_gt)):
            img_name = osp.basename(data_samples['meta']['img_path'][b])
            img = cv2.imread(osp.join(img_dir,img_name))
            # img = imgs[b].transpose(1,2,0)
            img1 = np.copy(img)
            img2 = np.copy(img)

            # pred_mask = pred_mask[b]
            # pred_points_num = np.sum(pred_logits[b]>0.5)
            pred_coords = pred_poly[b]
            try:
                # img1[pred_edge[b]>0.5] = np.array([0,0,255])
                img1 = cv2.polylines(img1,[pred_coords.reshape(-1,1,2)],color=(0,0,255),isClosed=True,thickness=2)
            except:
                # print(pred_edge[b].shape,img1.shape)
                print(pred_coords)
            img2 = cv2.polylines(img2,[real_gt[b].reshape(-1,1,2)],color=(0,0,255),isClosed=True,thickness=2)
            img = np.hstack([img1,img2])
            cv2.imwrite(osp.join(vis_dir,img_name),img)

            # input_mask = np.zeros(ori_shape[b]) 
            pred_mask = np.zeros(ori_shape[b])
            # pred_refine_mask = np.zeros(ori_shape[b])
            gt_mask = np.zeros(ori_shape[b])



            # # input_mask = cv2.fillPoly(input_mask,[input_poly[b].reshape(-1,1,2)],color=1)
            try:
                pred_mask = cv2.fillPoly(pred_mask,[pred_coords.reshape(-1,1,2).astype(np.int32)],color=1)
            except:
                pass
            # # pred_refine_mask = cv2.fillPoly(pred_refine_mask,[pred_poly[b][pred_cls[b]].reshape(-1,1,2)],color=1)
            gt_mask = cv2.fillPoly(gt_mask,[real_gt[b].reshape(-1,1,2)],color=1)

            # # input_intersect = (gt_mask * input_mask).sum()
            # # input_union = ((gt_mask + input_mask)>0).sum()
            pred_intersect = (gt_mask * pred_mask).sum()
            pred_union = ((gt_mask + pred_mask)>0).sum()
            # # pred_refine_intersect = (gt_mask * pred_refine_mask).sum()
            # # pred_refine_union = ((gt_mask + pred_refine_mask)>0).sum()
            # real_pts = len(real_gt[b])
            # before_pts = len(pred_poly[b])
            # # after_pts = pred_cls.sum()
            # self.results.append(dict(pred_intersect=pred_intersect,
            #                         pred_union=pred_union,
            #                         # pred_refine_intersect=pred_refine_intersect,
            #                         # pred_refine_union=pred_refine_union,
            #                         real_pts=real_pts,
            #                         before_pts=before_pts,
            #                         # after_pts=after_pts
            #                         ))
            gt = 1 - data_samples['real_gt']['pad_mask'][b,:,:1].cpu().numpy()
            length = gt.shape[0]
            new_gt = np.zeros((50,1))
            new_gt[:length] = gt
            acc = np.abs(len(pred_poly)-length)/length
            self.results.append(dict(pred_intersect=pred_intersect,pred_union=pred_union,acc = acc, gt=new_gt))

    def compute_metrics(self, results):
        # total_input_i = sum(result['input_intersect'] for result in self.results)
        # total_input_u = sum(result['input_union'] for result in self.results)
        total_pred_i = sum(result['pred_intersect'] for result in self.results)
        total_pred_u = sum(result['pred_union'] for result in self.results)
        # total_refine_i = sum(result['pred_refine_intersect'] for result in self.results)
        # total_refine_u = sum(result['pred_refine_union'] for result in self.results)
        # real_pts = sum(result['real_pts'] for result in self.results)
        # before_pts = sum(result['before_pts'] for result in self.results)
        # after_pts = sum(result['after_pts'] for result in self.results)

        # input_iou = total_input_i/total_input_u
        pred_iou = total_pred_i/total_pred_u
        # refine_iou = total_refine_i/total_refine_u
        # ab_rate_before = (before_pts-real_pts)/real_pts
        # ab_rate_after = (after_pts-real_pts)/real_pts
        # metrics = dict(input_iou=input_iou, pred_iou=pred_iou,ab_rate_after=ab_rate_after,ab_rate_before=ab_rate_before)
        # metrics = dict(pred_iou=pred_iou,ab_rate_before=ab_rate_before,total_pred_i=total_pred_i,total_pred_u=total_pred_u)
        # pred_scores = np.vstack([s['scores'].T for s in self.results])
        # np.save('/home/guning.wyx/code/mmengine/work_dirs/PolyGenDETR_sample50/pred_scores.npy',pred_scores)
        # gt_scores = np.vstack([s['gt'].T for s in self.results])
        # np.save('/home/guning.wyx/code/mmengine/work_dirs/PolyGenDETR_sample50/gt_scores.npy',gt_scores)
        acc = np.mean([res['acc'] for res in results])
        metrics = dict(acc=acc, pred_iou=pred_iou)
        print(metrics)
        return metrics 