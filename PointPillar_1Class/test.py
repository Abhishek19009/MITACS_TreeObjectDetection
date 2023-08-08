import argparse
import cv2
import numpy as np
import os
import torch
from models.pointpillar import PointPillars
from utils import read_label, read_points, setup_seed, lhw2wlh
from viso3d import vis_pc

def main(args):
    # CLASSES = {
    #     'pm': 0,
    #     'ab': 1
    # }

    CLASSES = {
        'tree': 0
    }

    setup_seed()
    
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    
    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES), voxel_size=[0.16, 0.16, 4], max_num_points=8).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
        
        
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError
    pc = read_points(args.pc_path)
#     print("pc shape: ", pc.shape)
    
    pc_torch = torch.from_numpy(pc)
    
    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    
    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()
            
            
        result_filter = model(batched_pts=[pc_torch],
                              mode='test')[0]
        
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']
    
    print("Predicted lidar bboxes: ", result_filter['lidar_bboxes'])
    
    
    annos_name = gt_label['name']
    annos_location = gt_label['location']
    annos_dimension = gt_label['dimensions']
    rotation_y = gt_label['rotation_y']

    gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
    gt_bboxes = lhw2wlh(gt_bboxes)

    print("Ground Truth lidar bboxes: ", gt_bboxes)
    gt_labels = [CLASSES[name] for name in annos_name]
    
    vis_pc(pc, pred_bboxes=lidar_bboxes, pred_labels=labels, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    # vis_pc(pc, bboxes=lidar_bboxes, labels=labels)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pillar_logs/checkpoints/treedata_epoch_180.pth', help='checkpoint for kitti')
#     parser.add_argument('--ckpt')
    parser.add_argument('--pc_path', default='/home/abhishek/Desktop/Work/TreeDetection/TreeData_bin/testing/velodyne/sub_0_plot_0_points.bin', help='point cloud path')
    parser.add_argument('--gt_path', default='/home/abhishek/Desktop/Work/TreeDetection/TreeData_bin/testing/label_2/sub_0_plot_0_bbox.txt', help='ground truth path')
    parser.add_argument('--no_cuda', default='', help='whether to use cuda')
    args = parser.parse_args()
    
    main(args)
    
    