import argparse
import numpy as np
import os
import torch
from tqdm import tqdm

from utils import setup_seed, write_pickle, write_label, iou2d, iou3d, iou_bev, wlh2hwl
# from utils import setup_seed, write_pickle, iou2d, iou3d_camera, iou_bev
from dataset.treedata import TreeData
from dataset.dataloader import get_dataloader
from models.pointpillar import PointPillars


def do_eval(det_results, gt_results, CLASSES, saved_path):
    '''
        det_results: list,
        gt_results: dict(id -> det_results)
        CLASSES: dict
    '''

    assert len(det_results) == len(gt_results)
    f = open(os.path.join(saved_path, 'eval_results.txt'), 'w')

    # calculate iou
    ious = {
        'bbox_bev': [],
        'bbox_3d': []
    }

    ious_mean = {
        'bbox_bev': [],
        'bbox_3d': []
    }

    ids = list(sorted(gt_results.keys()))

    for id in ids:
        gt_result = gt_results[id]['annos']
        det_result = det_results[id]

        gt_location = gt_result['location'].astype(np.float32)
        gt_dimensions = gt_result['dimensions'].astype(np.float32)
        gt_rotation_y = gt_result['rotation_y'].astype(np.float32)
        det_location = det_result['location'].astype(np.float32)
        det_dimensions = det_result['dimensions'].astype(np.float32)
        det_rotation_y = det_result['rotation_y'].astype(np.float32)


        gt_dimensions = wlh2hwl(gt_dimensions)


        # bev iou
        gt_bev = np.concatenate([gt_location[:, [0, 2]], gt_dimensions[:, [0, 2]], gt_rotation_y[:, None]], axis=-1)
        det_bev = np.concatenate([det_location[:, [0, 2]], det_dimensions[:, [0, 2]], det_rotation_y[:, None]], axis=-1)
        iou_bev_v = iou_bev(torch.from_numpy(gt_bev).cuda(), torch.from_numpy(det_bev).cuda())
        ious['bbox_bev'].append(iou_bev_v.cpu().numpy())

        # 3dbboxes iou
        gt_bboxes3d = np.concatenate([gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1)
        det_bboxes3d = np.concatenate([det_location, det_dimensions, det_rotation_y[:, None]], axis=-1)
        iou3d_v = iou3d(torch.from_numpy(gt_bboxes3d).cuda(), torch.from_numpy(det_bboxes3d).cuda())
        ious['bbox_3d'].append(iou3d_v.cpu().numpy())

    for iou in ious:
        eval_iou = ious[iou]
        for i, id in enumerate(ids):
            cur_eval_iou = eval_iou[i]
            ious_mean[iou].append(cur_eval_iou.max(axis=-1).mean())




    pass
    # print(np.array(ious['bbox_bev']))
    # print(np.array(ious['bbox_3d']))

def main(args):
    for canopy in ['all']:
        test_dataset = TreeData(data_root=args.data_root,
                                split='test',
                                canopy_type=canopy)

        print(f"Cur canopy: ", canopy, "num_scans: ", len(test_dataset))


        test_dataloader = get_dataloader(dataset=test_dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False)
        CLASSES = TreeData.CLASSES
        LABEL2CLASSES = {v: k for k, v in CLASSES.items()}

        if not args.no_cuda:
            model = PointPillars(nclasses=args.nclasses).cuda()
            model.load_state_dict(torch.load(args.ckpt))
        else:
            model = PointPillars(nclasses=args.nclasses)
            model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')))

        saved_path = args.saved_path
        os.makedirs(saved_path, exist_ok=True)
        saved_submit_path = os.path.join(saved_path, 'submit')
        os.makedirs(saved_submit_path, exist_ok=True)

        pcd_limit_range = [-5, -5, 0, 23, 23, 25]

        model.eval()
        with torch.no_grad():
            format_results = {}
            print("Predicting and formatting the results.")
            for i, data_dict in enumerate(tqdm(test_dataloader)):
                if not args.no_cuda:
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()

                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']
                batch_results = model(batched_pts=batched_pts,
                                      mode='test',
                                      batched_gt_bboxes=batched_gt_bboxes,
                                      batched_gt_labels=batched_labels)

                for j, result in enumerate(batch_results):
                    format_result = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'dimensions': [],
                        'location': [],
                        'rotation_y': [],
                        'score': []
                    }

                    # We don't have to limit point cloud ranges, so here result_filter is same as result
                    result_filter = result
                    lidar_bboxes = result_filter['lidar_bboxes']
                    labels, scores = result_filter['labels'], result_filter['scores']
                    idx = data_dict['batched_ids'][j]
                    for lidar_bbox, label, score in zip(lidar_bboxes, labels, scores):
                        format_result['name'].append(LABEL2CLASSES[label])
                        format_result['truncated'].append(0.0)
                        format_result['occluded'].append(0)
                        alpha = lidar_bbox[6] - np.arctan2(lidar_bbox[0], lidar_bbox[2])
                        format_result['alpha'].append(alpha)
                        format_result['dimensions'].append(lidar_bbox[3:6])
                        format_result['location'].append(lidar_bbox[:3])
                        format_result['rotation_y'].append(lidar_bbox[6])
                        format_result['score'].append(score)

                    # write_label(format_result, os.path.join(saved_submit_path, f'{}.txt'))

                    format_results[idx] = {k: np.array(v) for k, v in format_result.items()}

        do_eval(format_results, test_dataset.data_infos_filtered, CLASSES, saved_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/home/abhishek/Desktop/Work/TreeDetection/TreeData_bin',
                        help='your data root for kitti')
    parser.add_argument('--ckpt', default='./pillar_logs/checkpoints/treedata_epoch_180.pth',
                        help='your checkpoint for kitti')
    parser.add_argument('--saved_path', default='results', help='your saved path for predicted results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=1)
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)