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


def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)

    return score_thresholds


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

    # MIN_IOUS = {
    #     'pm': [0.5, 0.5],
    #     'ab': [0.5, 0.5],
    # }

    MIN_IOUS = {
        'tree': [0.5, 0.5]
    }

    overall_results = {}
    for e_ind, eval_type in enumerate(['bbox_bev', 'bbox_3d']):
        eval_ious = ious[eval_type]
        eval_ap_results, eval_aos_results = {}, {}
        for cls in CLASSES:
            eval_ap_results[cls] = []
            eval_aos_results[cls] = []
            CLS_MIN_IOU = MIN_IOUS[cls][e_ind]

            total_scores = []
            total_gt_alpha = []
            total_det_alpha = []

            total_gt_ignores, total_det_ignores, total_dc_bboxes = [], [], []

            for id in ids:
                gt_result = gt_results[id]['annos']
                det_result = det_results[id]

                # gt bbox property
                cur_gt_names = gt_result['name']
                cur_difficulty = gt_result['difficulty']
                gt_ignores, dc_bboxes = [], []

                for j, cur_gt_name in enumerate(cur_gt_names):
                    ignore = cur_difficulty[j] < 0 or cur_difficulty[j] > 0  # cur_difficulty[j] > difficulty, but here difficulty is replaced with 0, since we have only 1 value of difficulty i.e 0
                    if cur_gt_name == cls:
                        valid_class = 1
                    else:
                        valid_class = -1

                    if valid_class == 1 and not ignore:
                        gt_ignores.append(0)
                    elif valid_class == 0 or (valid_class == 1 and ignore):
                        gt_ignores.append(1)
                    else:
                        gt_ignores.append(-1)

                    if cur_gt_name == 'DontCare':
                        dc_bboxes.append(gt_result['bbox'][j])

                total_gt_ignores.append(gt_ignores)
                total_dc_bboxes.append(np.array(dc_bboxes))
                total_gt_alpha.append(gt_result['alpha'])

                # det bbox property

                cur_det_names = det_result['name']
                det_ignores = []

                for j, cur_det_name in enumerate(cur_det_names):
                    if cur_det_name == cls:
                        det_ignores.append(0)
                    else:
                        det_ignores.append(-1)


                total_det_ignores.append(det_ignores)
                total_scores.append(det_result['score'])
                total_det_alpha.append(det_result['alpha'])

            # calculate scores thresholds for PR Curve
            tp_scores = []
            for i, id in enumerate(ids):
                cur_eval_ious = eval_ious[i]
                gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]

                scores = total_scores[i]
                nn, mm = cur_eval_ious.shape
                assigned = np.zeros((mm, ), dtype=np.bool_)

                for j in range(nn):
                    if gt_ignores[j] == -1:
                        continue
                    match_id, match_score = -1, -1
                    for k in range(mm):
                        if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j, k] > CLS_MIN_IOU and scores[k] > match_score:
                            match_id = k
                            match_score = scores[k]
                    if match_id != -1:
                        assigned[match_id] = True
                        if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                            tp_scores.append(match_score)
            
            
            
            total_num_valid_gt = np.sum([np.sum(np.array(gt_ignores) == 0) for gt_ignores in total_gt_ignores])
            score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt)
#             print("Score thresholds: ", score_thresholds)

            # draw PR curve and calculate mAP
            tps, fns, fps, total_aos = [], [], [], []

            for score_threshold in score_thresholds:
                tp, fn, fp = 0, 0, 0
                aos = 0

                for i, id in enumerate(ids):
                    cur_eval_ious = eval_ious[i]
                    gt_ignores, det_ignores = total_gt_ignores[i] , total_det_ignores[i]
                    gt_alpha, det_alpha = total_gt_alpha[i], total_det_alpha[i]
                    scores = total_scores[i]

                    nn, mm = cur_eval_ious.shape
                    assigned = np.zeros((mm, ), dtype=np.bool_)
                    for j in range(nn):
                        if gt_ignores[j] == -1:
                            continue
                        match_id, match_iou = -1, -1
                        for k in range(mm):
                            if not assigned[k] and det_ignores[k] >= 0 and scores[k] >= score_threshold and cur_eval_ious[j, k] > CLS_MIN_IOU:
                                if det_ignores[k] == 0 and cur_eval_ious[j, k] > match_iou:
                                    match_iou = cur_eval_ious[j, k]
                                    match_id = k
                                elif det_ignores[k] == 1 and match_iou == 1:
                                    match_id = k

                        if match_id != -1:
                            assigned[match_id] = True
                            if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                tp += 1

                        else:
                            if gt_ignores[j] == 0:
                                fn += 1

                    for k in range(mm):
                        if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                            fp += 1


                tps.append(tp)
                fns.append(fn)
                fps.append(fp)

            
            tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)

            recalls = tps / (tps + fns)
            precisions = tps / (tps + fps)


            for i in range(len(score_thresholds)):
                precisions[i] = np.max(precisions[i:])

            sums_AP = 0
            for i in range(0, len(score_thresholds)):
                sums_AP += precisions[i]
            # mAP = sums_AP / 11 * 100
#             for i in range(0, len(score_thresholds)):
#                 sums_AP += precisions[i]
                
#             print(score_thresholds, len(score_thresholds))

            mAP = sums_AP / len(precisions) * 100
            eval_ap_results[cls].append(mAP)


        print(f'=========={eval_type.upper()}==========')
        print(f'=========={eval_type.upper()}==========', file=f)
        for k, v in eval_ap_results.items():
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f}')
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f}', file=f)


        overall_results[eval_type] = np.mean(list(eval_ap_results.values()), 0)

    print(f'\n==========Overall==========')
    print(f'\n==========Overall==========', file=f)
    for k, v in overall_results.items():
        print(f'{k} AP: {v[0]:.4f}')
        print(f'{k} AP: {v[0]:.4f}', file=f)
    f.close()



def main(args):
    for canopy in ['high_density', 'low_density', 'specie_specific', 'specie_mix', 'all']:
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