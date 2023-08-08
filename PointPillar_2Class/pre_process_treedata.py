import argparse
import pdb
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys

# CUR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(CUR)
print(sys.path)
from utils import read_points, write_points, read_label, write_pickle, get_points_num_in_bbox, points_in_bboxes_v2


def judge_difficulty(annotation_dict):
    # We can bypass this as we don't have a difficulty system.
    difficulties = [0 for name in annotation_dict['name']]
    
    return np.array(difficulties, dtype=int)
    
    
def create_data_info_pkl(data_root, data_type, prefix='treedata', label=True, db=False):
    sep = os.path.sep
    print(f"Processing {data_type} data..")
    
#     split = 'training' if label else 'testing'
    split = 'training' if data_type=='train' else 'testing'
    
    kitti_infos_dict = {}
    if db:
        kitti_dbinfos_train = {}
        db_points_saved_path = os.path.join(data_root, f'{prefix}_gt_database')
        os.makedirs(db_points_saved_path, exist_ok=True)        
        
    file_paths = sorted(os.listdir(os.path.join(data_root, split, 'velodyne')))
    ids = [os.path.basename(path).split('.')[0][:-7] for path in file_paths]
    
    for id in tqdm(ids):
        cur_info_dict={}
        lidar_path = os.path.join(data_root, split, 'velodyne', f'{id}_points.bin')
        cur_info_dict['velodyne_path'] = sep.join(lidar_path.split(sep)[-3:])
        lidar_points = read_points(lidar_path)
        
        
        if label:
            label_path = os.path.join(data_root, split, 'label_2', f'{id}_bbox.txt')
            annotation_dict = read_label(label_path)
            annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                points=lidar_points,
                annotation_dict=annotation_dict)
            cur_info_dict['annos'] = annotation_dict
            
            if db:
                indices, n_total_bbox, bboxes_lidar, name = points_in_bboxes_v2(
                    points=lidar_points,
                    annotation_dict=annotation_dict)
                for j in range(n_total_bbox):
                    db_points = lidar_points[indices[:, j]]
                    db_points[:, :3] -= bboxes_lidar[j, :3]
                    db_points_saved_name = os.path.join(db_points_saved_path, f'{id}_points_{name[j]}_{j}.bin')
                    write_points(db_points, db_points_saved_name)
                    
                    db_info={
                        'name': name[j],
                        'path': os.path.join(os.path.basename(db_points_saved_path), f'{id}_points_{name[j]}_{j}.bin'),
                        'box3d_lidar': bboxes_lidar[j],
                        'difficulty': annotation_dict['difficulty'][j],
                        'num_points_in_gt': len(db_points),                    
                    }
                    
                    if name[j] not in kitti_dbinfos_train:
                        kitti_dbinfos_train[name[j]] = [db_info]
                    else:
                        kitti_dbinfos_train[name[j]].append(db_info)
            
        kitti_infos_dict[id] = cur_info_dict
        
    saved_path = os.path.join(data_root, f'{prefix}_infos_{data_type}.pkl')
    write_pickle(kitti_infos_dict, saved_path)
    if db:
        saved_db_path = os.path.join(data_root, f'{prefix}_dbinfos_train.pkl')
        write_pickle(kitti_dbinfos_train, saved_db_path)
    return kitti_infos_dict



def main(args):
    data_root = args.data_root
    prefix = args.prefix
    
    # Create pkl file with data information for training set
    kitti_train_infos_dict = create_data_info_pkl(data_root, 'train', prefix, label=True, db=True)
    
    # Create pkl file with data information for test set
    kitti_test_infos_dict = create_data_info_pkl(data_root, 'test', prefix, label=True, db=False)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='../TreeData_bin/', 
                        help='your data root for kitti')
    parser.add_argument('--prefix', default='treedata', 
                        help='the prefix name for the saved .pkl file')
    args = parser.parse_args()

    main(args)
    