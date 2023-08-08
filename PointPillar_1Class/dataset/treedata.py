import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from utils import read_pickle, read_points, lhw2wlh
from dataset.data_aug import data_augment


class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret
    
    

    
class TreeData(Dataset):
    # CLASSES = {
    #     'pm': 0,
    #     'ab': 1
    # }

    CLASSES = {
        'tree': 0
    }
    
    def __init__(self, data_root, split, canopy_type, pts_prefix='velodyne'):
        assert split in ['train', 'test']
        self.data_root = data_root
        self.split = split
        self.canopy_type = canopy_type
        self.data_infos = read_pickle(os.path.join(data_root, f'treedata_infos_{split}.pkl'))
        # self.data_infos = {k: self.data_infos[k] for k in list(self.data_infos)[:100]}

        self.data_infos_filtered = {}

        if split == 'train':
            self.data_infos_filtered = self.data_infos

        if split == 'test' and canopy_type == 'all':
            self.data_infos_filtered = self.data_infos

        if split == 'test' and canopy_type != 'all':
            ids = sorted(self.data_infos.keys())

            for c_name in ids:
                c_name_list = c_name.split('_')
                sub_id = int(c_name_list[1])
                plot_id = int(c_name_list[3])

                if ((plot_id % 40 == 39) or (plot_id % 40 == 38) or (plot_id % 40 == 37) or (
                        plot_id % 40 == 36)) and canopy_type == 'high_density':
                    self.data_infos_filtered[c_name] = self.data_infos[c_name]

                if ((plot_id % 40 == 0) or (plot_id % 40 == 1) or (plot_id % 40 == 2) or (
                        plot_id % 40 == 3)) and canopy_type == 'low_density':
                    self.data_infos_filtered[c_name] = self.data_infos[c_name]

                if ((sub_id == 0) or (sub_id == 1) or (sub_id == 48) or (
                        sub_id == 49)) and canopy_type == 'specie_specific':
                    self.data_infos_filtered[c_name] = self.data_infos[c_name]

                if ((sub_id == 23) or (sub_id == 24) or (sub_id == 25) or (
                        sub_id == 26)) and canopy_type == 'specie_mix':
                    self.data_infos_filtered[c_name] = self.data_infos[c_name]

        self.sorted_ids = list(self.data_infos_filtered.keys())
        db_infos = read_pickle(os.path.join(data_root, 'treedata_dbinfos_train.pkl'))
        # db_infos = self.filter_db(db_infos)
        
        db_sampler = {}
        for cat_name in self.CLASSES:
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        self.data_aug_config=dict(
            db_sampler = dict(
                db_sampler=db_sampler,
                sample_groups=dict(pm=5, ab=5)
            ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
            ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
            )
        )
        
        
    def filter_db(self, db_infos):
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]
            
        filter_thrs = dict(pm=5, ab=5)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
            
        return db_infos
    
    def __getitem__(self, index):
        data_info = self.data_infos_filtered[self.sorted_ids[index]]
        annos_info = data_info['annos']
        
        # point cloud input 
        velodyne_path = data_info['velodyne_path']
        pts_path = os.path.join(self.data_root, velodyne_path)
        pts = read_points(pts_path)
        
        
        # annotations input 
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        
        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)

        gt_bboxes_new = lhw2wlh(gt_bboxes)
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_new,
            'gt_labels': np.array(gt_labels),
            'gt_names': annos_name,
            'difficulty': annos_info['difficulty'],
            'id': self.sorted_ids[index]
        }
        
        
        if self.split in ['train']:
            data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)
        
        
        return data_dict
    

    def __len__(self):
        return len(self.data_infos_filtered)
    

if __name__ == '__main__':
    tree_data = TreeData(data_root='/home/abhishek/Desktop/Work/TreeDetection/TreeData_bin', split='test', canopy_type='high_density')
    # print(tree_data.__getitem__(9))
    print(len(tree_data))
        
