import numpy as np
import numba
import copy
import random
import torch
import os
import pickle
from ops.iou3d_module import boxes_overlap_bev, boxes_iou_bev

# normalize_pts_z, normalize_label_z

CLASSES = {
        'pm': 0, 
        'ab': 1, 
        }


def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


###############   IO   #####################

def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


def write_points(lidar_points, file_path):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        with open(file_path, 'w') as f:
            lidar_points.tofile(f)
    else:
        raise NotImplementedError


def write_label(result, file_path, suffix='.txt'):
    '''
    result: dict,
    file_path: str
    '''
    assert os.path.splitext(file_path)[1] == suffix
    name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = \
        result['name'], result['truncated'], result['occluded'], result['alpha'], \
            result['bbox'], result['dimensions'], result['location'], result['rotation_y'], \
            result['score']

    with open(file_path, 'w') as f:
        for i in range(len(name)):
            bbox_str = ' '.join(map(str, bbox[i]))
            hwl = ' '.join(map(str, dimensions[i]))
            xyz = ' '.join(map(str, location[i]))
            line = f'{name[i]} {truncated[i]} {occluded[i]} {alpha[i]} {bbox_str} {hwl} {xyz} {rotation_y[i]} {score[i]}\n'
            f.writelines(line)

        
        
def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = {}
    annotation['name'] = np.array([line[0] for line in lines])

    # CAUTION: SUBSTITUTING NAME WITH TREE TO MAINTAIN ONE CLASS

    ####################################
    annotation['name'] = np.array(['tree' for line in lines])
    ####################################

    annotation['truncated'] = np.array([line[1] for line in lines], dtype=float)
    annotation['occluded'] = np.array([line[2] for line in lines], dtype=int)
    annotation['alpha'] = np.array([line[3] for line in lines], dtype=float)
    annotation['bbox'] = np.array([line[4:8] for line in lines], dtype=float)
    annotation['dimensions'] = np.array([line[8:11] for line in lines], dtype=float) # coordinates is (lhw)
    annotation['location'] = np.array([line[11:14] for line in lines], dtype=float)
    annotation['rotation_y'] = np.array([line[14] for line in lines], dtype=float)
    
    return annotation


def read_pc(file_path):
    pc = read_ply(file_path)
    pc_x = np.expand_dims(pc['x'], axis=-1)
    pc_y = np.expand_dims(pc['y'], axis=-1)
    pc_z = np.expand_dims(pc['z'], axis=-1)
    pc_class = np.expand_dims(pc['class'], axis=-1)
    pc_intensity = np.ones_like(pc_x)
    
    pc = np.concatenate((pc_x, pc_y, pc_z, pc_intensity), axis=-1)
    return pc



def save_pts(pts, file_path):
    pts.astype(np.float32).tofile(file_path)

def save_label(label, file_path):
    name = label['name']
    truncated = label['truncated']
    occluded = label['occluded']
    alpha = label['alpha']
    bbox = label['bbox']
    dimensions = label['dimensions']
    location = label['location']
    rotation_y = label['rotation_y']
    
    
    l_obj = len(name)

    file = open(file_path, 'w')

    for obj_idx in range(l_obj):
        string = str(name[obj_idx]) + ' ' + str(truncated[obj_idx]) + ' ' + str(int(occluded[obj_idx])) + ' ' + str(alpha[obj_idx]) + ' ' + ' '.join(str(x) for x in bbox[obj_idx]) + ' ' + ' '.join(str(x) for x in dimensions[obj_idx]) + ' ' + ' '.join(str(x) for x in location[obj_idx]) + ' ' + str(rotation_y[obj_idx])
        file.write(string + '\n')

    file.close()


#################################################


def lhw2wlh(bboxes):
    l, h, w = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    wlh_size = np.concatenate([w, l, h], axis=1) # (wlh)
    bboxes_lidar = np.concatenate([bboxes[:, :3], wlh_size, bboxes[:, 6:]], axis=1)  # [x, y, z, w, l, h, rot]
    return np.array(bboxes_lidar, dtype=np.float32)


def wlh2hwl(bboxes):
    w, l, h = bboxes[:, 0:1], bboxes[:, 1:2], bboxes[:, 2:3]
    hwl_size = np.concatenate([h, w, l], axis=1) # (hwl)
    return np.array(hwl_size, dtype=np.float32)



def group_rectangle_vertexs(bboxes_corners):
    '''
    bboxes_corners: shape=(n, 8, 3)
    return: shape=(n, 6, 4, 3)
    '''
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]], axis=1) # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]], axis=1) # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]], axis=1) # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]], axis=1) # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs


def group_plane_equation(bbox_group_rectangle_vertexs):
    '''
    bbox_group_rectangle_vertexs: shape=(n, 6, 4, 3)
    return: shape=(n, 6, 4)
    '''
    # 1. generate vectors for a x b
    vectors = bbox_group_rectangle_vertexs[:, :, :2] - bbox_group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1]) # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', bbox_group_rectangle_vertexs[:, :, 0], normal_vectors) # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params


@numba.jit(nopython=True)
def bevcorner2alignedbbox(bev_corners):
    '''
    bev_corners: shape=(N, 4, 2)
    return: shape=(N, 4)
    '''
    # xmin, xmax = np.min(bev_corners[:, :, 0], axis=-1), np.max(bev_corners[:, :, 0], axis=-1)
    # ymin, ymax = np.min(bev_corners[:, :, 1], axis=-1), np.max(bev_corners[:, :, 1], axis=-1)

    # why we don't implement like the above ? please see
    # https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html#calculation
    n = len(bev_corners)
    alignedbbox = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        cur_bev = bev_corners[i]
        alignedbbox[i, 0] = np.min(cur_bev[:, 0])
        alignedbbox[i, 2] = np.max(cur_bev[:, 0])
        alignedbbox[i, 1] = np.min(cur_bev[:, 1])
        alignedbbox[i, 3] = np.max(cur_bev[:, 1])
    return alignedbbox


# modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/data_augment_utils.py#L31
@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.
    Args:
        boxes (np.ndarray): Corners of current boxes. # (n1, 4, 2)
        qboxes (np.ndarray): Boxes to be avoid colliding. # (n2, 4, 2)
        clockwise (bool, optional): Whether the corners are in
            clockwise order. Default: True.
    return: shape=(n1, n2)
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = bevcorner2alignedbbox(boxes)
    qboxes_standup = bevcorner2alignedbbox(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                    min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                    max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                                       D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                                       D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret



def bbox3d2bevcorners(bboxes):
    '''
    bboxes: shape=(n, 7)

                ^ x (-0.5 * pi)
                |
                |                (bird's eye view)
       (-pi)  o |
        y <-------------- (0)
                 \ / (ag)
                  \ 
                   \ 

    return: shape=(n, 4, 2)
    '''
    
    centers, dims, angles = bboxes[:, :2], bboxes[:, 3:5], bboxes[:, 6]
    
    # 1.generate bbox corner coordinates, clockwise from minimal point
    bev_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    bev_corners = bev_corners[None, ...] * dims[:, None, :] # (1, 4, 2) * (n, 1, 2) -> (n, 4, 2)

    # 2. rotate
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (N, 2, 2)
    bev_corners = bev_corners @ rot_mat # (n, 4, 2)

    # 3. translate to centers
    bev_corners += centers[:, None, :] 
    return bev_corners.astype(np.float32)


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    '''
    np_bboxes = []
    for bbox in bboxes:
        # To return
        corner_boxes = np.zeros((8, 3))

        translation = bbox[0:3]
        w, l, h = bbox[3], bbox[4], bbox[5]
        rotation = bbox[6]

        # Create a bounding box outline
        bounding_box = np.array([
            [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

        # Standard 3x3 rotation matrix around the Z axis
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0.0],
            [np.sin(rotation), np.cos(rotation), 0.0],
            [0.0, 0.0, 1.0]])

        # Repeat the [x, y, z] eight times
        eight_points = np.tile(translation, (8, 1))

        # Translate the rotated bounding box by the
        # original center position to obtain the final box
        corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
        
#         corner_box = bounding_box + eight_points.transpose()
        
        np_bboxes.append(corner_box.transpose())

        
    np_bboxes = np.array(np_bboxes)
    return np_bboxes

def remove_pts_in_bboxes(points, bboxes, rm=True):
    '''
    points: shape=(N, 3)
    bboxes: shape=(n, 7)
    return: shape=(N, n), bool
    '''
    # 1. get 6 groups of rectangle vertexs
    bboxes_corners = bbox3d2corners(bboxes) # (n, 8, 3)
    bbox_group_rectangle_vertexs = group_rectangle_vertexs(bboxes_corners) # (n, 6, 4, 3)

    # 2. calculate plane equation: ax + by + cd + d = 0
    group_plane_equation_params = group_plane_equation(bbox_group_rectangle_vertexs)

    # 3. Judge each point inside or outside the bboxes
    # if point (x0, y0, z0) lies on the direction of normal vector(a, b, c), then ax0 + by0 + cz0 + d > 0.
    masks = points_in_bboxes(points, group_plane_equation_params) # (N, n)

    if not rm:
        return masks
        
    # 4. remove point insider the bboxes
    masks = np.any(masks, axis=-1)

    return points[~masks]


def nearest_bev(bboxes):
    '''
    bboxes: (n, 7), (x, y, z, w, l, h, theta)
    return: (n, 4), (x1, y1, x2, y2)
    '''    
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(bboxes[:, 6].cpu(), offset=0.5, period=np.pi).to(bboxes_bev)
    bboxes_bev = torch.where(torch.abs(bboxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)
    
    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat([bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2


def iou2d(bboxes1, bboxes2, metric=0):
    '''
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    '''
    bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :]) # (n, m)
    bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :]) # (n, m)
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

    iou_area = bboxes_w * bboxes_h # (n, m)
    
    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1] # (n, )
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1] # (m, )
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-8)
    return iou


def iou2d_nearest(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    '''
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou


def iou3d(bboxes1, bboxes2):
    '''

    Parameters
    ----------
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7)

    Returns: (n, m)
    -------

    '''
    # 1. height overlap
    bboxes1_bottom, bboxes2_bottom = bboxes1[:, 2], bboxes2[:, 2]  # (n, ), (m, )
    bboxes1_top, bboxes2_top = bboxes1[:, 2] + bboxes1[:, 5], bboxes2[:, 2] + bboxes2[:, 5]  # (n, ), (m, )
    bboxes_bottom = torch.maximum(bboxes1_bottom[:, None], bboxes2_bottom[None, :])  # (n, m)
    bboxes_top = torch.minimum(bboxes1_top[:, None], bboxes2_top[None, :])
    height_overlap = torch.clamp(bboxes_top - bboxes_bottom, min=0)

    # 2. bev overlap
    bboxes1_x1y1 = bboxes1[:, :2] - bboxes1[:, 3:5] / 2
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 3:5] / 2
    bboxes2_x1y1 = bboxes2[:, :2] - bboxes2[:, 3:5] / 2
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 3:5] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 6:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 6:]], dim=-1)
    bev_overlap = boxes_overlap_bev(bboxes1_bev, bboxes2_bev)  # (n, m)

    # 3. overlap and volume
    overlap = height_overlap * bev_overlap
    volume1 = bboxes1[:, 3] * bboxes1[:, 4] * bboxes1[:, 5]
    volume2 = bboxes2[:, 3] * bboxes2[:, 4] * bboxes2[:, 5]
    volume = volume1[:, None] + volume2[None, :]  # (n, m)

    # 4. iou
    iou = overlap / (volume - overlap + 1e-8)

    return iou



def iou_bev(bboxes1, bboxes2):
    '''
    bboxes1: (n, 5), (x, z, w, h, theta)
    bboxes2: (m, 5)
    return: (n, m)
    '''
    bboxes1_x1y1 = bboxes1[:, :2] - bboxes1[:, 2:4] / 2
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 2:4] / 2
    bboxes2_x1y1 = bboxes2[:, :2] - bboxes2[:, 2:4] / 2
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 2:4] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 4:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 4:]], dim=-1)
    bev_overlap = boxes_iou_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    return bev_overlap



# modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/utils.py#L11
def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def get_inputs(pts, annot):
    names = annot['name']
    locations = annot['location']
    dimensions = annot['dimensions']
    rotation_y = annot['rotation_y']
    difficulty = annot['occluded']
    
    gt_bboxes = np.concatenate([locations, dimensions, rotation_y[:, None]], axis=1).astype(np.float32)
    gt_labels = [CLASSES.get(name, -1) for name in names]
    
    data_dict = {
        'pts': pts,
        'gt_bboxes': gt_bboxes,
        'gt_labels': np.array(gt_labels),
        'gt_names': np.array(names),
        'difficulty': np.array(difficulty)
    }
    
    return data_dict



def normalize_z(pts, labels):
    # pts: (N, 4)
    min_z = pts[:, 2].min()
    pts_z = pts[:, 2] - min_z
    pts[:, 2] = pts_z
    
    # location: (n, 3)
    location = labels['location']
    location_z = location[:, 2] - min_z
    location[:, 2] = location_z
    
    labels['location'] = location
    
    return pts, labels



@numba.jit(nopython=True)
def points_in_bboxes(points, plane_equation_params):
    '''
    points: shape=(N, 3)
    plane_equation_params: shape=(n, 6, 4)
    return: shape=(N, n), bool
    '''
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks

    
def points_in_bboxes_v2(points, annotation_dict):
    '''
    points: shape=(N, 4) 
    dimensions: shape=(n, 3) 
    location: shape=(n, 3) 
    rotation_y: shape=(n, ) 
    name: shape=(n, )
    return:
        indices: shape=(N, n_valid_bbox), indices[i, j] denotes whether point i is in bbox j. 
        n_total_bbox: int. 
        bboxes_lidar: shape=(n_valid_bbox, 7) 
        name: shape=(n_valid_bbox, )
    '''
    
    dimensions = annotation_dict['dimensions']
    location = annotation_dict['location']
    rotation_y = annotation_dict['rotation_y']
    name = annotation_dict['name']
    n_total_bbox = len(dimensions)
    location, dimensions = location[:n_total_bbox], dimensions[:n_total_bbox]
    rotation_y, name = rotation_y[:n_total_bbox], name[:n_total_bbox]
    data_dict = get_inputs(points, annotation_dict)
    bboxes_lidar = data_dict['gt_bboxes']
    bboxes_corners = bbox3d2corners(bboxes_lidar)
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, n), N is points num, n is bboxes number
    return indices, n_total_bbox, bboxes_lidar, name


def get_points_num_in_bbox(points, annotation_dict):
    '''
    points: shape=(N, 4) 
    dimensions: shape=(n, 3) 
    location: shape=(n, 3) 
    rotation_y: shape=(n, ) 
    name: shape=(n, )
    return: shape=(n, )
    '''
    indices, n_total_bbox, bboxes_lidar, name = \
        points_in_bboxes_v2(
            points=points, 
            annotation_dict=annotation_dict)
    points_num = np.sum(indices, axis=0)
    return np.array(points_num, dtype=int)

    