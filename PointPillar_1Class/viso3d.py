import numpy as np
import open3d as o3d
import os
from ply_functions import read_ply, write_ply
from utils import bbox3d2corners, read_label, read_pc


# COLORS = [[1, 1, 1], [0, 0, 0]]
# COLORS = [[0, 0, 0], [0, 1, 0]]
PRED_COLORS = [[0, 0, 0]]
GT_COLORS = [[0, 1, 0]]

LINES = [
        [0, 1],
        [1, 2], 
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 6],
        [7, 3],
        [1, 5],
        [4, 0]
    ]



LINES = np.flip(np.array(LINES))



def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    return ply


def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def vis_core(plys):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for ply in plys:
        vis.add_geometry(ply)

    vis.run()
    vis.destroy_window()

    
def vis_pc(pc, pred_bboxes=None, pred_labels=None, gt_bboxes=None, gt_labels=None):
    '''
    pc: ply or np.ndarray (N, 4)
    bboxes: np.ndarray, (n, 7) or (n, 8, 3)
    labels: (n, )
    '''
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=10, origin=[0, 0, 0])

    if pred_bboxes is None:
        vis_core([pc, mesh_frame])
        return
    
    if len(pred_bboxes.shape) == 2:
        pred_bboxes = bbox3d2corners(pred_bboxes)

    if len(gt_bboxes.shape) == 2:
        gt_bboxes = bbox3d2corners(gt_bboxes)
    
    vis_objs = [pc]
    for i in range(len(pred_bboxes)):
        bbox = pred_bboxes[i]
        if pred_labels is None:
            color = [0, 0, 0]
        else:
            if pred_labels[i] >= 0 and pred_labels[i] < 3:
                color = PRED_COLORS[pred_labels[i]]
            else:
                color = PRED_COLORS[-1]
        vis_objs.append(bbox_obj(bbox, color=color))



    for i in range(len(gt_bboxes)):
        bbox = gt_bboxes[i]
        if gt_labels is None:
            color = [0, 0, 0]
        else:
            if gt_labels[i] >= 0 and gt_labels[i] < 3:
                color = GT_COLORS[gt_labels[i]]
            else:
                color = GT_COLORS[-1]
        vis_objs.append(bbox_obj(bbox, color=color))


    vis_core(vis_objs)
    

    
    


