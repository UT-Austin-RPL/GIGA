import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path
from collections import OrderedDict

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list, sample_iou_points, check_mesh_contains

class DatasetVoxelOccGeo(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point_occ=2048, augment=False):
        self.root = root
        self.num_point_occ = num_point_occ
        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)
        self.scene_list = OrderedDict()
        for i in range(len(self.df.index)):
            scene_id = self.df.loc[i, "scene_id"]
            self.scene_list[scene_id] = None
        self.scene_list = list(self.scene_list.keys())

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, i):
        scene_id = self.scene_list[i]
        voxel_grid = read_voxel_grid(self.root, scene_id)

        x = voxel_grid[0]

        occ_points, occ, scene = self.sample_occ(scene_id, self.num_point_occ)
        occ_points = occ_points / self.size - 0.5

        return x, occ_points, occ, scene

    def sample_occ(self, scene_id, num_point):
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True)
        points, occ = sample_iou_points(mesh_list, scene.bounds, num_point)
        return points, occ, scene

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene

class DatasetVoxelOccGeoROI(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 raw_root,
                 num_point_occ=2048,
                 ROI_scale=0.3,
                 uniform=True):
        self.root = root
        self.num_point_occ = num_point_occ
        self.raw_root = raw_root
        self.num_th = 32
        self.uniform = uniform
        # the thickness of region relative to finger depth
        self.ROI_scale = ROI_scale

        self.df = read_df(raw_root)
        self.size, _, _, self.finger_depth = read_setup(raw_root)
        self.scene_dict = OrderedDict()
        for i in range(len(self.df.index)):
            scene_id = self.df.loc[i, "scene_id"]
            if scene_id not in self.scene_dict.keys():
                self.scene_dict[scene_id] = []
            label = self.df.loc[i, "label"]
            if label:
                self.scene_dict[scene_id].append(i)
        pop_ks = []
        for k, v in self.scene_dict.items():
            if len(v) < 1:
                pop_ks.append(k)
        for k in pop_ks:
            self.scene_dict.pop(k)
        self.scene_list = list(self.scene_dict.keys())

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, i):

        scene_id = self.scene_list[i]
        voxel_grid = read_voxel_grid(self.root, scene_id)
        x = voxel_grid[0]

        pos_list = []
        width_list = []
        ori_list = []
        for grasp_idx in self.scene_dict[scene_id]:
            ori = Rotation.from_quat(
                self.df.loc[grasp_idx, "qx":"qw"].to_numpy(np.single))
            pos = self.df.loc[grasp_idx, "x":"z"].to_numpy(np.float64)
            width = self.df.loc[grasp_idx, "width"].astype(np.float64)
            pos += ori.apply(np.array([0, 0, 1])) * self.finger_depth
            ori_list.append(ori)
            pos_list.append(pos)
            width_list.append(width)

        # sample occ points proportional to volume
        num_occ_point_per_grasp = self.num_point_occ * np.array(width_list,
                                                                dtype=float)
        num_occ_point_per_grasp /= np.sum(np.array(width_list, dtype=float))
        num_occ_point_per_grasp = np.round(num_occ_point_per_grasp).astype(int)
        
        occ_points_list = []
        for num, pos, width, ori in zip(num_occ_point_per_grasp, pos_list,
                                        width_list, ori_list):
            occ_points = np.random.rand(num, 3)
            occ_points[:, 1] -= 0.5
            occ_points[:, 1] *= width
            occ_points[:, [0, 2]] -= 1
            occ_points[:, [0, 2]] *= self.finger_depth * self.ROI_scale
            occ_points = ori.as_matrix().dot(occ_points.T).T
            occ_points += pos
            occ_points_list.append(occ_points)
        occ_points_ROI = np.concatenate(occ_points_list, axis=0)
        occ_ROI, scene = self.check_occ(scene_id, occ_points_ROI)
        occ_points_ROI = occ_points_ROI / self.size - 0.5
        
        # uniform_occ_points
        occ_points, occ, _ = self.sample_occ(scene_id, self.num_point_occ)
        occ_points = occ_points / self.size - 0.5
        
        return x, occ_points, occ, occ_points_ROI, occ_ROI, scene

    def check_occ(self, scene_id, points):
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id +
                                                                  '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list,
                                                         return_list=True)
        occ = np.zeros(points.shape[0]).astype(bool)
        for mesh in mesh_list:
            occi = check_mesh_contains(mesh, points)
            occ = occ | occi
        return occ, scene

    def sample_occ(self, scene_id, num_point):
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id +
                                                                  '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list,
                                                         return_list=True)
        points, occ = sample_iou_points(mesh_list,
                                        scene.bounds,
                                        num_point,
                                        uniform=self.uniform,
                                        size=self.size, padding=0)
        return points, occ, scene

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id +
                                                                  '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list,
                                              return_list=False)
        return scene


def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]