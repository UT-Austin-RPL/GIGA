import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list

class DatasetVoxel(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, augment=False):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.long)
        voxel_grid = read_voxel_grid(self.root, scene_id)
        
        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)
        
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        x, y = voxel_grid[0], (label, rotations, width)

        return x, y, pos

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene


class DatasetVoxelOccFile(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, num_point_occ=2048, augment=False):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.num_point_occ = num_point_occ
        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.long)
        voxel_grid = read_voxel_grid(self.root, scene_id)
        
        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)
        
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        x, y = voxel_grid[0], (label, rotations, width)

        occ_points, occ = self.read_occ(scene_id, self.num_point_occ)
        occ_points = occ_points / self.size - 0.5

        return x, y, pos, occ_points, occ

    def read_occ(self, scene_id, num_point):
        occ_paths = list((self.raw_root / 'occ' / scene_id).glob('*.npz'))
        path_idx = torch.randint(high=len(occ_paths), size=(1,), dtype=int).item()
        occ_path = occ_paths[path_idx]
        occ_data = np.load(occ_path)
        points = occ_data['points']
        occ = occ_data['occ']
        points, idxs = sample_point_cloud(points, num_point, return_idx=True)
        occ = occ[idxs]
        return points, occ

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene


def apply_transform(voxel_grid, orientation, position):
    angle = np.pi / 2.0 * np.random.choice(4)
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    z_offset = np.random.uniform(6, 34) - position[2]

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position

def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]