import os
import trimesh
import numpy as np
from urdfpy import URDF
try:
    from vgn.ConvONets.utils.libmesh import check_mesh_contains
except:
    print('import libmesh failed!')

n_iou_points = 100000
n_iou_points_files = 10

## occupancy related code
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    The returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, visual=g.visual)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def get_mesh_pose_list_from_world(world, object_set, exclude_plane=True):
    mesh_pose_list = []
    # collect object mesh paths and poses
    for uid in world.bodies.keys():
        _, name = world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = world.bodies[uid]
        pose = body.get_pose().as_matrix()
        scale = body.scale
        visuals = world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, _, mesh_path, _, _, _ = visuals[0]
        mesh_path = mesh_path.decode('utf8')
        if mesh_path == '':
            mesh_path = os.path.join('./data/urdfs', object_set, name + '.urdf')
        mesh_pose_list.append((mesh_path, scale, pose))
    return mesh_pose_list

def get_scene_from_mesh_pose_list(mesh_pose_list, scene_as_mesh=True, return_list=False):
    # create scene from meshes
    scene = trimesh.Scene()
    mesh_list = []
    for mesh_path, scale, pose in mesh_pose_list:
        if os.path.splitext(mesh_path)[1] == '.urdf':
            obj = URDF.load(mesh_path)
            assert len(obj.links) == 1
            assert len(obj.links[0].visuals) == 1
            assert len(obj.links[0].visuals[0].geometry.meshes) == 1
            mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
        else:
            mesh = trimesh.load(mesh_path)

        mesh.apply_scale(scale)
        mesh.apply_transform(pose)
        scene.add_geometry(mesh)
        mesh_list.append(mesh)
    if scene_as_mesh:
        scene = as_mesh(scene)
    if return_list:
        return scene, mesh_list
    else:
        return scene

def sample_iou_points(mesh_list, bounds, num_point, padding=0.02, uniform=False, size=0.3):
    points = np.random.rand(num_point, 3).astype(np.float32)
    if uniform:
        points *= size + 2 * padding
        points -= padding
    else:
        points = points * (bounds[[1]] + 2 * padding - bounds[[0]]) + bounds[[0]] - padding
    occ = np.zeros(num_point).astype(bool)
    for mesh in mesh_list:
        occi = check_mesh_contains(mesh, points)
        occ = occ | occi

    return points, occ

def get_occ_from_world(world, object_set):
    mesh_pose_list = get_mesh_pose_list_from_world(world, object_set)
    scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True)
    points, occ = sample_iou_points(mesh_list, scene.bounds, n_iou_points * n_iou_points_files)
    return points, occ


# def get_occ_from_mesh(scene_mesh, world_size, object_count, voxel_resolution=120):
#     # voxelize scene
#     voxel_length = world_size / voxel_resolution
#     scene_voxel = scene_mesh.voxelized(voxel_length)

#     # collect near surface points occupancy
#     surface_points, _ = trimesh.sample.sample_surface(scene_mesh, object_count * 2048)
#     surface_points += np.random.randn(*surface_points.shape) * 0.002
#     occ_surface = scene_voxel.is_filled(surface_points)
#     # collect randomly distributed points occupancy
#     random_points = np.random.rand(object_count * 2048, 3)
#     random_points = random_points * (scene_voxel.bounds[[1]] - scene_voxel.bounds[[0]]) + scene_voxel.bounds[[0]]
#     occ_random = scene_voxel.is_filled(random_points)
#     surface_p_occ = np.concatenate((surface_points, occ_surface[:, np.newaxis]), axis=-1)
#     random_p_occ = np.concatenate((random_points, occ_random[:, np.newaxis]), axis=-1)
#     return surface_p_occ, random_p_occ
