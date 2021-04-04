import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import subprocess
import trimesh
import pyrender
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from vgn.grasp import Grasp
from vgn.utils.transform import Transform, Rotation
from vgn.utils.implicit import as_mesh

#########
# visualize affordance and graso
#########
cmap = plt.get_cmap('Reds')


def affordance_visual(qual_vol,
                      rot_vol,
                      scene_mesh,
                      size=0.3,
                      resolution=40,
                      th=0.5,
                      temp=150,
                      rad=0.02,
                      finger_depth=0.05,
                      finger_offset=0.5,
                      move_center=True,
                      aggregation='max'):
    # Transform voxel grid into point cloud
    x = np.linspace(0, size, num=resolution)
    y = np.linspace(0, size, num=resolution)
    z = np.linspace(0, size, num=resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    grid = np.stack((Y, X, Z), axis=-1)
    # move center_vol to grasp center
    if move_center:
        z_axis = np.stack([
            2 * rot_vol[:, :, :, 0] * rot_vol[:, :, :, 2] +
            2 * rot_vol[:, :, :, 1] * rot_vol[:, :, :, 3],
            2 * rot_vol[:, :, :, 1] * rot_vol[:, :, :, 2] -
            2 * rot_vol[:, :, :, 0] * rot_vol[:, :, :, 3],
            1 - 2 * rot_vol[:, :, :, 0] * rot_vol[:, :, :, 0] -
            2 * rot_vol[:, :, :, 1] * rot_vol[:, :, :, 1]
        ],
                          axis=-1)
        grid += z_axis * finger_depth * finger_offset

    grid = grid[qual_vol > th]
    if grid.shape[0] <= 0:
        return scene_mesh
    qual_vol = qual_vol[qual_vol > th]
    pc_coordinate = np.reshape(grid, (-1, 3))
    pc_vector = np.expand_dims(np.reshape(qual_vol, (-1, )), axis=1)
    qual_pc = np.concatenate((pc_coordinate, pc_vector), axis=1)

    # Calculate the affordance value for each trimesh face
    # sum(exp(-dist_i * 150) * aff_i) / sum(exp(-dist_i)) (using 150 as the temperature term for exp)
    mesh = scene_mesh.copy()
    triangles_center = mesh.triangles_center
    centers = np.reshape(triangles_center, (triangles_center.shape[0], 1, 3))
    qual_pc_coords = np.reshape(qual_pc[:, 0:3], (1, -1, 3))
    diff = centers - qual_pc_coords
    dist = np.sqrt((diff**2).sum(axis=-1))

    if aggregation == 'mean':
        weight = np.exp(-dist * temp)
        affordance = weight.dot(qual_pc[:, 3]) / weight.sum(axis=-1)
    elif aggregation == 'max':
        # num_face, num_points
        mask = dist <= rad
        affordance = mask * qual_pc[:, 3][np.newaxis]
        # num_face
        affordance = affordance.max(axis=1)
    elif aggregation == 'softmax':
        # num_face, num_points
        mask = dist <= rad
        affordance_mask = mask * qual_pc[:, 3][np.newaxis]
        # mask out points outside radiance
        affordance_mask[np.logical_not(mask)] = -1e10
        # softmax
        weight = np.exp(affordance_mask * temp)
        affordance = weight.dot(qual_pc[:, 3]) / (weight.sum(axis=-1) + 1e-5)

    affordance = np.clip(affordance, a_min=th, a_max=1)
    affordance = (affordance - th) / (1 - th)
    # affordance = (affordance - affordance.min()) / (affordance.max() -
    #                                                 affordance.min())
    # different colormaps if need to change
    # cmap = plt.get_cmap('rainbow')
    # cmap = plt.get_cmap('nipy_spectral')
    colors = cmap(affordance ** 4)
    mesh.visual.face_colors = colors
    return mesh


def grasp2mesh(grasp, score, finger_depth=0.05):
    # color = cmap(float(score))
    # color = (np.array(color) * 255).astype(np.uint8)
    color = np.array([0, 250, 0, 180]).astype(np.uint8)
    radius = 0.1 * finger_depth
    w, d = grasp.width, finger_depth
    scene = trimesh.Scene()
    # left finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, -w / 2, d / 2])
    scale = [radius, radius, d]
    left_finger = trimesh.creation.cylinder(radius,
                                            d,
                                            transform=pose.as_matrix())
    scene.add_geometry(left_finger, 'left_finger')

    # right finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, w / 2, d / 2])
    scale = [radius, radius, d]
    right_finger = trimesh.creation.cylinder(radius,
                                             d,
                                             transform=pose.as_matrix())
    scene.add_geometry(right_finger, 'right_finger')

    # wrist
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, 0.0, -d / 4])
    scale = [radius, radius, d / 2]
    wrist = trimesh.creation.cylinder(radius,
                                      d / 2,
                                      transform=pose.as_matrix())
    scene.add_geometry(wrist, 'wrist')

    # palm
    pose = grasp.pose * Transform(
        Rotation.from_rotvec(np.pi / 2 * np.r_[1.0, 0.0, 0.0]),
        [0.0, 0.0, 0.0])
    scale = [radius, radius, w]
    palm = trimesh.creation.cylinder(radius, w, transform=pose.as_matrix())
    scene.add_geometry(palm, 'palm')
    scene = as_mesh(scene)
    colors = np.repeat(color[np.newaxis, :], len(scene.faces), axis=0)
    scene.visual.face_colors = colors
    return scene

#########
# Render
#########
def get_camera_pose(radius, center=np.zeros(3), ax=0, ay=0, az=0):
    rotation = R.from_euler('xyz', (ax, ay, az)).as_matrix()
    vec = np.array([0, 0, radius])
    translation = rotation.dot(vec) + center
    camera_pose = np.zeros((4, 4))
    camera_pose[3, 3] = 1
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = translation
    return camera_pose

def render_mesh(mesh, camera, light, camera_pose, light_pose, renderer):
    r_scene = pyrender.Scene()
    o_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    r_scene.add(o_mesh)
    r_scene.add(camera, name='camera', pose=camera_pose)
    r_scene.add(light, name='light', pose=light_pose)
    color_img, _ = renderer.render(r_scene)
    return Image.fromarray(color_img)

#########
# Plot
#########


def plot_3d_point_cloud(x,
                        y,
                        z,
                        show=True,
                        show_axis=True,
                        in_u_sphere=False,
                        marker='.',
                        s=8,
                        alpha=.8,
                        figsize=(5, 5),
                        elev=10,
                        azim=240,
                        axis=None,
                        title=None,
                        lim=None,
                        *args,
                        **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if lim:
        ax.set_xlim3d(*lim[0])
        ax.set_ylim3d(*lim[1])
        ax.set_zlim3d(*lim[2])
    elif in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        lim = (min(np.min(x), np.min(y),
                   np.min(z)), max(np.max(x), np.max(y), np.max(z)))
        ax.set_xlim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_ylim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_zlim(1.3 * lim[0], 1.3 * lim[1])
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if show:
        plt.show()

    return fig


def plot_3d_point_cloud_dict(name_dict, lim, size=2):
    num_plots = len(name_dict)
    fig = plt.figure(figsize=(size * num_plots, size))
    ax = {}
    for i, (k, v) in enumerate(name_dict.items()):
        ax[k] = fig.add_subplot(1, num_plots, i + 1, projection='3d')
        plot_3d_point_cloud(v[2], -v[0], v[1], axis=ax[k], show=False, lim=lim)
        ax[k].set_title(k)
    plt.tight_layout()
    return fig


def plot_3d_voxel_cloud_dict(name_dict, size=5, *args, **kwargs):
    num_plots = len(name_dict)
    fig = plt.figure(figsize=(size * num_plots, size))
    ax = {}
    for i, (k, v) in enumerate(name_dict.items()):
        ax[k] = fig.add_subplot(1, num_plots, i + 1, projection='3d')
        plot_voxel_as_cloud(v, axis=ax[k], fig=fig, *args, **kwargs)
        ax[k].set_title(k)
    plt.tight_layout()
    return fig


def plot_voxel_as_cloud(voxel,
                        axis=None,
                        figsize=(5, 5),
                        marker='s',
                        s=8,
                        alpha=.8,
                        lim=[0.3, 0.3, 0.3],
                        elev=10,
                        azim=240,
                        fig=None,
                        *args,
                        **kwargs):
    cloud = convert_voxel_to_cloud(voxel, lim)
    points = cloud[:, :3]
    val = cloud[:, 3]
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis

    # color = cm.coolwarm(val)
    sc = ax.scatter(*points.T,
                    marker=marker,
                    s=s,
                    alpha=alpha,
                    c=val,
                    cmap=plt.cm.get_cmap('RdYlBu_r'),
                    *args,
                    **kwargs)
    plt.colorbar(sc, ax=ax)
    ax.set_xlim3d(0, lim[0])
    ax.set_ylim3d(0, lim[1])
    ax.set_zlim3d(0, lim[2])

    return fig

def plot_tsdf_with_grasps(tsdf,
                          grasps,
                          axis=None,
                          figsize=(5, 5),
                          marker='s',
                          s=8,
                          alpha=.8,
                          lim=[0.3, 0.3, 0.3],
                          elev=10,
                          azim=240,
                          fig=None,
                          *args,
                          **kwargs):
    cloud = convert_voxel_to_cloud(tsdf, lim)
    points = cloud[:, :3]
    val = cloud[:, 3]
    grasp_meshes = [
        grasp2mesh(grasps[idx], 1) for idx in range(len(grasps))
    ]
    gripper_points = [grasp_mesh.sample(512) for grasp_mesh in grasp_meshes]
    gripper_points = np.concatenate(gripper_points, axis=0)
    cmap = plt.cm.get_cmap('RdYlBu_r')
    color_tsdf = cmap(val)
    color_gripper_points = np.tile(np.array([[0, 1, 0, 0.3]]),
                                   (gripper_points.shape[0], 1))

    points = np.concatenate((points, gripper_points), axis=0)
    color = np.concatenate((color_tsdf, color_gripper_points), axis=0)

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis

    # color = cm.coolwarm(val)
    sc = ax.scatter(*points.T,
                    marker=marker,
                    s=s,
                    alpha=alpha,
                    c=color,
                    *args,
                    **kwargs)
    # plt.colorbar(sc, ax=ax)
    ax.set_xlim3d(0, lim[0])
    ax.set_ylim3d(0, lim[1])
    ax.set_zlim3d(0, lim[2])

    return fig

def convert_voxel_to_cloud(voxel, size):

    assert len(voxel.shape) == 3
    lx, ly, lz = voxel.shape
    lx = lx / voxel.shape[0] * size[0]
    ly = ly / voxel.shape[1] * size[1]
    lz = lz / voxel.shape[2] * size[2]
    points = []
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if voxel[x, y, z] > 0:
                    points.append([
                        x / voxel.shape[0] * size[0],
                        y / voxel.shape[1] * size[1],
                        z / voxel.shape[2] * size[2], voxel[x, y, z]
                    ])

    return np.array(points)
