import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

from vgn.detection_implicit import VGNImplicit
from vgn.experiments.clutter_removal import State
from vgn.perception import TSDFVolume, CameraIntrinsic
from vgn.utils.transform import Transform, Rotation
from vgn.utils.comm import receive_msg, send_msg
from vgn.utils.visual import grasp2mesh, plot_voxel_as_cloud, plot_tsdf_with_grasps

# input: depth image, camera intrinsics, camera extrinsics
# output: grasp lists

def main(args):
    planner = VGNImplicit(args.model,
                        args.type,
                        best=True,
                        qual_th=0.8,
                        rviz=False,
                        force_detection=True,
                        out_th=0.1,
                        resolution=args.resolution)
    while True:
        while True:
            msgs = receive_msg()
            if msgs[0] == 'input':
                input_path = msgs[1]
                break
            elif msgs[0] == 'err':
                print(f'Controller error: {msgs[1]}, exit!')
                exit(1)
            elif msgs[0] == 'finish':
                print(f'Finished. {msgs[1]}')
                exit(0)
        with open(msgs[1], 'rb') as f:
            data = pickle.load(f)
            print(f'Received {len(data)} images')
        grasps, scores, geometries = predict_grasp(args, planner, data)

        #o3d.visualization.draw_geometries(geometries, zoom=1.0, front=[0, -1, 0], up=[0, 0, 1], lookat=[0.15, 0.15, 0.05], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries(geometries)
        o3d.visualization.draw_geometries(geometries, 
                                         zoom=0.92000000000000015, 
                                         front=[-0.61855079398231205, -0.72632745971988766, 0.29973878047511043 ], 
                                         lookat=[ 0.11056985301069856, 0.1742554585446297, 0.13039254214633547  ],  
                                         up=[0.001, 0.001, 1.],# [-0.23421436533840864, 0.1936948246897317, 0.95269404635357136],
        )
        output_path = input_path.replace('.pkl', 'grasps.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump({'grasps': grasps, 'scores': scores}, f)
        send_msg(['output', output_path])

    
def predict_grasp(args, planner, data):
    if len(data[0]) == 3:
        high_res_tsdf = TSDFVolume(args.size, 120)
    else:
        high_res_tsdf = TSDFVolume(args.size, 120, color_type='rgb')
    tsdf = TSDFVolume(args.size, args.resolution)
    
    for sample in data:
        if len(sample) == 3:
            depth, intrinsics, extrinsics = sample
            rgb = None
        else:
            rgb, depth, intrinsics, extrinsics = sample
        intrinsics = CameraIntrinsic.from_dict(intrinsics)
        extrinsics = Transform.from_matrix(extrinsics)
        tsdf.integrate(depth, intrinsics, extrinsics)
        high_res_tsdf.integrate(depth, intrinsics, extrinsics, rgb_img=rgb)
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(args.lower, args.upper)
    pc = high_res_tsdf.get_cloud()
    pc = pc.crop(bounding_box)
    state = State(tsdf, pc)
    grasps, scores, _ = planner(state)
    print(len(grasps))
    # if len(grasps) > 0:
    #     fig = plot_tsdf_with_grasps(tsdf.get_grid()[0], [grasps[0]])
    #     print(scores)
    # else:
    #     fig = plot_voxel_as_cloud(tsdf.get_grid()[0])
    # fig.show()
    # while True:
    #     if plt.waitforbuttonpress():
    #         break
    # plt.close(fig)

    # grasp_meshes = [
    #     grasp2mesh(grasps[idx], 1).as_open3d for idx in range(len(grasps))
    # ]
    # geometries = [pc] + grasp_meshes

    # from copy import deepcopy
    # grasp_bck = deepcopy(grasps[0])
    # grasp_mesh_bck = grasp2mesh(grasp_bck, 1).as_open3d
    # grasp_mesh_bck.paint_uniform_color([0, 0.8, 0])

    # pos = grasps[0].pose.translation
    # # pos[2] += 0.05
    # angle = grasps[0].pose.rotation.as_euler('xyz')
    # print(pos, angle)
    # if angle[2] > np.pi / 2 or angle[2] < - np.pi / 2:
    #     reflect = Transform(Rotation.from_euler('xyz', (0, 0, np.pi)), np.zeros((3)))
    #     grasps[0].pose = grasps[0].pose * reflect
    # pos = grasps[0].pose.translation
    # angle = grasps[0].pose.rotation.as_euler('xyz')
    # print(pos, angle)
    # # grasps[0].pose = Transform(Rotation.from_euler('xyz', (angle[0], angle[1], angle[2])), pos)
    # grasp_mesh = grasp2mesh(grasps[0], 1).as_open3d
    # grasp_mesh.paint_uniform_color([0.8, 0, 0])
    # geometries = [high_res_tsdf.get_mesh(), grasp_mesh, grasp_mesh_bck]
    if len(grasps) == 0:
        return [], [], [high_res_tsdf.get_mesh()]
    pos = grasps[0].pose.translation
    # pos[2] += 0.05
    angle = grasps[0].pose.rotation.as_euler('xyz')
    print(pos, angle)
    if angle[2] > np.pi / 2 or angle[2] < - np.pi / 2:
        reflect = Transform(Rotation.from_euler('xyz', (0, 0, np.pi)), np.zeros((3)))
        grasps[0].pose = grasps[0].pose * reflect
    pos = grasps[0].pose.translation
    angle = grasps[0].pose.rotation.as_euler('xyz')
    print(pos, angle)
    # grasps[0].pose = Transform(Rotation.from_euler('xyz', (angle[0], angle[1], angle[2])), pos)
    grasp_mesh = grasp2mesh(grasps[0], 1).as_open3d
    grasp_mesh.paint_uniform_color([0, 0.8, 0])
    geometries = [high_res_tsdf.get_mesh(), grasp_mesh]
    #exit(0)
    return grasps, scores, geometries
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default="models/pile_convonet_v_cat_occ.pt")
    parser.add_argument("--type", type=str, default="giga")
    parser.add_argument("--size", type=float, default=0.3)
    parser.add_argument("--resolution", type=float, default=40)
    parser.add_argument("--lower", type=float, nargs=3, default=[0.02, 0.02, 0.005])
    parser.add_argument("--upper", type=float, nargs=3, default=[0.28, 0.28, 0.3])
    args = parser.parse_args()
    main(args)
