import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

from vgn.detection_implicit import VGNImplicit
from vgn.experiments.clutter_removal import State
from vgn.perception import TSDFVolume, CameraIntrinsic
from vgn.utils.transform import Transform
from vgn.utils.comm import receive_msg, send_msg
from vgn.utils.visual import grasp2mesh

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
        grasps, scores = predict_grasp(args, planner, data)
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
    # grasp_meshes = [
    #     grasp2mesh(grasps[idx], 1).as_open3d for idx in range(len(grasps))
    # ]
    # geometries = [pc] + grasp_meshes
    grasp_mesh = grasp2mesh(grasps[0], 1).as_open3d
    grasp_mesh.paint_uniform_color([0, 0.8, 0])
    geometries = [high_res_tsdf.get_mesh(), grasp_mesh]
    o3d.visualization.draw_geometries(geometries)
    return grasps, scores
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--size", type=float, default=0.3)
    parser.add_argument("--resolution", type=float, default=40)
    parser.add_argument("--lower", type=float, nargs=3, default=[0.02, 0.02, 0.005])
    parser.add_argument("--upper", type=float, nargs=3, default=[0.28, 0.28, 0.3])
    args = parser.parse_args()
    main(args)
