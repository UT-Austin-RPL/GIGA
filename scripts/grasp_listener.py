import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from vgn.detection_implicit import VGNImplicit
from vgn.experiments.clutter_removal import State
from vgn.perception import TSDFVolume
from vgn.utils.comm import receive_msg, send_msg
from vgn.utils.visual import plot_tsdf_with_grasps

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
    tsdf = TSDFVolume(args.size, args.resolution)
    high_res_tsdf = TSDFVolume(args.size, 120)
    for depth, intrinsics, extrinsics in data:
        tsdf.integrate(depth, intrinsics, extrinsics)
        high_res_tsdf.integrate(depth, intrinsics, extrinsics)
    
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(args.lower, args.upper)
    pc = high_res_tsdf.get_cloud()
    pc = pc.crop(bounding_box)
    state = State(tsdf, pc)
    grasps, scores, _ = planner(state)
    fig = plot_tsdf_with_grasps(tsdf.get_grid()[0], [grasps[0]])
    fig.show()
    while True:
        if plt.waitforbuttonpress():
            break
    plt.close(fig)
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