import argparse
import numpy as np
import pickle
from pathlib import Path

from vgn.grasp import Label
from vgn.simulation import ClutterRemovalSim
from vgn.utils.comm import receive_msg, send_msg
MAX_CONSECUTIVE_FAILURES = 2


def main(args):
    # sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, add_noise=add_noise, sideview=sideview)
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui, seed=args.seeds[0], add_noise=args.add_noise, sideview=args.sideview) 
    sim.reset(args.num_objects)
    consecutive_failures = 1
    
    last_label = None
    trial_id = -1
    cnt = 0
    total = sim.num_objects

    while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
        trial_id += 1
        timings = {}

        ### communicate with the grasp planner
        # scan the scene
        data = sim.acquire_obs(n=args.num_view, N=None, resolution=40)
        data = [(x[0], x[1], x[2].to_dict(), x[3].as_matrix()) for x in data]
        # data: list of (depth_img, intrinsic, extrinsic)
        # save data and send path
        obs_path = f'{args.save_obs_dir}/{trial_id}.pkl'
        with open(obs_path, 'wb') as f:
            pickle.dump(data, f)
        send_msg(['input', obs_path], port=12345)
        
        # receive path to predicted grasp
        _, result_path = receive_msg(port=12346)
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
        grasps = result['grasps']
        scores = result['scores']

        # execute grasp
        grasp, score = grasps[0], scores[0]
        label, _ = sim.execute_grasp(grasp, allow_contact=True)
        
        cnt += label
        
        print(f'Number of grasps: {len(grasps)}, score: {score}, result: {label}')

        if last_label == Label.FAILURE and label == Label.FAILURE:
            consecutive_failures += 1
        else:
            consecutive_failures = 1
        last_label = label
    send_msg(['finish', f'{cnt}/{total} grasped'], port=12345)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene",
                        type=str,
                        choices=["pile", "packed"],
                        default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--save-obs-dir", type=str)
    parser.add_argument(
        "--force",
        action="store_true",
        help=
        "When all grasps are under threshold, force the detector to select the best grasp"
    )
    parser.add_argument(
        "--add-noise",
        type=str,
        default='',
        help="Whether add noise to depth observation, trans | dex | norm | ''")
    parser.add_argument("--sideview",
                        action="store_true",
                        help="Whether to look from one side")
    parser.add_argument("--silence",
                        action="store_true",
                        help="Whether to disable tqdm bar")
    parser.add_argument("--vis",
                        action="store_true",
                        help="visualize and save affordance")

    args = parser.parse_args()
    main(args)
