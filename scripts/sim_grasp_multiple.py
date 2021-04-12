import argparse
import numpy as np
import json
from pathlib import Path

from vgn.detection import VGN
from vgn.detection_implicit import VGNImplicit
from vgn.experiments import clutter_removal
from vgn.utils.misc import set_random_seed


def main(args):

    if args.type in ['giga', 'giga_aff']:
        grasp_planner = VGNImplicit(args.model,
                                    args.type,
                                    best=args.best,
                                    qual_th=args.qual_th,
                                    force_detection=args.force,
                                    out_th=0.1,
                                    select_top=False,
                                    visualize=args.vis)
    elif args.type == 'vgn':
        grasp_planner = VGN(args.model,
                            args.type,
                            best=args.best,
                            qual_th=args.qual_th,
                            force_detection=args.force,
                            out_th=0.1,
                            visualize=args.vis)
    else:
        raise NotImplementedError(f'model type {args.type} not implemented!')

    gsr = []
    dr = []
    for seed in args.seeds:
        set_random_seed(seed)
        success_rate, declutter_rate = clutter_removal.run(
            grasp_plan_fn=grasp_planner,
            logdir=args.logdir,
            description=args.description,
            scene=args.scene,
            object_set=args.object_set,
            num_objects=args.num_objects,
            n=args.num_view,
            num_rounds=args.num_rounds,
            seed=seed,
            sim_gui=args.sim_gui,
            result_path=None,
            add_noise=args.add_noise,
            sideview=args.sideview,
            silence=args.silence,
            visualize=args.vis)
        gsr.append(success_rate)
        dr.append(declutter_rate)
    results = {
        'gsr': {
            'mean': np.mean(gsr),
            'std': np.std(gsr),
            'val': gsr
        },
        'dr': {
            'mean': np.mean(dr),
            'std': np.std(dr),
            'val': dr
        }
    }
    print('Average results:')
    print(f'Grasp sucess rate: {np.mean(gsr):.2f} ± {np.std(gsr):.2f} %')
    print(f'Declutter rate: {np.mean(dr):.2f} ± {np.std(dr):.2f} %')
    with open(args.result_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene",
                        type=str,
                        choices=["pile", "packed"],
                        default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument("--sim-gui", action="store_true")
    # parser.add_argument("--grad-refine", action="store_true")
    parser.add_argument("--qual-th", type=float, default=0.9)
    parser.add_argument("--eval-geo",
                        action="store_true",
                        help='whether evaluate geometry prediction')
    parser.add_argument(
        "--best",
        action="store_true",
        help="Whether to use best valid grasp (or random valid grasp)")
    parser.add_argument("--result-path", type=str)
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
