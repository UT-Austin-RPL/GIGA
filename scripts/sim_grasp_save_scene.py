import json
import argparse
import numpy as np
from pathlib import Path

from vgn.detection import VGN
from vgn.detection_implicit import VGNImplicit
from vgn.detection_implicit_top import VGNImplicitTop
from vgn.detection_implicit_pc import VGNImplicitPC
from vgn.experiments import clutter_removal_single
from vgn.utils.misc import set_random_seed


def main(args):

    if 'giga' in args.type:
        grasp_planner = VGNImplicit(args.model,
                                    args.type,
                                    best=args.best,
                                    qual_th=args.qual_th,
                                    force_detection=args.force,
                                    out_th=args.out_th,
                                    select_top=args.select_top,
                                    visualize=args.vis,
                                    resolution=args.res)
    elif args.type == 'vgn':
        grasp_planner = VGN(args.model,
                            args.type,
                            best=args.best,
                            qual_th=args.qual_th,
                            force_detection=args.force,
                            out_th=args.out_th,
                            visualize=args.vis)
    else:
        raise NotImplementedError(f'model type {args.type} not implemented!')

    set_random_seed(args.seed)

    results = {}
    for n in range(args.num_rounds):
        args.seed = np.random.randint(3000)
        save_dir = args.save_dir / f'round_{n:03d}'
        results[n] = clutter_removal_single.run(grasp_plan_fn=grasp_planner,
                                                save_dir=save_dir,
                                                scene=args.scene,
                                                object_set=args.object_set,
                                                num_objects=args.num_objects,
                                                n=args.num_view,
                                                seed=args.seed,
                                                sim_gui=args.sim_gui,
                                                add_noise=args.add_noise,
                                                sideview=args.sideview)
        print(f'Round {n} finished, result: {results[n]}')
    with open(args.save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--scene",
                        type=str,
                        choices=["pile", "packed"],
                        default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-gui", action="store_true")
    # parser.add_argument("--grad-refine", action="store_true")
    parser.add_argument("--qual-th", type=float, default=0.9)
    parser.add_argument(
        "--best",
        action="store_true",
        help="Whether to use best valid grasp (or random valid grasp)")
    parser.add_argument("--result-path", type=str, default=None)
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
    parser.add_argument("--simple-constrain",
                        action="store_true",
                        help="Whether to contrain grasp from backward")
    parser.add_argument("--res", type=int, default=40)
    parser.add_argument("--out-th", type=float, default=0.5)
    parser.add_argument("--silence",
                        action="store_true",
                        help="Whether to disable tqdm bar")
    parser.add_argument("--select-top",
                        action="store_true",
                        help="Use top heuristic")
    parser.add_argument("--vis",
                        action="store_true",
                        help="visualize and save affordance")

    args = parser.parse_args()
    main(args)
