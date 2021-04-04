import collections
import argparse
from datetime import datetime
import os


import numpy as np
import tqdm

from vgn import io#, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list

MAX_CONSECUTIVE_FAILURES = 2


State = collections.namedtuple("State", ["tsdf", "pc"])


def run(
    grasp_plan_fn,
    save_dir,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    seed=1,
    sim_gui=False,
    add_noise=False,
    sideview=False,
    resolution=40,
    silence=False,
    save_freq=8,
    
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    os.makedirs(save_dir, exist_ok=True)
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, add_noise=add_noise, sideview=sideview, save_dir=save_dir)
    cnt = 0
    success = 0
    left_objs = 0
    total_objs = 0
    cons_fail = 0
    no_grasp = 0

    sim.reset(num_objects)

    total_objs += sim.num_objects
    consecutive_failures = 1
    last_label = None
    trial_id = -1

    while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
        trial_id += 1
        timings = {}

        # scan the scene
        tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N, resolution=40)
        state = argparse.Namespace(tsdf=tsdf, pc=pc)
        if resolution != 40:
            extra_tsdf, _, _ = sim.acquire_tsdf(n=n, N=N, resolution=resolution)
            state.tsdf_process = extra_tsdf

        if pc.is_empty():
            break  # empty point cloud, abort this round TODO this should not happen

        grasps, scores, timings["planning"] = grasp_plan_fn(state)

        if len(grasps) == 0:
            no_grasp += 1
            break  # no detections found, abort this round

        # execute grasp
        grasp, score = grasps[0], scores[0]
        label, _ = sim.execute_grasp(grasp, allow_contact=True)
        cnt += 1
        if label != Label.FAILURE:
            success += 1

        if last_label == Label.FAILURE and label == Label.FAILURE:
            consecutive_failures += 1
        else:
            consecutive_failures = 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            cons_fail += 1
        last_label = label
    left_objs += sim.num_objects

    return success, cnt, total_objs