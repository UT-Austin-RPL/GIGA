import os
import json
import argparse
import numpy as np
import pickle
from pathlib import Path

from vgn.grasp import Label
from vgn.simulation import ClutterRemovalSim
from vgn.utils.comm import receive_msg, send_msg
from vgn.utils.transform import Transform, Rotation

from pyquaternion import Quaternion

from gprs import config_root
from gprs.franka_interface import FrankaInterface
from gprs.utils import YamlConfig
from gprs.utils.input_utils import input2action
from gprs.utils.io_devices import SpaceMouse
import gprs.utils.transform_utils as T

from rpl_vision_utils.networking.camera_redis_interface import CameraRedisSubInterface
from rpl_vision_utils.utils.transformation.transform_manager import TransformManager
import rpl_vision_utils.utils.transformation.transform_utils as T

MAX_CONSECUTIVE_FAILURES = 2

def move_to(robot_interface, controller_cfg, num_steps, num_additional_steps, grasp_pose, pregrasp_pose):
    while True:
        if len(robot_interface._state_buffer) > 0:
            if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q))) < 1e-3:
                continue
            else:
                break

    current_pose = np.array(robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
    current_pos = current_pose[:3, 3:]
    current_rot = current_pose[:3, :3]
    current_quat = T.mat2quat(current_rot)

    target_pos, target_rot = grasp_pose    
    target_pos, _ = pregrasp_pose

    target_quat = T.mat2quat(target_rot)
    if np.dot(target_quat, current_quat) < 0.0:
        current_quat = -current_quat
    target_axis_angle = T.quat2axisangle(target_quat)
    current_axis_angle = T.quat2axisangle(current_quat)    

    controller_type = "OSC_POSE"

    # print(current_pos)
    # print(target_pos)
    
    # print(current_axis_angle)
    # print(target_axis_angle)

    def osc_move(target_pose, num_iters=200):
        target_pos, target_quat = target_pose
        for _ in range(num_iters):
            current_pose = np.array(robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
            current_pos = current_pose[:3, 3:]
            current_rot = current_pose[:3, :3]
            current_quat = T.mat2quat(current_rot)
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            quat_diff = T.quat_distance(target_quat, current_quat)
            current_axis_angle = T.quat2axisangle(current_quat)
            axis_angle_diff = T.quat2axisangle(quat_diff)
            # print(np.round(current_pos.flatten(), 2))
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            print("Position error: ", np.round((target_pos - current_pos).flatten(), 3))
            print("Rotation error: ", np.round(axis_angle_diff, 3))
            action_pos = np.clip(action_pos, -0.8, 0.8)
            action_axis_angle = np.clip(action_axis_angle, -0.2, 0.2) # * np.sin(_ / num_iters * np.pi)

            action = action_pos.tolist() + action_axis_angle.tolist() + [-1.]
            # print(np.round(action, 2))
            robot_interface.control(
                control_type=controller_type, action=action, controller_cfg=controller_cfg
            )
        print("===================")
        print(current_pos.flatten())
        print(target_pos.flatten())

    osc_move((target_pos, target_quat), num_iters=100)
    current_pose = np.array(robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
    current_pos = current_pose[:3, 3:]
    current_rot = current_pose[:3, :3]
    current_quat = T.mat2quat(current_rot)
    target_pos, target_rot = grasp_pose
    target_pos = target_pos
    target_quat = T.mat2quat(target_rot)
    if np.dot(target_quat, current_quat) < 0.0:
        current_quat = -current_quat
    target_axis_angle = T.quat2axisangle(target_quat)
    current_axis_angle = T.quat2axisangle(current_quat)    
    osc_move((target_pos, target_quat), num_iters=40)

    # num_iters = 50
    # for _ in range(num_iters):
    #     current_pose = np.array(robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
    #     current_pos = current_pose[:3, 3:]
    #     current_rot = current_pose[:3, :3]
    #     current_quat = T.mat2quat(current_rot)
    #     if np.dot(target_quat, current_quat) < 0.0:
    #         current_quat = -current_quat
    #     quat_diff = T.quat_distance(target_quat, current_quat)
    #     current_axis_angle = T.quat2axisangle(current_quat)
    #     axis_angle_diff = T.quat2axisangle(quat_diff)
    #     # print(np.round(current_pos.flatten(), 2))
    #     action_pos = (target_pos - current_pos).flatten() * 10
    #     action_axis_angle = axis_angle_diff.flatten() * 1
    #     action_pos = np.clip(action_pos, -1., 1.)
    #     action_axis_angle = np.clip(action_axis_angle, -0.1, 0.1)
        
    #     action = action_pos.tolist() + action_axis_angle.tolist() + [-1.]
    #     # print(np.round(action, 2))
    #     robot_interface.control(
    #         control_type=controller_type, action=action, controller_cfg=controller_cfg
    #     )

    action = [0.] * 6 + [1.]
    for _ in range(30):
        robot_interface.control(
            control_type=controller_type, action=action, controller_cfg=controller_cfg
        )


def main(args):

    # Initialize camera and get its information
    calibration_method = "horaud"
    with open(
        os.path.join(
            args.config_folder,
            f"camera_{args.camera_id}_{args.camera_type}_{calibration_method}_extrinsics.json",
        ),
        "r",
    ) as f:
        extrinsics = json.load(f)
    
    extrinsics_pos = np.array(extrinsics["translation"])
    extrinsics_rot_in_matrix = np.array(extrinsics["rotation"])

    transform_manager = TransformManager()

    offset_x = -0.35
    offset_y = 0.13
    offset_z = -0.09

    extrinsics_pos += np.array([[offset_x], [offset_y], [offset_z]])

    transform_manager.add_transform("cam", "base", extrinsics_rot_in_matrix, extrinsics_pos)

    T_cam2base = transform_manager.get_transform("base", "cam")
    print(f"transformation is: {T_cam2base}")

    cr_interface = CameraRedisSubInterface(camera_id=args.camera_id, use_depth=True)
    cr_interface.start()

    import time; time.sleep(0.3)
    imgs = cr_interface.get_img()
    rgb_img = np.array(imgs["color"])[..., ::-1]
    depth_img = np.array(imgs["depth"])
    depth_img = depth_img.astype(np.float32) * 0.001
    # print(depth_img.max(), depth_img.min())
    # import cv2; cv2.imshow("", depth_img); cv2.waitKey(0)

    intrinsics_config = cr_interface.get_img_info()["intrinsics"]
    depth_intrinsics_cfg = intrinsics_config["color"]

    depth_intrinsics = {"width": depth_img.shape[1], "height": depth_img.shape[0], "K": [depth_intrinsics_cfg["fx"], 0, depth_intrinsics_cfg["cx"], 0, depth_intrinsics_cfg["fy"], depth_intrinsics_cfg["cy"], 0., 0., 1.]}

    consecutive_failures = 1

    trial_id = 0
    data = []
    data.append((rgb_img, depth_img, depth_intrinsics, T_cam2base))
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

    grasp_idx = 0
    if len(grasps) == 0:
        print('No grasp detected!')
        exit(0)

    while True:
        chosen_grasp = grasps[grasp_idx]
        # execute grasp
        grasp, score = chosen_grasp, scores[0]
        grasp_pose = chosen_grasp.pose.as_matrix()

        if (np.dot(chosen_grasp.pose.rotation.as_matrix(), np.array([0., 0., 1.])))[-1] > -0.05:
            grasp_idx += 1
            print("Rejecting the orientation of the chosen grasp!!!")
        else:
            break

    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
    T_world_pregrasp = (chosen_grasp.pose * T_grasp_pregrasp).as_matrix()

    T_final_grasp = Transform(Rotation.identity(), [0.0, 0.0, 0.015])
    T_world_grasp = (chosen_grasp.pose * T_final_grasp).as_matrix()
    
    final_grasp_pos = T_world_grasp[:3, 3:] - np.array([[offset_x], [offset_y], [offset_z]]) - np.array([[-0.03], [0.00], [0.0]])
    final_grasp_rot = T_world_grasp[:3, :3]

    final_pregrasp_pos = T_world_pregrasp[:3, 3:] - np.array([[offset_x], [offset_y], [offset_z]]) -np.array([[-0.015], [0.00], [0.0]])
    final_pregrasp_rot = T_world_grasp[:3, :3]

    print(final_pregrasp_pos)
    print(final_pregrasp_rot)
    
    print(final_grasp_pos)
    print(final_grasp_rot)

    num_steps = 20

    # number of additional steps to freeze final target and wait
    num_additional_steps = 40

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_cfg = YamlConfig("./" + f"/{args.controller_cfg}").as_easydict()

    # reset arm to known configuration with joint position controller
    # reset(robot_interface, controller_cfg)

    # use joint impedance controller to try and move to a new position using interpolated path
    move_to(
        robot_interface,
        controller_cfg,
        num_steps=num_steps,
        num_additional_steps=num_additional_steps,
        grasp_pose=(final_grasp_pos, final_grasp_rot),
        pregrasp_pose=(final_pregrasp_pos, final_pregrasp_rot)
    )

    joint_controller_cfg = YamlConfig(config_root + f"/osc-controller.yml").as_easydict()
    
    def move_to_joint_configuration(robot_interface, joint_configuration, gripper_close=False):
        # time.sleep(0.5)
        print(f"Moving to : {joint_configuration}")
        action = joint_configuration + [int(gripper_close) * 2 - 1]
        print(f"Moving to : {action}")
        start_time = time.time()        
        while True:
            end_time = time.time()            
            if len(robot_interface._state_buffer) > 0:
                if (
                        np.max(
                            np.abs(
                                np.array(robot_interface._state_buffer[-1].q)
                                - np.array(joint_configuration)
                            )
                        )
                        < 1e-3
                ) and (end_time - start_time) > 1.0:
                    break
            robot_interface.control(
                control_type="JOINT_POSITION", action=action, controller_cfg=joint_controller_cfg
            )
            if end_time - start_time > 6.:
                break
    
    top_joint_positions = [
        0.06695711197551554,
        -0.34349735836478174,
        -0.06032276350132912,
        -2.0003026779060264,
        -0.022716588261364475,
        1.642539553759179,
        0.8188037302796993 ]

    dispose_joint_positions = [0.486008438756645, -0.875851214873099, 0.21941024770488846, -2.3876419100510446, 0.09379814031389025, 1.5721408413482367, 1.231319760866463]

    current_joint_positions = np.array(robot_interface._state_buffer[-1].q).tolist()

    for _ in range(3):
        move_to_joint_configuration(robot_interface, current_joint_positions, gripper_close=True)
        time.sleep(0.2)    

    move_to_joint_configuration(robot_interface, top_joint_positions, gripper_close=True)
    move_to_joint_configuration(robot_interface, dispose_joint_positions, gripper_close=True)
    move_to_joint_configuration(robot_interface, dispose_joint_positions, gripper_close=False)
    time.sleep(0.5)
    move_to_joint_configuration(robot_interface, top_joint_positions, gripper_close=False)


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
    parser.add_argument("--save-obs-dir", type=str, default="/tmp")
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

    parser.add_argument(
        "-e",
        "--extrinsic",
        type=str,
        help="Path to extrinsic file",
        default="~/.rpl_vision_utils/calibration/camera_0_k4a_extrinsics.json",
    )
    parser.add_argument("--img-dir", type=str, help="name of image dir", default="imgs_1")
    parser.add_argument(
        "--joints", type=str, help="name of image dir", default="multiview_joints.json"
    )
    parser.add_argument(
        "--config-folder",
        type=str,
        default=os.path.expanduser("~/.rpl_vision_utils/calibration"),
    )

    parser.add_argument("--camera-id", type=int, default=0)

    parser.add_argument("--camera-type", type=str, default="k4a")

    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="giga-osc-controller.yml"
    )
    parser.add_argument(
        "--folder", type=Path, default="data_collection_example/example_data"
    )
    
    args = parser.parse_args()
    main(args)
