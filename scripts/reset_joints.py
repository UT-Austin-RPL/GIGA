"""Moving robot joint positions to initial pose for starting new experiments."""
import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gprs import config_root
from gprs.franka_interface import FrankaInterface
from gprs.utils import YamlConfig
from gprs.utils.input_utils import input2action
from gprs.utils.io_devices import SpaceMouse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="alice.yml")
    parser.add_argument("--controller-cfg", type=str, default="osc-controller.yml")
    parser.add_argument(
        "--folder", type=Path, default="data_collection_example/example_data"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

    controller_type = "JOINT_POSITION"

    # Golden resetting joints
    reset_joint_positions = [0.06813427352551457, -0.40089282268390314, 0.07713002235593189, -2.0168484201322876, -0.014262049357924194, 1.6024972743453487, 0.6649669447968954]

    dispose_joint_positions = [0.3097005563560829, -0.3212347640660315, 0.3062792126150351, -2.0068433089005318, 0.16912305284409368, 1.652241069890613, 0.8192343449475313]    
    # This is for varying initialization of joints a little bit to
    # increase data variation.
    reset_joint_positions = [
        e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
        for e in reset_joint_positions
    ]
    action = reset_joint_positions + [-1.0]

    while True:
        if len(robot_interface._state_buffer) > 0:
            print(robot_interface._state_buffer[-1].q)
            print(robot_interface._state_buffer[-1].q_d)
            print("-----------------------")

            if (
                np.max(
                    np.abs(
                        np.array(robot_interface._state_buffer[-1].q)
                        - np.array(reset_joint_positions)
                    )
                )
                < 1e-3
            ):
                break
        robot_interface.control(
            control_type=controller_type, action=action, controller_cfg=controller_cfg
        )
    robot_interface.close()


if __name__ == "__main__":
    main()
