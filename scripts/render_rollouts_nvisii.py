import os
import glob
import pickle
from tqdm import tqdm
import argparse
import nvisii
from vgn.utils.nvisii_render import NViSIIRenderer

def load_pkl(path):
    with open(path, 'rb') as f:
        content = pickle.load(f)
    return content

opt = {
    'spp': 100,
    'height': 480,
    'width': 480,
    'camera': {
        'position': [0.15, -0.3, 0.5],
        'look_at': [0.15, 0.2, 0.1]
    },
    'light': {
        'intensity': 80,
        'scale': [1, 1, 1],
        'position': [0, -2, 3],
        'look_at': [0.15, 0.15, 0.1]
    },
    'floor': {
        'texture':
        '/mnt/data0/zhenyu/robosuite-dev/robosuite/models/assets/textures/light-wood.png',
        'scale': [2, 2, 2],
        'position': [0.15, 0.15, 0.05],
    },
}

def main(args):
    root = args.root
    rollout_path_list = glob.glob(os.path.join(root, '*.pkl'))
    rollout_path_list.sort()
    print(f'Number of frames: {len(rollout_path_list)}')
    assert len(rollout_path_list) > 0
    os.makedirs(args.save, exist_ok=True)

    NViSIIRenderer.init()
    renderer = NViSIIRenderer(opt)
    renderer.reset()

    for idx, rollout_path in tqdm(enumerate(rollout_path_list), total=len(rollout_path_list), dynamic_ncols=True):
        mesh_pose_dict = load_pkl(rollout_path)
        renderer.update_objects(mesh_pose_dict)
        renderer.render(os.path.join(args.save, f'{idx:05d}.png'))
    NViSIIRenderer.deinit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-s', '--save', type=str, required=True)
    args = parser.parse_args()
    main(args)