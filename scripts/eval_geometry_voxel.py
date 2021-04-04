import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np

import torch
import tqdm
from torch.utils.data.dataloader import default_collate

from vgn.dataset_voxel_occ import DatasetVoxelOccGeo, DatasetVoxelOccGeoROI
from vgn.networks import load_network

from vgn.ConvONets.conv_onet.generation import Generator3D
from vgn.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list
from vgn.ConvONets.eval import MeshEvaluator
from vgn.ConvONets.utils.libmesh import check_mesh_contains
from vgn.dataset_pc import sample_point_cloud
from vgn.utils.misc import set_random_seed
from vgn.ConvONets.common import compute_iou

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    # create log directory
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "{}_eval_geo_dataset={},net={},th={},{}".format(
        time_stamp,
        args.dataset.name,
        args.type,
        args.th,
        args.description,
    ).strip(",")
    logdir = args.logdir / description
    logdir.mkdir()

    # create data loaders
    test_set, test_loader = create_test_loader(
        args.dataset, args.dataset_raw, kwargs, args.ROI
    )

    mean_dict = {
        'model_path': os.path.abspath(args.model_path),
        'iou': [], 
        'chamfer-L1': [], 
        'normals accuracy': [], 
        'f-score': [],
        }
    if args.ROI:
        for name in ['iou', 'precision', 'recall']:
            mean_dict[name + '_ROI'] = []
            mean_dict[name + '_ROI_infer'] = []


    # build the network
    net = load_network(args.model_path, device, model_type=args.type).eval()
    generator = Generator3D(
        net,
        device=device,
        threshold=args.th,
        input_type='pointcloud',
        padding=0,
    )

    with torch.no_grad():
        for idx, (data, gt_mesh) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader),  dynamic_ncols=True):
            if args.ROI:
                pc_in, points_occ, occ, occ_points_ROI, occ_ROI = data
                occ_ROI = occ_ROI.squeeze().numpy()
            else:
                pc_in, points_occ, occ = data
            pc_in = pc_in.float().to(device)
            points_occ = points_occ.float().to(device)
            occ = occ.float().to(device)
            gt_mesh = gt_mesh[0]
            gt_mesh.vertices = gt_mesh.vertices / test_set.size - 0.5
            pred_mesh = predict_mesh(generator, pc_in)
            out_dict = eval_mesh(pred_mesh, gt_mesh, points_occ, occ)
            if args.ROI and not 'empty' in out_dict.keys():
                ROI_dict = eval_points(pred_mesh, occ_points_ROI.squeeze().numpy(), occ_ROI)
                ROI_infer_dict = eval_points_infer(net, pc_in, occ_points_ROI.to(device), occ_ROI, args.th)

                out_dict.update(ROI_dict)
                out_dict.update(ROI_infer_dict)

            save_dir = logdir / ('%05d' % idx)
            save_dir.mkdir()
            if not 'empty' in out_dict.keys():
                for k, v in mean_dict.items():
                    if isinstance(v, list):
                        if out_dict[k] >= -1e5: # avoid nan
                            mean_dict[k].append(out_dict[k])
                gt_mesh.export(save_dir / 'gt_mesh.glb')
                pred_mesh.export(save_dir / 'pred_mesh.glb')
            else:
                print(f'{idx} empty mesh!')
            with open(save_dir / 'results.json', 'w') as f:
                json.dump({k: float(v) for k, v in out_dict.items()}, f, indent=4)

    print('Geometry prediction results:')
    for k, v in mean_dict.items():
        if isinstance(v, list):
            print('%s: %.6f' % (k, np.mean(v)))
            mean_dict[k] = float(np.mean(v))
    with open(logdir / 'mean_results.json', 'w') as f:
        json.dump({k: v for k, v in mean_dict.items()}, f, indent=4)


def create_test_loader(root, root_raw, kwargs, ROI, num_point_occ=100000):
    # load the dataset
    def collate_fn(batch):
        # remove audio from the batch
        meshes = [d[-1] for d in batch]
        if ROI:
            batch = [(d[0], d[1], d[2], d[3], d[4]) for d in batch]
        else:
            batch = [(d[0], d[1], d[2]) for d in batch]
        return default_collate(batch), meshes

    if ROI:
        dataset = DatasetVoxelOccGeoROI(root, root_raw, num_point_occ=num_point_occ)
    else:
        dataset = DatasetVoxelOccGeo(root, root_raw, num_point_occ=num_point_occ)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn, **kwargs
    )
    # it = iter(test_loader)
    # import pdb; pdb.set_trace()
    return dataset, test_loader

def predict_mesh(generator, pc_input):
    pred_mesh, _ = generator.generate_mesh({'inputs': pc_input})
    return pred_mesh

def eval_mesh(pred_mesh, gt_mesh, points_occ, occ):
    evaluator = MeshEvaluator()
    pointcloud_tgt, idx_tgt = gt_mesh.sample(evaluator.n_points, True)
    normals_tgt = gt_mesh.face_normals[idx_tgt]
    points_occ = points_occ[0].cpu().numpy()
    occ = occ[0].cpu().numpy()
    out = evaluator.eval_mesh(pred_mesh, pointcloud_tgt.astype(np.float32), normals_tgt, points_occ, occ)
    return out

def eval_points(pred_mesh, points_occ, occ):
    evaluator = MeshEvaluator()
    result = evaluator.eval_occ(pred_mesh, points_occ, occ, ext='_ROI')
    return result

def eval_points_infer(model, pc_input, points_occ, occ, th):
    with torch.no_grad():
        output = model.infer_geo(pc_input, points_occ)
        occ_pred = torch.sigmoid(output) > th
        occ_pred = occ_pred.squeeze().cpu().numpy()
    
    result = {}
    result['iou_ROI_infer'] = compute_iou(occ_pred, occ)
    result['precision_ROI_infer'] = 1.0 * np.sum(np.logical_and(occ_pred, occ)) / np.sum(occ_pred)
    result['recall_ROI_infer'] = 1.0 * np.sum(np.logical_and(occ_pred, occ)) / np.sum(occ)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--type", type=str)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset_raw", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/eval_geo")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--ROI", action='store_true', help='Use ROI occ point sampling')
    parser.add_argument("--th", type=float, default=0.5, help="level set threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)
    set_random_seed(args.seed)
    main(args)