# Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations

[Zhenyu Jiang](http://zhenyujiang.me), [Yifeng Zhu](https://zhuyifengzju.github.io/), [Maxwell Svetlik](https://maxsvetlik.github.io/), [Kuan Fang](https://ai.stanford.edu/~kuanfang/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)

RSS (Robotics: Science and Systems) 2021

[Project](https://sites.google.com/view/rpl-giga2021) | [arxiv](http://arxiv.org/abs/2104.01542) 

## Introduction

GIGA (Grasp detection via Implicit Geometry and Affordance) is a network that jointly detects 6 DOF grasp poses and reconstruct the 3D scene. GIGA takes advantage of deep implicit functions, a continuous and memory-efficient representation, to enable differentiable training of both tasks. GIGA takes as input a Truncated Signed Distance Function (TSDF) representation of the scene, and predicts local implicit functions for grasp affordance and 3D occupancy. By querying the affordance implict functions with grasp center candidates, we can get grasp quality, grasp orientation and gripper width at these centers. GIGA is trained on a synthetic grasping dataset generated with physics simulation.

If you find our work useful in your research, please consider [citing](#citing).

## Installation

1. Create a conda environment.

2. Install packages list in [requirements.txt](requirements.txt). Then install `torch-scatter` following [here](https://github.com/rusty1s/pytorch_scatter), based on `pytorch` version and `cuda` version.

3. Go to the root directory and install the project locally using `pip`

```
pip install -e .
```

4. Build ConvONets dependents by running `python scripts/convonet_setup.py build_ext --inplace`.

5. Download the [data](https://utexas.box.com/s/h3ferwjhuzy6ja8bzcm3nu9xq1wkn94s), then unzip and place the data folder under the repo's root. Pretrained models of GIGA, GIGA-Aff and VGN are in `data/models`.

## Self-supervised Data Generation

### Raw synthetic grasping trials

Pile scenario:

```bash
python scripts/generate_data_parallel.py --scene pile --object-set pile/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/pile/data_pile_train_random_raw_4M
```

Packed scenario:
```bash
python scripts/generate_data_parallel.py --scene packed --object-set packed/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/pile/data_packed_train_random_raw_4M
```

Please run `python scripts/generate_data_parallel.py -h` to print all options.

### Data clean and processing

First clean and balance the data using:

```bash
python scripts/clean_balance_data.py /path/to/raw/data
```

Then construct the dataset (add noise):

```bash
python scripts/construct_dataset_parallel.py --num-proc 40 --single-view --add-noise dex /path/to/raw/data /path/to/new/data
```

### Save occupancy data

Sampling occupancy data on the fly can be very slow and block the training, so I sample and store the occupancy data in files beforehand:

```bash
python scripts/save_occ_data_parallel.py /path/to/raw/data 100000 2 --num-proc 40
```

Please run `python scripts/save_occ_data_parallel.py -h` to print all options.


## Training

### Train GIGA

Run:

```bash
# GIGA
python scripts/train_giga.py --dataset /path/to/new/data --dataset_raw /path/to/raw/data
```

## Simulated grasping

Run:

```bash
python scripts/sim_grasp_multiple.py --num-view 1 --object-set (packed/test | pile/test) --scene （packed ｜ pile) --num-rounds 100 --sideview --add-noise dex --force --best --model /path/to/model --type (vgn | giga | giga_aff) --result-path /path/to/result
```

This commands will run experiment with each seed specified in the arguments.

Run `python scripts/sim_grasp_multiple.py -h` to print a complete list of optional arguments.

## Pre-trained models and pre-generated data

### Pre-trained models

Pretrained models are also in the [data.zip](https://utexas.box.com/s/h3ferwjhuzy6ja8bzcm3nu9xq1wkn94s). They are in `data/models`.

### Pre-generated data

As mentioned in the [issue](https://github.com/UT-Austin-RPL/GIGA/issues/3), data generation is very costly. So we upload the generated data. Because the occupancy data takes too much space (over 100G), we do not upload the occupancy data, you can generate them following the instruction in this [section](#save-occupancy-data). This generation won't take too long time.

| Scenario | Raw data | Processed data |
| ----------- | ----------- | ----------- |
| Pile | [link](https://utexas.box.com/s/w1abs6xfe8d2fo0h9k4bxsdgtnvuwprj) | [link](https://utexas.box.com/s/l3zpzlc1p6mtnu7ashiedasl2m3xrtg2) |
| Packed | [link](https://utexas.box.com/s/roaozwxiikr27rgeauxs3gsgpwry7gk7) | [link](https://utexas.box.com/s/h48jfsqq85gt9u5lvb82s5ft6k2hqdcn) |

## Related Repositories

1. Our code is largely based on [VGN](https://github.com/ethz-asl/vgn) 

2. We use [ConvONets](https://github.com/autonomousvision/convolutional_occupancy_networks) as our backbone.

## Citing

```
@article{jiang2021synergies,
 author = {Jiang, Zhenyu and Zhu, Yifeng and Svetlik, Maxwell and Fang, Kuan and Zhu, Yuke},
 journal = {Robotics: science and systems},
 title = {Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations},
 year = {2021}
}
```
