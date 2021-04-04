import torch
import torch.distributions as dist
from torch import nn
import os
from vgn.ConvONets.encoder import encoder_dict
from vgn.ConvONets.conv_onet import models, training
from vgn.ConvONets.conv_onet import generation
from vgn.ConvONets import data
from vgn.ConvONets import config
from vgn.ConvONets.common import decide_total_volume_range, update_reso
from torchvision import transforms
import numpy as np


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['decoder']
    encoder = cfg['encoder']
    c_dim = cfg['c_dim']
    decoder_kwargs = cfg['decoder_kwargs']
    encoder_kwargs = cfg['encoder_kwargs']
    padding = cfg['padding']
    if padding is None:
        padding = 0.1
    
    # for pointcloud_crop
    try: 
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg.keys():
        encoder_kwargs['local_coord'] = cfg['local_coord']
        decoder_kwargs['local_coord'] = cfg['local_coord']
    if 'pos_encoding' in cfg:
        encoder_kwargs['pos_encoding'] = cfg['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['pos_encoding']

    tsdf_only = 'tsdf_only' in cfg.keys() and cfg['tsdf_only']
    detach_tsdf = 'detach_tsdf' in cfg.keys() and cfg['detach_tsdf']

    if tsdf_only:
        decoders = []
    else:
        decoder_qual = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoder_rot = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=4,
            **decoder_kwargs
        )
        decoder_width = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders = [decoder_qual, decoder_rot, decoder_width]
    if cfg['decoder_tsdf'] or tsdf_only:
        decoder_tsdf = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders.append(decoder_tsdf)

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    if tsdf_only:
        model = models.ConvolutionalOccupancyNetworkGeometry(
            decoder_tsdf, encoder, device=device
        )
    else:
        model = models.ConvolutionalOccupancyNetwork(
            decoders, encoder, device=device, detach_tsdf=detach_tsdf
        )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    
    if cfg['data']['input_type'] == 'pointcloud_crop':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2**(cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']
        
        vol_info = decide_total_volume_range(query_vol_metric, recep_field, unit_size, depth)
        
        grid_reso = cfg['data']['query_vol_size'] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg['data']['query_vol_size'] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg['generation']['sliding_window']:
            vol_bound = {'query_crop_size': query_vol_size,
                         'input_crop_size': input_vol_size,
                         'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                         'reso': grid_reso}

    else: 
        vol_bound = None
        vol_info = None

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        else:
            fields['points'] = data.PatchPointsField(
                cfg['data']['points_file'], 
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )

    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = data.PatchPointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
                )
            else:
                fields['points_iou'] = data.PointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files']
                )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
