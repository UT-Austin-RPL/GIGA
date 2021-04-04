import os
import numpy as np
import random
import torch
import skimage.transform

def set_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def apply_noise(img, noise_type):
    if noise_type == 'dex':
        return apply_dex_noise(img)
    elif noise_type =='trans':
        return apply_translational_noise(img)
    elif noise_type == 'norm':
        return apply_gaussian_noise(img)
    else:
        return img


def apply_dex_noise(img,
                gamma_shape=1000,
                gamma_scale=0.001,
                gp_sigma=0.005,
                gp_scale=4.0,
                gp_rate=0.5):
    gamma_noise = np.random.gamma(gamma_shape, gamma_scale)
    img = img * gamma_noise
    if np.random.rand() < gp_rate:
        h, w = img.shape[:2]
        gp_sample_height = int(h / gp_scale)
        gp_sample_width = int(w / gp_scale)
        gp_num_pix = gp_sample_height * gp_sample_width
        gp_noise = np.random.randn(gp_sample_height, gp_sample_width) * gp_sigma
        gp_noise = skimage.transform.resize(gp_noise,
                                            img.shape[:2],
                                            order=1,
                                            anti_aliasing=False,
                                            mode="constant")
        
        img += gp_noise
    return img

def apply_translational_noise(img,
                              sigma_p=1,
                              sigma_d=0.005):
    h, w = img.shape[:2]
    hs = np.arange(h)
    ws = np.arange(w)
    ww, hh = np.meshgrid(ws, hs)
    hh = hh + np.random.randn(*hh.shape) * sigma_p
    ww = ww + np.random.randn(*ww.shape) * sigma_p
    hh = np.clip(np.round(hh), 0, h-1).astype(int)
    ww = np.clip(np.round(ww), 0, w-1).astype(int)
    new_img = img[hh, ww]
    new_img += np.random.randn(*new_img.shape) * sigma_d
    return new_img

def apply_gaussian_noise(img, sigma=0.005):
    img += np.random.randn(*img.shape) * sigma
    return img