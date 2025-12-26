#!/usr/bin/env python3
# coding: utf-8
import torch
import torch.fft as fft
import torch.nn.functional as f
import numpy as np
# import numpy.fft as fft
import cv2 as cv
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import yaml

## PyTorch 1.8.0 version

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def convolve_rgb(psf_pad, scene_pad):
    # print("psf_pad.shape: ", psf_pad.shape)
    # print("scene_pad.shape: ", scene_pad.shape)
    scene_fft = fft.fftn(scene_pad, dim=(0,1))
    psf_fft = fft.fftn(psf_pad, dim=(0,1))
    if psf_fft.ndim == 2:
        psf_fft = torch.repeat_interleave(psf_fft.unsqueeze(-1), 3, 2)
    else:
        pass
    # psf_fft = torch.repeat_interleave(psf_fft.unsqueeze(-1), 3, 2)
    print("scene_fft.shape: ", scene_fft.shape)
    print("psf_fft.shape: ", psf_fft.shape)
    return (torch.abs(fft.ifftshift(fft.ifftn(scene_fft*psf_fft, dim=(0, 1)), dim=(0, 1))))

# def convolve_rgb_Raw_Gen(psf_pad, scene_pad, batch_size):
#     # print("psf_pad.shape: ", psf_pad.shape)
#     # print("scene_pad.shape: ", scene_pad.shape)
#     scene_fft = fft.fftn(scene_pad, dim=(-2,-1))
#     psf_fft = fft.fftn(psf_pad, dim=(-2,-1))
#     # psf_fft = torch.repeat_interleave(psf_fft, batch_size, dim=0) #
#     psf_fft = torch.repeat_interleave(psf_fft, 1, dim=0) #
#     psf_fft = torch.repeat_interleave(psf_fft, 3, dim=1)
#     # print("scene_fft.shape: ", scene_fft.shape)
#     # print("psf_fft.shape: ", psf_fft.shape)
#     return torch.abs(fft.ifftshift(fft.ifftn(scene_fft*psf_fft, dim=(-2, -1)), dim=(-2, -1)))

# --- Convolution Function (FFT based) ---
def convolve_rgb_Raw_Gen(psf_padded, scene_padded):
    # psf_padded shape: (1, 1, fft_h, fft_w)
    # scene_padded shape: (batch, 3, fft_h, fft_w)
    scene_fft = torch.fft.fftn(scene_padded, dim=(-2,-1))
    psf_fft = torch.fft.fftn(psf_padded, dim=(-2,-1))
    psf_fft_matched = psf_fft.repeat(scene_fft.shape[0], scene_fft.shape[1], 1, 1)
    result_fft = scene_fft * psf_fft_matched
    result_spatial_shifted = torch.fft.ifftn(result_fft, dim=(-2, -1))
    result_spatial = torch.fft.ifftshift(result_spatial_shifted, dim=(-2,-1))
    return torch.abs(result_spatial) # Return magnitude

def convolve_rgb_Training(psf_pad, img_batch, batch_size):
    img_fft = fft.fftn(img_batch, dim=(2, 3))
    psf_fft = fft.fftn(psf_pad, dim=(2, 3))
    return (torch.abs(fft.ifftshift(fft.ifftn(img_fft*psf_fft, dim=(2, 3)), dim=(2, 3))))

def img_crop(M, full_size, sensor_size):
    top = (full_size[0] - sensor_size[0]) // 2
    bottom = (full_size[0] + sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    right = (full_size[1] + sensor_size[1]) // 2
    if len(M.shape)==3:
        return M[top:bottom,left:right,:]
    else:
        return M[top:bottom,left:right]
    
def img_crop_Training(M, full_size, sensor_size):
    top = (full_size[0] - sensor_size[0]) // 2
    bottom = (full_size[0] + sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    right = (full_size[1] + sensor_size[1]) // 2
    return M[:, :, top:bottom, left:right]

def img_pad(b, full_size):
    v_pad = int(np.ceil(abs(full_size[0] - b.shape[0]) // 2))
    h_pad = int(np.ceil(abs(full_size[1] - b.shape[1]) // 2))
    if b.ndim == 3:
        out = f.pad(b, (0, 0, h_pad, h_pad, v_pad, v_pad))
        return out[:full_size[0], :full_size[1], :]
    else:
        out = f.pad(b, (h_pad, h_pad, v_pad, v_pad))
        return out[:full_size[0], :full_size[1]]

# def Normalize(X):
#     max_X = torch.max(X)
#     if max_X == 0:
#         max_X = torch.ones_like(max_X).to(device)
#     X = (X / max_X * 255).to(torch.uint8)
#     return X

def norm_8bit_tensor(X):
    max_X = torch.max(X)
    if max_X == 0:
        max_X = torch.ones_like(max_X).to(device)
    X = (X / max_X * 255)
    return X.int()

def norm_8bit(X):
    max_X = X.max()
    if max_X == 0:
        max_X = 1
    A = X / max_X * 255
    return A.astype('uint8')

def img_resize(X, f):
    num = int(-np.log2(f))
    for i in range(num):
        X = 0.25*(X[::2,::2,...]+X[1::2,::2,...]+X[::2,1::2,...]+X[1::2,1::2,...])
    return X

def img_bias(x, bias=0):
    return non_neg(x-bias)

def non_neg(x):
    x = torch.maximum(x,torch.tensor((0)))
    return x

if __name__ == "__main__":
    print("This codes for image processing")

