import numpy as np
import numpy.fft as fft
import cv2
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import yaml

############################

def convolve_rgb_np(psf_pad, scene_pad, useGPU=False):
    scene_fft = fft.fft2(scene_pad,axes=(0, 1))
    psf_fft = fft.fft2(psf_pad,axes=(0, 1))
    psf_fft = np.repeat(psf_fft[:, :, np.newaxis], 3, axis=2)
    return np.abs(fft.ifftshift(fft.ifft2(scene_fft*psf_fft,axes=(0, 1)), axes=(0,1)))

def img_crop(M, full_size, sensor_size):
    top = (full_size[0] - sensor_size[0]) // 2
    bottom = (full_size[0] + sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    right = (full_size[1] + sensor_size[1]) // 2
    if len(M.shape)==3:
        return M[top:bottom,left:right,:]
    else:
        return M[top:bottom,left:right]

def img_pad_np(b, full_size):
    v_pad, vres = divmod(abs(full_size[0] - b.shape[0]), 2)
    h_pad, hres = divmod(abs(full_size[1] - b.shape[1]), 2)

    if b.ndim == 3:
        out = np.pad(b, ((v_pad, v_pad+vres), (h_pad, h_pad+hres),(0,0)), 'constant', constant_values=(0, 0))
        return out[:full_size[0], :full_size[1], :]
    else:
        out = np.pad(b, ((v_pad, v_pad+vres), (h_pad, h_pad+hres)), 'constant', constant_values=(0, 0))
        return out[:full_size[0], :full_size[1]]

def Normalize(X):
    max_X = np.max(X)
    if max_X == 0:
        max_X = 1
    X = (X/max_X * 255).astype('uint8')
    return X

def norm_8bit(X):
    max_X = X.max()
    if max_X == 0:
        max_X = 1
    return (X/max_X * 255).astype('uint8')


def img_resize(X, f):
    num = int(-np.log2(f))
    for i in range(num):
        X = 0.25*(X[::2,::2,...]+X[1::2,::2,...]+X[::2,1::2,...]+X[1::2,1::2,...])
    return X

def img_bias(x, bias=20):
    return non_neg(x-bias)

def non_neg(x):
    x = np.maximum(x,0)
    return x



if __name__ == "__main__":
    print("This codes for image processing")