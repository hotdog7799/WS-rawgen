import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import seaborn as sns
import numpy as np
import random
import copy
from glob import glob
from utils_Torch import *
from utils import *
from scipy import io
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, InterpolationMode

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# import imageio.v2 as imageio
from scipy.io import loadmat, savemat
from tqdm import tqdm
import cv2
import argparse  # 1. argparse 라이브러리 임포트
import matplotlib.pyplot as plt

# 2. (신규) Argument 파서 설정
parser = argparse.ArgumentParser(description="Raw image generator for lensless camera")
parser.add_argument('--psf_path', type=str, required=True, help='Path to the .mat PSF stack file')
parser.add_argument('--data_path', type=str, required=False, help='Path to the root of mini_dataset (containing train/image and train/label)')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the generated raw data')
parser.add_argument('--psf_idx_list', nargs='+', type=int, required=True, help='List of PSF indices to use for quantization')
parser.add_argument('--wd_list', nargs='+', type=float, required=True, help='List of working distances corresponding to psf_idx_list')
parser.add_argument('--data_idx', nargs='+', type=int, required=False, help='Start and end index of data to process (e.g., 0 500)')

# 3. (신규) 스크립트가 실행될 때 터미널로부터 인수를 읽어옴
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(DEVICE))

folder_name = '20251016_500_set_HJA_01'

HPARAMS = {
    'BATCH_SIZE': 1,
    'NUM_WORKERS': 1,
    # 'TRAINSET_SIZE' : 60000,
    # 'FOLDER_PATH': '/mnt/ssd1/depth_imaging/dataset_ssd1/current_blender_dataset/'+folder_name+'/', # ssd1
    'FOLDER_PATH': '/home/hotdog/files_251026/'+folder_name+'/',
    # 'IMAGE_PATH': '/home/hotdog/mini_dataset/'+folder_name+'/image/',
    # 'LABEL_PATH': '/home/hotdog/mini_dataset/'+folder_name+'/label/',
    'IMAGE_PATH': '/home/hotdog/ssd1/depth_imaging/dataset_ssd2/20251009_222218_10000_5/image/',
    'LABEL_PATH': '/home/hotdog/ssd1/depth_imaging/dataset_ssd2/20251009_222218_10000_5/label/',
    # 'FOLDER_PATH': '/mnt/ssd2/dataset_ssd2/'+folder_name+'/', # ssd2
    # 'IMAGE_PATH': '/mnt/ssd2/dataset_ssd2/'+folder_name+'/image/',
    # 'LABEL_PATH': '/mnt/ssd2/dataset_ssd2/'+folder_name+'/label/',
    'BG_PATH': '/mnt/ssd1/depth_imaging/dataset_ssd1/mirflickr25k/'
}


TPARAMS = {
    'crop': 150,}

START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # Gaussian noise
mean = 0
std_dev_min = 0
std_dev_max = 0.01

bg_train = sorted(glob(HPARAMS['BG_PATH']+'/*.jpg', recursive=True))
print('Number of background images: {}'.format(len(bg_train)))

background_level_max = 0.5

####### Save directories
suffix = '_bg_level_max'+str(background_level_max)

save_dir_raw_noisy_png = HPARAMS['FOLDER_PATH']+'aiobio_poission_disk_avgdist_140um_[real_210um]_with_simulated_2xPSF_in_250429_and_250408_modified_angular_response_250519_masked_rgb_png_raw_AWGN_'+ 'mean_'+ str(mean) + '_rand_std_'+ str(std_dev_min) + '_to_'+ str(std_dev_max) +suffix
if not os.path.isdir(save_dir_raw_noisy_png): # folder to save
    os.mkdir(save_dir_raw_noisy_png)

save_dir_raw_noisy_png = save_dir_raw_noisy_png+'/0'
if not os.path.isdir(save_dir_raw_noisy_png): # folder to save
    os.mkdir(save_dir_raw_noisy_png)

save_dir_image_png = HPARAMS['FOLDER_PATH']+'aiobio_image_gss_masked_png' +suffix
if not os.path.isdir(save_dir_image_png): # folder to save
    os.mkdir(save_dir_image_png)
    
save_dir_image_png = save_dir_image_png+'/0'
if not os.path.isdir(save_dir_image_png): # folder to save
    os.mkdir(save_dir_image_png)

save_dir_label_npz = HPARAMS['FOLDER_PATH']+'aiobio_label_npz' +suffix
if not os.path.isdir(save_dir_label_npz): # folder to save
    os.mkdir(save_dir_label_npz)

save_dir_label_npz = save_dir_label_npz+'/0'
if not os.path.isdir(save_dir_label_npz): # folder to save
    os.mkdir(save_dir_label_npz)


## 2025-07-09, dataset
# psf_dir = '/mnt/ssd1/depth_imaging/dataset_ssd1/psf_stack/20250709_aiobio/20250709_recentered_stack_bg_removed_centered_max_norm_layer_by_layer_resized_352x512x31.mat'
psf_dir ='/home/hotdog/files_251026/image_stack_cropped_2nd_251024.mat'

####################################################################################################
############################## Simulated_PSFs ####################################################
# psf_dir = '/mnt/ssd1/depth_imaging/phase_mask_design_file/241015_162843_PoissonDiskRandomDots/simulated_psf_stackfocalDist_0.006_upscale_2.35_crop_1/simulated_psf_resized_352x512x28.mat'

psf_mat_file = io.loadmat(psf_dir)
psf_stack = psf_mat_file['images']
# psf_stack = psf_mat_file['psf_stack_2x']
# psf_stack = psf_mat_file['psf_stack_2x_resized']
# psf_stack = psf_mat_file['psf_stack_2x_resized_rotated']
# psf_stack = psf_mat_file['psf_resized'] # arbitrary resized PSF (simulated PSF)
pad_size = tuple(2*x for x in psf_stack.shape)
print('='*30)
print('pad_size: ',pad_size)
print('Before padding - psf_stack shape: ',psf_stack.shape)
psf_stack = img_pad_np(psf_stack, pad_size) # no padding for 2x PSF, aiobio is needed
print('After padding - psf_stack shape: ',psf_stack)
print('='*30)

psf_stack = psf_stack.astype(np.float32)

psf_stack /= np.linalg.norm(psf_stack)
# print("Shape of PSF stack: {}".format(psf_stack.shape))
psf_stack = torch.from_numpy(psf_stack).permute(2,0,1).repeat(1,1,1,1,1).type(torch.FloatTensor).float().to(DEVICE) #BCDHW
psf_stack = psf_stack.clone().detach().to(dtype=torch.float64)

# psf_stack = psf_stack[:,:,5:,:,:] # 10 cm ~ 60 cm.. just use 9 channel
# psf_stack = img_pad(psf_stack,pad_size)
print('PSF stack shape: {}'.format(psf_stack.shape))
print("Max and Min of PSF stack: {} and {}".format(torch.max(psf_stack), torch.min(psf_stack)))
print("Data type of psf_stack: {}".format(psf_stack.dtype))

# gaussian_dir = '/mnt/ssd1/depth_imaging/dataset/20240221_2D_angular_response_IMX477_resized_704x1024_2XFOV.mat'

# gaussian_dir = '/mnt/ssd1/depth_imaging/dataset_ssd1/data_angular_response/20250408_2D_angular_response_IMX477_resized_704x1024_2XFOV.mat' # 20250408

# gaussian_mat_file = io.loadmat(gaussian_dir)
# gaussian_2D_filter = gaussian_mat_file['angular_2d_resized']
# gaussian_stack = torch.from_numpy(gaussian_2D_filter).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(DEVICE)
# gaussian_stack = gaussian_stack.clone().detach().to(dtype=torch.float32)
# gaussian_stack = gaussian_stack.repeat(1,1,1,1).unsqueeze(2)
# Make float32
# print("Shape of gaussian_stack: {}".format(gaussian_stack.shape))
# print("Max and Min of gaussian_stack: {} and {}".format(torch.max(gaussian_stack), torch.min(gaussian_stack)))
# print("Data type of gaussian_stack: {}".format(gaussian_stack.dtype))

quantize_num = np.shape(psf_stack)[2]
# print('Quantize number: {}'.format(quantize_num))
# range_log = torch.exp(torch.linspace(0, 1, steps=quantize_num + 1)) ** 5
# range_log = ((range_log - range_log.min()) / (range_log.max() - range_log.min()))

# wd_measured = np.concatenate([np.arange(9.5, 21), np.arange(23, 32,2), np.arange(37.5, 65, 5)]) # 20240420
# wd_measured = np.concatenate([[5], np.arange(5.5, 21), np.arange(23, 32,2), np.arange(37.5, 65, 5)]) # 20240602, add 5cm PSF

# wd_blender = np.concatenate(([5], np.arange(10, 71, 10))) #20240602 add [5]
wd_blender =  np.arange(5.0, 15.0, 0.1) # aiobio
# wd_measured = np.array([5,6,7,8,9,10,12,14,17,22,28,40,60]) # concentric PSF 13 channel in 20240618
# wd_measured = np.array([5,6,7,8,9,10,11,12,14,16,19,24,35,60]) # concentric PSF 14 channel in 20240620
# wd_measured = np.array([5,6,7,8,9,10,11,13, 15,17,20,24,30,40,60]) # RPC diffuser 15 channel in 20240720, resampled
# wd_measured = np.array([5,6,7,8,9,10,11,13,15,17,20,24,30,60]) # RPC diffuser 14 channel in 20240722, resampled without 40
# wd_measured = np.concatenate((np.arange(5, 21, 1), np.arange(22, 31, 2), np.arange(35, 66, 5)))  # concentric PSF 28 channel in 20240620

# wd_measured = np.concatenate((np.arange(7, 11, 0.5), np.arange(11, 17, 1), [17.5, 19, 21, 23, 26, 30, 36, 70]))  # RPC diffuser 22 channel in 20241130, resampled

# wd_measured = np.concatenate((
#     np.arange(5.0, 16.5, 0.5),  # 5.0 to 16.0 with step 0.5
#     np.arange(17.0, 24.0, 1.0), # 17.0 to 23.0 with step 1.0
#     np.arange(25.0, 34.0, 2.0), # 25.0 to 33.0 with step 2.0
#     [36.0, 38.0, 45.0, 50.0, 55.0, 60.0, 70.0] # Additional values
# )) # avg dist 140 um poission random dot 20250103
wd_measured = np.arange(5.0, 15.0, 0.1)  # 5.0 to 15.0 with step 0.1, aiobio

# (xp) 맵의 X축: argument로 받은 11개의 대표 거리
# args.wd_list는 문자열 리스트이므로 float 배열로 변환합니다.
wd_map_x = np.array(args.wd_list, dtype=np.float32) # (len 11)

# (fp) 맵의 Y축: 11개의 대표 거리에 1:1로 매핑될 11개의 정규화된 깊이 (0.0 ~ 1.0)
wd_map_y = np.linspace(0.0, 1.0, len(wd_map_x)) # (len 11)

# 100개의 전체 거리(wd_measured)를 11개 맵(wd_map_x, wd_map_y)을 기준으로 보간하여
# 최종 100개짜리 조회 테이블(range_log)을 생성합니다.
print("--- Interpolation Shapes ---")
print("Query points (x) shape: ", wd_measured.shape)      # (100,)
print("Map X-axis (xp) shape: ", wd_map_x.shape)         # (11,)
print("Map Y-axis (fp) shape: ", wd_map_y.shape)         # (11,)

range_log = np.interp(wd_measured, wd_map_x, wd_map_y)

print("Generated lookup table shape: ", range_log.shape) # (100,)
print("----------------------------")



# 5, 10, 20, 30, 40, 50, 60, 70 in blender environment
# depth_value = [0, 0.0769231, 0.230769, 0.384615, 0.537981, 0.691827, 0.846154, 1] # 20240420, inverted
# depth_value = np.linspace(0.0, 1.0, 31)

# Interpolate depth values for wd_measured using wd_blender and depth_value
# print("wd_measured.shape: ",wd_measured.shape)
# print("wd_blender.shape: ",wd_blender.shape)
# print("depth_value.shape: ",depth_value.shape)
# range_log = np.interp(wd_measured, wd_blender, depth_value)
print("Length of range_log: ", len(range_log))
range_log = torch.from_numpy(range_log).float().to(DEVICE)


print("WD measured [cm]:")
print(wd_measured)
print("Length of WD measured: ", len(wd_measured))

print("Interpolated Depth Values:")
print(range_log)
print("Length of Interpolated Depth Values: ", len(range_log))


# def quantize_rgb_by_depth(quantize_num, sample_img, sample_bg_img, sample_depth, max_val=1):
#     mask_stack = torch.zeros_like(sample_depth).to(DEVICE)
#     mask_stack.unsqueeze_(2)
#     # mask_stack = mask_stack.repeat(1, 1, quantize_num, 1, 1)
#     sample_img = sample_img.unsqueeze(2)
#     # print("Max and Min of sample_img: {} and {}".format(torch.max(sample_img), torch.min(sample_img)))
#     # if torch.min(sample_img) > 0:
#     #     sample_img[sample_img<=torch.min(sample_img)] = 0
#     sample_img = torch.multiply(sample_img, gaussian_stack) # multiply gaussian filter
#     sample_img_bwmask = torch.where(sample_depth > 0, 1, 0).unsqueeze(2) # binary mask

#     # sample_img_bwmask = sample_img_bwmask.repeat(1, 3, 1, 1, 1) # repeat to match the channel
#     # print("Shape of sample_img: ", sample_img.shape)
#     # print("Shape of sample_img_bwmask: ", sample_img_bwmask.shape)
    
#     # sample_img = torch.multiply(sample_img, sample_img_bwmask) # (2025-05-19) multiply binary mask to remove unwanted boundary of the object
#     sample_img = sample_img * sample_img_bwmask
    
#     sample_depth = 1.0 - sample_depth # invert to calculate depth
    
#     for i in range(quantize_num):
#         if i == 0:
#             mask_stack[:,:,i] = torch.where(sample_depth <= range_log[i + 1], 1, 0) # modified
#         elif i == quantize_num - 1:
#             mask_stack[:,:,i] = torch.where((sample_depth > range_log[i]) & (sample_depth <= max_val), 1, 0)
#         else:
#             mask_stack[:,:,i] = torch.where((sample_depth > range_log[i]) & (sample_depth <= range_log[i + 1]), 1, 0)
#     mask_3channel = mask_stack.repeat(1, 3, 1, 1, 1)
#     scene_stack = torch.multiply(sample_img, mask_3channel)
    
#     scene_gray = torch.mean(sample_img, dim=1, keepdim=False) # 8,1,352,512
#     scene_gray_negative = torch.abs(1-scene_gray)
#     scene_gray_negative[scene_gray_negative<1] = 0
#     scene_gray_negative = torch.repeat_interleave(scene_gray_negative,3, dim=1)
#     sample_bg_masked = torch.multiply(sample_bg_img, scene_gray_negative).to(DEVICE)
#     sample_bg_masked[sample_bg_masked<0] = 0
#     scene_stack[:,:,-1,:,:] += sample_bg_masked # last slice is background
#     return scene_stack, sample_img.squeeze(2)
wd_map_y_tensor = torch.from_numpy(wd_map_y).float().to(DEVICE) # (11,)

# def quantize_rgb_by_depth(quantize_num, sample_img, sample_bg_img, sample_depth, max_val=1):
#     # Get batch size, height, width from sample_depth
#     batch_size = sample_depth.shape[0]
#     height = sample_depth.shape[2]
#     width = sample_depth.shape[3]

#     # 1. mask_stack을 (batch, 1, quantize_num, H, W) 크기로 초기화
#     mask_stack = torch.zeros(batch_size, 1, quantize_num, height, width, device=DEVICE, dtype=torch.float32)

#     # sample_img 준비 (이전과 동일)
#     sample_img = sample_img.unsqueeze(2) # (batch, 3, 1, H, W)
#     sample_img = torch.multiply(sample_img, gaussian_stack) # multiply gaussian filter
#     sample_img_bwmask = torch.where(sample_depth > 0, 1, 0).unsqueeze(2).float() # binary mask, ensure float
#     sample_img = sample_img * sample_img_bwmask

#     # sample_depth 준비 (이전과 동일, invert)
#     # sample_depth는 (batch, 1, H, W) 형태여야 함
#     sample_depth = 1.0 - sample_depth

#     # --- 2. 양자화 루프 수정 ---
#     for i in range(quantize_num): # 0부터 10까지 (11번)
#         # wd_map_y_tensor 사용 (0.0, 0.1, ..., 1.0)
#         lower_bound = wd_map_y_tensor[i]

#         if i == quantize_num - 1: # 마지막 구간 (i=10)
#             # lower_bound (1.0) 보다 크거나 같은 모든 값 (max_val 고려 불필요)
#             # Note: sample_depth는 1.0 - original_depth 이므로, 원본 depth가 min일 때 1.0이 됨
#             # 따라서 마지막 구간은 wd_map_y[10] (1.0) 보다 크거나 같은 값이 아니라,
#             # wd_map_y[9] (0.9) 보다 크고 wd_map_y[10] (1.0) 보다 작거나 같은 값이어야 함.
#             # 코드를 더 명확하게 수정:
#             if i == 0: # 첫 번째 구간 (가장 가까운 거리, sample_depth 값이 가장 큼)
#                  upper_bound = wd_map_y_tensor[i+1] # 0.1
#                  mask = (sample_depth >= lower_bound) & (sample_depth < upper_bound) # 예: 1.0 >= depth > 0.9 (틀림!)
#                  # -> sample_depth는 1-depth 이므로, 0.0 ~ 1.0 사이 값.
#                  # wd_map_y도 0.0(가까움) ~ 1.0(멈)으로 가정해야 함.
#                  # 첫 구간(i=0): 0.0 <= sample_depth < 0.1
#                  # 마지막 구간(i=10): 0.9 <= sample_depth <= 1.0
#                  lower_threshold = wd_map_y_tensor[i] # 0.0
#                  upper_threshold = wd_map_y_tensor[i+1] # 0.1
#                  mask = (sample_depth >= lower_threshold) & (sample_depth < upper_threshold)

#             elif i == quantize_num - 1: # 마지막 구간 (가장 먼 거리, i=10)
#                  lower_threshold = wd_map_y_tensor[i] # 1.0 (틀림!) -> wd_map_y_tensor[i]는 1.0
#                  # 마지막 구간 경계는 wd_map_y[10] = 1.0
#                  lower_threshold = wd_map_y_tensor[i] # 1.0
#                  mask = (sample_depth >= lower_threshold) # depth >= 1.0 인 모든 값 -> 1.0만 해당

#                  # 재수정: linspace(0, 1, 11) -> [0.0, 0.1, ..., 0.9, 1.0]
#                  # i = 0:  0.0 <= depth < 0.1
#                  # i = 1:  0.1 <= depth < 0.2
#                  # ...
#                  # i = 9:  0.9 <= depth < 1.0
#                  # i = 10: depth == 1.0 (경계 문제 발생 가능) -> depth >= 1.0 으로 처리하는게 안전

#                  lower_threshold = wd_map_y_tensor[i] # i=10 이면 1.0
#                  mask = (sample_depth >= lower_threshold) # depth >= 1.0 (사실상 depth == 1.0)

#                  # 더 안전한 방법: 마지막 구간은 이전 경계보다 크거나 같은 모든 값
#                  lower_threshold = wd_map_y_tensor[i] # i = 10 -> 1.0 이므로, i = 9 사용
#                  lower_threshold = wd_map_y_tensor[i-1] # i = 10 -> 0.9
#                  mask = (sample_depth >= lower_threshold) # depth >= 0.9 인 모든 값

#                  # 최종 수정안: 경계를 명확히 하자
#                  if i == 0:
#                      # 0.0 <= depth < (0.0 + 0.1)/2 = 0.05
#                      # mid_point = (wd_map_y_tensor[i] + wd_map_y_tensor[i+1]) / 2.0
#                      # mask = sample_depth < mid_point
#                      # 더 쉬운 방법: 첫번째 경계값보다 작거나 같은 값
#                      mask = sample_depth <= wd_map_y_tensor[i] # depth <= 0.0 (사실상 depth == 0.0) -> 가장 가까운 값

#                      # Baek 논문 방식 추정: depth 값을 가장 가까운 대표값 인덱스로 변환 후 one-hot 인코딩
#                      # 이 방식이 더 직관적일 수 있음. 현재 코드는 경계 기반 할당.

#                      # 경계 기반 할당 유지 시 수정:
#                      threshold_upper = wd_map_y_tensor[i+1] # 0.1
#                      mask = sample_depth < threshold_upper # depth < 0.1

#                  elif i == quantize_num - 1: # 마지막 구간 (i=10)
#                      threshold_lower = wd_map_y_tensor[i] # 1.0
#                      mask = sample_depth >= threshold_lower # depth >= 1.0

#                  else: # 중간 구간 (i=1 ~ 9)
#                      threshold_lower = wd_map_y_tensor[i] # e.g., i=1 -> 0.1
#                      threshold_upper = wd_map_y_tensor[i+1] # e.g., i=1 -> 0.2
#                      mask = (sample_depth >= threshold_lower) & (sample_depth < threshold_upper) # e.g., 0.1 <= depth < 0.2

#             # 생성된 마스크(True/False)를 float32로 변환하여 저장
#             mask_stack[:, :, i, :, :] = mask.float() # mask는 (batch, 1, H, W) 이므로 인덱싱 주의

#     # 3. 마스크를 3채널로 복제 (이전과 동일)
#     mask_3channel = mask_stack.repeat(1, 3, 1, 1, 1) # shape: (batch, 3, 11, H, W)

#     # 4. scene_stack 계산 (sample_img 브로드캐스팅 활용)
#     # sample_img: (batch, 3, 1, H, W)
#     # mask_3channel: (batch, 3, 11, H, W)
#     scene_stack = sample_img * mask_3channel # shape: (batch, 3, 11, H, W) -> Correct!

#     # 5. 배경 처리 (이전과 동일)
#     scene_gray = torch.mean(sample_img.squeeze(2), dim=1, keepdim=True) # (batch, 1, H, W)
#     # Binary mask for background based on where the object *isn't*
#     # (Use the original depth before inversion, or use the bwmask directly)
#     object_mask = torch.where(sample_depth_original > 0, 1.0, 0.0) # Assume sample_depth_original exists
#     background_mask = 1.0 - object_mask # Where object isn't
#     background_mask = background_mask.repeat(1, 3, 1, 1) # Repeat for RGB channels

#     sample_bg_masked = sample_bg_img * background_mask
#     sample_bg_masked[sample_bg_masked < 0] = 0

#     # Add background to the last depth slice (index 10)
#     scene_stack[:, :, -1, :, :] = scene_stack[:, :, -1, :, :] + sample_bg_masked

#     # 6. 반환값 (sample_img는 squeeze 불필요 -> 차원 유지해야 할 수도 있음)
#     # return scene_stack, sample_img.squeeze(2) # Original
#     return scene_stack, sample_img # Return sample_img keeping the dimension (batch, 3, 1, H, W)
# def quantize_rgb_by_depth(quantize_num, sample_img, sample_bg_img, sample_depth, max_val=1):
#     # Get batch size, height, width from sample_depth
#     # Note: Using sample_img shape might be safer if depth map has different initial size
#     batch_size = sample_img.shape[0]
#     # height = sample_img.shape[2] # Use target height/width instead
#     # width = sample_img.shape[3]

#     # --- Add Resizing Here ---
#     # Get target size from gaussian_stack (Height, Width)
#     # target_h = gaussian_stack.shape[-2] # 704
#     # target_w = gaussian_stack.shape[-1] # 1024

#     # Resize sample_img if its size doesn't match gaussian_stack
#     if sample_img.shape[-2:] != (target_h, target_w):
#         print(f"Resizing sample_img from {sample_img.shape[-2:]} to {(target_h, target_w)}")
#         # Ensure sample_img is float before interpolation if it isn't already
#         sample_img = F.interpolate(sample_img.float(), size=(target_h, target_w), mode='bilinear', align_corners=False)

#     # Also resize sample_depth to match the new image dimensions
#     if sample_depth.shape[-2:] != (target_h, target_w):
#         print(f"Resizing sample_depth from {sample_depth.shape[-2:]} to {(target_h, target_w)}")# 1. 채널 차원 추가 (N, H, W) -> (N, C=1, H, W)
#         # --- 수정 시작 ---
#         # 1. Use torchvision.transforms.Resize
#         #    Input needs to be (C, H, W) or (N, C, H, W). Add channel dim first.
#         depth_with_channel = sample_depth.float().unsqueeze(1) # Shape: (N, 1, H, W)

#         # 2. Define the resize transform
#         #    InterpolationMode.NEAREST is equivalent to mode='nearest'
#         resize_transform = transforms.Resize(size=(target_h, target_w),
#                                         interpolation=InterpolationMode.NEAREST,
#                                         antialias=False) # antialias=False 권장 (nearest 모드)

#         # 3. Apply the transform
#         resized_depth_with_channel = resize_transform(depth_with_channel) # Shape: (N, 1, new_H, new_W)

#         # 4. Remove the channel dimension
#         sample_depth = resized_depth_with_channel.squeeze(1) # Shape: (N, new_H, new_W)
#         # depth_with_channel = sample_depth.float().unsqueeze(1)
#         # # 2. interpolate 수행
#         # resized_depth = F.interpolate(depth_with_channel, size=(target_h, target_w), mode='nearest') # mode='nearest' 유지
#         # # 3. 채널 차원 제거 (N, C=1, H, W) -> (N, H, W)
#         # sample_depth = resized_depth.squeeze(1)
         
#          # Depth maps often use 'nearest' interpolation to avoid creating new depth values
#         #  sample_depth = F.interpolate(sample_depth.float().unsqueeze(1), size=(target_h, target_w), mode='nearest').squeeze(1)
#          # Make sure label is back to original dtype if needed, or keep as float

#     # Now assign the correct height/width after resizing
#     # height = target_h
#     # width = target_w
#     # --- Resizing Added ---


#     # mask_stack 초기화 (Size uses the NEW height/width)
#     # mask_stack = torch.zeros(batch_size, 1, quantize_num, height, width, device=DEVICE, dtype=torch.float32)

#     # sample_img 준비 (unsqueeze happens AFTER resizing)
#     sample_img = sample_img.unsqueeze(2) # (batch, 3, 1, H=704, W=1024)

#     # Multiply gaussian filter (NOW shapes match)
#     # sample_img = torch.multiply(sample_img, gaussian_stack) # <-- Error should be gone

#     # sample_img_bwmask should also use the resized sample_depth
#     sample_img_bwmask = torch.where(sample_depth > 0, 1.0, 0.0).unsqueeze(1).unsqueeze(2).float() # (b, 1, 1, H, W) float
#     sample_img = sample_img * sample_img_bwmask

#     # sample_depth 준비 (invert)
#     sample_depth = 1.0 - sample_depth.unsqueeze(1) # Add channel dim back for comparison (b, 1, H, W)

#     # 양자화 루프 (Logic from previous step)
#     for i in range(quantize_num): # 0부터 10까지 (11번)
#         if i == 0:
#             threshold_upper = wd_map_y_tensor[i+1] # 0.1
#             mask = sample_depth < threshold_upper # depth < 0.1
#         elif i == quantize_num - 1: # 마지막 구간 (i=10)
#             threshold_lower = wd_map_y_tensor[i] # 1.0
#             mask = sample_depth >= threshold_lower # depth >= 1.0
#         else: # 중간 구간 (i=1 ~ 9)
#             threshold_lower = wd_map_y_tensor[i] # e.g., i=1 -> 0.1
#             threshold_upper = wd_map_y_tensor[i+1] # e.g., i=1 -> 0.2
#             mask = (sample_depth >= threshold_lower) & (sample_depth < threshold_upper) # e.g., 0.1 <= depth < 0.2

#         mask_stack[:, :, i, :, :] = mask.float()


#     # 마스크를 3채널로 복제
#     mask_3channel = mask_stack.repeat(1, 3, 1, 1, 1) # shape: (batch, 3, 11, H, W)

#     # scene_stack 계산
#     scene_stack = sample_img * mask_3channel # shape: (batch, 3, 11, H, W)

#     # 배경 처리
#     # Use bwmask directly (already calculated and resized)
#     object_mask = sample_img_bwmask.squeeze(2) # (b, 1, H, W)
#     background_mask = 1.0 - object_mask # Where object isn't
#     background_mask = background_mask.repeat(1, 3, 1, 1) # Repeat for RGB channels (b, 3, H, W)

#     # Ensure sample_bg_img is also resized if needed (assuming it was handled before calling this func)
#     sample_bg_masked = sample_bg_img * background_mask
#     sample_bg_masked[sample_bg_masked < 0] = 0

#     # Add background to the last depth slice (index 10)
#     # Ensure dimensions match before adding
#     scene_stack[:, :, -1, :, :] = scene_stack[:, :, -1, :, :] + sample_bg_masked

#     # 반환값
#     return scene_stack, sample_img.squeeze(2) # Squeeze back the depth dim from sample_img
def quantize_rgb_by_depth(quantize_num, sample_img, sample_bg_img, sample_depth, max_val=1):
    batch_size = sample_img.shape[0]
    height = sample_img.shape[2] # Original height (e.g., 648)
    width = sample_img.shape[3]  # Original width (e.g., 1152)

    mask_stack = torch.zeros(batch_size, 1, quantize_num, height, width, device=DEVICE, dtype=torch.float32)

    sample_img_prep = sample_img.float().unsqueeze(2) # (batch, 3, 1, H, W)

    if sample_depth.dim() == 3: # If (N, H, W)
        sample_depth_ch = sample_depth.unsqueeze(1).float() # (N, 1, H, W)
    elif sample_depth.dim() == 4: # If (N, 1, H, W) already
         sample_depth_ch = sample_depth.float()
    else:
        raise ValueError(f"Unexpected sample_depth shape: {sample_depth.shape}")

    sample_img_bwmask = torch.where(sample_depth_ch > 0, 1.0, 0.0).unsqueeze(2) # (b, 1, 1, H, W)
    sample_img_masked_prep = sample_img_prep * sample_img_bwmask
    image_masked_return = sample_img.float() * sample_img_bwmask.squeeze(2)

    sample_depth_inverted = 1.0 - sample_depth_ch # (b, 1, H, W)

    for i in range(quantize_num):
        if i == 0:
            threshold_upper = wd_map_y_tensor[i+1]
            mask = sample_depth_inverted < threshold_upper
        elif i == quantize_num - 1:
            threshold_lower = wd_map_y_tensor[i]
            mask = sample_depth_inverted >= threshold_lower
        else:
            threshold_lower = wd_map_y_tensor[i]
            threshold_upper = wd_map_y_tensor[i+1]
            mask = (sample_depth_inverted >= threshold_lower) & (sample_depth_inverted < threshold_upper)
        mask_stack[:, :, i, :, :] = mask.float()

    mask_3channel = mask_stack.repeat(1, 3, 1, 1, 1)
    scene_stack = sample_img_masked_prep * mask_3channel

    # 배경 처리 (Resize bg_image to match ORIGINAL sample_img size)
    if sample_bg_img.shape[-2:] != (height, width):
         print(f"Resizing bg_image from {sample_bg_img.shape[-2:]} to {(height, width)}")
         sample_bg_img = F.interpolate(sample_bg_img.float(), size=(height, width), mode='bilinear', align_corners=False)

    object_mask = sample_img_bwmask.squeeze(2) # (b, 1, H, W)
    background_mask = 1.0 - object_mask
    background_mask = background_mask.repeat(1, 3, 1, 1)

    sample_bg_masked = sample_bg_img * background_mask
    sample_bg_masked[sample_bg_masked < 0] = 0

    scene_stack[:, :, -1, :, :] = scene_stack[:, :, -1, :, :] + sample_bg_masked

    return scene_stack, image_masked_return

def run_generator_object_torch(psf, scene):
    raw = convolve_rgb_Raw_Gen(psf,scene,batch_size=HPARAMS['BATCH_SIZE'])
    return raw

def run_generator_bg_torch(psf, scene_stack):
    raw_bg = convolve_rgb_Raw_Gen(psf,scene_stack[:,:,-1,:,:],batch_size=HPARAMS['BATCH_SIZE'])
    RGB_sum = scene_stack.sum(dim=2)
    b,c,w,h = RGB_sum.shape
    RGB_sum_crop = RGB_sum[:,:,w//4:3*w//4,h//4:3*h//4] # crop
    return raw_bg, RGB_sum_crop

def simul_raw_generation_torch(quantize_num, psf_stack, scene_stack):
    raw_stack = torch.zeros_like(scene_stack).to(DEVICE)
    for i in range(quantize_num):
        # generate sub-unit images
        if i == quantize_num-1:
            # Get simulated raw from background
            raw_stack[:,:,i], RGB_sum_crop = run_generator_bg_torch(psf_stack[:,:,i], scene_stack)
        else:
            raw_stack[:,:,i] = run_generator_object_torch(psf_stack[:,:,i], scene_stack[:,:,i])
    raw_sum = torch.sum(raw_stack, dim=2) # sum out
    b,c,w,h = raw_sum.shape
    raw_final = raw_sum[:,:,w//4:3*w//4,h//4:3*h//4] # crop
    return raw_final, RGB_sum_crop

def npz_loader(path):
    sample = torch.from_numpy(np.load(path)['dmap'])
    return sample

def add_gaussian_noise(image, mean=None, std_dev_min = None, std_dev_max = None):
    gaussian_noise = np.random.normal(mean, np.random.uniform(std_dev_min, std_dev_max), image.shape)
    # print("Max and min of image: ", np.max(image), np.min(image))
    # print("Max and min of noise: ", np.max(gaussian_noise), np.min(gaussian_noise))
    # Add the Gaussian noise to the image
    # noisy_image = image + gaussian_noise + bg
    noisy_image = image + gaussian_noise
    # print("Max and min of noisy image: ", np.max(noisy_image), np.min(noisy_image))
    # Convert back to uint8
    # noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    # print("Max and min of noisy image after clip: ", np.max(noisy_image), np.min(noisy_image))  
    return noisy_image

def calculate_snr(original_image, noisy_image):
    # Calculate signal power
    signal_power = np.sum(original_image ** 2)

    # Calculate noise power
    noise = original_image - noisy_image
    noise_power = np.sum(noise ** 2)

    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power, where=signal_power > 0)

    return snr

# Update ImageDataset
class ImageDataset(Dataset):
    """Custom image file dataset loader."""

    def __init__(self, image_path, label_path):
        self.images = image_path
        self.labels = label_path
        self.len = len(image_path)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.images[index][0]
        label = self.labels[index][0]
        return (image, label)

def train_batch(train_parameters, trainset_loader):
    for batch_index, (image, label) in tqdm(enumerate(trainset_loader), total=len(trainset_loader),
                                             desc="Processing Batches", unit="batch", dynamic_ncols=True):
        image, label = image.float().to(DEVICE), label.float().to(DEVICE)
        print(f"Batch {batch_index}: Image shape: {image.shape}, Label shape: {label.shape}") # <-- Shape 확인 추가
        label = label.unsqueeze(1)
        label[label < 0] = 0
        bg_image = torch.zeros_like(image)
        # print("Max and min of image[0]: ", torch.max(image[0]), torch.min(image[0]))
        for i in range(image.size(0)):
            random_idx_bg = np.uint16(np.round(np.random.rand() * (25000 - 1), 1))
            bg_tmp = plt.imread(bg_train[random_idx_bg])
            bg_tmp = cv2.resize(bg_tmp, dsize=(image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_CUBIC)
            bg_tmp[bg_tmp<0] = 0
            bg_tmp = torch.from_numpy(bg_tmp).permute(2, 0, 1).unsqueeze(0).to(device)
            bg_tmp = bg_tmp/torch.max(bg_tmp)
            bg_tmp = bg_tmp*background_level_max*torch.rand(1).to(device)
            bg_image[i] = bg_tmp
        
        # Get target size from gaussian_stack (Height, Width)
        # target_h = gaussian_stack.shape[-2] # 704
        # target_w = gaussian_stack.shape[-1] # 1024
        # Resize bg_image to match gaussian_stack using interpolate
        # (align_corners=False is usually recommended for general resizing)
        # if bg_image.shape[-2:] != (target_h, target_w):
        #     print(f"Resizing bg_image from {bg_image.shape[-2:]} to {(target_h, target_w)}")
        #     bg_image = F.interpolate(bg_image, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # bg_image = bg_image * gaussian_stack.squeeze(2)
        scene_stack, image_masked = quantize_rgb_by_depth(quantize_num, image, bg_image, label)
        raw_simulated, RGB_sum_crop = simul_raw_generation_torch(quantize_num, psf_stack, scene_stack)
        # raw_simulated is not max normalized
        [b, c, w, h] = image.shape
        for i in range(image.size(0)):
            idx = str(batch_index * HPARAMS["BATCH_SIZE"] + i).zfill(5)
            raw_simulated_norm = (raw_simulated[i] / torch.max(raw_simulated[i])).squeeze().permute(1, 2, 0).cpu().numpy()
            raw_png = cv2.cvtColor(np.uint8(raw_simulated_norm * 255), cv2.COLOR_BGR2RGB)
            raw_noisy = add_gaussian_noise(raw_simulated_norm, mean, std_dev_min, std_dev_max)
            snr = calculate_snr(raw_simulated_norm, raw_noisy)
            raw_noisy_png = (255*raw_noisy/np.max(raw_noisy))
            raw_noisy_png = np.clip(raw_noisy_png, 0, 255).astype(np.uint8)
            # save
            # np.save(os.path.join(save_dir_raw_npy, '{}.npy'.format(idx)), raw_simulated_norm)
            # cv2.imwrite(os.path.join(save_dir_raw_png, '{}.png'.format(idx)),raw_png) # raw wo noise
            # np.save(os.path.join(save_dir_raw_noisy_npy, '{}_{}dB.npy'.format(idx,snr.round(2))), raw_noisy)    
            
            image_masked_png =image_masked[i].squeeze().permute(1, 2, 0).cpu().numpy()
            image_masked_png = image_masked_png[w//4:3*w//4,h//4:3*h//4,:] # center crop 
            # print("Shape of image_masked_png: ", image_masked_png.shape)
            image_masked_png = cv2.cvtColor(np.uint8(image_masked_png * 255), cv2.COLOR_BGR2RGB)
            image_masked_png = np.clip(image_masked_png, 0, 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(save_dir_raw_noisy_png, '{}_{}dB.png'.format(idx, snr.round(2))),cv2.cvtColor(raw_noisy_png, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(save_dir_image_png, '{}.png'.format(idx)), image_masked_png)
            
            # cv2.imwrite(os.path.join(save_dir_image_cc_sum, '{}.png'.format(idx)), cv2.cvtColor(np.uint8(RGB_sum_crop[i].squeeze().permute(1, 2, 0).cpu().numpy() * 255), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(os.path.join(save_dir_noisy, '{}.png'.format(idx)),raw_noisy)
            # cv2.imwrite(os.path.join(save_dir_bg, '{}.png'.format(idx)), cv2.cvtColor(np.uint8(scene_stack[i,:,-1,:,:].squeeze().permute(1, 2, 0).cpu().numpy() * 255), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(os.path.join(save_dir_image_sum, '{}.png'.format(idx)), cv2.cvtColor(np.uint8(RGB_sum[i].squeeze().permute(1, 2, 0).cpu().numpy() * 255), cv2.COLOR_BGR2RGB))
            
def train(train_parameters):
    with ThreadPoolExecutor() as executor:
        futures = []
        
        for batch_index in tqdm(range(len(train_parameters['trainset_loader'])), desc="Processing Batches", unit="batch", dynamic_ncols=True):
            future = executor.submit(train_batch, train_parameters, batch_index)
            futures.append(future)

        # Wait for all futures to complete
        for future in tqdm(futures, desc="Waiting for Futures", unit="future", dynamic_ncols=True):
            future.result()
    print("Complete!")
        
# def main():
#     # Define transformer
#     transformer = transforms.Compose([
#         transforms.ToTensor(), # then convert to tensor
#         # transforms.Lambda(lambda x: x - 1.0/255.0), # subtract 1 from the image with 2X FOV
#     ])
    
#     trainset_image = torchvision.datasets.ImageFolder(root=HPARAMS['IMAGE_PATH'],
#                                                          transform=transformer)
#     trainset_label = torchvision.datasets.DatasetFolder(root=HPARAMS['LABEL_PATH'],
#                                                         loader=npz_loader,
#                                                         extensions=['.npz'],)
#     data_size = len(trainset_image)
#     print('Total dataset size: {}'.format(data_size))
#     # trainset_size = HPARAMS['TRAINSET_SIZE']
#     train_load = ImageDataset(trainset_image, trainset_label)
    
#     TPARAMS['trainset_loader'] = torch.utils.data.DataLoader(
#         train_load,
#         batch_size=HPARAMS['BATCH_SIZE'],
#         shuffle=False,
#         num_workers=HPARAMS['NUM_WORKERS'],
#         pin_memory=False,
#     )
    
#     print('Raw image simulation start!')
#     tot_time = time.time()
#     # Call train_batch with trainset_loader
#     train_batch(TPARAMS, TPARAMS['trainset_loader'])
#     # end_time = time.time()
#     total_time = time.time() - tot_time
#     print('Total Elapsed : {} hrs for {} images'.format(total_time/3600,data_size))
#     send_slack_alert(total_time,data_size)

def main():
    # Define transformer
    transformer = transforms.Compose([
        transforms.ToTensor(), # then convert to tensor
        # transforms.Lambda(lambda x: x - 1.0/255.0), # subtract 1 from the image with 2X FOV
    ])
    
    # 1. Load the full datasets as before
    trainset_image_full = torchvision.datasets.ImageFolder(root=HPARAMS['IMAGE_PATH'],
                                                           transform=transformer)
    trainset_label_full = torchvision.datasets.DatasetFolder(root=HPARAMS['LABEL_PATH'],
                                                             loader=npz_loader,
                                                             extensions=['.npz'])

    # 2. Define the number of images to use and create subsets
    num_images_to_process = 5
    total_data_size = len(trainset_image_full)
    print(f'Total dataset size: {total_data_size}')
    
    # Ensure you don't request more images than available
    if total_data_size < num_images_to_process:
        print(f"Warning: Dataset size ({total_data_size}) is smaller than requested. Using all {total_data_size} images.")
        num_images_to_process = total_data_size
        
    indices = list(range(num_images_to_process))
    trainset_image_subset = torch.utils.data.Subset(trainset_image_full, indices)
    trainset_label_subset = torch.utils.data.Subset(trainset_label_full, indices)
    
    print(f'Processing the first {len(trainset_image_subset)} images.')

    # 3. Pass these new subset objects to your ImageDataset
    # Your custom ImageDataset class works perfectly with Subset without any changes.
    train_load = ImageDataset(trainset_image_subset, trainset_label_subset)
    
    TPARAMS['trainset_loader'] = torch.utils.data.DataLoader(
        train_load,
        batch_size=HPARAMS['BATCH_SIZE'],
        shuffle=False,  # shuffle=False ensures you process the first 5000 in order
        num_workers=HPARAMS['NUM_WORKERS'],
        pin_memory=False,
    )
    
    print('Raw image simulation start!')
    tot_time = time.time()
    # Call train_batch with the new, smaller trainset_loader
    train_batch(TPARAMS, TPARAMS['trainset_loader'])
    
    total_time = time.time() - tot_time
    print(f'Total Elapsed : {total_time/3600:.2f} hrs for {num_images_to_process} images')
    # send_slack_alert(total_time, num_images_to_process)
    
if __name__ == "__main__":
    main()
