import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import torch
import cv2
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import seaborn as sns
import numpy as np
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
import imageio

# 2. (신규) Argument 파서 설정
# parser = argparse.ArgumentParser(description="Raw image generator for lensless camera")
# parser.add_argument('--psf_path', type=str, required=True, help='Path to the .mat PSF stack file')
# parser.add_argument('--data_path', type=str, required=False, help='Path to the root of mini_dataset (containing train/image and train/label)')
# parser.add_argument('--save_path', type=str, required=True, help='Path to save the generated raw data')
# parser.add_argument('--psf_idx_list', nargs='+', type=int, required=True, help='List of PSF indices to use for quantization')
# parser.add_argument('--wd_list', nargs='+', type=float, required=True, help='List of working distances corresponding to psf_idx_list')
# parser.add_argument('--data_idx', nargs='+', type=int, required=False, help='Start and end index of data to process (e.g., 0 500)')
# --- Argument Parser (Simplified) ---
parser = argparse.ArgumentParser(description="Raw image generator for lensless camera")
parser.add_argument('--psf_path', type=str, required=False, help='Path to the .mat PSF stack file (21x512x512)')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the generated raw data')
parser.add_argument('--data_idx', nargs=2, type=int, required=False, default=[0, 10000], help='Start and end index (exclusive) of data to process (e.g., 0 500)')
# 3. (신규) 스크립트가 실행될 때 터미널로부터 인수를 읽어옴
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(DEVICE))

folder_name = '20251028_21psf_set_HJA_02'

HPARAMS = {
    'BATCH_SIZE': 1,
    'NUM_WORKERS': 1,
    # 'TRAINSET_SIZE' : 60000,
    # 'FOLDER_PATH': '/mnt/ssd1/depth_imaging/dataset_ssd1/current_blender_dataset/'+folder_name+'/', # ssd1
    'FOLDER_PATH': '/home/hotdog/files_251026/'+folder_name+'/',
    # 'IMAGE_PATH': '/home/hotdog/mini_dataset/'+folder_name+'/image/',
    # # 'LABEL_PATH': '/home/hotdog/mini_dataset/'+folder_name+'/label/',
    # 'IMAGE_PATH': '/home/hotdog/ssd1/depth_imaging/dataset_ssd2/20251009_222218_10000_5/image/',
    # 'LABEL_PATH': '/home/hotdog/ssd1/depth_imaging/dataset_ssd2/20251009_222218_10000_5/label/',
    'IMAGE_PATH': '/home/hotdog/ssd1/depth_imaging/dataset_ssd2/20251028_141843_10000_5/image/',
    'LABEL_PATH': '/home/hotdog/ssd1/depth_imaging/dataset_ssd2/20251028_141843_10000_5/label/',
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

# --- Quantization Setup ---
quantize_num = 21 # Explicitly set based on the new PSF stack
print(f"Using quantize_num = {quantize_num}")
# Create thresholds for quantization (0.0 to 1.0 in quantize_num+1 steps)
# These will be used inside quantize_rgb_by_depth
quantization_thresholds = torch.linspace(0.0, 1.0, quantize_num + 1, device=DEVICE).float()


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
psf_dir ="/home/hotdog/files_251026/image_stack_selected_resized_512_060_080.mat"

####################################################################################################
############################## Simulated_PSFs ####################################################
# psf_dir = '/mnt/ssd1/depth_imaging/phase_mask_design_file/241015_162843_PoissonDiskRandomDots/simulated_psf_stackfocalDist_0.006_upscale_2.35_crop_1/simulated_psf_resized_352x512x28.mat'


# # --- PSF Loading and Processing (Corrected) ---
# print(f"Loading PSF stack: {psf_dir}")
# psf_mat_file = io.loadmat(psf_dir)
# psf_stack = psf_mat_file['images']
# # psf_stack = psf_mat_file['psf_stack_2x']
# # psf_stack = psf_mat_file['psf_stack_2x_resized']
# # psf_stack = psf_mat_file['psf_stack_2x_resized_rotated']
# # psf_stack = psf_mat_file['psf_resized'] # arbitrary resized PSF (simulated PSF)
# pad_size = tuple(2*x for x in psf_stack.shape)

# print('='*30)
# print('pad_size: ',pad_size)
# print('Before padding - psf_stack shape: ',psf_stack.shape) # (101, 1300, 1300)
# psf_stack = img_pad_np(psf_stack, pad_size) # no padding for 2x PSF, aiobio is needed
# print('After padding - psf_stack shape: ',psf_stack.shape) # (202, 2600, 2600)
# print('='*30)


# # psf_stack = img_pad_np(psf_stack, pad_size) # no padding for 2x PSF, aiobio is needed

# psf_stack = psf_stack.astype(np.float32)

# psf_stack /= np.linalg.norm(psf_stack)
# # print("Shape of PSF stack: {}".format(psf_stack.shape))
# psf_stack = torch.from_numpy(psf_stack).permute(2,0,1).repeat(1,1,1,1,1).type(torch.FloatTensor).float().to(DEVICE) #BCDHW
# psf_stack = psf_stack.clone().detach().to(dtype=torch.float64)

# # psf_stack = psf_stack[:,:,5:,:,:] # 10 cm ~ 60 cm.. just use 9 channel
# # psf_stack = img_pad(psf_stack,pad_size)
# print('PSF stack shape: {}'.format(psf_stack.shape))
# print("Max and Min of PSF stack: {} and {}".format(torch.max(psf_stack), torch.min(psf_stack)))
# print("Data type of psf_stack: {}".format(psf_stack.dtype))
# --- PSF Loading and Processing (Corrected) ---
print(f"Loading PSF stack: {psf_dir}")
psf_mat_file = io.loadmat(psf_dir)
# *** IMPORTANT: Check the actual key name in your .mat file! ***
psf_key = 'images' # Or 'psf_stack', 'psf_stack_resized', etc.
if psf_key not in psf_mat_file:
    raise ValueError(f"Key '{psf_key}' not found in {psf_dir}. Check .mat file structure.")
psf_stack_np = psf_mat_file[psf_key] # Should be (21, 512, 512)

print(f"Loaded PSF stack shape: {psf_stack_np.shape}")
if psf_stack_np.shape != (quantize_num, 512, 512):
    raise ValueError(f"Expected PSF shape ({quantize_num}, 512, 512), but got {psf_stack_np.shape}")

# Convert to tensor, add batch and channel dims, ensure float32
# Shape: (Batch=1, Channel=1, Depth=21, Height=512, Width=512)
psf_stack = torch.from_numpy(psf_stack_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

# Normalize the stack (consider if needed, maybe normalize per slice later?)
psf_stack /= torch.linalg.norm(psf_stack)

print(f"Processed PSF stack shape: {psf_stack.shape}")
print(f"Max and Min of PSF stack: {torch.max(psf_stack)} and {torch.min(psf_stack)}")
print(f"Data type of psf_stack: {psf_stack.dtype}") # Should be float32
# gaussian_dir = '/mnt/ssd1/depth_imaging/dataset/20240221_2D_angular_response_IMX477_resized_704x1024_2XFOV.mat'

gaussian_dir = '/mnt/ssd1/depth_imaging/dataset_ssd1/data_angular_response/20250408_2D_angular_response_IMX477_resized_704x1024_2XFOV.mat' # 20250408

gaussian_mat_file = io.loadmat(gaussian_dir)
gaussian_2D_filter = gaussian_mat_file['angular_2d_resized']
gaussian_stack = torch.from_numpy(gaussian_2D_filter).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(DEVICE)
gaussian_stack = gaussian_stack.clone().detach().to(dtype=torch.float32)
gaussian_stack = gaussian_stack.repeat(1,1,1,1).unsqueeze(2)
# Make float32
# print("Shape of gaussian_stack: {}".format(gaussian_stack.shape))
# print("Max and Min of gaussian_stack: {} and {}".format(torch.max(gaussian_stack), torch.min(gaussian_stack)))
# print("Data type of gaussian_stack: {}".format(gaussian_stack.dtype))

# quantize_num = np.shape(psf_stack)[2]
# print('Quantize number: {}'.format(quantize_num))
# range_log = torch.exp(torch.linspace(0, 1, steps=quantize_num + 1)) ** 5
# range_log = ((range_log - range_log.min()) / (range_log.max() - range_log.min()))

# wd_measured = np.concatenate([np.arange(9.5, 21), np.arange(23, 32,2), np.arange(37.5, 65, 5)]) # 20240420
# wd_measured = np.concatenate([[5], np.arange(5.5, 21), np.arange(23, 32,2), np.arange(37.5, 65, 5)]) # 20240602, add 5cm PSF

# wd_blender = np.concatenate(([5], np.arange(10, 71, 10))) #20240602 add [5]
wd_blender =  np.arange(5.0, 15.0, 0.1) # aiobio
# wd_measured = np.concatenate((np.arange(7, 11, 0.5), np.arange(11, 17, 1), [17.5, 19, 21, 23, 26, 30, 36, 70]))  # RPC diffuser 22 channel in 20241130, resampled

wd_measured = np.arange(5.0, 15.0, 0.1)  # 5.0 to 15.0 with step 0.1, aiobio

# (xp) 맵의 X축: argument로 받은 11개의 대표 거리
# args.wd_list는 문자열 리스트이므로 float 배열로 변환합니다.
# wd_map_x = np.array(args.wd_list, dtype=np.float32) # (len 11)

# (fp) 맵의 Y축: 11개의 대표 거리에 1:1로 매핑될 11개의 정규화된 깊이 (0.0 ~ 1.0)
# wd_map_y = np.linspace(0.0, 1.0, len(wd_map_x)) # (len 11)
# wd_map_y_tensor = torch.from_numpy(wd_map_y).float().to(DEVICE) # (11,)

# 100개의 전체 거리(wd_measured)를 11개 맵(wd_map_x, wd_map_y)을 기준으로 보간하여
# 최종 100개짜리 조회 테이블(range_log)을 생성합니다.
# print("--- Interpolation Shapes ---")
# print("Query points (x) shape: ", wd_measured.shape)      # (100,)
# print("Map X-axis (xp) shape: ", wd_map_x.shape)         # (11,)
# print("Map Y-axis (fp) shape: ", wd_map_y.shape)         # (11,)

# range_log = np.interp(wd_measured, wd_map_x, wd_map_y)

# print("Generated lookup table shape: ", range_log.shape) # (100,)
# print("----------------------------")



# 5, 10, 20, 30, 40, 50, 60, 70 in blender environment
# depth_value = [0, 0.0769231, 0.230769, 0.384615, 0.537981, 0.691827, 0.846154, 1] # 20240420, inverted
# depth_value = np.linspace(0.0, 1.0, 31)

# Interpolate depth values for wd_measured using wd_blender and depth_value
# print("wd_measured.shape: ",wd_measured.shape)
# print("wd_blender.shape: ",wd_blender.shape)
# print("depth_value.shape: ",depth_value.shape)
# range_log = np.interp(wd_measured, wd_blender, depth_value)
# print("Length of range_log: ", len(range_log))
# range_log = torch.from_numpy(range_log).float().to(DEVICE)


print("WD measured [cm]:")
print(wd_measured)
print("Length of WD measured: ", len(wd_measured))

print("Interpolated Depth Values:")
print(range_log)
print("Length of Interpolated Depth Values: ", len(range_log))

# def quantize_rgb_by_depth(quantize_num, sample_img, sample_bg_img, sample_depth, max_val=1):
#     mask_stack = torch.zeros_like(sample_depth).to(DEVICE)
#     mask_stack.unsqueeze_(2)
#     mask_stack = mask_stack.repeat(1, 1, quantize_num, 1, 1)
#     sample_img = sample_img.unsqueeze(2)
#     # print("Max and Min of sample_img: {} and {}".format(torch.max(sample_img), torch.min(sample_img)))
#     # if torch.min(sample_img) > 0:
#     #     sample_img[sample_img<=torch.min(sample_img)] = 0
#     # sample_img = torch.multiply(sample_img, gaussian_stack) # multiply gaussian filter
#     sample_img_bwmask = torch.where(sample_depth > 0, 1, 0).unsqueeze(2) # binary mask

#     # sample_img_bwmask = sample_img_bwmask.repeat(1, 3, 1, 1, 1) # repeat to match the channel
#     # print("Shape of sample_img: ", sample_img.shape)
#     # print("Shape of sample_img_bwmask: ", sample_img_bwmask.shape)
    
#     # sample_img = torch.multiply(sample_img, sample_img_bwmask) # (2025-05-19) multiply binary mask to remove unwanted boundary of the object
#     sample_img = sample_img * sample_img_bwmask
    
#     sample_depth_inverted = 1.0 - sample_depth # invert to calculate depth
    
#     # for i in range(quantize_num):
#     #     if i == 0:
#     #         mask_stack[:,:,i] = torch.where(sample_depth <= range_log[i + 1], 1, 0) # modified
#     #     elif i == quantize_num - 1:
#     #         mask_stack[:,:,i] = torch.where((sample_depth > range_log[i]) & (sample_depth <= max_val), 1, 0)
#     #     else:
#     #         mask_stack[:,:,i] = torch.where((sample_depth > range_log[i]) & (sample_depth <= range_log[i + 1]), 1, 0)
#     # mask_3channel = mask_stack.repeat(1, 3, 1, 1, 1)

#     for i in range(quantize_num): # Loop 11 times (i = 0 to 10)
#         # print('-'*30)
#         # print('quantized_num:',quantize_num)
#         # print('-'*30)
#         # wd_map_y_tensor has 11 values [0.0, 0.1, ..., 1.0]

#         if i == 0: # First bin (closest distance, inverted depth near 0.0)
#             # Use midpoint between 0.0 and 0.1 as upper bound, or just use 0.1?
#             # Let's use the next threshold as the strict upper bound
#             threshold_upper = wd_map_y_tensor[i+1] # Index 1 (value 0.1)
#             # Condition: inverted_depth < 0.1
#             mask = (sample_depth_inverted < threshold_upper)
#             # Special case for exact 0.0? Might need <= threshold_upper depending on logic.
#             # Let's refine: 0.0 <= inverted_depth < 0.1
#             # threshold_lower = wd_map_y_tensor[i] # Index 0 (value 0.0)
#             # mask = (sample_depth_inverted >= threshold_lower) & (sample_depth_inverted < threshold_upper)
#             # Simpler: If it's the first bin, just check against the next threshold
#             mask = sample_depth_inverted < threshold_upper

#         elif i == quantize_num - 1: # Last bin (farthest distance, inverted depth near 1.0)
#             # Condition: inverted_depth >= 1.0 (or >= 0.9 depending on exact bounds)
#             threshold_lower = wd_map_y_tensor[i] # Index 10 (value 1.0)
#             # Condition: inverted_depth >= 1.0
#             mask = (sample_depth_inverted >= threshold_lower)

#         else: # Intermediate bins (i = 1 to 9)
#             threshold_lower = wd_map_y_tensor[i]     # Index i   (e.g., 0.1 when i=1)
#             threshold_upper = wd_map_y_tensor[i+1]   # Index i+1 (e.g., 0.2 when i=1)
#             # Condition: 0.1 <= inverted_depth < 0.2 (when i=1)
#             mask = (sample_depth_inverted >= threshold_lower) & (sample_depth_inverted < threshold_upper)

#         # Assign the boolean mask (converted to float) to the correct slice
#         mask_stack[:, :, i, :, :] = mask.float()
#     scene_stack = torch.multiply(sample_img, mask_stack)
    
#     scene_gray = torch.mean(sample_img, dim=1, keepdim=False) # 8,1,352,512
#     scene_gray_negative = torch.abs(1-scene_gray)
#     scene_gray_negative[scene_gray_negative<1] = 0
#     scene_gray_negative = torch.repeat_interleave(scene_gray_negative,3, dim=1)
#     # print('='*20)
#     # print('sample_bg_img.shape: ',sample_bg_img.shape)
#     # print('scene_gray_negative.shape: ',scene_gray_negative.shape)
#     # print('='*20)
#     sample_bg_masked = torch.multiply(sample_bg_img, scene_gray_negative).to(DEVICE)
#     sample_bg_masked[sample_bg_masked<0] = 0
#     scene_stack[:,:,-1,:,:] += sample_bg_masked # last slice is background
#     return scene_stack, sample_img.squeeze(2)


# --- quantize_rgb_by_depth (Simplified Thresholds) ---
def quantize_rgb_by_depth(quantize_num_in, sample_img, sample_bg_img, sample_depth):
    batch_size, _, height, width = sample_img.shape # Shape (B, 3, 512, 512)
    # quantize_num_in should be 21

    mask_stack = torch.zeros(batch_size, 1, quantize_num_in, height, width, device=DEVICE, dtype=torch.float32)

    sample_img_prep = sample_img.float().unsqueeze(2) # (B, 3, 1, H, W)

    # Ensure sample_depth is (B, 1, H, W) and float
    if sample_depth.dim() == 3: # If (B, H, W)
        sample_depth_ch = sample_depth.unsqueeze(1).float()
    elif sample_depth.dim() == 4 and sample_depth.shape[1] == 1: # If (B, 1, H, W)
        sample_depth_ch = sample_depth.float()
    else:
        raise ValueError(f"Unexpected sample_depth shape: {sample_depth.shape}")

    sample_img_bwmask = torch.where(sample_depth_ch > 0, 1.0, 0.0).unsqueeze(2) # (B, 1, 1, H, W)
    sample_img_masked_prep = sample_img_prep * sample_img_bwmask
    image_masked_return = sample_img.float() * sample_img_bwmask.squeeze(2)

    sample_depth_inverted = 1.0 - sample_depth_ch # (B, 1, H, W)

    # Quantization Loop using simplified thresholds
    for i in range(quantize_num_in): # i = 0 to 20
        threshold_lower = quantization_thresholds[i]     # Value from linspace (e.g., 0.0, 1/21, 2/21...)
        threshold_upper = quantization_thresholds[i+1]   # Next value (e.g., 1/21, 2/21, 3/21...)

        # Create mask for the bin [lower, upper)
        # Handle the last bin separately to include 1.0
        if i == quantize_num_in - 1: # Last bin (i=20)
             # inverted_depth >= threshold_lower (e.g., >= 20/21)
             # Include 1.0 exactly: inverted_depth >= lower AND inverted_depth <= 1.0
            mask = (sample_depth_inverted >= threshold_lower) & (sample_depth_inverted <= 1.0)
        else:
             # inverted_depth >= lower AND inverted_depth < upper
            mask = (sample_depth_inverted >= threshold_lower) & (sample_depth_inverted < threshold_upper)

        mask_stack[:, :, i, :, :] = mask.float()

    mask_3channel = mask_stack.repeat(1, 3, 1, 1, 1) # (B, 3, 21, H, W)
    scene_stack = sample_img_masked_prep * mask_3channel # (B, 3, 21, H, W)

    # --- Background Processing ---
    # Resize bg_image to match Scene size (512x512)
    if sample_bg_img.shape[-2:] != (height, width):
         # print(f"Resizing bg_image from {sample_bg_img.shape[-2:]} to {(height, width)}")
         sample_bg_img = F.interpolate(sample_bg_img.float(), size=(height, width), mode='bilinear', align_corners=False)

    object_mask = sample_img_bwmask.squeeze(2) # (B, 1, H, W)
    background_mask = 1.0 - object_mask
    background_mask = background_mask.repeat(1, 3, 1, 1) # (B, 3, H, W)

    sample_bg_masked = sample_bg_img * background_mask
    sample_bg_masked[sample_bg_masked < 0] = 0

    # Add background to the last depth slice (index 20)
    scene_stack[:, :, -1, :, :] = scene_stack[:, :, -1, :, :] + sample_bg_masked

    return scene_stack, image_masked_return # image_masked is (B, 3, H, W)

def run_generator_object_torch(psf, scene):
    raw = convolve_rgb_Raw_Gen(psf,scene,batch_size=HPARAMS['BATCH_SIZE'])
    return raw

def run_generator_bg_torch(psf, scene_stack):
    raw_bg = convolve_rgb_Raw_Gen(psf,scene_stack[:,:,-1,:,:],batch_size=HPARAMS['BATCH_SIZE'])
    RGB_sum = scene_stack.sum(dim=2)
    b,c,w,h = RGB_sum.shape
    RGB_sum_crop = RGB_sum[:,:,w//4:3*w//4,h//4:3*h//4] # crop
    return raw_bg, RGB_sum_crop

# def simul_raw_generation_torch(quantize_num, psf_stack, scene_stack):
#     raw_stack = torch.zeros_like(scene_stack).to(DEVICE)
#     psf_stack = psf_stack.permute(0, 1, 3, 2, 4)
#     # psf_resized_stack = F.interpolate(psf_stack, size=(scene_stack.shape[-2], scene_stack.shape[-1]), mode='bilinear', align_corners=False)
#     # PSF 차원 변환: (N, C, D, H, W) -> (N * D, C, H, W)
#     n, c, d, h, w = psf_stack.shape
#     psf_stack_reshaped = psf_stack.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
    
#     # PSF 크기 조정
#     psf_resized = F.interpolate(psf_stack_reshaped, size=(scene_stack.shape[-2], scene_stack.shape[-1]), mode='bilinear', align_corners=False)
    
#     # PSF 차원 복원: (N * D, C, H, W) -> (N, C, D, H, W)
#     psf_resized_stack = psf_resized.reshape(n, d, c, scene_stack.shape[-2], scene_stack.shape[-1]).permute(0, 2, 1, 3, 4)
#     for i in range(quantize_num):
#         # generate sub-unit images
#         if i == quantize_num-1:
#             # Get simulated raw from background
#             raw_stack[:,:,i], RGB_sum_crop = run_generator_bg_torch(psf_resized_stack[:,:,i], scene_stack)
#         else:
#             # print("="*25)
#             # print("psf_stack.shape: ",psf_stack.shape) #torch.Size([1, 1, 1300, 101, 1300])
#             # print("psf_resized_stack.shape: ",psf_resized_stack.shape) #torch.Size([1, 1, 101, 648, 1152])
#             # print("scene_stack.shape: ",scene_stack.shape) #torch.Size([1, 3, 11, 648, 1152])
#             # print("="*25)
#             raw_stack[:,:,i] = run_generator_object_torch(psf_resized_stack[:,:,i], scene_stack[:,:,i])
#     raw_sum = torch.sum(raw_stack, dim=2) # sum out
#     b,c,w,h = raw_sum.shape
#     raw_final = raw_sum[:,:,w//4:3*w//4,h//4:3*h//4] # crop
#     return raw_final, RGB_sum_crop

# def simul_raw_generation_torch(quantize_num, psf_stack_in, scene_stack_in):
#     # psf_stack_in shape: (1, 1, 11, H_psf=1300, W_psf=1300)
#     # scene_stack_in shape: (Batch, 3, 11, H_scene=648, W_scene=1152)
#     batch_size, _, _, scene_h, scene_w = scene_stack_in.shape
#     _, _, _, psf_h, psf_w = psf_stack_in.shape

#     # --- 1. 결정: FFT 그리드 크기 ---
#     # 선배가 사용했던 4608x2592 를 사용하거나,
#     # 최소 크기 (scene + psf - 1) 이상인 2의 거듭제곱 등을 사용
#     # 여기서는 Scene 크기의 약 2배로 설정 (일반적인 방법)
#     # fft_h = scene_h * 2 # 
#     # fft_w = scene_w * 2 # 
#     # -> FFT 크기는 반드시 PSF 크기보다 크거나 같아야 함!
#     # -> 최소 필요 크기: H=1947, W=2451
#     # -> 안전하게 2의 거듭제곱 사용: fft_h = 2048, fft_w = 4096 (or 2560?)
#     fft_h = 2800
#     fft_w = 2800
#     # print(f"Using FFT grid size: H={fft_h}, W={fft_w}")

#     target_h_final = 1500
#     target_w_final = 1500

#     raw_stack_list = []

#     # print("="*25)
#     # print("Inside simul_raw_generation_torch")
#     # print("psf_stack_in.shape: ", psf_stack_in.shape) # (1, 1, 11, 1300, 1300)
#     # print("scene_stack_in.shape: ", scene_stack_in.shape) # (Batch, 3, 11, 648, 1152)
#     # print("="*25)

#     for i in range(quantize_num): # i from 0 to 10
#         # --- 2. 슬라이스 가져오기 ---
#         current_psf_slice = psf_stack_in[0, 0, i, :, :] # (1300, 1300)
#         current_scene_slice = scene_stack_in[:, :, i, :, :] # (Batch, 3, 648, 1152)

#         # --- 3. PSF 패딩 ---
#         # Add batch and channel dims for padding function: (1, 1, 1300, 1300)
#         psf_to_pad = current_psf_slice.unsqueeze(0).unsqueeze(0)
#         # Calculate padding amounts for PSF (left, right, top, bottom)
#         pad_h_total = fft_h - psf_h # 2048 - 1300 = 748
#         pad_w_total = fft_w - psf_w # 4096 - 1300 = 2796
#         pad_top = pad_h_total // 2
#         pad_bottom = pad_h_total - pad_top
#         pad_left = pad_w_total // 2
#         pad_right = pad_w_total - pad_left
#         # Apply padding
#         psf_padded = F.pad(psf_to_pad, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
#         # psf_padded shape: (1, 1, fft_h, fft_w)

#         # --- 4. Scene 슬라이스 패딩 ---
#         # Calculate padding amounts for Scene
#         pad_h_total_scene = fft_h - scene_h # 2048 - 648 = 1400
#         pad_w_total_scene = fft_w - scene_w # 4096 - 1152 = 2944
#         pad_top_scene = pad_h_total_scene // 2
#         pad_bottom_scene = pad_h_total_scene - pad_top_scene
#         pad_left_scene = pad_w_total_scene // 2
#         pad_right_scene = pad_w_total_scene - pad_left_scene
#         # Apply padding (works on batch dimension N, C, H, W)
#         scene_padded = F.pad(current_scene_slice, (pad_left_scene, pad_right_scene, pad_top_scene, pad_bottom_scene), "constant", 0)
#         # scene_padded shape: (Batch, 3, fft_h, fft_w)

#         # --- 5. FFT 컨볼루션 수행 ---
#         # Pass the padded tensors to the convolution function
#         raw_slice_padded = convolve_rgb_Raw_Gen(psf_padded, scene_padded,batch_size=HPARAMS['BATCH_SIZE']) # Output: (Batch, 3, fft_h, fft_w)

#         # --- 6. 결과 크롭 ---
#         # Crop the center part matching the original scene size
#         # Cropping indices match the padding amounts for the scene# Calculate center cropping indices for 1500x1500 output
#         center_h = fft_h // 2
#         center_w = fft_w // 2
#         crop_h_half = target_h_final // 2
#         crop_w_half = target_w_final // 2
        
#         crop_top = center_h - crop_h_half
#         crop_bottom = center_h + (target_h_final - crop_h_half) # Handle odd sizes
#         crop_left = center_w - crop_w_half
#         crop_right = center_w + (target_w_final - crop_w_half) # Handle odd sizes

#         # Perform center cropping to 1500x1500
#         raw_slice = raw_slice_padded[:, :, crop_top:crop_bottom, crop_left:crop_right]
#         # raw_slice shape: (Batch, 3, 1500, 1500)

#         raw_stack_list.append(raw_slice)

#     # --- 7. 스택 쌓고 합산 ---
#     raw_stack_tensor = torch.stack(raw_stack_list, dim=2) # Shape: (Batch, 3, 11, scene_h, scene_w)
#     raw_sum = torch.sum(raw_stack_tensor, dim=2) # Sum across depth dim -> (Batch, 3, scene_h, scene_w)

#     # --- 최종 Raw 이미지 크롭 (주석 처리됨) ---
#     raw_final = raw_sum

#     # RGB_sum_crop은 현재 사용되지 않으므로 None 반환
#     return raw_final, None

def simul_raw_generation_torch(quantize_num_in, psf_stack_in, scene_stack_in):
    # psf_stack_in shape: (1, 1, 21, H_psf=512, W_psf=512)
    # scene_stack_in shape: (Batch, 3, 21, H_scene=512, W_scene=512)
    batch_size, _, _, scene_h, scene_w = scene_stack_in.shape # 512, 512
    _, _, _, psf_h, psf_w = psf_stack_in.shape # 512, 512

    # --- FFT Grid Size ---
    fft_h = 1024 # >= 512+512-1 = 1023
    fft_w = 1024
    # print(f"Using FFT grid size: H={fft_h}, W={fft_w}")

    # --- Target Final Size ---
    target_h_final = 512
    target_w_final = 512

    raw_stack_list = []

    # print("="*25)
    # print("Inside simul_raw_generation_torch")
    # print("psf_stack_in.shape: ", psf_stack_in.shape)
    # print("scene_stack_in.shape: ", scene_stack_in.shape)
    # print("="*25)

    for i in range(quantize_num_in): # i from 0 to 20
        current_psf_slice = psf_stack_in[0, 0, i, :, :] # (512, 512)
        current_scene_slice = scene_stack_in[:, :, i, :, :] # (Batch, 3, 512, 512)

        # --- PSF Padding ---
        psf_to_pad = current_psf_slice.unsqueeze(0).unsqueeze(0) # (1, 1, 512, 512)
        pad_h_total = fft_h - psf_h # 1024 - 512 = 512
        pad_w_total = fft_w - psf_w # 1024 - 512 = 512
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        psf_padded = F.pad(psf_to_pad, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0) # (1, 1, 1024, 1024)

        # --- Scene Padding ---
        pad_h_total_scene = fft_h - scene_h # 1024 - 512 = 512
        pad_w_total_scene = fft_w - scene_w # 1024 - 512 = 512
        pad_top_scene = pad_h_total_scene // 2
        pad_bottom_scene = pad_h_total_scene - pad_top_scene
        pad_left_scene = pad_w_total_scene // 2
        pad_right_scene = pad_w_total_scene - pad_left_scene
        scene_padded = F.pad(current_scene_slice, (pad_left_scene, pad_right_scene, pad_top_scene, pad_bottom_scene), "constant", 0) # (Batch, 3, 1024, 1024)

        # --- FFT Convolution ---
        raw_slice_padded = convolve_rgb_Raw_Gen(psf_padded, scene_padded) # Output: (Batch, 3, 1024, 1024)

        # --- Result Cropping (512x512) ---
        center_h, center_w = fft_h // 2, fft_w // 2
        crop_h_half, crop_w_half = target_h_final // 2, target_w_final // 2
        crop_top = center_h - crop_h_half
        crop_bottom = center_h + (target_h_final - crop_h_half)
        crop_left = center_w - crop_w_half
        crop_right = center_w + (target_w_final - crop_w_half)
        raw_slice = raw_slice_padded[:, :, crop_top:crop_bottom, crop_left:crop_right] # (Batch, 3, 512, 512)

        raw_stack_list.append(raw_slice)

    # --- Stack and Sum ---
    raw_stack_tensor = torch.stack(raw_stack_list, dim=2) # (Batch, 3, 21, 512, 512)
    raw_sum = torch.sum(raw_stack_tensor, dim=2) # (Batch, 3, 512, 512)

    raw_final = raw_sum
    return raw_final, None


def npz_loader(path):
    try:
        data = np.load(path)
        key = 'label' if 'label' in data else ('depth' if 'depth' in data else ('dmap' if 'dmap' in data else None))
        if key is None: raise KeyError("No 'label', 'depth', or 'dmap' key")
        # Return NumPy array directly, convert to tensor later
        return data[key].astype(np.float32)
    except Exception as e:
        print(f"Error loading npz {path}: {e}")
        return None

# def add_gaussian_noise(image, mean=None, std_dev_min = None, std_dev_max = None):
#     gaussian_noise = np.random.normal(mean, np.random.uniform(std_dev_min, std_dev_max), image.shape)
#     # print("Max and min of image: ", np.max(image), np.min(image))
#     # print("Max and min of noise: ", np.max(gaussian_noise), np.min(gaussian_noise))
#     # Add the Gaussian noise to the image
#     # noisy_image = image + gaussian_noise + bg
#     noisy_image = image + gaussian_noise
#     # print("Max and min of noisy image: ", np.max(noisy_image), np.min(noisy_image))
#     # Convert back to uint8
#     # noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
#     # print("Max and min of noisy image after clip: ", np.max(noisy_image), np.min(noisy_image))  
#     return noisy_image

# --- Noise and SNR Functions ---
def add_gaussian_noise(image_tensor, std_dev_min=0, std_dev_max=0.01):
    noise_std_dev = np.random.uniform(std_dev_min, std_dev_max)
    noise = torch.randn_like(image_tensor) * noise_std_dev
    return image_tensor + noise

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

    # quantize_num = len(args.psf_idx_list) #여기 수정해야될거 같애 21개로.
    quantize_num = 21 # explicit하게
    print(f"Using quantize_num = {quantize_num}")

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
        target_h = image.shape[-2] # 
        target_w = image.shape[-1] # 
        # Resize bg_image to match gaussian_stack using interpolate
        # (align_corners=False is usually recommended for general resizing)
        if bg_image.shape[-2:] != (target_h, target_w):
            print(f"Resizing bg_image from {bg_image.shape[-2:]} to {(target_h, target_w)}")
            bg_image = F.interpolate(bg_image, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # bg_image = bg_image * gaussian_stack.squeeze(2)
        scene_stack, image_masked = quantize_rgb_by_depth(quantize_num, image, bg_image, label)
        raw_simulated, RGB_sum_crop = simul_raw_generation_torch(quantize_num, psf_stack, scene_stack)
        # raw_simulated is not max normalized
        [b, c, w, h] = image.shape
        
        # Save results (iterate through batch)
        for i in range(image.size(0)):
            idx = str(batch_index * HPARAMS["BATCH_SIZE"] + i).zfill(5) # Use batch_index

            # --- Convert Tensors to Numpy for saving ---
            # Raw image (already calculated as raw_noisy_png)
            # print('data type : raw_simulated: ',raw_simulated.dtype) # torch.float32
            raw_simulated_norm = (raw_simulated[i] / torch.max(raw_simulated[i])).squeeze().permute(1, 2, 0).cpu().numpy()
            # print('data type : raw_simulated_norm: ',raw_simulated_norm.dtype) # float32
            raw_noisy = add_gaussian_noise(raw_simulated_norm, mean, std_dev_min, std_dev_max)
            # print('data type : raw_noisy :', raw_noisy.dtype) # float64
            # raw_noisy_np = raw_noisy[i].cpu().detach().numpy().transpose(1, 2, 0) # For SNR calc & saving prep

            # Original GT RGB Image
            img_np = image[i].cpu().detach().numpy().transpose(1, 2, 0) # Use the input 'image' directly

            # Original GT Label (Depth)
            lbl_np = label[i].cpu().detach().numpy() # Use the input 'label' directly

            # Clean Raw Image (for SNR calculation)
            raw_clean_np = raw_simulated[i].cpu().detach().numpy().transpose(1, 2, 0)

            # Calculate SNR
            snr = calculate_snr(raw_clean_np, raw_noisy)

            # --- Prepare for saving (Scale 0-1 to 0-255, change type) ---
            # Raw noisy image for saving as PNG
            raw_to_save = np.clip(raw_noisy * 255, 0, 255).astype(np.uint8)

            # Original GT RGB image for saving as PNG
            img_to_save = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            # Label (lbl_np) is already numpy array, save as is in npz


            # --- Define Save Paths (Make sure these are correct) ---
            save_raw_path = os.path.join(args.save_path, 'train', 'raw')
            save_img_path = os.path.join(args.save_path, 'train', 'image')
            save_lbl_path = os.path.join(args.save_path, 'train', 'label')
            # Create directories if they don't exist (should be done before loop ideally)
            os.makedirs(save_raw_path, exist_ok=True)
            os.makedirs(save_img_path, exist_ok=True)
            os.makedirs(save_lbl_path, exist_ok=True)


            # --- Save the files ---
            try:
                # 1. Save Raw Image (PNG)
                # Note: cv2 reads/writes BGR by default. imageio handles RGB better.
                # Using imageio assuming RGB format internally
                imageio.imwrite(os.path.join(save_raw_path, f'{idx}_{snr:.2f}dB.png'), raw_to_save)

                # 2. Save Original GT RGB Image (PNG) - CORRECTED
                imageio.imwrite(os.path.join(save_img_path, f'{idx}.png'), img_to_save) # Use img_to_save

                # 3. Save Original GT Label (NPZ) - ADDED
                np.savez_compressed(os.path.join(save_lbl_path, f'{idx}.npz'), label=lbl_np) # Use key 'label'

            except Exception as e:
                 print(f"Error saving file for index {idx}: {e}")

            # file_counter += 1 # Not needed if using batch_index calculation for idx
        # for i in range(image.size(0)):
        #     idx = str(batch_index * HPARAMS["BATCH_SIZE"] + i).zfill(5)
        #     raw_simulated_norm = (raw_simulated[i] / torch.max(raw_simulated[i])).squeeze().permute(1, 2, 0).cpu().numpy()
        #     raw_png = cv2.cvtColor(np.uint8(raw_simulated_norm * 255), cv2.COLOR_BGR2RGB)
        #     raw_noisy = add_gaussian_noise(raw_simulated_norm, mean, std_dev_min, std_dev_max)
        #     snr = calculate_snr(raw_simulated_norm, raw_noisy)
        #     raw_noisy_png = (255*raw_noisy/np.max(raw_noisy))
        #     raw_noisy_png = np.clip(raw_noisy_png, 0, 255).astype(np.uint8)
        #     # save
        #     # np.save(os.path.join(save_dir_raw_npy, '{}.npy'.format(idx)), raw_simulated_norm)
        #     # cv2.imwrite(os.path.join(save_dir_raw_png, '{}.png'.format(idx)),raw_png) # raw wo noise
        #     # np.save(os.path.join(save_dir_raw_noisy_npy, '{}_{}dB.npy'.format(idx,snr.round(2))), raw_noisy)    
            
        #     image_masked_png =image_masked[i].squeeze().permute(1, 2, 0).cpu().numpy()
        #     image_masked_png = image_masked_png[w//4:3*w//4,h//4:3*h//4,:] # center crop 
        #     # print("Shape of image_masked_png: ", image_masked_png.shape)
        #     image_masked_png = cv2.cvtColor(np.uint8(image_masked_png * 255), cv2.COLOR_BGR2RGB)
        #     image_masked_png = np.clip(image_masked_png, 0, 255).astype(np.uint8)
            
        #     cv2.imwrite(os.path.join(save_dir_raw_noisy_png, '{}_{}dB.png'.format(idx, snr.round(2))),cv2.cvtColor(raw_noisy_png, cv2.COLOR_BGR2RGB))
        #     cv2.imwrite(os.path.join(save_dir_image_png, '{}.png'.format(idx)), image_masked_png)
            
        #     # cv2.imwrite(os.path.join(save_dir_image_cc_sum, '{}.png'.format(idx)), cv2.cvtColor(np.uint8(RGB_sum_crop[i].squeeze().permute(1, 2, 0).cpu().numpy() * 255), cv2.COLOR_BGR2RGB))
        #     # cv2.imwrite(os.path.join(save_dir_noisy, '{}.png'.format(idx)),raw_noisy)
        #     # cv2.imwrite(os.path.join(save_dir_bg, '{}.png'.format(idx)), cv2.cvtColor(np.uint8(scene_stack[i,:,-1,:,:].squeeze().permute(1, 2, 0).cpu().numpy() * 255), cv2.COLOR_BGR2RGB))
        #     # cv2.imwrite(os.path.join(save_dir_image_sum, '{}.png'.format(idx)), cv2.cvtColor(np.uint8(RGB_sum[i].squeeze().permute(1, 2, 0).cpu().numpy() * 255), cv2.COLOR_BGR2RGB))
            
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
    num_images_to_process = 10000
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
#/home/hotdog/ssd1/depth_imaging/dataset_ssd2/HJA_1/train