import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Or your desired GPU ID
import datetime
import torch
import cv2
import matplotlib.pyplot as plt
# import matplotlib.cm as cm # Unused
# import seaborn as sns # Unused
import numpy as np
from glob import glob
# from utils_Torch import * # Assuming convolve_rgb_Raw_Gen is defined below
# from utils import * # Assuming npz_loader is defined below
from scipy import io
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode # Keep if needed for background resizing

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
# import concurrent.futures # Not used in the final version
# from concurrent.futures import ThreadPoolExecutor # Not used
import torch.nn as nn
import torch.nn.functional as F
# import imageio.v2 as imageio # Use imageio instead of cv2 for saving
import imageio
import argparse
import time
import random

# --- Argument Parser (Simplified) ---
# parser = argparse.ArgumentParser(description="Raw image generator for lensless camera")
# parser.add_argument('--psf_path', type=str, required=True, help='Path to the .mat PSF stack file (21x512x512)')
# parser.add_argument('--save_path', type=str, required=True, help='Path to save the generated raw data')
# parser.add_argument('--data_idx', nargs=2, type=int, required=False, default=[0, 10000], help='Start and end index (exclusive) of data to process (e.g., 0 500)')

# args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(DEVICE))

# --- HPARAMS (Simplified Paths) ---
HPARAMS = {
    'IMAGE_PATH': '/home/hotdog/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/image/0', # 576x1024 Scene PNGs
    'LABEL_PATH': '/home/hotdog/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/label/0', # 576x1024 Depth NPZs
    'BG_PATH': '/home/hotdog/mnt/ssd1/depth_imaging/dataset_ssd1/mirflickr25k/',
    'PSF_DIR': "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/251223_HJC/gray_center_psf/",
    'SAVE_PATH': "/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/syn_raw_image/",
    'BATCH_SIZE': 1, # Start with 1 due to large FFT size
    'NUM_WORKERS': 0, # Start with 0 for easier debugging
    'SCENE_SIZE': (576,1024),
    'FFT_SIZE': (1152, 2048),
    'QUANTIZE_NUM': 20,
}
background_level_max = 0.5 # Max background intensity relative to object

TPARAMS = {}

PSF_FILE_LIST = [ # 20개
    "05p0.png", "05p2.png", "05p4.png", "05p6.png", "05p8.png",
    "06p0.png", "06p1.png", "06p4.png", "06p6.png", "06p8.png",
    "07p0.png", "07p2.png", "07p4.png", "07p8.png", "08p1.png",
    "08p3.png", "08p7.png", "09p0.png", "09p4.png", "09p9.png",
]

# 4. Super-Gaussian Mask 생성 함수 (Torch 버전)
def get_super_gaussian_mask(shape, sigma=50, order=4, device='cuda'):
    Hp, Wp = shape
    y = torch.linspace(-1, 1, Hp, device=device)
    x = torch.linspace(-1, 1, Wp, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    aspect_ratio = Wp / Hp
    X_scaled = X / (sigma / 100.0)
    Y_scaled = Y * aspect_ratio / (sigma / 100.0)

    R2 = X_scaled**2 + Y_scaled**2
    mask = torch.exp(-(R2**(order / 2)))
    return mask / mask.max()

# --- PSF 로더 (개별 PNG 로드 및 정규화) ---
def load_psf_stack(psf_dir, file_list, target_size, device='cuda'):
    psf_stack = []
    for fname in file_list:
        path = os.path.join(psf_dir, fname)
        psf_img = Image.open(path).convert('L') # Monochromatic
        psf_np = np.array(psf_img).astype(np.float32)
        psf_tensor = torch.from_numpy(psf_np).to(device)
        
        # 크기가 다를 경우 리사이즈
        if psf_tensor.shape != target_size:
            psf_tensor = F.interpolate(psf_tensor.unsqueeze(0).unsqueeze(0), 
                                     size=target_size, mode='bilinear').squeeze()
            
        # 에너지 정규화 (중요)
        psf_tensor /= torch.sum(psf_tensor)
        psf_stack.append(psf_tensor)
        
    return torch.stack(psf_stack).unsqueeze(1) # (20, 1, H, W)

# --- 수정된 Forward Model (Super-Gaussian 적용) ---
def simul_raw_generation_torch_v2(psf_stack, scene_stack, hparams, device='cuda'):
    # psf_stack: (20, 1, 576, 1024), scene_stack: (B, 3, 20, 576, 1024)
    B, C, D, H, W = scene_stack.shape
    fft_h, fft_w = hparams['FFT_SIZE']
    
    # Super-Gaussian Mask 생성
    fade_mask = get_super_gaussian_mask((fft_h, fft_w), sigma=50, order=4, device=device)
    
    raw_accum = torch.zeros((B, C, H, W), device=device)
    
    for d in range(D):
        curr_scene = scene_stack[:, :, d, :, :] # (B, 3, 576, 1024)
        curr_psf = psf_stack[d, :, :, :]       # (1, 576, 1024)
        
        # 1. Replicate Padding
        # 상하좌우 각각 50%씩 패딩하여 2배 크기로 만듦 (Linear Conv 모사)
        pad_h, pad_w = H // 2, W // 2
        scene_padded = F.pad(curr_scene, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        psf_padded = F.pad(curr_psf.unsqueeze(0), (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        
        # 2. Super-Gaussian Fade 적용 (Boundary Artifact 억제)
        # 사용자님의 'recon.py' 전략: 양쪽에 모두 마스크를 씌워 줌
        scene_faded = scene_padded * fade_mask
        psf_faded = psf_padded * fade_mask
        
        # 3. FFT Convolution
        scene_fft = torch.fft.rfft2(scene_faded)
        psf_fft = torch.fft.rfft2(psf_faded, s=(fft_h, fft_w))
        
        raw_fft = scene_fft * psf_fft
        raw_faded = torch.fft.irfft2(raw_fft, s=(fft_h, fft_w))
        raw_faded = torch.fft.ifftshift(raw_faded, dim=(-2, -1))
        
        # 4. Center Crop (576, 1024)
        raw_cropped = raw_faded[:, :, pad_h:pad_h+H, pad_w:pad_w+W]
        raw_accum += raw_cropped
        
    return torch.clamp(raw_accum, 0, 1)

#--------------------------------------------

#---below is the original code----
# --- Quantization Setup ---
quantize_num = 20 # Explicitly set based on the new PSF stack
print(f"Using quantize_num = {quantize_num}")
# Create thresholds for quantization (0.0 to 1.0 in quantize_num+1 steps)
# These will be used inside quantize_rgb_by_depth
quantization_thresholds = torch.linspace(0.0, 1.0, quantize_num + 1, device=DEVICE).float()


# --- Background image list ---
bg_train = sorted(glob(HPARAMS['BG_PATH']+'/*.jpg', recursive=True))
print('Number of background images: {}'.format(len(bg_train)))

# --- PSF Loading and Processing (Corrected) ---
print(f"Loading PSF stack: {args.psf_path}")
psf_mat_file = io.loadmat(args.psf_path)
# *** IMPORTANT: Check the actual key name in your .mat file! ***
psf_key = 'images' # Or 'psf_stack', 'psf_stack_resized', etc.
if psf_key not in psf_mat_file:
    raise ValueError(f"Key '{psf_key}' not found in {args.psf_path}. Check .mat file structure.")
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

# --- Noise Parameters ---
mean = 0
std_dev_min = 0
std_dev_max = 0.01

# --- Helper Functions ---

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

# --- simul_raw_generation_torch (Padding/Cropping Adjusted for 512x512) ---
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

# --- Noise and SNR Functions ---
def add_gaussian_noise(image_tensor, std_dev_min=0, std_dev_max=0.01):
    noise_std_dev = random.uniform(std_dev_min, std_dev_max)
    noise = torch.randn_like(image_tensor) * noise_std_dev
    return image_tensor + noise

def calculate_snr(original_image_np, noisy_image_np):
    signal_power = np.sum(original_image_np ** 2)
    noise = original_image_np - noisy_image_np
    noise_power = np.sum(noise ** 2)
    if noise_power < 1e-10: return float('inf') # Avoid division by zero/very small noise
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# --- Custom Dataset (Using glob results) ---
class CustomImageDataset(Dataset):
    def __init__(self, image_files, label_files, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform
        if len(image_files) != len(label_files):
            raise ValueError("Mismatch between image and label file counts!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            lbl_path = self.label_files[idx]

            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image).float()

            label = npz_loader(lbl_path) # Returns numpy array or None
            if label is None: return None, None # Handle loading error

            label = torch.from_numpy(label).float() # Convert numpy to tensor
            if label.dim() != 2: raise ValueError("Label is not 2D")

            return image, label
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            print(f"Paths: {self.image_files[idx]}, {self.label_files[idx]}")
            return None, None

# --- train_batch Function (Simplified, uses quantize_num=21) ---
def train_batch(train_parameters, trainset_loader):
    global psf_stack # Use global processed psf_stack
    # quantize_num = len(args.psf_idx_list) # No longer use args
    quantize_num_local = quantize_num # Use the global variable (21)

    # --- Save Path Setup ---
    save_raw_path = os.path.join(HPARAMS['SAVE_PATH'], 'train', 'raw')
    save_img_path = os.path.join(HPARAMS['SAVE_PATH'], 'train', 'image')
    save_lbl_path = os.path.join(HPARAMS['SAVE_PATH'], 'train', 'label')
    os.makedirs(save_raw_path, exist_ok=True)
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    file_counter = args.data_idx[0] # Start numbering from start index

    for batch_idx, data in enumerate(tqdm(trainset_loader, desc="Processing Batches", unit="batch", dynamic_ncols=True)):
        if not data or data[0] is None or data[1] is None:
             print(f"Skipping batch {batch_idx} due to data loading error.")
             continue

        image, label = data # image: (B, 3, 512, 512), label: (B, 512, 512)
        image, label = image.float().to(DEVICE), label.float().to(DEVICE)

        print(f"Batch {batch_idx}: Image shape: {image.shape}, Label shape: {label.shape}")
        if label.dim() != 3 or label.shape[0] != HPARAMS['BATCH_SIZE']:
            print(f"Skipping batch {batch_idx}: Unexpected label shape {label.shape}. Expected ({HPARAMS['BATCH_SIZE']}, H, W)")
            continue

        label[label < 0] = 0
        batch_size = image.shape[0]

        # --- Load and Prepare Background Image ---
        bg_image_batch = []
        for i in range(batch_size):
            try:
                bg_idx = random.choice(range(len(bg_train)))
                bg_tmp = Image.open(bg_train[bg_idx]).convert('RGB')
                transformer_bg = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((image.shape[2], image.shape[3]), interpolation=InterpolationMode.BILINEAR, antialias=True)
                ])
                bg_tmp_tensor = transformer_bg(bg_tmp).float().to(DEVICE)
                bg_tmp_tensor[bg_tmp_tensor < 0] = 0
                bg_tmp_tensor = bg_tmp_tensor / (torch.max(bg_tmp_tensor) + 1e-6)
                bg_tmp_tensor = bg_tmp_tensor * background_level_max * random.random()
                bg_image_batch.append(bg_tmp_tensor)
            except Exception as e:
                print(f"Error loading background: {e}, using zeros.")
                bg_image_batch.append(torch.zeros_like(image[i]))
        bg_image = torch.stack(bg_image_batch) # (B, 3, 512, 512)

        # Quantize (uses quantize_num_local = 21)
        scene_stack, image_masked = quantize_rgb_by_depth(quantize_num_local, image, bg_image, label)

        # Generate raw image (uses quantize_num_local = 21)
        raw_simulated, _ = simul_raw_generation_torch(quantize_num_local, psf_stack, scene_stack)

        # Add Gaussian noise
        raw_noisy = add_gaussian_noise(raw_simulated, std_dev_min, std_dev_max)

        # Clip
        raw_image_final = torch.clamp(raw_noisy, 0.0, 1.0)

        # --- Save results ---
        for i in range(batch_size):
            idx = str(file_counter).zfill(5)
            try:
                raw_np = raw_image_final[i].cpu().detach().numpy().transpose(1, 2, 0)
                img_np = image[i].cpu().detach().numpy().transpose(1, 2, 0)
                lbl_np = label[i].cpu().detach().numpy()
                raw_clean_np = raw_simulated[i].cpu().detach().numpy().transpose(1, 2, 0)
                raw_noisy_np_for_snr = raw_noisy[i].cpu().detach().numpy().transpose(1, 2, 0)

                snr = calculate_snr(raw_clean_np, raw_noisy_np_for_snr)

                raw_to_save = np.clip(raw_np * 255, 0, 255).astype(np.uint8)
                img_to_save = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                imageio.imwrite(os.path.join(save_raw_path, f'{idx}_{snr:.2f}dB.png'), raw_to_save)
                imageio.imwrite(os.path.join(save_img_path, f'{idx}.png'), img_to_save)
                np.savez_compressed(os.path.join(save_lbl_path, f'{idx}.npz'), label=lbl_np)
            except Exception as e:
                 print(f"Error saving file for index {idx}: {e}")
            file_counter += 1


# --- main Function (Using glob and Custom Dataset) ---
def main():
    # Transformer for GT images
    transformer = transforms.Compose([transforms.ToTensor()])

    # --- Get file lists ---
    image_files = sorted(glob(os.path.join(HPARAMS['IMAGE_PATH'], '*.png')))
    label_files = sorted(glob(os.path.join(HPARAMS['LABEL_PATH'], '*.npz')))

    if not image_files: raise FileNotFoundError(f"No .png in {HPARAMS['IMAGE_PATH']}")
    if not label_files: raise FileNotFoundError(f"No .npz in {HPARAMS['LABEL_PATH']}")
    total_data_size = len(image_files)
    print(f'Total dataset size: {total_data_size}')
    if len(image_files) != len(label_files): raise ValueError("Image/Label count mismatch.")

    # --- Determine indices to process ---
    data_idx_start = args.data_idx[0]
    data_idx_end = args.data_idx[1]
    if data_idx_end > total_data_size: data_idx_end = total_data_size
    if data_idx_start >= data_idx_end: raise ValueError("data_idx start >= end")
    num_images_to_process = data_idx_end - data_idx_start

    image_files_subset = image_files[data_idx_start:data_idx_end]
    label_files_subset = label_files[data_idx_start:data_idx_end]
    print(f'Processing images {data_idx_start} to {data_idx_end-1} ({num_images_to_process} images).')

    # --- Create Custom Dataset ---
    train_load = CustomImageDataset(image_files_subset, label_files_subset, transform=transformer)

    # --- DataLoader Setup ---
    def collate_fn_filter_none(batch):
        batch = [sample for sample in batch if sample is not None and all(s is not None for s in sample)]
        if not batch: return None
        try: return torch.utils.data.dataloader.default_collate(batch)
        except RuntimeError as e: print(f"Error during collate_fn: {e}"); return None

    TPARAMS['trainset_loader'] = DataLoader(
        train_load,
        batch_size=HPARAMS['BATCH_SIZE'], # Should be 1
        shuffle=False, # No need to shuffle for generation
        num_workers=HPARAMS['NUM_WORKERS'], # Should be 0
        pin_memory=False, # Keep false if NUM_WORKERS is 0
        collate_fn=collate_fn_filter_none
    )

    print('Raw image simulation start!')
    tot_time = time.time()
    train_batch(TPARAMS, TPARAMS['trainset_loader'])
    total_time = time.time() - tot_time
    print(f'Total Elapsed : {total_time/60:.2f} min for {num_images_to_process} images')

if __name__ == "__main__":
    main()