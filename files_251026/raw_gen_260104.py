import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
import numpy as np
from glob import glob
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import imageio
import time
import random
import shutil
from datetime import datetime

# --- 1. 하이퍼파라미터 및 경로 설정 ---
HPARAMS = {
    'IMAGE_PATH': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/image/0',
    'LABEL_PATH': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/label/0',
    'BG_PATH': '/home/hjahn/mnt/ssd1/depth_imaging/dataset_ssd1/mirflickr25k/',
    'PSF_DIR': "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/251223_HJC/gray_center_psf/",
    'SAVE_PATH': "/home/hjahn/mnt/nas/Research/HJA/",
    'BATCH_SIZE': 16, # GPU 메모리 24GB 고려하여 상향 조정
    'NUM_WORKERS': 16, # CPU 병렬 처리 활용
    'SCENE_SIZE': (576, 1024),
    'FFT_SIZE': (1152, 2048),
    'QUANTIZE_NUM': 51,
}

timestamp = datetime.now().strftime("%m%d_%H%M%S")
HPARAMS['SAVE_PATH'] = os.path.join(HPARAMS['SAVE_PATH'], timestamp)

# 51개 PSF 리스트 (05p0 ~ 10p0)
PSF_FILE_LIST = [f"{str(d).zfill(2)}p{p}.png" for d in range(5, 10) for p in range(10)] + ["10p0.png"]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. PSF 로드 및 정규화 ---
def load_psf_stack(psf_dir, file_list, target_size, device='cuda'):
    psf_stack = []
    for fname in file_list:
        path = os.path.join(psf_dir, fname)
        psf_img = Image.open(path).convert('L')
        psf_tensor = torch.from_numpy(np.array(psf_img).astype(np.float32)).to(device)
        if psf_tensor.shape != target_size:
            psf_tensor = F.interpolate(psf_tensor.unsqueeze(0).unsqueeze(0), 
                                     size=target_size, mode='bilinear', align_corners=False).squeeze()
        psf_tensor /= (torch.sum(psf_tensor) + 1e-8)
        psf_stack.append(psf_tensor)
    return torch.stack(psf_stack).unsqueeze(1)

# [cite_start]--- 3. Forward Model (FlatNet 스타일: 커널 곱하기 삭제) [cite: 1] ---
def quantize_and_simulate(image, label, bg_batch, psf_stack, hparams, device='cuda'):
    B, C, H, W = image.shape
    D = hparams['QUANTIZE_NUM']
    fft_h, fft_w = hparams['FFT_SIZE']
    pad_h, pad_w = H // 2, W // 2
    
    depth_inverted = torch.clamp(1.0 - label.unsqueeze(1), 0, 1) 
    raw_accum = torch.zeros((B, C, H, W), device=device)
    thresholds = torch.linspace(0.0, 1.0, D + 1, device=device)

    for i in range(D):
        if i == D - 1: mask = (depth_inverted >= thresholds[i]) & (depth_inverted <= thresholds[i+1])
        else: mask = (depth_inverted >= thresholds[i]) & (depth_inverted < thresholds[i+1])
        
        curr_scene = image * mask.float()
        
        if i == D - 1: # 배경 합성
            bg_mask = (label.unsqueeze(1) == 0).float()
            curr_scene += bg_batch * bg_mask
            
        if torch.sum(mask) == 0 and i != D - 1: continue

        # Padding (Linear Convolution 모사)
        scene_padded = F.pad(curr_scene, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        psf_padded = F.pad(psf_stack[i:i+1], (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        # FFT Convolution
        scene_fft = torch.fft.rfft2(scene_padded)
        psf_fft = torch.fft.rfft2(psf_padded, s=(fft_h, fft_w))
        res_spatial = torch.fft.irfft2(scene_fft * psf_fft, s=(fft_h, fft_w))
        res_spatial = torch.fft.ifftshift(res_spatial, dim=(-2, -1))
        
        raw_accum += res_spatial[:, :, pad_h:pad_h+H, pad_w:pad_w+W]
        
    return torch.clamp(raw_accum, 0, 1)

# --- 4. 데이터셋 정의 (배경 로딩 포함) ---
class SceneDataset(Dataset):
    def __init__(self, img_paths, lbl_paths, bg_paths, scene_size):
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths
        self.bg_paths = bg_paths
        self.transform = transforms.ToTensor()
        # 배경 전처리를 미리 정의하여 일꾼들이 수행하게 함
        self.bg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(scene_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        # Scene & Label
        img = self.transform(Image.open(self.img_paths[idx]).convert('RGB'))
        lbl = torch.from_numpy(np.load(self.lbl_paths[idx])['dmap']).float()
        
        # 배경 랜덤 로드 및 전처리 (일꾼들이 미리 처리)
        bg_path = random.choice(self.bg_paths)
        bg_img = self.bg_transform(Image.open(bg_path).convert('RGB')) * 0.2
        
        return img, lbl, bg_img, self.img_paths[idx], self.lbl_paths[idx]

def main():
    # 경로 초기화
    sub_folders = ['raw/0', 'image/0', 'label/0']
    for sub in sub_folders:
        os.makedirs(os.path.join(HPARAMS['SAVE_PATH'], sub), exist_ok=True)

    # 초기 준비
    psf_stack = load_psf_stack(HPARAMS['PSF_DIR'], PSF_FILE_LIST, HPARAMS['SCENE_SIZE'], DEVICE)
    
    img_files = sorted(glob(os.path.join(HPARAMS['IMAGE_PATH'], '*.png')))
    lbl_files = sorted(glob(os.path.join(HPARAMS['LABEL_PATH'], '*.npz')))
    bg_files = sorted(glob(os.path.join(HPARAMS['BG_PATH'], '*.jpg')))

    dataset = SceneDataset(img_files, lbl_files, bg_files, HPARAMS['SCENE_SIZE'])
    # pin_memory=True는 CPU 재료를 GPU 요리판으로 빠르게 옮겨줌
    loader = DataLoader(dataset, batch_size=HPARAMS['BATCH_SIZE'], 
                        num_workers=HPARAMS['NUM_WORKERS'], shuffle=False, pin_memory=True)

    print(f"Starting simulation for {len(dataset)} scenes...")
    
    for i, (image_batch, label_batch, bg_batch, img_paths, lbl_paths) in enumerate(tqdm(loader, desc="Raw Generation", unit="batch")):
        image_batch = image_batch.to(DEVICE)
        label_batch = label_batch.to(DEVICE)
        bg_batch = bg_batch.to(DEVICE)

        # 1. Forward Simulation (배치 단위 처리)
        raw_img = quantize_and_simulate(image_batch, label_batch, bg_batch, psf_stack, HPARAMS, DEVICE)
        
        # 2. 정규화 및 노이즈 추가
        for b in range(image_batch.shape[0]):
            curr_raw = raw_img[b:b+1]
            raw_max = torch.max(curr_raw)
            if raw_max > 0:
                curr_raw = curr_raw / raw_max
            
            # 랜덤 노이즈 (0.01~0.03)
            raw_noisy = torch.clamp(curr_raw + torch.randn_like(curr_raw) * random.uniform(0.01, 0.03), 0, 1)
            
            # 저장
            global_idx = i * HPARAMS['BATCH_SIZE'] + b
            idx_str = str(global_idx).zfill(5)
            
            raw_np = (raw_noisy[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(HPARAMS['SAVE_PATH'], 'raw/0', f'{idx_str}.png'), raw_np)
            shutil.copy(img_paths[b], os.path.join(HPARAMS['SAVE_PATH'], 'image/0', f'{idx_str}.png'))
            shutil.copy(lbl_paths[b], os.path.join(HPARAMS['SAVE_PATH'], 'label/0', f'{idx_str}.npz'))

    print("Generation complete!")

if __name__ == "__main__":
    main()