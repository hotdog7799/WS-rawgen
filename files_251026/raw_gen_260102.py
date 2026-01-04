import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import imageio
import time
import random
import shutil
from datetime import datetime

# --- 1. 하이퍼파라미터 및 경로 설정 (사용자 환경에 맞게 수정) ---
HPARAMS = {
    'IMAGE_PATH': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/image/0', # 576x1024 Scene PNGs
    'LABEL_PATH': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/label/0', # 576x1024 Depth NPZs
    'BG_PATH': '/home/hjahn/mnt/ssd1/depth_imaging/dataset_ssd1/mirflickr25k/',
    'PSF_DIR': "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/251223_HJC/gray_center_psf/",
    'SAVE_PATH': "/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/syn_raw_image/",
    'BATCH_SIZE': 4, # Start with 1 due to large FFT size
    'NUM_WORKERS': 8, # Start with 0 for easier debugging
    'SCENE_SIZE': (576,1024),
    'FFT_SIZE': (1152, 2048),
    'QUANTIZE_NUM': 20,
}

# 현재 시간 → YYYYMMDD_HHMM 형식으로 문자열 생성
timestamp = datetime.now().strftime("%Y%m%d_%HH%MM")
# SAVE_PATH 뒤에 붙이기
HPARAMS['SAVE_PATH'] = os.path.join(HPARAMS['SAVE_PATH'], timestamp)

# 유사도 기반으로 선정하신 20개의 PSF 파일명 리스트 (순서대로 5.0mm ~ 10.0mm 매핑)
# 리스트 순서가 뎁스 0(가까움)에서 1(멀음)로 대응됩니다.
PSF_FILE_LIST = [ # 20개
    "05p0.png", "05p2.png", "05p4.png", "05p6.png", "05p8.png",
    "06p0.png", "06p1.png", "06p4.png", "06p6.png", "06p8.png",
    "07p0.png", "07p2.png", "07p4.png", "07p8.png", "08p1.png",
    "08p3.png", "08p7.png", "09p0.png", "09p4.png", "09p9.png",
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Super-Gaussian Mask (recon.py 로직 기반) ---
def get_super_gaussian_mask(shape, sigma=50, order=2, device='cuda'):
    Hp, Wp = shape
    y = torch.linspace(-1, 1, Hp, device=device)
    x = torch.linspace(-1, 1, Wp, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    aspect_ratio = Wp / Hp
    X_scaled = X / (sigma / 100.0)
    Y_scaled = Y * aspect_ratio / (sigma / 100.0)
    R2 = X_scaled**2 + Y_scaled**2
    mask = torch.exp(-(R2**(order / 2)))
    return (mask / mask.max()).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

# --- 3. PSF 로드 및 정규화 (에너지 보존) ---
def load_psf_stack(psf_dir, file_list, target_size, device='cuda'):
    print("loading psf stack")
    psf_stack = []
    print(f"Loading {len(file_list)} PSFs and normalizing...")
    for fname in file_list:
        path = os.path.join(psf_dir, fname)
        psf_img = Image.open(path).convert('L')
        psf_np = np.array(psf_img).astype(np.float32)
        psf_tensor = torch.from_numpy(psf_np).to(device)
        if psf_tensor.shape != target_size:
            psf_tensor = F.interpolate(psf_tensor.unsqueeze(0).unsqueeze(0), 
                                     size=target_size, mode='bilinear', align_corners=False).squeeze()
        psf_tensor /= (torch.sum(psf_tensor) + 1e-8) # L1 Normalization 
        psf_stack.append(psf_tensor)
    # print("psf stack shape: ",psf_stack.shape)
    return torch.stack(psf_stack).unsqueeze(1) # (20, 1, H, W) 

# --- 4. Quantization & Forward Model (Linear Convolution) ---
def quantize_and_simulate(image, label, bg_img, psf_stack, hparams, fade_mask, device='cuda'):
    # print("quantize and simulation start.\nscene shape: ",image.shape)
    B, C, H, W = image.shape
    D = hparams['QUANTIZE_NUM']
    fft_h, fft_w = hparams['FFT_SIZE']
    pad_h, pad_w = H // 2, W // 2
    depth_inverted = torch.clamp(1.0 - label.unsqueeze(1), 0, 1) # Blender dmap 대응 
    raw_accum = torch.zeros((B, C, H, W), device=device)
    thresholds = torch.linspace(0.0, 1.0, D + 1, device=device)

    for i in range(D):
        if i == D - 1: mask = (depth_inverted >= thresholds[i]) & (depth_inverted <= thresholds[i+1])
        else: mask = (depth_inverted >= thresholds[i]) & (depth_inverted < thresholds[i+1])
        curr_scene = image * mask.float()
        if i == D - 1: # 배경을 가장 먼 레이어에 합성
            bg_mask = (label.unsqueeze(1) == 0).float()
            curr_scene += bg_img * bg_mask
        if torch.sum(mask) == 0 and i != D - 1: continue

        # Replicate Padding & Super-Gaussian Fade
        scene_padded = F.pad(curr_scene, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        psf_padded = F.pad(psf_stack[i:i+1], (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        scene_faded = scene_padded * fade_mask
        psf_faded = psf_padded * fade_mask

        # FFT Convolution
        scene_fft = torch.fft.rfft2(scene_faded)
        psf_fft = torch.fft.rfft2(psf_faded, s=(fft_h, fft_w))
        raw_fft = scene_fft * psf_fft
        res_spatial = torch.fft.irfft2(raw_fft, s=(fft_h, fft_w))
        res_spatial = torch.fft.ifftshift(res_spatial, dim=(-2, -1))
        
        # Center Crop (576, 1024) 
        raw_accum += res_spatial[:, :, pad_h:pad_h+H, pad_w:pad_w+W]
    return torch.clamp(raw_accum, 0, 1)

# --- 5. 데이터셋 정의 (기존 dmap 키 유지) ---
class SceneDataset(Dataset):
    def __init__(self, img_paths, lbl_paths):
        self.img_paths, self.lbl_paths = img_paths, lbl_paths
        self.transform = transforms.ToTensor()
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_paths[idx]).convert('RGB'))
        lbl = torch.from_numpy(np.load(self.lbl_paths[idx])['dmap']).float() # 
        return img, lbl, self.img_paths[idx], self.lbl_paths[idx]

def main():
    # 1. 경로 존재 여부 우선 확인
    if not os.path.exists(HPARAMS['IMAGE_PATH']):
        print(f"Error: Image path does not exist: {HPARAMS['IMAGE_PATH']}")
        return
    if not os.path.exists(HPARAMS['LABEL_PATH']):
        print(f"Error: Label path does not exist: {HPARAMS['LABEL_PATH']}")
        return
    # 경로 생성 
    sub_folders = ['raw/0', 'image/0', 'label/0']
    for sub in sub_folders:
        os.makedirs(os.path.join(HPARAMS['SAVE_PATH'], sub), exist_ok=True)

    # 초기 준비
    psf_stack = load_psf_stack(HPARAMS['PSF_DIR'], PSF_FILE_LIST, HPARAMS['SCENE_SIZE'], DEVICE)
    print("psf_stack.shape: ",psf_stack.shape)
    fade_mask = get_super_gaussian_mask(HPARAMS['FFT_SIZE'], sigma=50, order=4, device=DEVICE)
    
    img_files = sorted(glob(os.path.join(HPARAMS['IMAGE_PATH'], '*.png')))
    lbl_files = sorted(glob(os.path.join(HPARAMS['LABEL_PATH'], '*.npz')))
    bg_files = sorted(glob(os.path.join(HPARAMS['BG_PATH'], '*.jpg')))

    # # --- 테스트용: 5개만 선택 ---
    # img_files = img_files[:5]
    # lbl_files = lbl_files[:5]
    # print(f"Testing with {len(img_files)} scenes...")
    # # --- 테스트용: 5개만 선택 ---

    dataset = SceneDataset(img_files, lbl_files)
    loader = DataLoader(dataset, batch_size=HPARAMS['BATCH_SIZE'], num_workers=HPARAMS['NUM_WORKERS'], shuffle=False,pin_memory=True)

    print(f"Starting simulation for {len(dataset)} scenes...")
    
    # tqdm으로 진행 상황 및 예상 시간 확인 
    for i, (image, label, img_path, lbl_path) in enumerate(tqdm(loader, desc="Raw Generation", unit="scene")):
        image, label = image.to(DEVICE), label.to(DEVICE)
        
        # 배경 랜덤 선택 및 리사이즈
        bg_raw = Image.open(random.choice(bg_files)).convert('RGB')
        bg_img = transforms.ToTensor()(bg_raw).to(DEVICE).unsqueeze(0)
        bg_img = F.interpolate(bg_img, size=HPARAMS['SCENE_SIZE'], mode='bilinear') * 0.2

        # Forward Simulation (20개 PSF Quantization 적용)
        raw_img = quantize_and_simulate(image, label, bg_img, psf_stack, HPARAMS, fade_mask, DEVICE)
        
        # Gaussian Noise 추가 (표준편차 0.01)
        raw_noisy = torch.clamp(raw_img + torch.randn_like(raw_img) * 0.01, 0, 1)
        
        # --- [추가] Max-Normalization 로직 ---
        raw_max = torch.max(raw_noisy)
        if raw_max > 0:
            raw_noisy = raw_noisy / raw_max # 0~1 스케일링
        raw_noisy = torch.clamp(raw_noisy, 0, 1)
                    
        # 결과 저장 
        idx_str = str(i).zfill(5)
        
        # 1. Raw Image 저장 (정수형 변환)
        raw_np = (raw_noisy[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(HPARAMS['SAVE_PATH'], 'raw/0', f'{idx_str}.png'), raw_np)

        # 2. GT Image 및 Label 복사 (저장 포맷 유지) 
        shutil.copy(img_path[0], os.path.join(HPARAMS['SAVE_PATH'], 'image/0', f'{idx_str}.png'))
        shutil.copy(lbl_path[0], os.path.join(HPARAMS['SAVE_PATH'], 'label/0', f'{idx_str}.npz'))

    print("Generation complete!")

if __name__ == "__main__":
    main()