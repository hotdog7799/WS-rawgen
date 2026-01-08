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
import random
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor # 비동기 저장을 위한 라이브러리
import time

# 저장은 SSD에
# shutil 같은 copy는 나중에
# queue size 조절



# --- 1. 하이퍼파라미터 설정 ---
HPARAMS = {
    # 'IMAGE_PATH': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/image/0',
    # 'LABEL_PATH': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/label/0',
    'IMAGE_PATH': "/home/hjahn/mnt/ssd1/data/hjahn/scene_and_label/image/0",
    'LABEL_PATH': "/home/hjahn/mnt/ssd1/data/hjahn/scene_and_label/label/0",
    'BG_PATH': '/home/hjahn/mnt/nas/_datasets/mirflickr',
    'PSF_DIR': "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/251223_HJC/gray_center_psf/",
    # 'SAVE_PATH': "/home/hjahn/mnt/nas/Research/HJA/",
    'SAVE_PATH': "/home/hjahn/mnt/ssd1/data/hjahn/syn_raw_image/",
    'BATCH_SIZE': 32, 
    'NUM_WORKERS': 12, # 너무 높으면 오히려 CPU 오버헤드 발생 가능, 12~16 권장
    'SCENE_SIZE': (576, 1024),
    'FFT_SIZE': (1152, 2048),
    'QUANTIZE_NUM': 51,
}

timestamp = datetime.now().strftime("%m%d_%H%M%S")
HPARAMS['SAVE_PATH'] = os.path.join(HPARAMS['SAVE_PATH'], timestamp)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 51개 PSF 리스트 자동 생성
PSF_FILE_LIST = [f"{str(d).zfill(2)}p{p}.png" for d in range(5, 10) for p in range(10)] + ["10p0.png"]

# --- 2. PSF 로드 및 FFT 미리 계산 (속도 향상 핵심) ---
def load_psf_fft_stack(psf_dir, file_list, target_size, fft_size, device='cuda'):
    print("Pre-calculating PSF FFTs...")
    psf_fft_stack = []
    pad_h, pad_w = target_size[0] // 2, target_size[1] // 2
    
    for fname in file_list:
        path = os.path.join(psf_dir, fname)
        psf_img = Image.open(path).convert('L')
        psf_tensor = torch.from_numpy(np.array(psf_img).astype(np.float32)).to(device)
        if psf_tensor.shape != target_size:
            psf_tensor = F.interpolate(psf_tensor.unsqueeze(0).unsqueeze(0), 
                                     size=target_size, mode='bilinear').squeeze()
        psf_tensor /= (torch.sum(psf_tensor) + 1e-8)
        
        # Padding 후 FFT 미리 수행
        psf_padded = F.pad(psf_tensor.unsqueeze(0), (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        psf_fft = torch.fft.rfft2(psf_padded, s=fft_size)
        psf_fft_stack.append(psf_fft)
        
    return torch.stack(psf_fft_stack) # (51, 1, H_freq, W_freq)

# --- 3. Forward Model (최적화 버전) ---
def quantize_and_simulate_optimized(image, label, bg_batch, psf_fft_stack, hparams, device='cuda'):
    B, C, H, W = image.shape
    D = hparams['QUANTIZE_NUM']
    fft_h, fft_w = hparams['FFT_SIZE']
    pad_h, pad_w = H // 2, W // 2
    
    depth_inverted = torch.clamp(1.0 - label.unsqueeze(1), 0, 1) 
    raw_accum = torch.zeros((B, C, H, W), device=device)
    thresholds = torch.linspace(0.0, 1.0, D + 1, device=device)

    for i in range(D):
        mask = (depth_inverted >= thresholds[i]) & (depth_inverted <= thresholds[i+1]) if i == D-1 \
               else (depth_inverted >= thresholds[i]) & (depth_inverted < thresholds[i+1])
        
        curr_scene = image * mask.float()
        if i == D - 1:
            bg_mask = (label.unsqueeze(1) == 0).float()
            curr_scene += bg_batch * bg_mask
            
        if torch.sum(mask) == 0 and i != D - 1: continue

        # Scene Padding & FFT
        scene_padded = F.pad(curr_scene, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        scene_fft = torch.fft.rfft2(scene_padded)
        
        # 미리 계산된 PSF FFT 사용 (이 부분에서 매 루프당 FFT 1번씩 절약)
        res_spatial = torch.fft.irfft2(scene_fft * psf_fft_stack[i], s=(fft_h, fft_w))
        res_spatial = torch.fft.ifftshift(res_spatial, dim=(-2, -1))
        
        raw_accum += res_spatial[:, :, pad_h:pad_h+H, pad_w:pad_w+W]
        
    return torch.clamp(raw_accum, 0, 1)

# --- 4. 데이터셋 정의 ---
class SceneDataset(Dataset):
    def __init__(self, img_paths, lbl_paths, bg_paths, scene_size):
        self.img_paths, self.lbl_paths, self.bg_paths = img_paths, lbl_paths, bg_paths
        self.transform = transforms.ToTensor()
        self.bg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(scene_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_paths[idx]).convert('RGB'))
        lbl = torch.from_numpy(np.load(self.lbl_paths[idx])['dmap']).float()
        bg_img = self.bg_transform(Image.open(random.choice(self.bg_paths)).convert('RGB')) * 0.2
        return img, lbl, bg_img, self.img_paths[idx], self.lbl_paths[idx]

# --- 5. 비동기 저장을 위한 함수 ---
def save_worker(idx_str, raw_np, img_src, lbl_src, save_path):
    # 1. Raw 저장
    imageio.imwrite(os.path.join(save_path, 'raw/0', f'{idx_str}.png'), raw_np)
    # 2. GT 및 Label 복사
    # shutil.copy(img_src, os.path.join(save_path, 'image/0', f'{idx_str}.png'))
    # shutil.copy(lbl_src, os.path.join(save_path, 'label/0', f'{idx_str}.npz'))

def main():
    # 폴더 생성
    for sub in ['raw/0', 'image/0', 'label/0']:
        os.makedirs(os.path.join(HPARAMS['SAVE_PATH'], sub), exist_ok=True)

    # PSF FFT 미리 준비
    psf_fft_stack = load_psf_fft_stack(HPARAMS['PSF_DIR'], PSF_FILE_LIST, HPARAMS['SCENE_SIZE'], HPARAMS['FFT_SIZE'], DEVICE)
    
    img_files = sorted(glob(os.path.join(HPARAMS['IMAGE_PATH'], '*.png')))
    lbl_files = sorted(glob(os.path.join(HPARAMS['LABEL_PATH'], '*.npz')))
    bg_files = sorted(glob(os.path.join(HPARAMS['BG_PATH'], '*.jpg')))

    dataset = SceneDataset(img_files, lbl_files, bg_files, HPARAMS['SCENE_SIZE'])
    loader = DataLoader(dataset,
        batch_size=HPARAMS['BATCH_SIZE'],
        num_workers=HPARAMS['NUM_WORKERS'], 
        shuffle=False, 
        pin_memory=True, 
        prefetch_factor=4)

    # 비동기 처리를 위한 스레드풀 (8개 스레드가 저장 담당)
    executor = ThreadPoolExecutor(max_workers=16)
    
    # [수정] 대기열 모니터링을 위한 리스트
    futures = []
    MAX_QUEUE_SIZE = 250 # 최대 100개 이미지까지만 대기 허용 (메모리 상황에 따라 조절)

    print(f"Starting optimized simulation for {len(dataset)} scenes...")
    
    for i, (image_batch, label_batch, bg_batch, img_paths, lbl_paths) in enumerate(tqdm(loader, desc="Raw Gen", unit="batch")):
        image_batch, label_batch, bg_batch = image_batch.to(DEVICE), label_batch.to(DEVICE), bg_batch.to(DEVICE)

        # 1. Forward Simulation
        raw_batch = quantize_and_simulate_optimized(image_batch, label_batch, bg_batch, psf_fft_stack, HPARAMS, DEVICE)
        
        # 2. 배치 후처리 및 비동기 저장 요청
        current_batch_size = image_batch.shape[0]
        for b in range(current_batch_size):
            global_idx = i * HPARAMS['BATCH_SIZE'] + b
            idx_str = str(global_idx).zfill(5)
            
            single_raw = raw_batch[b:b+1]
            raw_max = torch.max(single_raw)
            if raw_max > 0: single_raw = single_raw / raw_max
            
            # 노이즈 및 uint8 변환 (CPU로 넘기기 전 최소한의 연산)
            raw_noisy = torch.clamp(single_raw + torch.randn_like(single_raw) * random.uniform(0.01, 0.03), 0, 1)
            raw_np = (raw_noisy[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # [수정] 대기열이 너무 많으면 처리될 때까지 기다림 (메모리 보호)
            while len(futures) > MAX_QUEUE_SIZE:
                # 완료된 작업들을 리스트에서 제거
                futures = [f for f in futures if not f.done()]
                if len(futures) > MAX_QUEUE_SIZE:
                    time.sleep(0.1) # 0.1초 쉬면서 대기
            
            # 작업 제출 및 추적 리스트에 추가
            fut = executor.submit(save_worker, idx_str, raw_np, img_paths[b], lbl_paths[b], HPARAMS['SAVE_PATH'])
            futures.append(fut)
            # # 스레드풀에 저장 작업 던지기 (비동기)
            # executor.submit(save_worker, idx_str, raw_np, img_paths[b], lbl_paths[b], HPARAMS['SAVE_PATH'])

    # 모든 저장 작업이 끝날 때까지 대기
    executor.shutdown(wait=True)
    print("Generation complete!")

if __name__ == "__main__":
    main()