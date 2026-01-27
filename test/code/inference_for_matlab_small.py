import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
import torch.nn as nn
import time

# 기존 모델 및 유틸 불러오기 (경로 주의)
from mwdnet_cpsf_rgbd_model import MWDNet_CPSF_RGBD_large_w_softmax_change_wiener_reg
from forHJ import MWDNet_CPSF_depth
from train_utils import get_depth_range
from train_the_thomas import load_psf_for_train

timestamp = time.strftime("%y%m%d-%H%M%S")
# 1. 설정 (HPARAMS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/hjahn/depth/WS-rawgen/pth_256_color/model_51ch/20260113-023527_model_aiobio.pth"
RAW_IMAGE_DIR = "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/validate_jig_uv/0/" # 실제 이미지 경로
SAVE_DIR = "../inference_results/"
PSF_DIR = "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/260112/psf_color_aligned/"
IMG_SIZE = (256, 256)

SAVE_DIR = os.path.join(SAVE_DIR,timestamp)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_inference():
    print("Loading Model...")
    psf_tensor = load_psf_for_train(PSF_DIR, target_size=IMG_SIZE).to(DEVICE)
    depth_range = get_depth_range().to(DEVICE).view(1, -1, 1, 1)

    model = MWDNet_CPSF_depth(n_channels=3, n_classes=51, psf=psf_tensor, height=256, width=256)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(DEVICE).eval()

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(IMG_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
    ])

    raw_files = [f for f in os.listdir(RAW_IMAGE_DIR) if f.endswith(('.png', '.jpg'))]
    
    with torch.no_grad():
        for fname in raw_files:
            img_path = os.path.join(RAW_IMAGE_DIR, fname)
            raw_img = Image.open(img_path).convert("RGB")

            orig_width, orig_height = raw_img.size

            input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

            # 모델 추론
            intensity, soft_max_depth_stack = model(input_tensor)
            
            # 1. Normalized Depth (0~1)
            depth_norm = 1.0 - torch.sum(depth_range * soft_max_depth_stack, dim=1, keepdim=True)
            
            # [핵심] 원래 해상도(9:16)로 보간 (Bicubic 사용 권장)
            # intensity: [1, 3, 256, 256] -> [1, 3, orig_height, orig_width]
            intensity_resized = torch.nn.functional.interpolate(
                intensity, size=(orig_height, orig_width), mode='bicubic', align_corners=False
            )
            # depth_norm: [1, 1, 256, 256] -> [1, 1, orig_height, orig_width]
            depth_resized = torch.nn.functional.interpolate(
                depth_norm, size=(orig_height, orig_width), mode='bicubic', align_corners=False
            )
            
            depth_norm_np = depth_norm.squeeze().cpu().numpy()
            
            # 2. Physical Depth Conversion (5mm ~ 10mm)
            # 수식: 1.0 -> 5mm, 0.0 -> 10mm
            depth_mm_np = 10.0 - (depth_norm_np * 5.0)
            
            color_np = np.clip(intensity.squeeze().permute(1, 2, 0).cpu().numpy(), 0, 1)
            base_name = os.path.splitext(fname)[0]

            # --- 시각화 저장 ---
            # 1. 복원된 RGB 이미지 저장 (학습 시 보던 것과 동일)
            # intensity 이미지를 그대로 저장합니다.
            plt.imsave(os.path.join(SAVE_DIR, f"{base_name}_restored_color.png"), color_np)
            
            # A) 기존 Normalized Depth 시각화
            plt.figure(figsize=(6, 5))
            plt.imshow(depth_norm_np, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label='Normalized Depth (0:Far, 1:Close)')
            plt.title(f"Normalized: {fname}")
            plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_depth_norm.png"))
            plt.close()

            # B) 실제 Physical Depth (mm) 시각화 - 요청하신 부분
            plt.figure(figsize=(6, 5))
            # 컬러맵 범위를 5~10mm로 고정하여 비교하기 쉽게 함
            plt.imshow(depth_mm_np, cmap='viridis_r', vmin=5, vmax=10) # _r은 가까운게 빨간색 계열로 보이게 뒤집은 것
            plt.colorbar(label='Physical Depth (mm)')
            plt.title(f"Distance: {fname} (mm)")
            plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_depth_mm.png"))
            plt.close()

            # C) MATLAB용 데이터 저장 (mm 단위 추가)
            sio.savemat(os.path.join(SAVE_DIR, f"{base_name}_data.mat"), {
                'depth_norm': depth_norm_np,
                'depth_mm': depth_mm_np, # MATLAB에서 바로 쓸 수 있는 mm 값
                'restored_rgb': color_np
            })

    print(f"Inference completed! Results saved in {SAVE_DIR}")

if __name__ == "__main__":
    run_inference()