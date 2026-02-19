import os
os.environ["WANDB_MODE"] = "disabled"
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
from forHJ_light_modulated import MWDNet_CPSF_depth_light as MWDNet_CPSF_depth
from train_utils import get_depth_range
from train_the_thomas import load_psf_for_train

timestamp = time.strftime("%y%m%d-%H%M%S")
# 1. 설정 (HPARAMS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/hjahn/HJA_nas/lensless-depth/WS-rawgen/pth_512_uv-light-newobj/model_51ch/20260214-090327_model_aiobio.pth"
RAW_IMAGE_DIR = "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/validate_jig_uv/0/" # 실제 이미지 경로
SAVE_DIR = "../inference_results/"
PSF_DIR = "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/260112/psf_color_aligned/"
IMG_SIZE = (512, 512)

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

            # [해결책] 용량 최적화를 위한 새로운 사이즈 계산
            # 긴 쪽(Height)을 512로 잡고 비율을 계산합니다.
            target_w = 512
            target_h = int(target_w * (9 / 16)) # 약 288 픽셀
            # target_w = 512
            # 만약 9:16이 가로가 더 긴 형태라면 target_w = 512, target_h = int(512 * 9/16)으로 바꾸세요.

            input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

            # 모델 추론
            intensity, soft_max_depth_stack = model(input_tensor)
            
            # 1. Normalized Depth 계산
            depth_norm = 1.0 - torch.sum(depth_range * soft_max_depth_stack, dim=1, keepdim=True)
            
            # [수정] 원본 해상도가 아닌, 계산한 '적당한' 9:16 사이즈로 보간
            intensity_resized = torch.nn.functional.interpolate(
                intensity, size=(target_h, target_w), mode='bicubic', align_corners=False
            )
            depth_resized = torch.nn.functional.interpolate(
                depth_norm, size=(target_h, target_w), mode='bicubic', align_corners=False
            )
            
            # Numpy 변환
            depth_norm_np = depth_resized.squeeze().cpu().numpy()
            depth_mm_np = 10.0 - (depth_norm_np * 5.0)
            color_np = np.clip(intensity_resized.squeeze().permute(1, 2, 0).cpu().numpy(), 0, 1)
            
            base_name = os.path.splitext(fname)[0]

            # --- 저장 로직 (이제 용량이 매우 가볍습니다) ---
            plt.imsave(os.path.join(SAVE_DIR, f"{base_name}_restored_color.png"), color_np)
            
            # A) 9:16 비율의 뎁스맵 시각화
            plt.figure(figsize=(target_w/100, target_h/100)) # 비율에 맞는 피규어 크기
            plt.imshow(depth_mm_np, cmap='viridis_r', vmin=5, vmax=10)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_depth_mm.png"), bbox_inches='tight')
            plt.close()

            # C) MATLAB용 데이터 저장 (가벼운 용량)
            sio.savemat(os.path.join(SAVE_DIR, f"{base_name}_data.mat"), {
                'depth_mm': depth_mm_np,
                'restored_rgb': color_np,
                'target_size': [target_h, target_w]
            })

    print(f"Inference completed! 용량 최적화 완료: {SAVE_DIR}")

if __name__ == "__main__":
    run_inference()