import os
from typing import Any, Dict  # 상단 import에 추가하세요

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# import config
from utils import *

# from MWDNs import MWDNet_CPSF_RGBD_large_w_softmax_output_add_DepthRefine
from mwdnet_cpsf_rgbd_model import MWDNet_CPSF_RGBD_large_w_softmax_change_wiener_reg
from forHJ import MWDNet_CPSF_depth

import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

# from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder, DatasetFolder

# from torchsummary import summary # To identify network structure
from torchinfo import summary
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from pytorch_msssim import MS_SSIM, SSIM
import lpips

# from accelerate import Accelerator
# from accelerate.utils import tqdm
from tqdm import tqdm
import wandb

import warnings

warnings.filterwarnings("ignore")

from einops import rearrange, reduce, repeat
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(DEVICE))
scaler = GradScaler()

# Define network hyperparameters:
HPARAMS = {
    "IN_CHANNEL": 3,
    "OUT_CHANNEL": 51,
    "BATCH_SIZE": 4,
    "NUM_WORKERS": 4,
    "TRAINSET_SIZE": 18000,  # 전체 2만장 중 18,000장 학습용
    "EPOCHS_NUM": 1000,
    "LR": 5e-4,
    # [수정] SSD2에 저장된 실제 경로 (마지막 /0/ 제외)
    "DATA_ROOT_RAW": "/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/syn_raw_image/0104_113023/raw/",
    "DATA_ROOT_IMAGE": "/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/image/",
    "DATA_ROOT_LABEL": "/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/label/",
    "PSF_DIR": "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/251223_HJC/gray_center_psf/",
    "WANDB_LOG": True,
    "WEIGHT_SAVE_PATH": "/home/hjahn/WS-rawgen/test/weight_fig/",
    "CHECKPOINT_PATH": "",
}

# Check if the weight save path and fig save path exist, if not create them
# if HPARAMS['WEIGHT_SAVE']:
if not os.path.exists(HPARAMS["WEIGHT_SAVE_PATH"]):
    os.makedirs(HPARAMS["WEIGHT_SAVE_PATH"])
    # if not os.path.exists(HPARAMS['FIG_SAVE_PATH']):
    #     os.makedirs(HPARAMS['FIG_SAVE_PATH'])

# Define training parameters:
TPARAMS = {}


# --- 2. PSF 로딩 함수 (PNG 스택 버전) ---
def load_psf_for_train(psf_dir, target_size=(256, 256)):
    # 51개 파일 리스트 생성
    file_list = [
        f"{str(d).zfill(2)}p{p}.png" for d in range(5, 10) for p in range(10)
    ] + ["10p0.png"]
    psf_stack = []
    print(f"Loading {len(file_list)} PSFs from PNGs...")
    for fname in file_list:
        path = os.path.join(psf_dir, fname)
        psf_img = Image.open(path).convert("L")
        # v2.functional 사용
        psf_t = v2.functional.to_image(psf_img).to(torch.float32)
        psf_t = v2.functional.resize(psf_t, target_size)
        psf_t /= torch.sum(psf_t) + 1e-8  # L1 Normalization
        psf_stack.append(psf_t)
    return torch.stack(psf_stack)  # (51, 1, 512, 512)


START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

TPARAMS["depth_range"] = get_depth_range().to(DEVICE)
print("Shape of Depth Range: ", TPARAMS["depth_range"].shape)
print("Depth Range: ", TPARAMS["depth_range"])

name_tmp = (
    START_DATE + "aiobio-5mm~10mm objects"
)  # Notation for individual wandb log name
NOTES = (
    name_tmp
    + "_[COLOR]_LPIPS_[SMOOTH_L1_BETA_0.1]_[DEPTH]_LPIPS_[SMOOTH_L1_BETA_0.1]+SI_LOG_V2_with_lambda_0.5_softmax_based_calculation"
)  # Notation for individual wandb log name

PROJECT_NAME = "Lensless depth imaging-HJA-2601"

if HPARAMS["WANDB_LOG"] == True:
    wandb.require("core")
    wandb.init(
        project=PROJECT_NAME,
        config=HPARAMS,
        notes=NOTES,
        name=name_tmp,
        save_code=True,
    )
else:
    print("Wandb logging is disabled")


def wandb_log(loglist, epoch, note):
    for key, val in loglist.items():
        if key in ["output_color", "output_depth", "label", "label_color", "image"]:
            try:
                # 1. 텐서를 CPU로 옮기고 그라디언트 제거
                item = val.detach().cpu()

                # 2. 배치 차원 제거 (예: [1, 1, 256, 256] -> [1, 256, 256])
                if item.dim() == 4:
                    item = item[0]
                elif item.dim() == 5:  # 테스트 에러 메시지 중 dim=5 대응
                    item = item[0, 0]

                # 3. 데이터 범위 정규화 (0~1 사이로 클리핑)
                item = torch.clamp(item, 0, 1)

                # 4. WandB 이미지 객체 생성 및 로깅
                log = wandb.Image(item)
                wandb.log(
                    {f"{note.capitalize()}/{key.capitalize()}": log}, step=epoch + 1
                )

            except Exception as e:
                print(f"Error logging {key} in {note}: {e}")
        else:
            # 수치 데이터 (Loss, RMSE 등) 로깅
            try:
                wandb.log(
                    {f"{note.capitalize()}/{key.capitalize()}": val}, step=epoch + 1
                )
            except Exception as e:
                print(f"Error logging {key}: {e}")


# class LossFunction(nn.Module):
#     """Loss function class for multiple loss function."""
#     def __init__(self):
#         super().__init__()
#         self.criterion_l1 = nn.L1Loss()
#         self.criterion_l2 =  nn.MSELoss() # L2 loss
#         self.criterion_lpips = nn.DataParallel(lpips.LPIPS(net='vgg')).to(DEVICE)
#         # self.criterion_lpips = nn.DataParallel(lpips.LPIPS(net='alex')).to(DEVICE)
#         # self.criterion_msssim_color = MS_SSIM(data_range=1.0, size_average=True, channel=3)
#         # self.criterion_msssim_depth = MS_SSIM(data_range=1.0, size_average=True, channel=1)
#         self.criterion_smooth_l1  = nn.SmoothL1Loss(beta=0.1) # default beta=1.0
#         # self.criterion_bce = nn.BCELoss()  # Binary Cross-Entropy loss for adversarial loss
#         # self.criterion_silog_loss = scale_invariant_log_loss_v2_lambda(min_scale=0.05, max_scale=0.7, lambda_value=0.5) # min and max scale in meters

#     def forward(self, output_color, output_depth, label_color, label, epoch=0):

#         lpips_color_loss = torch.mean(self.criterion_lpips(output_color, label_color)) # take average of batch
#         smooth_l1_color = self.criterion_smooth_l1(output_color, label_color)
#         # blur_color = self.criterion_blur(output_color, label_color)

#         output_depth_rgb = output_depth.repeat(1,3,1,1) # make RGB depth
#         print(f"Output shape: {output_depth.shape}, Label shape: {label.shape}")
#         label_depth_rgb = label.repeat(1,3,1,1) # make RGB depth
#         lpips_depth_loss = torch.mean(self.criterion_lpips(output_depth_rgb, label_depth_rgb))

#         smooth_l1_depth = self.criterion_smooth_l1(output_depth, label)
#         si_log_loss = self.criterion_silog_loss(output_depth, label)

#         total_loss = lpips_color_loss + smooth_l1_color + smooth_l1_depth + si_log_loss + lpips_depth_loss

#         rmse_depth = torch.sqrt(torch.mean((output_depth - label)**2))

#         # Check for NaN losses
#         nan_losses = []
#         if torch.isnan(lpips_color_loss):
#             nan_losses.append("LPIPS color loss")
#         if torch.isnan(smooth_l1_color):
#             nan_losses.append("Smooth L1 color loss")
#         if torch.isnan(lpips_depth_loss):
#             nan_losses.append("LPIPS depth loss")
#         if torch.isnan(smooth_l1_depth):
#             nan_losses.append("Smooth L1 depth loss")
#         if torch.isnan(si_log_loss):
#             nan_losses.append("SI Log loss")
#         if torch.isnan(rmse_depth):
#             nan_losses.append("RMSE depth")
#         if torch.isnan(total_loss):
#             nan_losses.append("Total loss")


#         if nan_losses:
#             error_message = f"NaN loss detected at epoch {epoch} in: {', '.join(nan_losses)}. Terminating training."
#             print(error_message)
#             # Terminate the script
#             import sys
#             sys.exit(1)
#         return total_loss, lpips_color_loss, lpips_depth_loss, smooth_l1_color, smooth_l1_depth, rmse_depth, si_log_loss
class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        # 수치적으로 가장 안정한 SmoothL1만 활성화
        self.criterion_smooth_l1 = nn.SmoothL1Loss(beta=0.1)

        # 나머지는 일단 주석 처리 (메모리 및 초기화 오류 방지)
        # self.criterion_lpips = nn.DataParallel(lpips.LPIPS(net='vgg')).to(DEVICE)
        # self.criterion_silog_loss = scale_invariant_log_loss_v2_lambda(...)

    def forward(self, output_color, output_depth, label_color, label, epoch=0):
        # 1. 실제 계산할 로스 (Color와 Depth 모두 SmoothL1 적용)
        smooth_l1_color = self.criterion_smooth_l1(output_color, label_color)
        smooth_l1_depth = self.criterion_smooth_l1(output_depth, label)

        # 2. 나머지 로스는 0으로 처리 (변수 유지를 위해)
        # .detach()를 써서 역전파에 영향을 주지 않게 합니다.
        lpips_color_loss = torch.tensor(0.0, device=DEVICE)
        lpips_depth_loss = torch.tensor(0.0, device=DEVICE)
        si_log_loss = torch.tensor(0.0, device=DEVICE)

        # 3. Total Loss (현재는 SmoothL1들의 합)
        total_loss = smooth_l1_color + smooth_l1_depth

        # 4. 성능 지표용 RMSE
        rmse_depth = torch.sqrt(torch.mean((output_depth - label) ** 2) + 1e-8)

        # NaN 체크 로직 유지
        if torch.isnan(total_loss):
            print(f"NaN detected at epoch {epoch}!")
            import sys

            sys.exit(1)

        return (
            total_loss,
            lpips_color_loss,
            lpips_depth_loss,
            smooth_l1_color,
            smooth_l1_depth,
            rmse_depth,
            si_log_loss,
        )


class ImageDataset(Dataset):
    """Custom image file dataset loader."""

    def __init__(self, raw_path, label_path, label_path_color):
        self.raw = raw_path
        self.label = label_path
        self.label_color = label_path_color
        self.len = len(raw_path)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.raw[index][0], self.label[index][0], self.label_color[index][0])


class ImageDataset_val(Dataset):
    """Custom image file dataset loader."""

    def __init__(self, raw_path, transform_image=None):
        self.raw = raw_path
        self.len = len(raw_path)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.raw[index][0]


def train(train_parameters):
    train_parameters["model"].train()

    # 1. Pylance 에러 방지: Dict[str, Any] 명시
    result: Dict[str, Any] = {
        k: 0.0 for k in ["loss_sum", "PSNR_color", "PSNR_depth", "RMSE_depth"]
    }

    ACCUM_STEPS = 4  # 가상 배치 사이즈: 4개마다 업데이트
    optimizer = train_parameters["optimizer"]
    optimizer.zero_grad()  # 루프 시작 전 초기화

    bar = tqdm(
        train_parameters["trainset_loader"], position=1, leave=False, desc="Train Loop"
    )
    total_steps_per_epoch = len(train_parameters["trainset_loader"])
    epoch = train_parameters.get("epoch_now", 0) # main에서 넘겨줘야 함
    loop_start = time.time() # 데이터 로딩 측정용
    for i, (image, label, label_color) in enumerate(bar):
        if torch.isnan(image).any():
            print(f"!!! Error: Input image contains NaN at index {i} !!!")
            continue
        data_time = time.time() - loop_start # 진짜 데이터 로딩 시간
        global_step = epoch * total_steps_per_epoch + i
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        label_color = label_color.to(DEVICE)

        # 2. 최신 autocast 문법 적용 (device_type 명시)
        with autocast(device_type="cuda"):
            model_start = time.time()
            intensity, soft_max_depth_stack = train_parameters["model"](image)
            torch.cuda.synchronize()
            forward_time = time.time() - model_start
            depth_range = TPARAMS["depth_range"].view(1, -1, 1, 1)
            output_depth = 1.0 - torch.sum(
                depth_range * soft_max_depth_stack, dim=1, keepdim=True
            )

            total_loss, _, _, _, _, rmse_depth, _ = train_parameters["loss_function"](
                intensity, output_depth, label_color, label, i
            )
            # Accumulation을 위해 로스를 나눔
            loss_to_backward = total_loss / ACCUM_STEPS

        scaler.scale(loss_to_backward).backward()

        # 3. 가상 배치 가중치 업데이트 (ACCUM_STEPS 마다 수행)
        if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(bar):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                train_parameters["model"].parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # 업데이트 후 초기화

        # 결과 기록
        result["loss_sum"] += total_loss.item()
        result["RMSE_depth"] += rmse_depth.item()

        # 4. 실시간 WandB 차트 로깅 (100 배치마다)
        if i % 100 == 0 and HPARAMS["WANDB_LOG"]:
            wandb.log(
                {
                    "Train/Batch_Loss": total_loss.item(),
                    "Train/Batch_RMSE": rmse_depth.item(),
                    "Train/Realtime_Forward_Time": forward_time,
                    "Train/Realtime_Data_Time": data_time,
                    "Train/Batch_Loss": total_loss.item()
                }
            )

        # 5. 실시간 WandB 이미지 로깅 (1000 배치마다)
        if i % 1000 == 0 and HPARAMS["WANDB_LOG"]:
            # 배치 차원 제거를 위해 [0] 인덱싱 적용
            wandb.log(
                {
                    "Realtime/Output_Depth": wandb.Image(torch.clamp(output_depth[0], 0, 1).cpu()),
                    "Realtime/Input_Raw": wandb.Image(torch.clamp(image[0], 0, 1).cpu()),
                    "Realtime/Intensity": wandb.Image(torch.clamp(intensity[0], 0, 1).cpu())
                }, step=global_step
            ) # global_step을 관리하면 연속적으로 보입니다.
        # 캐시 정리
        if i % 100 == 0:
            print(f"Data: {data_time:.4f}s | Forward: {forward_time:.4f}s")
            torch.cuda.empty_cache()

        bar.set_description(
            f"Loss: {total_loss.item():.5f} | RMSE: {rmse_depth.item():.5f}"
        )
        loop_start = time.time() # 다음 루프를 위한 시작점

    # 에폭 평균 계산
    for key in ["loss_sum", "RMSE_depth"]:
        result[key] /= len(train_parameters["trainset_loader"])

    # 로깅용 이미지 데이터 담기 (Pylance 에러 해결)
    visuals = {
        "image": image,
        "label": label,
        "label_color": label_color,
        "output_color": intensity,
        "output_depth": output_depth,
    }
    result.update(visuals)

    return result


def test(test_parameters):
    test_parameters["model"].eval()
    result = {k: 0.0 for k in ["loss_sum", "PSNR_color", "PSNR_depth", "RMSE_depth"]}

    with torch.no_grad():
        bar = tqdm(
            test_parameters["testset_loader"], position=2, leave=False, desc="Test Loop"
        )
        for i, (image, label, label_color) in enumerate(bar):
            image, label, label_color = (
                image.to(DEVICE),
                label.to(DEVICE),
                label_color.to(DEVICE),
            )
            # label = label.unsqueeze(1) # 이미 npz loader에서 처리함

            intensity, soft_max_depth_stack = test_parameters["model"](image)
            depth_range = TPARAMS["depth_range"].view(1, -1, 1, 1)
            output_depth = 1.0 - torch.sum(
                depth_range * soft_max_depth_stack, dim=1, keepdim=True
            )

            total_loss, _, _, _, _, rmse_depth, _ = test_parameters["loss_function"](
                intensity, output_depth, label_color, label, i
            )

            result["loss_sum"] += total_loss.item()
            result["RMSE_depth"] += rmse_depth.item()

    for key in result:
        result[key] /= len(test_parameters["testset_loader"])
    result["image"], result["label"], result["label_color"] = image, label, label_color
    result["output_color"], result["output_depth"] = intensity, output_depth
    return result


# 1. Hook 함수 정의
def nan_hook(self, input, output):
    # output이 튜플인 경우(우리 모델처럼 intensity, x 두 개를 반환할 때) 처리
    if isinstance(output, tuple):
        for i, out in enumerate(output):
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(
                    f"[NaN/Inf Detected!] Layer: {self.__class__.__name__} | Output index: {i}"
                )
    else:
        if torch.isnan(output).any() or torch.isinf(output).any():
            # 구체적인 레이어 경로를 알기 위해 아래와 같이 출력
            print(f"[NaN/Inf Detected!] Layer: {self}")


# 2. 모델의 모든 모듈에 Hook 등록 (main 함수 내부)
def register_debug_hooks(model):
    for name, module in model.named_modules():
        # 모든 서브 모듈(Conv, ReLU, W2 등)의 출력을 감시합니다.
        module.register_forward_hook(nan_hook)
    print("All debug hooks registered.")


def main():
    global TPARAMS
    # 1. 초기화 (GPU 메모리 청소)
    torch.cuda.empty_cache()

    psf_tensor = load_psf_for_train(HPARAMS["PSF_DIR"], target_size=(256, 256))
    TPARAMS["psf"] = psf_tensor.to(DEVICE)
    TPARAMS["depth_range"] = get_depth_range().to(DEVICE)

    # 2. 뎁스 물리 범위 로드
    TPARAMS["depth_range"] = get_depth_range().to(DEVICE)
    print(
        f"Depth range (normalized) step: {TPARAMS['depth_range'][1] - TPARAMS['depth_range'][0]:.4f}"
    )

    # --- 3. 데이터셋 준비 (Random Split 로직) ---
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
        ]
    )

    # 전체 데이터셋 로드
    full_raw = ImageFolder(root=HPARAMS["DATA_ROOT_RAW"], transform=transform)
    full_label = DatasetFolder(
        root=HPARAMS["DATA_ROOT_LABEL"], loader=npz_loader, extensions=[".npz"]
    )
    full_image = ImageFolder(root=HPARAMS["DATA_ROOT_IMAGE"], transform=transform)

    # 일관된 랜덤 시드로 분할 (0104_113023 데이터셋 2만장 기준)
    total_len = len(full_raw)
    train_len = HPARAMS["TRAINSET_SIZE"]
    test_len = total_len - train_len

    # 각각 동일한 시드로 쪼개서 짝이 맞도록 함
    seed = 4716
    train_raw, test_raw = random_split(
        full_raw, [train_len, test_len], generator=torch.Generator().manual_seed(seed)
    )
    train_lbl, test_lbl = random_split(
        full_label, [train_len, test_len], generator=torch.Generator().manual_seed(seed)
    )
    train_img, test_img = random_split(
        full_image, [train_len, test_len], generator=torch.Generator().manual_seed(seed)
    )

    # 커스텀 ImageDataset 클래스를 이용해 묶어줌
    train_data = ImageDataset(train_raw, train_lbl, train_img)
    test_data = ImageDataset(test_raw, test_lbl, test_img)

    train_loader = DataLoader(
        train_data,
        batch_size=HPARAMS["BATCH_SIZE"],
        shuffle=True,
        num_workers=HPARAMS["NUM_WORKERS"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=HPARAMS["BATCH_SIZE"],
        shuffle=False,
        num_workers=HPARAMS["NUM_WORKERS"],
        pin_memory=True,
    )

    TPARAMS["trainset_loader"] = train_loader
    TPARAMS["testset_loader"] = test_loader

    # --- 4. 모델 및 학습 설정 -- -
    # n_classes=51 확인
    model = MWDNet_CPSF_depth(
        n_channels=3, n_classes=51, psf=TPARAMS["psf"], height=256, width=256
    )
    # 학습 전 모델 파라미터 및 구조 확인
    stats = summary(
        model,
        input_size=(HPARAMS["BATCH_SIZE"], 3, 256, 256),
        device=DEVICE,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3,
    )  # depth를 조절해 얼마나 상세히 볼지 결정
    print(stats)
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model).to(DEVICE) #GPU 2개
    register_debug_hooks(model)
    model = model.to(DEVICE)
    TPARAMS["model"] = model

    optimizer = optim.AdamW(model.parameters(), lr=HPARAMS["LR"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    TPARAMS["optimizer"] = optimizer
    TPARAMS["scheduler"] = scheduler
    TPARAMS["loss_function"] = LossFunction()

    # WandB 시작
    if HPARAMS["WANDB_LOG"]:
        wandb.init(project="Lensless-Depth-AIOBIO", config=HPARAMS)
        wandb.watch(model)

    print("Training Start!")
    for epoch in range(HPARAMS["EPOCHS_NUM"]):
        TPARAMS["epoch_now"] = epoch
        train_result = train(TPARAMS)  # 루프 내부에서 wandb_log 호출
        test_result = test(TPARAMS)

        # 로그 기록
        if HPARAMS["WANDB_LOG"]:
            wandb_log(train_result, epoch, "train")
            wandb_log(test_result, epoch, "test")

        # 가중치 저장
        if (epoch + 1) % 10 == 0:
            weight_save(
                epoch, "HJA_Exp", HPARAMS["WEIGHT_SAVE_PATH"], TPARAMS, "model_51ch"
            )


if __name__ == "__main__":
    main()
