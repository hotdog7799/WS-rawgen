import os
from typing import Any, Dict  # 상단 import에 추가하세요

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

# import config
from train_utils import *

# from MWDNs import MWDNet_CPSF_RGBD_large_w_softmax_output_add_DepthRefine
from mwdnet_cpsf_rgbd_model import MWDNet_CPSF_RGBD_large_w_softmax_change_wiener_reg
from forHJ_light_modulated import MWDNet_CPSF_depth_light as MWDNet_CPSF_depth

import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

# from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.utils import make_grid

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(DEVICE))
scaler = GradScaler()

# Define network hyperparameters:
HPARAMS = {
    "IN_CHANNEL": 3,
    "OUT_CHANNEL": 51,
    "BATCH_SIZE": 8, 
    "NUM_WORKERS": 8,
    "TRAINSET_SIZE": 18000,  # 전체 2만장 중 18,000장 학습용
    "EPOCHS_NUM": 1000,
    "LR": 1e-4,
    "H":512,
    "W":512,
    # [수정] SSD2에 저장된 실제 경로 (마지막 /0/ 제외)
    # "DATA_ROOT_RAW": "/home/hjahn/mnt/nas/Research/HJA/syn_raw_light_object_gen_cam2/0212_123338/raw/",
    'DATA_ROOT_RAW': "/home/hjahn/mnt/nas/Research/HJA/syn_raw_light_object_gen_cam2/0219_064527/raw/",
    "DATA_ROOT_IMAGE": "/home/hjahn/mnt/nas/Research/HJA/rendering/20260211_192340_20000/image/",
    "DATA_ROOT_LABEL": "/home/hjahn/mnt/nas/Research/HJA/rendering/20260211_192340_20000/label/",
    "DATA_ROOT_VAL_REAL": "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/cam2/260205-raw/",
    "PSF_DIR": "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/cam2/260205_psf_color_aligned/",
    "WEIGHT_SAVE_PATH": "/home/hjahn/mnt/nas/homes/HJA/lensless-depth/WS-rawgen/pth_cam2_512_uv-light-newobj/",
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
def load_psf_for_train(psf_dir, target_size=(HPARAMS["H"], HPARAMS["W"])):
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
        # print(psf_t.max())
        psf_t = psf_t / 255.
        # psf_t /= torch.sum(psf_t) + 1e-8  # L1 Normalization
        psf_stack.append(psf_t)
    return torch.stack(psf_stack)  # (51, 1, 512, 512)


START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

TPARAMS["depth_range"] = get_depth_range().to(DEVICE)
print("Shape of Depth Range: ", TPARAMS["depth_range"].shape)
print("Depth Range: ", TPARAMS["depth_range"])

name_tmp = (
    START_DATE + "cam2"
)  # Notation for individual wandb log name
NOTES = (
    name_tmp
    + "_[COLOR]_LPIPS_[SMOOTH_L1_BETA_0.1]_[DEPTH]_LPIPS_[SMOOTH_L1_BETA_0.1]+SI_LOG_V2_with_lambda_0.5_softmax_based_calculation"
)  # Notation for individual wandb log name

PROJECT_NAME = "Lensless depth imaging-HJA-2601"

wandb.require("core")
wandb.init(
    project=PROJECT_NAME,
    config=HPARAMS,
    notes=NOTES,
    # mode="disabled",
    name=name_tmp,
    save_code=True,
)

def wandb_log(loglist, epoch, note):
    for key, val in loglist.items():
        try:
            try:
                item = val.cpu().detach()
            except:
                item = val
            log = wandb.Image(item)
        except:
            log = val
        wandb.log(
            {
                "{0} {1}".format(note.capitalize(), key.capitalize()): log,
            },
            step=epoch + 1,
        )

class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        # 수치적으로 가장 안정한 SmoothL1만 활성화
        self.criterion_smooth_l1 = nn.SmoothL1Loss(beta=0.1, reduction='none')
        self.criterion_silog = scale_invariant_log_loss_v2_lambda(
            min_scale=0.05, max_scale=0.7, lambda_value=0.5
        )

        # 나머지는 일단 주석 처리 (메모리 및 초기화 오류 방지)
        self.criterion_lpips = nn.DataParallel(lpips.LPIPS(net='vgg')).to(DEVICE)
        # self.criterion_silog_loss = scale_invariant_log_loss_v2_lambda(...)
    # def get_gradient(self, x):
    #     # depth의 급격한 변화를 감지하도록 하는 함수
    #     h_x = x.size()[-2]
    #     w_x = x.size()[-1]
    #     # gradient in x direction
    #     grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    #     # gradient in y direction
    #     grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    #     return grad_x, grad_y
    
    def forward(self, output_color, output_depth, label_color, label, epoch=0):
        # 1. color loss
        smooth_l1_color = self.criterion_smooth_l1(output_color, label_color)
        lpips_color_loss = torch.mean(self.criterion_lpips(output_color, label_color))
        smooth_l1_depth = self.criterion_smooth_l1(output_depth, label)
        # 2. ROI mask
        # roi_mask = (label > 0.05).float()

        # # 3. weighted Depth SmoothL1 loss
        # raw_l1_depth = self.criterion_smooth_l1(output_depth, label)
        # # [수정] ROI Mask + 비대칭 페널티 (Asymmetric Penalty)
        # # 치아 영역(roi_mask=1)인데, 예측값(output_depth)이 정답(label)보다 작다면(즉, 배경인 0쪽으로 간다면)
        # # *참고: 깊이 정의에 따라 부호는 반대일 수 있음. 
        # # (여기서는 output_depth가 1.0이 가까운 것, 0.0이 먼 배경이라고 가정 시 -> output_depth < label 인 경우 페널티)
        # # 만약 반대라면 (output > label)로 수정 필요.

        # diff = label - output_depth # 정답(치아) - 예측(배경) > 0 이면 구멍이 난 상태
        # under_estimation_mask = (diff > 0).float() * roi_mask

        # # 구멍이 난 곳에는 기존 7배 + 추가 10배 = 총 17배 가중치 부여
        # penalty_weight = 1.0 + under_estimation_mask * 10.0 
        # depth_weight = (roi_mask * 7.0 + (1 - roi_mask) * 1.0) * penalty_weight
        # smooth_l1_depth = torch.mean(raw_l1_depth * depth_weight)

        # # 4. depth gradient loss
        # out_grad_x, out_grad_y = self.get_gradient(output_depth)
        # target_grad_x, target_grad_y = self.get_gradient(label)
        # loss_grad_x = torch.mean(torch.abs(out_grad_x - target_grad_x))
        # loss_grad_y = torch.mean(torch.abs(out_grad_y - target_grad_y))
        # grad_loss = loss_grad_x + loss_grad_y
        lpips_depth_loss = torch.tensor(0.0, device=DEVICE)       
        silog_loss = self.criterion_silog(output_depth, label)
        # total_loss = smooth_l1_color + smooth_l1_depth + 0.7 *silog_loss + lpips_color_loss

        total_loss = 0.01 * smooth_l1_color + \
                     smooth_l1_depth + \
                     0.9 * silog_loss + \
                        lpips_color_loss
                    #  0.2 * grad_loss + \
                        

        total_loss = total_loss.mean()

        # 4. 성능 지표용 RMSE
        rmse_depth = torch.sqrt(torch.mean((output_depth - label) ** 2) + 1e-8)

        # NaN 체크 로직 유지
        if torch.isnan(total_loss).any():
            print(f"NaN detected at epoch {epoch}!")
            import sys

            sys.exit(1)

        return (
            total_loss,
            lpips_color_loss,
            #grad_loss,
            lpips_depth_loss,
            smooth_l1_color,
            smooth_l1_depth,
            rmse_depth,
            silog_loss,
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

    result = {}
    result["loss"] = 0
    # result["grad_loss"] = 0
    result["RMSE_depth"] = 0
    # result["LPIPS_color"] = 0
    # result["LPIPS_depth"] = 0
    # result["PSNR_color"] = 0
    # result["PSNR_depth"] = 0
    # ACCUM_STEPS = 4  # 가상 배치 사이즈: 4개마다 업데이트
    ACCUM_STEPS = 1

    bar = tqdm(
        train_parameters["trainset_loader"], position=1, leave=False, desc="Train Loop"
    )
    for i, (image, label, label_color) in enumerate(bar):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        label_color = label_color.to(DEVICE)
        # breakpoint()
        with autocast(device_type="cuda"):
            intensity, soft_max_depth_stack = train_parameters["model"](image)
            depth_range = TPARAMS["depth_range"].view(1, -1, 1, 1)
            output_depth = 1.0 - torch.sum(
                depth_range * soft_max_depth_stack, dim=1, keepdim=True
            )

            total_loss, _, _, _, _, rmse_depth, _  = train_parameters["loss_function"](
                intensity, output_depth, label_color, label, i
            )
            loss_to_backward = total_loss / ACCUM_STEPS

        scaler.scale(loss_to_backward).backward()

        if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(bar):
            scaler.unscale_(train_parameters["optimizer"])
            torch.nn.utils.clip_grad_norm_(
                train_parameters["model"].parameters(), max_norm=1.0
            )
            scaler.step(train_parameters["optimizer"])
            scaler.update()
            train_parameters["optimizer"].zero_grad()  # 업데이트 후 초기화

        result["loss"] += total_loss.item()
        # result["grad_loss"] += grad_loss.item()
        result["RMSE_depth"] += rmse_depth.item()
        # result['LPIPS_color'] += lpips_color_loss
        # result['LPIPS_depth'] += lpips_depth_loss
        # result['PSNR_color'] += PSNR(result['output_color'], label_color)
        # result['PSNR_depth'] += PSNR(result['output_depth'], label)

        bar.set_description(
            f"Loss: {total_loss.item():.5f} | RMSE: {rmse_depth.item():.5f}"
        )
        # break

    result["loss"] /= len(train_parameters["trainset_loader"])
    # result["grad_loss"] /= len(train_parameters["trainset_loader"])
    result["RMSE_depth"] /= len(train_parameters["trainset_loader"])
    result["input"] = image[0]
    result["label"] = label[0]
    result["label_color"] = label_color[0]
    result["output_color"] = intensity[0]
    result["output_depth"] = output_depth[0]

    return result


def test(test_parameters):
    test_parameters["model"].eval()
    result = {}
    result["loss"] = 0
    # result["grad_loss"] = 0
    result["RMSE_depth"] = 0
    with torch.no_grad():
        bar = tqdm(
            test_parameters["testset_loader"], position=2, leave=False, desc="Test Loop"
        )
        for i, (image, label, label_color) in enumerate(bar):
            image, label, label_color = image.to(DEVICE),label.to(DEVICE),label_color.to(DEVICE)

            intensity, soft_max_depth_stack = test_parameters["model"](image)
            depth_range = TPARAMS["depth_range"].view(1, -1, 1, 1)
            output_depth = 1.0 - torch.sum(
                depth_range * soft_max_depth_stack, dim=1, keepdim=True
            )

            total_loss, _, _, _, _, rmse_depth, _ = test_parameters["loss_function"](
                intensity, output_depth, label_color, label, i
            )

            result["loss"] += total_loss.item()
            # result["grad_loss"] += grad_loss.item()
            result["RMSE_depth"] += rmse_depth.item()
            # break

    result["loss"] /= len(test_parameters["testset_loader"])
    # result["grad_loss"] /= len(test_parameters["testset_loader"])
    result["RMSE_depth"] /= len(test_parameters["testset_loader"])
    result["input"] = image[0]
    result["label"] = label[0]
    result["label_color"] = label_color[0]
    result["output_color"] = intensity[0]
    result["output_depth"] = output_depth[0]

    return result

def validate_real(val_parameters):
    val_parameters["model"].eval()
    
    all_inputs = []
    all_intensities = []
    all_depths = []
    
    with torch.no_grad():
        real_loader = val_parameters["real_val_loader"]
        
        for image, _ in real_loader:
            image = image.to(DEVICE)
            
            # 모델 추론
            intensity, soft_max_depth_stack = val_parameters["model"](image)
            
            # 뎁스 계산
            depth_range = TPARAMS["depth_range"].view(1, -1, 1, 1).to(DEVICE)
            output_depth = 1.0 - torch.sum(
                depth_range * soft_max_depth_stack, dim=1, keepdim=True
            )
            
            all_inputs.append(image.detach())
            all_intensities.append(intensity.detach())
            all_depths.append(output_depth.detach())

            del image, intensity, soft_max_depth_stack, output_depth

    # 시각화를 위해 모든 배치를 하나로 합침
    # (10장 내외이므로 메모리에 충분히 올라갑니다)
    cat_inputs = torch.cat(all_inputs, dim=0).cpu()
    cat_intensities = torch.cat(all_intensities, dim=0).cpu()
    cat_depths = torch.cat(all_depths, dim=0).cpu()

    # 한 줄에 5장씩 배치하여 그리드 생성 (nrow=5)
    # padding을 주어 이미지 사이 경계를 만듭니다.
    result = {}
    # make_grid 결과는 [C, H_grid, W_grid] 형태의 3D 텐서가 됩니다.
    result["input"] = make_grid(cat_inputs, nrow=8, padding=2)
    result["output_color"] = make_grid(cat_intensities, nrow=8, padding=2)
    result["output_depth"] = make_grid(cat_depths, nrow=8, padding=2)
    
    return result

def main():
    global TPARAMS
    torch.cuda.empty_cache()

    psf_tensor = load_psf_for_train(HPARAMS["PSF_DIR"], target_size=(HPARAMS["H"], HPARAMS["W"]))
    TPARAMS["psf"] = psf_tensor.to(DEVICE)
    TPARAMS["depth_range"] = get_depth_range().to(DEVICE)

    TPARAMS["depth_range"] = get_depth_range().to(DEVICE)
    print(
        f"Depth range (normalized) step: {TPARAMS['depth_range'][1] - TPARAMS['depth_range'][0]:.4f}"
    )

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((HPARAMS["H"], HPARAMS["W"]), interpolation=v2.InterpolationMode.BICUBIC),
        ]
    )

    # 전체 데이터셋 로드
    full_raw = ImageFolder(root=HPARAMS["DATA_ROOT_RAW"], transform=transform)
    full_label = DatasetFolder(
        root=HPARAMS["DATA_ROOT_LABEL"], loader=npz_loader, extensions=[".npz"]
    )
    full_image = ImageFolder(root=HPARAMS["DATA_ROOT_IMAGE"], transform=transform)
    # 실제 촬영 이미지는 레이블이 없으므로 ImageFolder를 쓰되 레이블은 무시합니다.
    real_val_dataset = ImageFolder(root=HPARAMS["DATA_ROOT_VAL_REAL"], transform=transform)
    
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
    real_val_loader = DataLoader(
        real_val_dataset,
        batch_size=4, # 시각화용이므로 작게 설정
        shuffle=False,
        num_workers=HPARAMS["NUM_WORKERS"],
        pin_memory=True,
    )
    TPARAMS["trainset_loader"] = train_loader
    TPARAMS["testset_loader"] = test_loader
    TPARAMS["real_val_loader"] = real_val_loader

    # --- 4. 모델 및 학습 설정 -- -
    # n_classes=51 확인
    model = MWDNet_CPSF_depth(
        n_channels=3, n_classes=51, psf=TPARAMS["psf"], height=HPARAMS["H"], width=HPARAMS["W"]
    )
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model).to(DEVICE)  # GPU 2개
    TPARAMS["model"] = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=HPARAMS["LR"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    TPARAMS["optimizer"] = optimizer
    TPARAMS["scheduler"] = scheduler
    TPARAMS["loss_function"] = LossFunction()
    print("Training Start!")

    min_loss = float('inf')
    
    BAR = tqdm(range(HPARAMS["EPOCHS_NUM"]), position=0, leave=True)
    for epoch in BAR:
        TPARAMS["epoch_now"] = epoch

        # Train 실행 (내부에서 global_step 로깅)
        train_result = train(TPARAMS)
        # Test 실행
        test_result = test(TPARAMS)
        current_test_loss = test_result["loss"]
        

        if min_loss > current_test_loss:
            min_loss = current_test_loss
            weight_save(epoch, START_DATE, HPARAMS["WEIGHT_SAVE_PATH"], TPARAMS, "model_51ch")
            print(f"Weight saved at epoch {epoch} with loss {min_loss:.5f}")
        
        torch.cuda.empty_cache()
        try:
            # 3. 리얼 데이터 Inference 실행 (눈으로 확인용)
            real_result = validate_real(TPARAMS)

            # current_test_loss = test_result["loss"]

            wandb_log(train_result, epoch, "Train")
            wandb_log(test_result, epoch, "Test")
            wandb_log(real_result, epoch, "Real")
        except Exception as e:
            print(f"Error during real data validation at epoch {epoch}: {e}")
            wandb_log(train_result, epoch, "Train")
            wandb_log(test_result, epoch, "Test")


if __name__ == "__main__":
    main()