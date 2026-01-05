import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from torch.cuda.amp import autocast, GradScaler

from pytorch_msssim import MS_SSIM, SSIM
import lpips

# from accelerate import Accelerator
# from accelerate.utils import tqdm
from tqdm import tqdm
import wandb

import warnings
warnings.filterwarnings('ignore')

from einops import rearrange, reduce, repeat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(DEVICE))        
scaler = GradScaler()

# Define network hyperparameters:
HPARAMS = {
    'IN_CHANNEL' : 3,
    'OUT_CHANNEL': 51, 
    'BATCH_SIZE': 1,
    'NUM_WORKERS': 0,
    'TRAINSET_SIZE' : 18000, # 전체 2만장 중 18,000장 학습용
    'EPOCHS_NUM': 1000,
    'LR': 5e-4,
    
    # [수정] SSD2에 저장된 실제 경로 (마지막 /0/ 제외)
    'DATA_ROOT_RAW': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/syn_raw_image/0104_113023/raw/',
    'DATA_ROOT_IMAGE': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/image/',
    'DATA_ROOT_LABEL': '/home/hjahn/mnt/ssd2/dataset_ssd2/HJA/HJA_data/20251230_152424_20000_8/label/',
    'PSF_DIR': "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/251223_HJC/gray_center_psf/",

    'WANDB_LOG': True,
    'WEIGHT_SAVE_PATH' : '/home/hjahn/WS-rawgen/test/weight_fig/',
    'CHECKPOINT_PATH': '',
}

# Check if the weight save path and fig save path exist, if not create them
# if HPARAMS['WEIGHT_SAVE']: 
if not os.path.exists(HPARAMS['WEIGHT_SAVE_PATH']):
    os.makedirs(HPARAMS['WEIGHT_SAVE_PATH'])
    # if not os.path.exists(HPARAMS['FIG_SAVE_PATH']):
    #     os.makedirs(HPARAMS['FIG_SAVE_PATH'])

# Define training parameters:
TPARAMS = {}
# --- 2. PSF 로딩 함수 (PNG 스택 버전) ---
def load_psf_for_train(psf_dir, target_size=(256, 256)):
    # 51개 파일 리스트 생성
    file_list = [f"{str(d).zfill(2)}p{p}.png" for d in range(5, 10) for p in range(10)] + ["10p0.png"]
    psf_stack = []
    print(f"Loading {len(file_list)} PSFs from PNGs...")
    for fname in file_list:
        path = os.path.join(psf_dir, fname)
        psf_img = Image.open(path).convert('L')
        # v2.functional 사용
        psf_t = v2.functional.to_image(psf_img).to(torch.float32)
        psf_t = v2.functional.resize(psf_t, target_size)
        psf_t /= (torch.sum(psf_t) + 1e-8) # L1 Normalization
        psf_stack.append(psf_t)
    return torch.stack(psf_stack) # (51, 1, 512, 512)




START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

TPARAMS['depth_range'] = get_depth_range().to(DEVICE)
print("Shape of Depth Range: ", TPARAMS['depth_range'].shape)
print("Depth Range: ", TPARAMS['depth_range'])

name_tmp = START_DATE +'aiobio-5mm~10mm objects' # Notation for individual wandb log name
NOTES = name_tmp + \
    '_[COLOR]_LPIPS_[SMOOTH_L1_BETA_0.1]_[DEPTH]_LPIPS_[SMOOTH_L1_BETA_0.1]+SI_LOG_V2_with_lambda_0.5_softmax_based_calculation' # Notation for individual wandb log name
    
PROJECT_NAME = 'Lensless depth imaging-HJA-2601'

if HPARAMS['WANDB_LOG'] == True:
    wandb.require("core")
    wandb.init(project=PROJECT_NAME,
            config = HPARAMS,
            notes = NOTES,
            name = name_tmp,
            save_code=True,)
else:
    print("Wandb logging is disabled")

def wandb_log(loglist, epoch, note):
    for key, val in loglist.items():
        if key in ['output_color', 'output_depth', 'label', 'label_color', 'image']:
            # Log only the first N batches
            try:
                # item = val[:8].cpu().detach()  # Get first 8 batches and move to CPU
                item = val.cpu().detach()
                log = wandb.Image(item)
                wandb.log({
                    f"{note.capitalize()} {key.capitalize()}": log,
                }, step=epoch+1)
            except Exception as e:
                print(f"Error logging {key}: {e}")
        else:
            try:
                log = val
                wandb.log({
                    f"{note.capitalize()} {key.capitalize()}": log,
                }, step=epoch+1)
            except Exception as e:
                print(f"Error logging {key}: {e}")

class LossFunction(nn.Module):
    """Loss function class for multiple loss function."""
    def __init__(self):
        super().__init__()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 =  nn.MSELoss() # L2 loss
        self.criterion_lpips = nn.DataParallel(lpips.LPIPS(net='vgg')).to(DEVICE)
        # self.criterion_lpips = nn.DataParallel(lpips.LPIPS(net='alex')).to(DEVICE)
        self.criterion_msssim_color = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.criterion_msssim_depth = MS_SSIM(data_range=1.0, size_average=True, channel=1)
        self.criterion_smooth_l1  = nn.SmoothL1Loss(beta=0.1) # default beta=1.0
        # self.criterion_bce = nn.BCELoss()  # Binary Cross-Entropy loss for adversarial loss
        self.criterion_silog_loss = scale_invariant_log_loss_v2_lambda(min_scale=0.05, max_scale=0.7, lambda_value=0.5) # min and max scale in meters
        
    def forward(self, output_color, output_depth, label_color, label, epoch=0):
        
        lpips_color_loss = torch.mean(self.criterion_lpips(output_color, label_color)) # take average of batch
        smooth_l1_color = self.criterion_smooth_l1(output_color, label_color)
        # blur_color = self.criterion_blur(output_color, label_color)
        
        output_depth_rgb = output_depth.repeat(1,3,1,1) # make RGB depth
        print(f"Output shape: {output_depth.shape}, Label shape: {label.shape}")
        label_depth_rgb = label.repeat(1,3,1,1) # make RGB depth
        lpips_depth_loss = torch.mean(self.criterion_lpips(output_depth_rgb, label_depth_rgb))       
        
        smooth_l1_depth = self.criterion_smooth_l1(output_depth, label)
        si_log_loss = self.criterion_silog_loss(output_depth, label)
        
        total_loss = lpips_color_loss + smooth_l1_color + smooth_l1_depth + si_log_loss + lpips_depth_loss
        
        rmse_depth = torch.sqrt(torch.mean((output_depth - label)**2))
        
        # Check for NaN losses
        nan_losses = []
        if torch.isnan(lpips_color_loss):
            nan_losses.append("LPIPS color loss")
        if torch.isnan(smooth_l1_color):
            nan_losses.append("Smooth L1 color loss")
        if torch.isnan(lpips_depth_loss):
            nan_losses.append("LPIPS depth loss")
        if torch.isnan(smooth_l1_depth):
            nan_losses.append("Smooth L1 depth loss")
        if torch.isnan(si_log_loss):
            nan_losses.append("SI Log loss")
        if torch.isnan(rmse_depth):
            nan_losses.append("RMSE depth")
        if torch.isnan(total_loss):
            nan_losses.append("Total loss")
        
        if nan_losses:
            error_message = f"NaN loss detected at epoch {epoch} in: {', '.join(nan_losses)}. Terminating training."
            print(error_message)
            # Terminate the script
            import sys
            sys.exit(1)
        return total_loss, lpips_color_loss, lpips_depth_loss, smooth_l1_color, smooth_l1_depth, rmse_depth, si_log_loss

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
        return (self.raw[index][0])

def train(train_parameters):
    train_parameters['model'].train()
    result = {k: 0 for k in ['loss_sum', 'PSNR_color', 'PSNR_depth', 'RMSE_depth']}
    
    bar = tqdm(train_parameters['trainset_loader'], position=1, leave=False, desc="Train Loop")
    for i, (image, label, label_color) in enumerate(bar):
        train_parameters['optimizer'].zero_grad()
        
        image = image.to(DEVICE)
        # label = label.to(DEVICE).unsqueeze(1)
        label = label.to(DEVICE)
        label_color = label_color.to(DEVICE)
        
        with autocast():
            # [메모리 최적화] 모델 연산 시 checkpointing 적용 고려 가능
            # 여기서는 우선 autocast만 유지
            intensity, soft_max_depth_stack = train_parameters['model'](image)
            
            depth_range = TPARAMS['depth_range'].view(1, -1, 1, 1)
            output_depth = 1.0 - torch.sum(depth_range * soft_max_depth_stack, dim=1, keepdim=True)

            total_loss, _, _, _, _, rmse_depth, _ = train_parameters['loss_function'](
                intensity, output_depth, label_color, label, i)
        
        scaler.scale(total_loss).backward()
        scaler.step(train_parameters['optimizer'])
        scaler.update()
        
        result['loss_sum'] += total_loss.item()
        result['RMSE_depth'] += rmse_depth.item()
        
        # [메모리 최적화] 100 배치마다 캐시 강제 정리
        if i % 100 == 0:
            torch.cuda.empty_cache()
            
        bar.set_description(f"Loss: {total_loss.item():.5f} | RMSE: {rmse_depth.item():.5f}")

    # 평균값 계산 및 텐서 정리
    for key in result: result[key] /= len(train_parameters['trainset_loader'])
    
    # WandB 로깅용 결과물 담기
    result.update({
        'image': image, 'label': label, 'label_color': label_color,
        'output_color': intensity, 'output_depth': output_depth
    })
    return result

def test(test_parameters):
    test_parameters['model'].eval()
    result = {k: 0 for k in ['loss_sum', 'PSNR_color', 'PSNR_depth', 'RMSE_depth']}
    
    with torch.no_grad():
        bar = tqdm(test_parameters['testset_loader'], position=2, leave=False, desc="Test Loop")
        for i, (image, label, label_color) in enumerate(bar):
            image, label, label_color = image.to(DEVICE), label.to(DEVICE), label_color.to(DEVICE)
            label = label.unsqueeze(1)
            
            intensity, soft_max_depth_stack = test_parameters['model'](image)
            depth_range = TPARAMS['depth_range'].view(1, -1, 1, 1)
            output_depth = 1.0 - torch.sum(depth_range * soft_max_depth_stack, dim=1, keepdim=True)
            
            total_loss, _, _, _, _, rmse_depth, _ = test_parameters['loss_function'](
                intensity, output_depth, label_color, label, i)
            
            result['loss_sum'] += total_loss.item()
            result['RMSE_depth'] += rmse_depth.item()

    for key in result: result[key] /= len(test_parameters['testset_loader'])
    result['image'], result['label'], result['label_color'] = image, label, label_color
    result['output_color'], result['output_depth'] = intensity, output_depth
    return result

def main():
    global TPARAMS
    # 1. 초기화 (GPU 메모리 청소)
    torch.cuda.empty_cache()
    
    psf_tensor = load_psf_for_train(HPARAMS['PSF_DIR'], target_size=(256, 256))
    TPARAMS['psf'] = psf_tensor.to(DEVICE)
    TPARAMS['depth_range'] = get_depth_range().to(DEVICE)

    # 2. 뎁스 물리 범위 로드
    TPARAMS['depth_range'] = get_depth_range().to(DEVICE)
    print(f"Depth range (normalized) step: {TPARAMS['depth_range'][1] - TPARAMS['depth_range'][0]:.4f}")

    # --- 3. 데이터셋 준비 (Random Split 로직) ---
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
    ])

    # 전체 데이터셋 로드
    full_raw = ImageFolder(root=HPARAMS['DATA_ROOT_RAW'], transform=transform)
    full_label = DatasetFolder(root=HPARAMS['DATA_ROOT_LABEL'], loader=npz_loader, extensions=['.npz'])
    full_image = ImageFolder(root=HPARAMS['DATA_ROOT_IMAGE'], transform=transform)

    # 일관된 랜덤 시드로 분할 (0104_113023 데이터셋 2만장 기준)
    total_len = len(full_raw)
    train_len = HPARAMS['TRAINSET_SIZE']
    test_len = total_len - train_len

    # 각각 동일한 시드로 쪼개서 짝이 맞도록 함
    seed = 4716
    train_raw, test_raw = random_split(full_raw, [train_len, test_len], generator=torch.Generator().manual_seed(seed))
    train_lbl, test_lbl = random_split(full_label, [train_len, test_len], generator=torch.Generator().manual_seed(seed))
    train_img, test_img = random_split(full_image, [train_len, test_len], generator=torch.Generator().manual_seed(seed))

    # 커스텀 ImageDataset 클래스를 이용해 묶어줌
    train_data = ImageDataset(train_raw, train_lbl, train_img)
    test_data = ImageDataset(test_raw, test_lbl, test_img)

    train_loader = DataLoader(train_data, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True, num_workers=HPARAMS['NUM_WORKERS'], pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=HPARAMS['BATCH_SIZE'], shuffle=False, num_workers=HPARAMS['NUM_WORKERS'], pin_memory=True)

    TPARAMS['trainset_loader'] = train_loader
    TPARAMS['testset_loader'] = test_loader

    # --- 4. 모델 및 학습 설정 -- -
    # n_classes=51 확인
    model = MWDNet_CPSF_depth(n_channels=3, n_classes=51, psf=TPARAMS['psf'], height=256, width=256)
    # model = nn.DataParallel(model).to(DEVICE)
    model = model.to(DEVICE)
    TPARAMS['model'] = model

    optimizer = optim.AdamW(model.parameters(), lr=HPARAMS['LR'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    TPARAMS['optimizer'] = optimizer
    TPARAMS['scheduler'] = scheduler
    TPARAMS['loss_function'] = LossFunction()

    # WandB 시작
    if HPARAMS['WANDB_LOG']:
        wandb.init(project='Lensless-Depth-AIOBIO', config=HPARAMS)
        wandb.watch(model)

    print('Training Start!')
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        train_result = train(TPARAMS) # 루프 내부에서 wandb_log 호출
        test_result = test(TPARAMS)
        
        # 로그 기록
        if HPARAMS['WANDB_LOG']:
            wandb_log(train_result, epoch, 'train')
            wandb_log(test_result, epoch, 'test')

        # 가중치 저장
        if (epoch + 1) % 10 == 0:
            weight_save(epoch, "HJA_Exp", HPARAMS['WEIGHT_SAVE_PATH'], TPARAMS, "model_51ch")

if __name__ == "__main__":
    main()