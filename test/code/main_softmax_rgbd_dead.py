import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# import config
from utils import *

# from MWDNs import MWDNet_CPSF_RGBD_large_w_softmax_output_add_DepthRefine
from mwdnet_cpsf_rgbd_model import MWDNet_CPSF_RGBD_large_w_softmax_change_wiener_reg

import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder, DatasetFolder
# from torchsummary import summary # To identify network structure
from torchinfo import summary

from pytorch_msssim import MS_SSIM, SSIM
import lpips

# from accelerate import Accelerator
# from accelerate.utils import tqdm
from tqdm import tqdm
# import wandb

import warnings
warnings.filterwarnings('ignore')

from einops import rearrange, reduce, repeat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(DEVICE))        
        
# Define network hyperparameters:
HPARAMS = {
    'IN_CHANNEL' : 3,
    # 'OUT_CHANNEL': 42,
    'OUT_CHANNEL': 11, #14
    'BATCH_SIZE': 2,
    'NUM_WORKERS': 4,
    # 'TRAINSET_SIZE' : 10000,
    'EPOCHS_NUM': 100,
    'LR': 1e-4,
    
   # --- 경로 수정 (가장 중요) ---
    'DATASET_ROOT': '/mnt/ssd1/depth_imaging/dataset_ssd2/HJA_1/1500x1500/train/', # <-- 데이터셋 부모 폴더
    'TRAIN_PATH': 'raw/', # <-- DATASET_ROOT 기준 하위 경로
    'TRAIN_LABEL_PATH_COLOR': 'image/', # <-- DATASET_ROOT 기준 하위 경로
    'TRAIN_LABEL_PATH': 'label/', # <-- DATASET_ROOT 기준 하위 경로
    # 'TRAIN_PATH': '/mnt/ssd1/depth_imaging/dataset_ssd2/HJA_1/1500x1500/train/raw/',
    
    # 'TRAIN_LABEL_PATH_COLOR': '/mnt/ssd1/depth_imaging/dataset_ssd2/HJA_1/1500x1500/train/image/',
    
    # 'TRAIN_LABEL_PATH': '/mnt/ssd1/depth_imaging/dataset_ssd2/HJA_1/1500x1500/train/label/',

    # 'TEST_PATH': '/mnt/ssd1/depth_imaging/dataset_ssd1/current_blender_dataset/20240909_testdataset_210obj_2X_FOV_6K_clipping_0.1_1100_cm_random_select_from_5_to_20_obj_scale_from_0.5_to_1.5_resize_add_linear_scale_y_l_10_65/v14_poission_disk_avgdist_140um_[real_210um]_with_simulated_2xPSF_in_250429_and_250408_modified_angular_response_250519_masked_rgb_png_raw_AWGN_mean_0_rand_std_0_to_0.01/',
    
    
    # 'TEST_LABEL_PATH_COLOR': '/mnt/ssd1/depth_imaging/dataset_ssd1/current_blender_dataset/20240909_testdataset_210obj_2X_FOV_6K_clipping_0.1_1100_cm_random_select_from_5_to_20_obj_scale_from_0.5_to_1.5_resize_add_linear_scale_y_l_10_65/image_gss_masked_png/',
    
    
    # 'TEST_LABEL_PATH': '/mnt/ssd1/depth_imaging/dataset_ssd1/current_blender_dataset/20240909_testdataset_210obj_2X_FOV_6K_clipping_0.1_1100_cm_random_select_from_5_to_20_obj_scale_from_0.5_to_1.5_resize_add_linear_scale_y_l_10_65/label_cc/',
# --- PSF Indices ---
    'psf_idx_list': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    
    'VAL_PATH': '/mnt/ssd1/depth_imaging/dataset_ssd1/current_realworld_dataset/20250429_validation_selected_batch_10/',
    'PSF_PATH': '/home/hotdog/files_251026/image_stack_cropped_2nd_251024.mat',

    
    'WEIGHT_SAVE': True,
    'WANDB_LOG': False,
    'SLACK_ALERT': False,

    # 'WEIGHT_SAVE_PATH' : '/mnt/ssd1/depth_imaging/output/mwdn_weight/MWDN3D_weight_v14_poisson_140um_rand_dot/',
    # 'FIG_SAVE_PATH' : '/mnt/ssd1/depth_imaging/output/mwdn_figure/MWDN3D_fig_v14_poisson_140um/',
    
    # --- 저장 경로 ---
    'WEIGHT_SAVE_PATH' : '/mnt/ssd1/depth_imaging/dataset_ssd2/HJA/training_results/weights/', # SSD 경로
    'FIG_SAVE_PATH' : '/mnt/ssd1/depth_imaging/dataset_ssd2/HJA/training_results/figures/', # SSD 경로

    'CHECKPOINT_PATH': '',

}

# Check if the weight save path and fig save path exist, if not create them
# --- 저장 경로 생성 확인 ---
if HPARAMS['WEIGHT_SAVE']:
    # Optional: Add model name subdir
    model_save_name = "MWDN_CPSF_HJA_FailureCase_1500px" # Use START_DATE
    weight_save_subdir = os.path.join(HPARAMS['WEIGHT_SAVE_PATH'], model_save_name)
    fig_save_subdir = os.path.join(HPARAMS['FIG_SAVE_PATH'], model_save_name)
    if not os.path.exists(weight_save_subdir): os.makedirs(weight_save_subdir)
    if not os.path.exists(fig_save_subdir): os.makedirs(fig_save_subdir)
    # Update HPARAMS if using subdirs for saving weights/figs later
    # HPARAMS['WEIGHT_SAVE_PATH'] = weight_save_subdir
    # HPARAMS['FIG_SAVE_PATH'] = fig_save_subdir

def npz_loader(path):
    try:
        data = np.load(path)
        if 'label' in data:
            sample = torch.from_numpy(data['label']).float() # float32
            # Optional: Add dimension check/squeeze if needed here
            return sample
        else:
            raise KeyError(f"'label' key not found in {path}")
    except Exception as e:
        print(f"Error loading npz {path}: {e}")
        return None

# Define training parameters:
TPARAMS = {}

# Load PSF stack as .mat file
# psf = psf_Load(HPARAMS['PSF_PATH'],'psf_stack_resized_rotated')
psf = psf_Load(HPARAMS['PSF_PATH'],'psf_stack')
# TPARAMS['psf'] = psf[:,-1,:,:].unsqueeze(1) # Select the last channel and add channel dimension

TPARAMS['psf'] = psf
print("Shape of PSF: ", TPARAMS['psf'].shape)
print("Device of PSF: ", psf.device)
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

TPARAMS['depth_range'] = get_depth_range().to(DEVICE)
print("Shape of Depth Range: ", TPARAMS['depth_range'].shape)
print("Depth Range: ", TPARAMS['depth_range'])

name_tmp = START_DATE +'_nnDP_20K_v14_MWDN3D_increase_params_w_UpG_ConvT_softmax_single_decoder_DepthRefine' # Notation for individual wandb log name
NOTES = name_tmp + \
    '_[COLOR]_LPIPS_[SMOOTH_L1_BETA_0.1]_[DEPTH]_LPIPS_[SMOOTH_L1_BETA_0.1]+SI_LOG_V2_with_lambda_0.5_softmax_based_calculation' # Notation for individual wandb log name
    
PROJECT_NAME = 'Lensless depth imaging'

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
            
            # Send Slack alert
            if HPARAMS['SLACK_ALERT']:
                # send_slack_alert(epoch, "NaN", "NaN", "NaN", error_message)
                send_slack_alert(epoch, "NaN", "NaN", "NaN", "NaN", "NaN", error_message)
            
            # Terminate the script
            import sys
            sys.exit(1)
        
        return total_loss, lpips_color_loss, lpips_depth_loss, smooth_l1_color, smooth_l1_depth, rmse_depth, si_log_loss

def train(train_parameters):

    train_parameters['model'].train()
    
    result = {k: 0 for k in [
        'loss_sum', 'PSNR_color', 'PSNR_depth', 'LPIPS_color', 'LPIPS_depth',
        'SSIM_color', 'SSIM_depth', 'RMSE_depth', 'SMOOTH_L1_color', 'SMOOTH_L1_depth', 'SI_LOG_depth',]}
    
    train_parameters['train_bar'] = tqdm(train_parameters['trainset_loader'], position=1, leave=False, disable=False)
    for i, (image, label, label_color) in enumerate(train_parameters['train_bar']):
        train_parameters['optimizer'].zero_grad()
        # train_parameters['optimizer_discriminator'].zero_grad()
        image, label, label_color = image.to(DEVICE), label.to(DEVICE), label_color.to(DEVICE)
        label = label.unsqueeze(1)
        
        # result['output_color'], result['output_depth'] = train_parameters['model'](image) # depth            
        aif, soft_max_depth_stack = train_parameters['model'](image) # depth       
        result['output_color']= aif
        
        # Broadcast depth_range to match the shape of soft_max_depth_stack
        depth_range_expanded = TPARAMS['depth_range'].unsqueeze(0).unsqueeze(2).unsqueeze(3)  # shape (1, 42, 1, 1)
        depth_range_expanded = depth_range_expanded.expand(soft_max_depth_stack.size(0), -1, soft_max_depth_stack.size(2), soft_max_depth_stack.size(3))  # shape (B, 42, H, W)

        # Now compute the weighted sum (depth map) using the softmax probabilities
        output_depth_map = torch.sum(depth_range_expanded * soft_max_depth_stack, dim=1, keepdim=True).type(torch.FloatTensor).to(DEVICE)
        result['output_depth'] = (1.0 - output_depth_map) # reverse the depth map

        total_loss, lpips_color_loss, lpips_depth_loss, smooth_l1_color, smooth_l1_depth, rmse_depth, si_log_loss = \
            train_parameters['loss_function'](result['output_color'], result['output_depth'], label_color, label, i)
        
        # Update generator
        total_loss.backward()
        train_parameters['optimizer'].step()
        
        train_parameters['train_bar'].set_description(
            'Train :: total loss: {:.5}. silog: {:.5e}. sl1_depth: {:.5e}. sl1_color: {:.5e}. psnr_c: {:.5}. psnr_d: {:.5}. rmse_d: {:.5e}.'.format(
                result['loss_sum'] / (i+1),
                result['SI_LOG_depth'] / (i+1),
                result['SMOOTH_L1_depth'] / (i+1),
                result['SMOOTH_L1_color'] / (i+1),
                result['PSNR_color'] / (i+1),
                result['PSNR_depth'] / (i+1),
                result['RMSE_depth'] / (i+1),
            ))
        
        with torch.no_grad():
            result['loss_sum'] += total_loss.sum().item()
            result['PSNR_color'] += PSNR(result['output_color'], label_color)
            result['PSNR_depth'] += PSNR(result['output_depth'], label)
            result['LPIPS_color'] += lpips_color_loss
            result['LPIPS_depth'] += lpips_depth_loss
            # result['BLUR_color'] += blur_color
            result['SSIM_color'] += train_parameters['loss_function'].criterion_msssim_color(result['output_color'], label_color)
            result['SSIM_depth'] += train_parameters['loss_function'].criterion_msssim_depth(result['output_depth'], label)
            result['RMSE_depth'] += rmse_depth
            result['SMOOTH_L1_color'] += smooth_l1_color
            result['SMOOTH_L1_depth'] += smooth_l1_depth
            result['SI_LOG_depth'] += si_log_loss

    train_parameters['scheduler'].step(result['loss_sum']) # ReduceLROnPlateau
    result['label'] = label
    result['label_color'] = label_color
    result['image'] = image

    for key in result:
        if key not in ['label', 'label_color', 'image']:
            result[key] /= len(train_parameters['trainset_loader'])
    
    return result

def test(test_parameters):

    test_parameters['model'].eval()
    
    result = {k: 0 for k in [
            'loss_sum', 'PSNR_color', 'PSNR_depth', 'LPIPS_color', 'LPIPS_depth',
            'SSIM_color', 'SSIM_depth', 'RMSE_depth', 'SMOOTH_L1_color', 'SMOOTH_L1_depth', 'SI_LOG_depth']}
    
    test_parameters['test_bar'] = tqdm(test_parameters['testset_loader'], position=2, leave=False, disable=True)
    
    with torch.no_grad():
        for i, (image, label, label_color) in enumerate(test_parameters['test_bar']):
            test_parameters['optimizer'].zero_grad()
            image, label, label_color = image.to(DEVICE), label.to(DEVICE), label_color.to(DEVICE)
            label = label.unsqueeze(1)
            
            # result['output_color'], result['output_depth'] = train_parameters['model'](image) # depth            
            aif, soft_max_depth_stack = test_parameters['model'](image) # depth       
            result['output_color']= aif
            
            # Broadcast depth_range to match the shape of soft_max_depth_stack
            depth_range_expanded = TPARAMS['depth_range'].unsqueeze(0).unsqueeze(2).unsqueeze(3)  # shape (1, 42, 1, 1)
            depth_range_expanded = depth_range_expanded.expand(soft_max_depth_stack.size(0), -1, soft_max_depth_stack.size(2), soft_max_depth_stack.size(3))  # shape (B, 42, H, W)

            # Now compute the weighted sum (depth map) using the softmax probabilities
            output_depth_map = torch.sum(depth_range_expanded * soft_max_depth_stack, dim=1, keepdim=True).type(torch.FloatTensor).to(DEVICE)
            result['output_depth'] = (1.0 - output_depth_map) # reverse the depth map
            
            total_loss, lpips_color_loss, lpips_depth_loss, smooth_l1_color, smooth_l1_depth, rmse_depth, si_log_loss = \
            test_parameters['loss_function'](result['output_color'], result['output_depth'], label_color, label, i)

            result['loss_sum'] += total_loss.sum().item()
            result['PSNR_color'] += PSNR(result['output_color'], label_color)
            result['PSNR_depth'] += PSNR(result['output_depth'], label)
            result['LPIPS_color'] += lpips_color_loss
            result['LPIPS_depth'] += lpips_depth_loss
            result['SSIM_color'] += test_parameters['loss_function'].criterion_msssim_color(result['output_color'], label_color)
            result['SSIM_depth'] += test_parameters['loss_function'].criterion_msssim_depth(result['output_depth'], label)
            result['RMSE_depth'] += rmse_depth
            result['SMOOTH_L1_color'] += smooth_l1_color
            result['SMOOTH_L1_depth'] += smooth_l1_depth
            result['SI_LOG_depth'] += si_log_loss
            
            test_parameters['test_bar'].set_description('Test :: loss: {:.5}.'.format(result['loss_sum'] / (i+1)))
    
    test_parameters['test_bar'].close()
    
    result['label'] = label
    result['label_color'] = label_color
    result['image'] = image
    
    for key in result:
        if key not in ['label', 'label_color', 'image']:
            result[key] /= len(test_parameters['testset_loader'])
    
    return result

def validation(val_parameters):
    # Model evaluation function.
    val_parameters['model'].eval()
    result = {} 
    val_parameters['eval_bar'] = tqdm(val_parameters['valset_loader'], position=2, leave=False, disable=True, colour='CYAN')
    with torch.no_grad():
        for i, (image) in enumerate(val_parameters['eval_bar']):
            image= image.to(DEVICE)
            for j in range(len(image)):
                image[j] = image[j] / torch.max(image[j])
            # result['output_color'], result['output_depth']  = val_parameters['model'](image)
            aif, soft_max_depth_stack = val_parameters['model'](image) # depth       
            result['output_color']= aif
            
            # Broadcast depth_range to match the shape of soft_max_depth_stack
            depth_range_expanded = TPARAMS['depth_range'].unsqueeze(0).unsqueeze(2).unsqueeze(3)  # shape (1, 42, 1, 1)
            depth_range_expanded = depth_range_expanded.expand(soft_max_depth_stack.size(0), -1, soft_max_depth_stack.size(2), soft_max_depth_stack.size(3))  # shape (B, 42, H, W)

            # Now compute the weighted sum (depth map) using the softmax probabilities
            output_depth_map = torch.sum(depth_range_expanded * soft_max_depth_stack, dim=1, keepdim=True).type(torch.FloatTensor).to(DEVICE)
            result['output_depth'] = (1.0 - output_depth_map) # reverse the depth map
    result['image'] = image
    return result

# # 코드 내부에 있는 ImageDataset 클래스 정의를 찾아서 수정
class ImageDataset(Dataset):
    """Loads data from Subset objects containing (raw_path, label_obj, label_color_obj) tuples."""
    def __init__(self, subset_raw, subset_label, subset_label_color):
        # subset_raw contains tuples like (<PIL Image obj>, 0) from ImageFolder
        # subset_label contains tuples like (<loaded npz tensor>, 0) from DatasetFolder
        # subset_label_color contains tuples like (<PIL Image obj>, 0) from ImageFolder
        self.subset_raw = subset_raw
        self.subset_label = subset_label
        self.subset_label_color = subset_label_color

        if not (len(subset_raw) == len(subset_label) == len(subset_label_color)):
            raise ValueError("Subset length mismatch!")
        self.len = len(subset_raw)
        print(f"Initialized ImageDataset with {self.len} samples.")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Subset objects return the original dataset's item when indexed
        # Raw image is already loaded and transformed by ImageFolder's transform
        raw_img, _ = self.subset_raw[index] # Get the transformed tensor

        # Label depth is already loaded by DatasetFolder's loader and converted to tensor
        label_depth, _ = self.subset_label[index] # Get the loaded tensor

        # Label color is already loaded and transformed by ImageFolder's transform
        label_color, _ = self.subset_label_color[index] # Get the transformed tensor

        # --- 차원 일관성 확인 (필요시) ---
        # if label_depth is not None and label_depth.dim() != 2:
        #     print(f"Warning: Unexpected depth label dim {label_depth.dim()} at index {index}")
        #     # Attempt to fix or return None
        #     if label_depth.dim() == 3 and label_depth.shape[0] == 1:
        #         label_depth = label_depth.squeeze(0)
        #     else:
        #         return None, None, None # Skip this sample

        if raw_img is None or label_depth is None or label_color is None:
             print(f"Warning: Found None data at index {index}. Skipping.")
             return None, None, None

        return raw_img, label_depth, label_color
    
class ImageDataset_val(Dataset):
    """Custom image file dataset loader."""
    
    def __init__(self, raw_path, transform_image=None):
        self.raw = raw_path
        self.len = len(raw_path)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.raw[index][0])

def main():

    # Define transformer
    transformer_raw = v2.Compose([
        v2.ToTensor(),
    ])
    
    transformer_label_color = v2.Compose([
        v2.ToTensor(),

    ])
    
        # Define transformer
    transformer_val = v2.Compose([
        v2.ToTensor(),
        v2.Resize((352,512), interpolation=Image.BICUBIC),
        # transforms.Grayscale(num_output_channels=1),
    ])
    
    
# --- 수정: ImageFolder/DatasetFolder 경로 설정 ---
    # root 경로는 각 데이터 타입의 부모 폴더까지만 지정
    # ImageFolder/DatasetFolder가 하위 폴더('raw', 'image', 'label')를 클래스로 인식하려 할 수 있으나,
    # 하위에 폴더가 하나뿐이면 보통 그 폴더 안의 파일을 모두 로드함.
    # 만약 'dummy' 같은 클래스 폴더를 만들어야 한다면 구조 변경 필요.
    # 우선 이대로 시도.

    try:
        # Load Raw Images (Input)
        # Assuming files are directly inside 'raw' folder (no sub-class folders)
        trainset_raw_full = ImageFolder(root=os.path.join(HPARAMS['DATASET_ROOT'], HPARAMS['TRAIN_PATH']), # .../train/raw/
                                        transform=transformer_raw)
        # Load Depth Labels (NPZ)
        trainset_label_full = DatasetFolder(root=os.path.join(HPARAMS['DATASET_ROOT'], HPARAMS['TRAIN_LABEL_PATH']), # .../train/label/
                                            loader=npz_loader,
                                            extensions=['.npz'])
        # Load Color Labels (GT RGB)
        trainset_label_color_full = ImageFolder(root=os.path.join(HPARAMS['DATASET_ROOT'], HPARAMS['TRAIN_LABEL_PATH_COLOR']), # .../train/image/
                                                transform=transformer_label_color)
    except FileNotFoundError as e:
         print(f"Error loading initial dataset: {e}")
         print("Please ensure HPARAMS paths are correct and data exists.")
         return # Exit if initial loading fails

    data_size = len(trainset_raw_full)
    print(f'Total dataset size found: {data_size}')
    if not (data_size == len(trainset_label_full) == len(trainset_label_color_full)):
        raise ValueError("Initial dataset sizes mismatch! Check folder contents.")


    # --- Train/Test 분할 (기존 코드 유지) ---
    testset_ratio = 0.1 # 10% for testing
    testset_size = int(data_size * testset_ratio)
    trainset_size = data_size - testset_size
    HPARAMS['TRAINSET_SIZE'] = trainset_size # Update HPARAMS

    print(f'Splitting into Trainset size: {trainset_size}, Testset size: {testset_size}')

    # split the trainset and testset respectively
    # Use the full datasets loaded above
    trainset_raw, testset_raw = torch.utils.data.random_split(trainset_raw_full, [trainset_size, testset_size],\
                                                                generator=torch.Generator().manual_seed(4716))
    trainset_label, testset_label = torch.utils.data.random_split(trainset_label_full, [trainset_size, testset_size],\
                                                                generator=torch.Generator().manual_seed(4716))
    trainset_label_color, testset_label_color = torch.utils.data.random_split(trainset_label_color_full, [trainset_size, testset_size],\
                                                                generator=torch.Generator().manual_seed(4716))

    # --- 수정: 수정된 ImageDataset 사용 (Subset 객체 전달) ---
    train_load = ImageDataset(trainset_raw, trainset_label, trainset_label_color)
    test_load = ImageDataset(testset_raw, testset_label, testset_label_color)

    # Validation dataset 로딩 (기존 방식 유지)
    try:
        valset_raw_folder = ImageFolder(root=HPARAMS['VAL_PATH'], transform=transformer_val)
        val_load = ImageDataset_val(valset_raw_folder) # Assuming ImageDataset_val takes ImageFolder
    except FileNotFoundError:
        print(f"Validation path {HPARAMS['VAL_PATH']} not found. Skipping validation.")
        val_load = None

    # --- DataLoader 생성 (collate_fn 추가) ---
    def collate_fn_filter_none(batch):
        batch = [sample for sample in batch if sample is not None and all(s is not None for s in sample)]
        if not batch: return None
        try:
             return torch.utils.data.dataloader.default_collate(batch)
        except RuntimeError as e:
             print(f"Error during collate_fn: {e}")
             # Print shapes of items in batch for debugging
             for i, sample in enumerate(batch):
                 print(f" Batch item {i} shapes: {[s.shape if hasattr(s, 'shape') else type(s) for s in sample]}")
             return None # Skip batch if collate fails

    TPARAMS['trainset_loader'] = DataLoader(
        train_load, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True,
        num_workers=HPARAMS['NUM_WORKERS'], pin_memory=True,
        collate_fn=collate_fn_filter_none)
    TPARAMS['testset_loader'] = DataLoader(
        test_load, batch_size=HPARAMS['BATCH_SIZE'], shuffle=False,
        num_workers=HPARAMS['NUM_WORKERS'], pin_memory=True,
        collate_fn=collate_fn_filter_none)
    if val_load:
        TPARAMS['valset_loader'] = DataLoader(
            val_load, batch_size=HPARAMS['BATCH_SIZE'], shuffle=False,
            num_workers=HPARAMS['NUM_WORKERS'], pin_memory=True,
            collate_fn=lambda b: collate_fn_filter_none(b) if b[0] is not None else None) # Adapt for val loader if needed
    else:
        TPARAMS['valset_loader'] = None

    # --- 모델 초기화 (psf_indices 전달 추가) ---
    # !! 모델 코드 수정 필요할 수 있음 !!
    TPARAMS['model'] =  MWDNet_CPSF_RGBD_large_w_softmax_change_wiener_reg(
        in_channels=HPARAMS['IN_CHANNEL'],
        out_channels=HPARAMS['OUT_CHANNEL'], # 값 확인 필수!
        psf = TPARAMS['psf']
        # psf_indices=HPARAMS['psf_idx_list'] # <-- 인덱스 전달
    )

    # Print model summary
    # summary(TPARAMS['model'].to(DEVICE), (HPARAMS['BATCH_SIZE'], 3, 352, 512))
    # print("DEVICE: ", DEVICE)
    # # parallelize the models
    TPARAMS['model'] = nn.DataParallel(TPARAMS['model']).to(DEVICE)

    TPARAMS['optimizer'] = optim.AdamW(TPARAMS['model'].parameters(), lr=HPARAMS['LR'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    
    TPARAMS['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(TPARAMS['optimizer'], mode='min', factor=0.1, patience=5,
                                                                threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0,
                                                                eps=1e-8)
    # 예시: summary(TPARAMS['model'].to(DEVICE), input_size=(HPARAMS['BATCH_SIZE'], 3, 1500, 1500)) # <-- Input size 수정!
    # summary(TPARAMS['model'].to(DEVICE), input_size=(HPARAMS['BATCH_SIZE'], 3, 1500, 1500)) # Assumes 1500x1500
    # print("DEVICE: ", DEVICE)
    # TPARAMS['model'] = nn.DataParallel(TPARAMS['model']).to(DEVICE)
    # TPARAMS['optimizer'] = optim.AdamW(...)
    # TPARAMS['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(...)

    if HPARAMS['CHECKPOINT_PATH'] != '':
        print(f"Resuming training from checkpoint")
        checkpoint = torch.load(HPARAMS['CHECKPOINT_PATH'])
        TPARAMS['model'].load_state_dict(checkpoint['model_state_dict'])
        TPARAMS['epoch_now'] = checkpoint['epoch'] + 1  # Start the next epoch
    else:
        print("Starting training from scratch")
        TPARAMS['epoch_now'] = 0
        
    
    TPARAMS['loss_function'] = LossFunction()
    
    
    if HPARAMS['WANDB_LOG'] == True:
        wandb.watch((TPARAMS['model']), log='all')

    print('Train Start!')
    
    BAR = tqdm(range(HPARAMS['EPOCHS_NUM']), position=0, leave=True, colour='YELLOW')
    
    try:
        # accelerator.log({"Epoch": 1}, step=0)
        
        if HPARAMS['WANDB_LOG'] == True:
            wandb.log({"Epoch": 1}, step=0)
    
        for epoch in BAR:
            TPARAMS['epoch_now'] = epoch
            train_result = train(TPARAMS)
            test_result = test(TPARAMS)
            TPARAMS['scheduler'].step(test_result["loss_sum"])
            BAR.set_description('{0} Epoch - Train Loss : {1:.5}. - Test Loss : {2:.5}. - RMSE : {3:.5}. - Scheduler LR : {4:.5e}'.format(epoch, train_result["loss_sum"], test_result["loss_sum"], test_result["RMSE_depth"], TPARAMS['optimizer'].param_groups[0]['lr']))
            val_result = validation(TPARAMS)
            # draw_fig_validation(epoch, START_DATE, HPARAMS['FIG_SAVE_PATH'], val_result)
            
            if HPARAMS['WANDB_LOG'] == True:
                wandb_log(train_result, epoch, 'train')
                wandb_log(test_result, epoch, 'test')
                wandb_log(val_result, epoch, 'validation')
                wandb.log({"Epoch": epoch+1}, step=epoch+2)
            
            if HPARAMS['WEIGHT_SAVE'] == True:
                weight_save(epoch,START_DATE,HPARAMS['WEIGHT_SAVE_PATH'],TPARAMS, name_tmp)
            
            
            if HPARAMS['SLACK_ALERT'] == True:
                if epoch == 0:
                    message = f"Training started with the following NOTES: {NOTES}"
                    try:
                        response = client.chat_postMessage(channel=slack_channel, text=message)
                        assert response["message"]["text"] == message
                    except SlackApiError as e:
                        print(f"Error posting message to Slack: {e.response['error']}")
                    
                # Send Slack alert
                send_slack_alert(epoch, train_result["loss_sum"], test_result["loss_sum"], train_result["RMSE_depth"], test_result["RMSE_depth"], TPARAMS['optimizer'].param_groups[0]['lr'])
            
            # torch.cuda.empty_cache()
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Ending training...")
        # accelerator.end_training()
    
    finally:
        # Perform any cleanup or final operations here
        # accelerator.end_training()
        print("Training finished.")
        
if __name__ == "__main__":
    main()
