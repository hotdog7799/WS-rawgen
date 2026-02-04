import os
from typing import Any, Dict
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from train_utils import *

# [MODIFIED] Import the light model
# from forHJ import MWDNet_CPSF_depth
from forHJ_light import MWDNet_CPSF_depth_light as MWDNet_CPSF_depth

import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.utils import make_grid

from torchinfo import summary
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from pytorch_msssim import MS_SSIM, SSIM
import lpips

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
    "BATCH_SIZE": 32, # You might be able to increase this with the light model!
    "NUM_WORKERS": 16,
    "TRAINSET_SIZE": 18000,
    "EPOCHS_NUM": 1000,
    "LR": 1e-4,
    "H":256,
    "W":256,
    "DATA_ROOT_RAW": "/home/hjahn/mnt/ssd1/data/hjahn/syn_raw_image_color/0113_001134/raw/",
    "DATA_ROOT_IMAGE": "/home/hjahn/mnt/ssd1/data/hjahn/scene_and_label/image/",
    "DATA_ROOT_LABEL": "/home/hjahn/mnt/ssd1/data/hjahn/scene_and_label/label/",
    "DATA_ROOT_VAL_REAL": "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/260112/rawimage_uv/",
    "PSF_DIR": "/home/hjahn/mnt/nas/Grants/25_AIOBIO/experiment/260112/psf_color_aligned/",
    "WEIGHT_SAVE_PATH": "/home/hjahn/depth/WS-rawgen/pth_256_color_light/", # [MODIFIED] Changed path
    "CHECKPOINT_PATH": "",
}

if not os.path.exists(HPARAMS["WEIGHT_SAVE_PATH"]):
    os.makedirs(HPARAMS["WEIGHT_SAVE_PATH"])

TPARAMS = {}

# --- 2. PSF Loader ---
def load_psf_for_train(psf_dir, target_size=(HPARAMS["H"], HPARAMS["W"])):
    file_list = [
        f"{str(d).zfill(2)}p{p}.png" for d in range(5, 10) for p in range(10)
    ] + ["10p0.png"]
    psf_stack = []
    print(f"Loading {len(file_list)} PSFs from PNGs...")
    for fname in file_list:
        path = os.path.join(psf_dir, fname)
        psf_img = Image.open(path).convert("L")
        psf_t = v2.functional.to_image(psf_img).to(torch.float32)
        psf_t = v2.functional.resize(psf_t, target_size)
        # print(psf_t.max())
        psf_t = psf_t / 255.
        psf_stack.append(psf_t)
    return torch.stack(psf_stack)

START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

TPARAMS["depth_range"] = get_depth_range().to(DEVICE)
print("Shape of Depth Range: ", TPARAMS["depth_range"].shape)
print("Depth Range: ", TPARAMS["depth_range"])

name_tmp = (
    START_DATE + "aiobio-5mm~10mm-LIGHT-MODEL" # [MODIFIED] Name
)
NOTES = (
    name_tmp
    + "_[COLOR]_LPIPS_[SMOOTH_L1_BETA_0.1]_[DEPTH]_LPIPS_[SMOOTH_L1_BETA_0.1]+SI_LOG_V2_with_lambda_0.5_softmax_based_calculation"
)

PROJECT_NAME = "Lensless depth imaging-HJA-2601"

wandb.require("core")
wandb.init(
    project=PROJECT_NAME,
    config=HPARAMS,
    notes=NOTES,
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
        self.criterion_smooth_l1 = nn.SmoothL1Loss(beta=0.1)
        self.criterion_silog = scale_invariant_log_loss_v2_lambda(
            min_scale=0.05, max_scale=0.7, lambda_value=0.5
        )
        self.criterion_lpips = nn.DataParallel(lpips.LPIPS(net='vgg')).to(DEVICE)

    def forward(self, output_color, output_depth, label_color, label, epoch=0):
        smooth_l1_color = self.criterion_smooth_l1(output_color, label_color)
        smooth_l1_depth = self.criterion_smooth_l1(output_depth, label)

        lpips_color_loss = torch.mean(self.criterion_lpips(output_color, label_color))
        lpips_depth_loss = torch.tensor(0.0, device=DEVICE)
        
        silog_loss = self.criterion_silog(output_depth, label)

        total_loss = smooth_l1_color + smooth_l1_depth + 0.7 *silog_loss + lpips_color_loss

        rmse_depth = torch.sqrt(torch.mean((output_depth - label) ** 2) + 1e-8)

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
            silog_loss,
        )


class ImageDataset(Dataset):
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
    result["RMSE_depth"] = 0
    ACCUM_STEPS = 1

    bar = tqdm(
        train_parameters["trainset_loader"], position=1, leave=False, desc="Train Loop"
    )
    for i, (image, label, label_color) in enumerate(bar):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        label_color = label_color.to(DEVICE)
        
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
            train_parameters["optimizer"].zero_grad()

        result["loss"] += total_loss.item()
        result["RMSE_depth"] += rmse_depth.item()

        bar.set_description(
            f"Loss: {total_loss.item():.5f} | RMSE: {rmse_depth.item():.5f}"
        )

    result["loss"] /= len(train_parameters["trainset_loader"])
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
            result["RMSE_depth"] += rmse_depth.item()

    result["loss"] /= len(test_parameters["testset_loader"])
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
            
            intensity, soft_max_depth_stack = val_parameters["model"](image)
            
            depth_range = TPARAMS["depth_range"].view(1, -1, 1, 1)
            output_depth = 1.0 - torch.sum(
                depth_range * soft_max_depth_stack, dim=1, keepdim=True
            )
            
            all_inputs.append(image.cpu())
            all_intensities.append(intensity.cpu())
            all_depths.append(output_depth.cpu())

    cat_inputs = torch.cat(all_inputs, dim=0)
    cat_intensities = torch.cat(all_intensities, dim=0)
    cat_depths = torch.cat(all_depths, dim=0)

    result = {}
    result["input"] = make_grid(cat_inputs, nrow=5, padding=2)
    result["output_color"] = make_grid(cat_intensities, nrow=5, padding=2)
    result["output_depth"] = make_grid(cat_depths, nrow=5, padding=2)
    
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

    full_raw = ImageFolder(root=HPARAMS["DATA_ROOT_RAW"], transform=transform)
    full_label = DatasetFolder(
        root=HPARAMS["DATA_ROOT_LABEL"], loader=npz_loader, extensions=[".npz"]
    )
    full_image = ImageFolder(root=HPARAMS["DATA_ROOT_IMAGE"], transform=transform)
    real_val_dataset = ImageFolder(root=HPARAMS["DATA_ROOT_VAL_REAL"], transform=transform)
    
    total_len = len(full_raw)
    train_len = HPARAMS["TRAINSET_SIZE"]
    test_len = total_len - train_len

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
        batch_size=4,
        shuffle=False,
        num_workers=HPARAMS["NUM_WORKERS"],
        pin_memory=True,
    )
    TPARAMS["trainset_loader"] = train_loader
    TPARAMS["testset_loader"] = test_loader
    TPARAMS["real_val_loader"] = real_val_loader

    # --- 4. Model ---
    # [MODIFIED] Using light model (dim=16 default in class)
    model = MWDNet_CPSF_depth(
        n_channels=3, n_classes=51, psf=TPARAMS["psf"], height=HPARAMS["H"], width=HPARAMS["W"], dim=16
    )
    print(f"Using {torch.cuda.device_count()} GPUs!")
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TOTAL PARAMETERS: {total_params:,}")

    model = nn.DataParallel(model).to(DEVICE)
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

        train_result = train(TPARAMS)
        test_result = test(TPARAMS)
        real_result = validate_real(TPARAMS)

        current_test_loss = test_result["loss"]

        wandb_log(train_result, epoch, "Train")
        wandb_log(test_result, epoch, "Test")
        wandb_log(real_result, epoch, "Real")

        if min_loss > current_test_loss:
            min_loss = current_test_loss
            weight_save(epoch, START_DATE, HPARAMS["WEIGHT_SAVE_PATH"], TPARAMS, "model_light_16ch")

if __name__ == "__main__":
    main()
