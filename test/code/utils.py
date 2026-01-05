import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
from scipy import io # to read .mat file
from einops import rearrange, reduce, repeat
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from pytorch_msssim import MS_SSIM, SSIM
    
def norm_8bit_tensor(X):
    max_X = 1
    max_X = torch.max(X)
    if max_X == 0:
        max_X = torch.ones_like(max_X).to(DEVICE)
    X = (X / max_X * 255)
    return X
    
# alternating axis of image for Display...(C,W,H to W,H,C)
def alt_axis(X):
    return np.einsum('ijk->jki', X)

def tensor_norm_double(X):
    return (X/torch.max(X))

def center_half_crop(input_tensor):
    """
    Crop the center half of the input tensor.
    
    Args:
    - input_tensor (torch.Tensor): Input tensor of shape (C, H, W).
    
    Returns:
    - torch.Tensor: Cropped tensor of shape (C, H//2, W//2).
    """

    h, w = input_tensor.shape[-2], input_tensor.shape[-1]
    
    # Calculate the cropping box for half the height and width
    top = h // 4
    left = w // 4
    bottom = top + h // 2
    right = left + w // 2
    
    # Crop and return the center half
    return input_tensor[:, :, top:bottom, left:right]

def get_depth_range():
    # wd_blender = np.concatenate(([5], np.arange(10, 71, 10))) #20240602 add [5]
    # wd_measured = np.concatenate((
    # np.arange(5.0, 16.5, 0.5),  # 5.0 to 16.0 with step 0.5
    # np.arange(17.0, 24.0, 1.0), # 17.0 to 23.0 with step 1.0
    # np.arange(25.0, 34.0, 2.0), # 25.0 to 33.0 with step 2.0
    # [36.0, 38.0, 45.0, 50.0, 55.0, 60.0, 70.0] # Additional values
    # )) # avg dist 140 um poission random dot 20250103
    wd_blender = np.linspace(5.0, 10.0, 51)
    wd_measured = np.linspace(5.0, 10.0, 51)
    
    
    # depth value measure by blender environment
    # depth_value = [0, 0.0769231, 0.230769, 0.384615, 0.537981, 0.691827, 0.846154, 1] # 20240420, inverted
    depth_value = np.linspace(0.0, 1.0, 51)  # 하드코딩
    # depth_value = np.linspace(0.0, 1.0, n_depth_levels)
    
    # Interpolate depth values for wd_measured using wd_blender and depth_value
    # range_log = np.interp(wd_measured, wd_blender, depth_value)
    # range_log = torch.from_numpy(range_log)
    
    range_log = torch.from_numpy(depth_value).float()
    return range_log

def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * torch.log10(1. / mse)

# def npz_loader(path):
#     sample = torch.from_numpy(np.load(path)['dmap'])
#     return sample.float()
def npz_loader(path):
    # 1. 파일 로드
    sample = np.load(path)['dmap']
    sample = torch.from_numpy(sample).float() # (H, W)
    
    # 2. 채널 차원 추가 (H, W) -> (1, H, W)
    if len(sample.shape) == 2:
        sample = sample.unsqueeze(0)
    
    # 3. [핵심] 정답 라벨도 모델 입력과 똑같이 256x256으로 리사이즈
    # Loss 계산 시 차원을 맞추기 위해 반드시 필요합니다.
    sample = F.interpolate(sample.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
    
    return sample

def npy_loader(path):
    sample = torch.from_numpy(np.load(path)).permute(2,0,1).type(torch.FloatTensor)
    return sample

def npy_loader_grayscale(path):
    sample = torch.from_numpy(np.load(path)).permute(2,0,1).type(torch.FloatTensor)
    sample = sample.mean(dim=0, keepdim=True)  # Average across color channels to get grayscale
    return sample

def psf_Load(path, variable = 'psf_stack'):
    # Load the .mat file
    psf_mat_file = io.loadmat(path)
    
    # Convert the loaded PSF stack to a numpy array and then to a PyTorch tensor
    
    # psf = np.asarray(psf_mat_file['psf_resized']).astype('float32')
    # psf = np.asarray(psf_mat_file['psf_stack_2x_resized']).astype('float32')
    # psf = np.asarray(psf_mat_file['psf_stack_resized']).astype('float32')
    
    
    if variable == 'psf_stack':
        psf = np.asarray(psf_mat_file['images']).astype('float32')
    elif variable == 'psf_resized':
        psf = np.asarray(psf_mat_file['psf_resized']).astype('float32')
    elif variable == 'psf_stack_resized':
        psf = np.asarray(psf_mat_file['psf_stack_resized']).astype('float32')
    elif variable == 'psf_stack_2x_resized':
        psf = np.asarray(psf_mat_file['psf_stack_2x_resized']).astype('float32')
    elif variable == 'psf_stack_resized_rotated':
        psf = np.asarray(psf_mat_file['psf_stack_resized_rotated']).astype('float32')
    psf = torch.from_numpy(psf)
    
    # Print the shape for debugging
    print("Original PSF Shape:", psf.shape)
    
    # Check the dimensionality of the PSF
    if psf.ndim == 2:
        # Unsqueeze twice to get the shape (1, 1, H, W) and move to the device
        psf = psf.unsqueeze(0).unsqueeze(0)
    elif psf.ndim == 3:
        # Permute dimensions to the form (1, H, W, D), then unsqueeze and move to the device
        psf = psf.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected PSF dimensions: {psf.ndim}")

    # Ensure PSF is of type FloatTensor
    # psf = psf.type(torch.FloatTensor)
    
    # Return the PSF tensor on the correct device
    return psf

def make_dir(path):
    if not os.path.isdir("{}".format(path)):
        os.mkdir("{}".format(path))
    return None

def weight_save(epoch,START_DATE,weight_save_path,TPARAMS,name_tmp):
    weight_save_path = weight_save_path+'/{}'.format(name_tmp)
    make_dir(weight_save_path)
    # Network save for inference
    save_filename = "{}/model_{}_epoch_{}.pth".format(weight_save_path,START_DATE,epoch)
    
    # torch.save(TPARAMS['model'].state_dict(), save_filename)
    torch.save({
        'epoch': epoch,
        'epoch_now': TPARAMS['epoch_now'], # 2024-05-01 added
        'optimizer_state_dict': TPARAMS['optimizer'].state_dict(),
        'scheduler_state_dict': TPARAMS['scheduler'].state_dict(),
        'model_state_dict': TPARAMS['model'].state_dict(),
    }, save_filename)
    
    print('{}epoch weight saved'.format(epoch))
    return None

def weight_save_with_discriminator(epoch, START_DATE, weight_save_path, TPARAMS, name_tmp):
    """
    Saves the state dictionaries of the model (generator), discriminator,
    their optimizers, and the scheduler.
    """
    # Create the full path for this specific checkpoint folder
    # This ensures each run/checkpoint has its own folder for organization
    checkpoint_dir = os.path.join(weight_save_path, name_tmp)
    make_dir(checkpoint_dir)
    
    # Define filenames for the generator's and discriminator's checkpoints
    generator_save_filename = os.path.join(checkpoint_dir, f"model_{START_DATE}_epoch_{epoch}.pth")
    discriminator_save_filename = os.path.join(checkpoint_dir, f"discriminator_{START_DATE}_epoch_{epoch}.pth") # New filename for discriminator

    # Save Generator (model) state and related info
    torch.save({
        'epoch': epoch,
        'epoch_now': TPARAMS['epoch_now'],
        'optimizer_state_dict': TPARAMS['optimizer'].state_dict(),
        'scheduler_state_dict': TPARAMS['scheduler'].state_dict(),
        'model_state_dict': TPARAMS['model'].state_dict(),
    }, generator_save_filename)
    
    # print(f'Epoch {epoch} generator weight saved to {generator_save_filename}')

    # --- NEW: Save Discriminator state and its optimizer state ---
    torch.save({
        'epoch': epoch,
        'epoch_now': TPARAMS['epoch_now'],
        'optimizer_discriminator_state_dict': TPARAMS['optimizer_discriminator'].state_dict(),
        'discriminator_state_dict': TPARAMS['discriminator'].state_dict(),
    }, discriminator_save_filename)

    # print(f'Epoch {epoch} discriminator weight saved to {discriminator_save_filename}')
    
    return None

# def wandb_log(loglist, epoch, note):
#     for key, val in loglist.items():
#         try:
#             try:
#                 item = val.cpu().detach()
#             except:
#                 item = val
#             log = wandb.Image(item)
#         except:
#             log = val
#         wandb.log({
#             "{0} {1}".format(note.capitalize(), key.capitalize()): log,
#         }, step=epoch+1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model, model_name="Model"):
    total_params = count_parameters(model)
    print(f"Total trainable parameters in {model_name}: {total_params:,}")


def draw_fig_validation(epoch, START_DATE, fig_save_path, val_result,name_tmp):
    fig_save_path = fig_save_path+'/{}'.format(name_tmp)
    make_dir(fig_save_path)
    
    # make dir to save figure for each epoch
    fig_save_path = fig_save_path+'/'+START_DATE+'_epoch_{}'.format(epoch)
    make_dir(fig_save_path)
    
    scale_transform = ((1 - val_result['output_depth']) * (700 - 50)+ 50) / 10 # unit:cm, inverted
        
    for i in range(len(val_result['image'])):
        fig = plt.figure(figsize=(12, 4))
        grid = ImageGrid(fig, 111,
                        nrows_ncols = (1,3),
                        axes_pad = 0.05,
                        cbar_location = "right",
                        cbar_mode="single",
                        cbar_size="5%",
                        cbar_pad=0.05
                        )
        
        grid[0].imshow(val_result['image'][i].cpu().squeeze().numpy().transpose(1,2,0).clip(0,1))
        grid[0].axis('off')
        grid[0].set_title('Raw')
        
        # convert to 8bit int
        grid[1].imshow(val_result['output_color'][i].cpu().squeeze().numpy().transpose(1,2,0).clip(0,1))
        grid[1].axis('off')
        grid[1].set_title('Color')
        
        imc = grid[2].imshow(scale_transform[i].cpu().squeeze().numpy(),cmap='turbo_r',vmin=10, vmax=60)
        grid[2].axis('off')
        grid[2].set_title('Depth')
        
        cbar = plt.colorbar(imc, cax=grid.cbar_axes[0])
        cbar.set_label('Absolute depth (cm)', horizontalalignment='center')
        cbar.mappable.set_clim(10, 60) #this works

        plt.savefig(fig_save_path+'/epoch_{}_sample_{}.png'.format(epoch, i))
        plt.close()
        

#################################################################
################ CUSTOM LOSS FUNCTIONS ##########################
#################################################################
# Confidence map from photometric error
def compute_confidence_map(photo_error, alpha=10.0):
    """
    photo_error: (B, 1, H, W)
    returns: (B, 1, H, W) - confidence map (0~1)
    """
    confidence = torch.exp(-alpha * photo_error)
    return confidence.clamp(0.0, 1.0)

class Smoothness_based_on_Depth(nn.Module):
    def __init__(self,beta=2.5):
        super(Smoothness_based_on_Depth, self).__init__()
        self.beta = beta
        
    def forward(self, output_depth):
        depth_grad_x_exp = torch.exp(-self.beta*torch.abs(self.gradient(output_depth)[0]))
        depth_grad_y_exp = torch.exp(-self.beta*torch.abs(self.gradient(output_depth)[1]))
        
        dx,dy = self.gradient(output_depth)
        d_Depth_x = dx.abs() * depth_grad_x_exp
        d_Depth_y = dy.abs() * depth_grad_y_exp
        smooth_loss = (d_Depth_x+d_Depth_y).mean()
        return smooth_loss
        
    def gradient(self, img):
        D_dy = img - F.pad(img[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = img - F.pad(img[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy

class Smoothness_based_on_RGB(nn.Module):
    def __init__(self,beta=2.5):
        super(Smoothness_based_on_RGB, self).__init__()
        self.beta = beta
        
    def forward(self, output_color, output_depth):
        color_grad_x_exp = torch.exp(-self.beta*torch.abs(self.gradient(output_color)[0]))
        color_grad_y_exp = torch.exp(-self.beta*torch.abs(self.gradient(output_color)[1]))
        
        dx,dy = self.gradient(output_depth)
        d_Depth_x = dx.abs() * color_grad_x_exp
        d_Depth_y = dy.abs() * color_grad_y_exp
        smooth_loss = (d_Depth_x+d_Depth_y).mean()
        return smooth_loss
    
    def gradient(self, img):
        D_dy = img - F.pad(img[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = img - F.pad(img[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy
    
class Blur(nn.Module):
    def __init__(self, sigma=1.0, window_size=7, beta=0.01):
        super(Blur, self).__init__()
        self.kernel = self.gen_LoG_kernel(sigma, window_size).repeat(1, 3, 1, 1)
        self.beta = beta

    def forward(self, img, _):
        B, C, H, W = img.shape
        img_lap = F.conv2d(img, self.kernel.to(torch.get_device(img)), padding='same')
        blur_loss = - torch.log (torch.sum(img_lap ** 2, dim=[1, 2, 3]) / (H*W - torch.mean(img, dim=[1,2,3])**2) + 1e-8)
        return blur_loss.mean() * self.beta

    def gen_LoG_kernel(self, sigma, window_size):
        X = np.arange(window_size//2, -window_size//2, -1)
        Y = np.arange(window_size//2, -window_size//2, -1)
        xx, yy = np.meshgrid(X, Y)
        LoG_kernel = 1 / (np.pi * sigma ** 4) * (1 - (xx ** 2 + yy ** 2) / (2 * sigma ** 2)) * np.exp(- (xx ** 2 + yy ** 2) / (2 * sigma ** 2))    
        return torch.from_numpy(LoG_kernel).type(torch.float32).view(1, 1, window_size, window_size)

class OrdinalLoss(nn.Module):
    def __init__(self, delta=0.1, sample_size=2500):
        super(OrdinalLoss, self).__init__()
        self.delta = delta
        self.sample_size = sample_size

    def forward(self, depth_pred, depth_gt):
        """
        Args:
            depth_pred (tensor): Predicted depth map (B, 1, H, W).
            depth_gt (tensor): Ground truth depth map (B, 1, H, W).
        """
        # Flatten depth maps: (B, 1, H, W) -> (B, H*W)
        B, _, H, W = depth_pred.size()
        depth_pred = depth_pred.view(B, -1)  # (B, H*W)
        depth_gt = depth_gt.view(B, -1)  # (B, H*W)
        # print("Shape of depth_pred:", depth_pred.shape)
        # print("Shape of depth_gt:", depth_gt.shape)
        # Randomly sample pixel pairs (indices are randomly selected pairs)
        indices = torch.randint(0, H * W, (self.sample_size, 2))  # (sample_size, 2) for pairs

        # print("Shape of indices:", indices.shape)
        # print("First few indices:", indices[:5])
        
        # Extract the depth values of the sampled pixel pairs
        sampled_pred = depth_pred.view(B, -1)[:, indices]  # (B, 2, sample_size)
        sampled_gt = depth_gt.view(B, -1)[:, indices]  # (B, 2, sample_size)
        # print("Shape of sampled_pred:", sampled_pred.shape)
        # print("Shape of sampled_gt:", sampled_gt.shape)
        
        # print("Shape of sampled_pred[:, :, 0]:", sampled_pred[:, :, 0].shape)
        # print("Shape of sampled_pred[:, :, 1]:", sampled_pred[:, :, 1].shape)
        # print("Shape of sampled_gt[:, :, 0]:", sampled_gt[:, :, 0].shape)
        # print("Shape of sampled_gt[:, :, 1]:", sampled_gt[:, :, 1].shape)
        # Calculate the depth difference for the sampled pixel pairs
        depth_diff = sampled_pred[:, :, 0] - sampled_pred[:, :, 1]  # (B, sample_size) 
        gt_diff = sampled_gt[:, :, 0] - sampled_gt[:, :, 1]  # (B, sample_size)

        # print("Shape of depth_diff:", depth_diff.shape)
        # print("Shape of gt_diff:", gt_diff.shape)
        
        # Apply ordinal loss
        loss = torch.zeros_like(depth_diff)
        # print("Shape of loss:", loss.shape)
        
        # If |ΔOij| < δ, use squared difference (if gt_diff is smaller than delta)
        mask = torch.abs(gt_diff) < self.delta
        # print("Shape of mask:", mask.shape)
        loss[mask] = torch.pow(depth_diff[mask], 2)
        
        # If |ΔOij| >= δ, apply hinge loss without margin
        hinge_mask = ~mask
        loss[hinge_mask] = torch.relu(-depth_diff[hinge_mask] * torch.sign(gt_diff[hinge_mask]))

        # Return the mean loss over the sampled pairs (along the indice dimension)
        return torch.mean(loss, dim=1).mean()  # Average over batch size
    
   
class silog_loss_original(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(silog_loss_original, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class scale_invariant_log_loss_v1_add_sqrt(nn.Module):
    
    """scale-invariant loss from https://arxiv.org/abs/1406.2283"""
    """ set lambda as 0.5"""
    def __init__(self,min_scale=0.05,max_scale=0.7, lambda_value=0.5):
        super(scale_invariant_log_loss_v1_add_sqrt, self).__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.lambda_value = lambda_value
        
    def forward(self, output_depth, label_depth):
        scaled_output_depth = (1.0 - output_depth) * (self.max_scale - self.min_scale) + self.min_scale
        scaled_label_depth = (1.0 - label_depth) * (self.max_scale - self.min_scale) + self.min_scale
        d = torch.log(scaled_output_depth) - torch.log(scaled_label_depth)
        return torch.sqrt((d ** 2).mean() + self.lambda_value * d.mean() ** 2+ 1e-8) # add sqrt to avoid log(0) in the denominator
    
class scale_invariant_log_loss_v1(nn.Module):
    
    """scale-invariant loss from https://arxiv.org/abs/1406.2283"""
    """ set lambda as 0.5"""
    def __init__(self,min_scale=0.05,max_scale=0.7, lambda_value=0.5):
        super(scale_invariant_log_loss_v1, self).__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.lambda_value = lambda_value
        
    def forward(self, output_depth, label_depth):
        scaled_output_depth = (1.0 - output_depth) * (self.max_scale - self.min_scale) + self.min_scale
        scaled_label_depth = (1.0 - label_depth) * (self.max_scale - self.min_scale) + self.min_scale
        d = torch.log(scaled_output_depth) - torch.log(scaled_label_depth)
        return (d ** 2).mean() + self.lambda_value * d.mean() ** 2
        

class scale_invariant_log_loss_v2_lambda(nn.Module):
    
    """scale-invariant loss from https://arxiv.org/abs/1406.2283"""
    """from https://medium.com/@omarbarakat1995/depth-estimation-with-deep-neural-networks-part-1-5fa6d2237d0d"""
    """ v2: add gradient term to the scale-invariant loss following medium"""
    
    
    def __init__(self,min_scale=0.05,max_scale=0.7, lambda_value=0.5):
        super(scale_invariant_log_loss_v2_lambda, self).__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.lambda_value = lambda_value
    
    def gradient(self, img):
        D_dy = img - F.pad(img[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = img - F.pad(img[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy
        
    def forward(self, output_depth, label_depth):
        scaled_output_depth = (1.0 - output_depth) * (self.max_scale - self.min_scale) + self.min_scale
        scaled_label_depth = (1.0 - label_depth) * (self.max_scale - self.min_scale) + self.min_scale
        
        d = torch.log(scaled_output_depth) - torch.log(scaled_label_depth)
        D_dx, D_dy = self.gradient(d) # v2 added
        grad = (D_dx ** 2 + D_dy ** 2 + 1e-8) # v2 added
        return (d ** 2).mean() + self.lambda_value * d.mean() ** 2 + grad.mean()

class scale_invariant_log_loss_v2(nn.Module):
    
    """scale-invariant loss from https://arxiv.org/abs/1406.2283"""
    """from https://medium.com/@omarbarakat1995/depth-estimation-with-deep-neural-networks-part-1-5fa6d2237d0d"""
    """ v2: add gradient term to the scale-invariant loss following medium"""
    
    
    def __init__(self,min_scale=0.05,max_scale=0.7):
        super(scale_invariant_log_loss_v2, self).__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def gradient(self, img):
        D_dy = img - F.pad(img[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = img - F.pad(img[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy
        
    def forward(self, output_depth, label_depth):
        scaled_output_depth = (1.0 - output_depth) * (self.max_scale - self.min_scale) + self.min_scale
        scaled_label_depth = (1.0 - label_depth) * (self.max_scale - self.min_scale) + self.min_scale
        
        d = torch.log(scaled_output_depth) - torch.log(scaled_label_depth)
        D_dx, D_dy = self.gradient(d) # v2 added
        grad = (D_dx ** 2 + D_dy ** 2 + 1e-8) # v2 added
        return (d ** 2).mean() + d.mean() ** 2 + grad.mean()

class scale_invariant_log_loss_v3(nn.Module):
    
    """scale-invariant loss from https://arxiv.org/abs/1406.2283"""
    """from https://medium.com/@omarbarakat1995/depth-estimation-with-deep-neural-networks-part-1-5fa6d2237d0d"""
    """ v2: add gradient term to the scale-invariant loss """
    """ v3: clamp the scaled output depth into (1e-8, max_scale) to avoid log(0) """
    
    def __init__(self,min_scale=0.05,max_scale=0.7):
        super(scale_invariant_log_loss_v3, self).__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def gradient(self, img):
        D_dy = img - F.pad(img[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = img - F.pad(img[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy
        
    def forward(self, output_depth, label_depth):
        scaled_output_depth = (1.0 - output_depth) * (self.max_scale - self.min_scale) + self.min_scale
        scaled_label_depth = (1.0 - label_depth) * (self.max_scale - self.min_scale) + self.min_scale
        
        scaled_output_depth = torch.clamp(scaled_output_depth, min=1e-8, max=self.max_scale)
        scaled_label_depth = torch.clamp(scaled_label_depth, min=1e-8, max=self.max_scale)
        
        d = torch.log(scaled_output_depth) - torch.log(scaled_label_depth)
        D_dx, D_dy = self.gradient(d) # v2 added
        grad = (D_dx ** 2 + D_dy ** 2 + 1e-8) # v2 added
        return (d ** 2).mean() + d.mean() ** 2 + grad.mean()
    

class MultiScaleLocalVarianceLoss(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7], weights=None):
        """
        Args:
            kernel_sizes (list of int): list of square window sizes for computing local variance.
            weights (list of float or None): weights for each scale. If None, all scales are equally weighted.
        """
        super(MultiScaleLocalVarianceLoss, self).__init__()
        self.kernel_sizes = kernel_sizes
        if weights is None:
            # equal weighting if not specified
            self.weights = [1.0] * len(kernel_sizes)
        else:
            assert len(weights) == len(kernel_sizes), "weights and kernel_sizes must have same length"
            self.weights = weights

    def forward(self, pred, target):
        """
        pred, target: tensors of shape (B, C, H, W)
        Returns:
            scalar: weighted sum of mean absolute differences of local variances across scales
        """
        total_loss = 0.0
        weight_sum = 0.0

        for k, w in zip(self.kernel_sizes, self.weights):
            pad = k // 2

            # local means
            mean_pred   = F.avg_pool2d(pred,  k, stride=1, padding=pad)
            mean_target = F.avg_pool2d(target, k, stride=1, padding=pad)

            # local second moments
            mean2_pred   = F.avg_pool2d(pred * pred,   k, stride=1, padding=pad)
            mean2_target = F.avg_pool2d(target * target, k, stride=1, padding=pad)

            # variance = E[x^2] - (E[x])^2
            var_pred   = mean2_pred   - mean_pred   * mean_pred
            var_target = mean2_target - mean_target * mean_target

            # per-scale loss
            scale_loss = torch.mean(torch.abs(var_pred - var_target))
            total_loss += w * scale_loss
            weight_sum  += w

        return total_loss / weight_sum

class local_variance_loss(nn.Module):
    
    def __init__(self, kernel_size=3):
        super(local_variance_loss, self).__init__()
        self.kernel_size = kernel_size
        
    def forward(self, pred, target):
                
        # Compute local means using a simple average pooling
        mean_pred = F.avg_pool2d(pred, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        mean_target = F.avg_pool2d(target, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        
        # Compute local squared differences
        var_pred = F.avg_pool2d(pred ** 2, self.kernel_size, stride=1, padding=self.kernel_size // 2) - mean_pred ** 2
        var_target = F.avg_pool2d(target ** 2, self.kernel_size, stride=1, padding=self.kernel_size // 2) - mean_target ** 2
        
        return torch.mean(torch.abs(var_pred - var_target))

    

        
### Inspired by  Godard et al., "Unsupervised Monocular Depth Estimation with Left-Right Consistency"

def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def compute_smoothness_loss(depth, image):
    """Edge-aware smoothness loss."""
    depth_grad_x = gradient_x(depth)
    depth_grad_y = gradient_y(depth)

    image_grad_x = gradient_x(image)
    image_grad_y = gradient_y(image)

    weights_x = torch.exp(-torch.mean(torch.abs(image_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_grad_y), 1, keepdim=True))

    smoothness_x = depth_grad_x * weights_x
    smoothness_y = depth_grad_y * weights_y

    return smoothness_x.abs().mean() + smoothness_y.abs().mean()

class BilateralDepthLoss(nn.Module):
    def __init__(self, alpha=0.85, smooth_weight=0.1):
        super(BilateralDepthLoss, self).__init__()
        self.alpha = alpha  # SSIM weight
        self.smooth_weight = smooth_weight
        self.ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, pred_depth, target_depth):
        pred_depth = torch.clamp(pred_depth, 0, 1)
        target_depth = torch.clamp(target_depth, 0, 1)

        # SSIM loss (returns similarity, so subtract from 1)
        ssim_loss = 1 - self.ssim_fn(pred_depth, target_depth)
        l1_loss = F.l1_loss(pred_depth, target_depth)

        # Appearance loss
        appearance_loss = self.alpha * ssim_loss + (1 - self.alpha) * l1_loss

        # Smoothness loss (edge-aware)
        smooth_loss = compute_smoothness_loss(pred_depth, target_depth)

        return appearance_loss + self.smooth_weight * smooth_loss



def rgb_confidence_map(image):
    """
    Compute confidence map from RGB gradients.
    Args:
        image: Tensor [B, 3, H, W] with values in [0, 1]
    Returns:
        conf: Tensor [B, 1, H-1, W-1]
    """
    image_grad_x = image - F.pad(image[:, :, :-1, :], (0, 0, 1, 0))
    image_grad_y = image - F.pad(image[:, :, :, :-1], (1, 0, 0, 0))

    # Per-pixel gradient magnitude weighted by inverse contrast (edges → lower weight)
    weights_x = torch.exp(-torch.mean(torch.abs(image_grad_x), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_grad_y), dim=1, keepdim=True))

    grad_x = image_grad_x * weights_x  # [B, 3, H, W-1]
    grad_y = image_grad_y * weights_y  # [B, 3, H-1, W]

    # print("grad_x shape:", grad_x.shape)
    # print("grad_y shape:", grad_y.shape)
    # # Crop to overlapping region [B, 3, H-1, W-1]
    # grad_x = grad_x[:, :, :-1, :]
    # grad_y = grad_y[:, :, :, :-1]

    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # [B, 3, H-1, W-1]

    # Final confidence map (inverse gradient magnitude)
    conf = torch.mean(grad_mag, dim=1, keepdim=True)  # [B, 1, H-1, W-1]
    conf = 1.0 - torch.clamp(conf, 0.0, 1.0)  # Higher gradient → lower confidence

    return conf


def compute_confidence_from_softmax(softmax_depth_u):
    """
    softmax_depth_u: Tensor of shape [B, 42, H, W]
    Returns:
        conf_depth: [B, 1, H, W] with values in [0, 1]
    """
    entropy = -torch.sum(softmax_depth_u * torch.log(softmax_depth_u + 1e-8), dim=1)  # [B, H, W]
    entropy_normalized = entropy / math.log(softmax_depth_u.size(1))  # Normalize to [0, 1]
    conf_depth = 1.0 - entropy_normalized  # Higher entropy = lower confidence
    return conf_depth.unsqueeze(1)  # Add channel dimension