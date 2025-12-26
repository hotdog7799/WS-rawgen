import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from einops import rearrange, reduce, repeat
import time
from tabulate import tabulate
from timm.models.layers import trunc_normal_, DropPath


# =============================
#        Deconv Components
# =============================

# def inv_gaussian_kernel_2d_non_square(shape, sigma, device):
#     """
#     Create a 2D inverse Gaussian kernel for a non-square shape.

#     Args:
#         shape (tuple): Dimensions of the kernel as (height, width).
#         sigma (float): Standard deviation of the Gaussian.
#         device (torch.device): Device to place the kernel on.

#     Returns:
#         torch.Tensor: 2D inverse Gaussian kernel.
#     """
#     height, width = shape
#     y = torch.arange(height, dtype=torch.float32, device=device) - height // 2
#     x = torch.arange(width, dtype=torch.float32, device=device) - width // 2
#     yy, xx = torch.meshgrid(y, x, indexing="ij")

#     # Compute the 2D Gaussian kernel
#     sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=device)
#     kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma_tensor**2))
#     kernel /= kernel.sum()  # Normalize the kernel to sum to 1

#     # Compute the inverse kernel
#     kernel_inv = 1 / (kernel + 1e-8)
#     return kernel_inv

# def gaus_t(x, fwhm=3):
#     b, c, w, h = x.size()
#     device = x.device
#     dtype = x.dtype

#     # 가우시안 윈도우 생성
#     def gaussian_window(size, fwhm):
#         sigma = size / fwhm
#         x = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
#         gauss = torch.exp(-0.5 * (x / sigma) ** 2)
#         return gauss

#     ga_w = gaussian_window(w, fwhm)
#     ga_h = gaussian_window(h, fwhm)

#     # 외적을 이용하여 2D 가우시안 윈도우 생성
#     ga = ga_w.unsqueeze(1) * ga_h.unsqueeze(0)
#     ga = ga.unsqueeze(0).unsqueeze(0)  # (1, 1, w, h)
#     ga = ga.expand(b, c, w, h)  # 입력 텐서와 크기 맞추기

#     return x * ga.detach()


# def WieNer_replicate_pad(raw, psf, delta):
#     # get size of raw and psf
#     B, C, H, W = raw.shape
#     pad_H = H
#     pad_W = W

#     # replicate pad raw
#     raw_padded = F.pad(raw, (pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2), mode='replicate')
#     raw_padded = gaus_t(raw_padded)
#     # zero pad psf
#     psf_padded = F.pad(psf, (pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2), mode='constant')

#     raw_fft = torch.fft.rfft2(raw_padded, dim=(-2, -1))
#     psf_fft = torch.fft.rfft2(psf_padded, dim=(-2, -1))

#     psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta)

#     img_2x = torch.fft.ifftshift(torch.fft.irfft2(psf_fft * raw_fft), (-2, -1))

#     start_H = pad_H // 2
#     start_W = pad_W // 2

#     img = img_2x[...,start_H:start_H + H, start_W:start_W + W] # center crop
#     return img.real


# --- New Generative Module for Depth Refinement (Remains unchanged) ---
class DepthRefinementModule(nn.Module):
    """
    A generative module designed to refine the predicted depth map.
    It takes the RGB intensity output and the initial depth prediction as input,
    concatenates them, and uses convolutional layers to learn a refined depth output.
    This module aims to compensate for errors in dark regions of the RGB image.
    """

    def __init__(self, in_channels, out_channels, num_layers=3):
        """
        Initializes the DepthRefinementModule.

        Args:
            in_channels (int): The total number of input channels (RGB channels + initial depth channels).
                                If RGB is 3 and initial depth is 'out_channels', then in_channels = 3 + out_channels.
            out_channels (int): The number of output channels for the refined depth map.
                                This should match the expected depth map channels (e.g., 42 for multi-channel depth).
            num_layers (int): Number of DoubleConvG blocks to use in the refinement module.
                              More layers can capture more complex relationships.
        """
        super().__init__()
        # Initial convolution to process the concatenated RGB and initial depth features
        self.initial_conv = DoubleConvG(in_channels, 64)  # Start with 64 features

        # Sequential block of DoubleConvG layers for feature extraction and refinement
        layers = []
        current_channels = 64
        for _ in range(num_layers - 1):  # Create `num_layers-1` intermediate blocks
            layers.append(DoubleConvG(current_channels, current_channels))
        self.feature_extractor = nn.Sequential(*layers)

        # Final convolution to map the refined features to the desired output depth channels
        self.final_conv = OutConv(current_channels, out_channels)

        # Apply softmax to the final refined depth, consistent with the original depth output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb_intensity, initial_depth):
        """
        Forward pass for the DepthRefinementModule.

        Args:
            rgb_intensity (torch.Tensor): The predicted RGB intensity map from the main network.
                                          Expected shape: (batch_size, 3, H, W)
            initial_depth (torch.Tensor): The initial depth map predicted by the main network.
                                          Expected shape: (batch_size, out_channels, H, W)

        Returns:
            torch.Tensor: The refined depth map after processing.
        """
        # Concatenate the RGB intensity and the initial depth prediction along the channel dimension.
        # This allows the refinement module to learn from both visual cues and initial depth estimates.
        x = torch.cat([rgb_intensity, initial_depth], dim=1)

        # Process through the initial convolutional layer
        x = self.initial_conv(x)

        # Extract and refine features through multiple convolutional layers
        x = self.feature_extractor(x)

        # Map the refined features to the output depth channels
        refined_depth = self.final_conv(x)

        # Apply softmax for the final depth output
        return self.softmax(refined_depth)


class SaturationAdaptiveLayer(nn.Module):
    """
    A lightweight layer to adaptively re-calibrate feature channels.
    It learns a scaling factor for each channel based on the global average
    of the feature map, allowing the network to become more robust to
    saturation artifacts by de-emphasizing corrupted channels or emphasizing
    reliable ones.
    """

    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        # Small network to predict a channel-wise scaling factor
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                1
            ),  # Squeeze spatial to get global feature descriptor for current map
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid(),  # Output a sigmoid to get a gating factor between 0 and 1
        )

    def forward(self, x):
        # Generate a per-channel scaling factor
        scaling_factors = self.gate(x)
        # Apply channel-wise scaling to the input features
        return x * scaling_factors


class MSCBAM(nn.Module):
    def __init__(self, in_channels, out_channels=512, reduction=16, use_dilation=True):
        super(MSCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )

        self.use_dilation = use_dilation
        if use_dilation:
            self.spatial = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.spatial = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid()
            )

        # Final fusion layer to match expected channel size
        self.channel_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        x = self.channel_reduce(x)  # Ensure output has correct number of channels
        return x


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        return x


class PixelWiseDepthAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(PixelWiseDepthAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.attn(x)  # shape: (B, C, H, W)
        return x * weights  # apply channel attention at each pixel


# def WieNer_replicate_pad_raw_only(raw, psf, delta, fwhm=3):
#     # Get sizes from raw (assumed to be [B, C, H, W])
#     B, C, H, W = raw.shape
#     pad_H, pad_W = H, W

#     # Replicate pad raw
#     raw_padded = F.pad(
#         raw,
#         (pad_W // 2, pad_W - pad_W // 2,
#          pad_H // 2, pad_H - pad_H // 2),
#         mode='replicate'
#     )

#     # Compute Gaussian window on the padded dimensions without expanding per channel/batch.
#     padded_H, padded_W = raw_padded.shape[-2], raw_padded.shape[-1]
#     device, dtype = raw.device, raw.dtype

#     # Create 1D coordinate vectors
#     x_coords = torch.arange(padded_H, device=device, dtype=dtype) - (padded_H - 1) / 2.
#     y_coords = torch.arange(padded_W, device=device, dtype=dtype) - (padded_W - 1) / 2.
#     sigma_x = padded_H / fwhm
#     sigma_y = padded_W / fwhm

#     # Compute 2D Gaussian window via meshgrid
#     xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
#     gauss_window = torch.exp(-0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2))
#     # Reshape to [1, 1, H_pad, W_pad] so that multiplication broadcasts to [B, C, H_pad, W_pad]
#     gauss_window = gauss_window.unsqueeze(0).unsqueeze(0)
#     # Apply the window in-place to reduce memory overhead.
#     raw_padded.mul_(gauss_window.detach())

#     # Zero pad psf
#     # psf_padded = F.pad(
#     #     psf,
#     #     (pad_W // 2, pad_W - pad_W // 2,
#     #      pad_H // 2, pad_H - pad_H // 2),
#     #     mode='constant'
#     # )

#     # Perform the FFT-based operations as before.
#     raw_fft = torch.fft.rfft2(raw_padded, dim=(-2, -1))
#     psf_fft = torch.fft.rfft2(psf, dim=(-2, -1)) # no padding

#     psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta)
#     img_2x = torch.fft.ifftshift(
#         torch.fft.irfft2(psf_fft * raw_fft), (-2, -1)
#     )

#     start_H = pad_H // 2
#     start_W = pad_W // 2
#     img = img_2x[..., start_H:start_H + H, start_W:start_W + W]  # center crop
#     return img.real

# def WieNer_replicate_pad(raw, psf, delta, fwhm=3):
#     # Get sizes from raw (assumed to be [B, C, H, W])
#     B, C, H, W = raw.shape
#     pad_H, pad_W = H, W

#     # Replicate pad raw
#     raw_padded = F.pad(
#         raw,
#         (pad_W // 2, pad_W - pad_W // 2,
#          pad_H // 2, pad_H - pad_H // 2),
#         mode='replicate'
#     )

#     # Compute Gaussian window on the padded dimensions without expanding per channel/batch.
#     padded_H, padded_W = raw_padded.shape[-2], raw_padded.shape[-1]
#     device, dtype = raw.device, raw.dtype

#     # Create 1D coordinate vectors
#     x_coords = torch.arange(padded_H, device=device, dtype=dtype) - (padded_H - 1) / 2.
#     y_coords = torch.arange(padded_W, device=device, dtype=dtype) - (padded_W - 1) / 2.
#     sigma_x = padded_H / fwhm
#     sigma_y = padded_W / fwhm

#     # Compute 2D Gaussian window via meshgrid
#     xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
#     gauss_window = torch.exp(-0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2))
#     # Reshape to [1, 1, H_pad, W_pad] so that multiplication broadcasts to [B, C, H_pad, W_pad]
#     gauss_window = gauss_window.unsqueeze(0).unsqueeze(0)
#     # Apply the window in-place to reduce memory overhead.
#     raw_padded.mul_(gauss_window.detach())

#     # Zero pad psf
#     psf_padded = F.pad(
#         psf,
#         (pad_W // 2, pad_W - pad_W // 2,
#          pad_H // 2, pad_H - pad_H // 2),
#         mode='constant'
#     )

#     # Perform the FFT-based operations as before.
#     raw_fft = torch.fft.rfft2(raw_padded, dim=(-2, -1))
#     psf_fft = torch.fft.rfft2(psf_padded, dim=(-2, -1))

#     psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta)
#     img_2x = torch.fft.ifftshift(
#         torch.fft.irfft2(psf_fft * raw_fft), (-2, -1)
#     )

#     start_H = pad_H // 2
#     start_W = pad_W // 2
#     img = img_2x[..., start_H:start_H + H, start_W:start_W + W]  # center crop
#     return img.real


def WieNerPad(blur, psf, delta):
    # Get sizes from raw (assumed to be [B, C, H, W])
    _, _, H, W = blur.shape
    _, _, H_p, W_p = psf.shape
    pad_H, pad_W = H, W
    print(psf.size())
    print(blur.size())
    # Replicate pad raw
    blur = F.pad(
        blur,
        (H_p // 2, H_p - H_p // 2, W_p // 2, W_p - W_p // 2),
        mode="replicate",
    )

    psf = F.pad(
        psf,
        (H // 2, H - H // 2, W // 2, W - W // 2),
        mode="constant",
    )
    print(psf.size())
    print(blur.size())
    blur_fft = torch.fft.rfft2(blur)
    psf_fft = torch.fft.rfft2(psf)
    print(blur_fft.size())
    psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta)
    img = torch.fft.ifftshift(torch.fft.irfft2(psf_fft * blur_fft), (-2, -1))
    # center crop the image to original size
    img = img[..., H // 2 : -H // 2, W // 2 : -W // 2]
    return img.real


def WieNer(blur, psf, delta):
    blur_fft = torch.fft.rfft2(blur)
    psf_fft = torch.fft.rfft2(psf)
    psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta)
    img = torch.fft.ifftshift(torch.fft.irfft2(psf_fft * blur_fft), (-2, -1))
    return img.real


# def WieNer_Reg(blur, psf, delta, sigma):
#     """
#     Perform Wiener regularization with a Gaussian kernel adjustment for non-square PSF.

#     Args:
#         blur (torch.Tensor): Blurred input image.
#         psf (torch.Tensor): Point spread function (PSF).
#         delta (float): Regularization parameter.
#         sigma (float): Standard deviation for Gaussian kernel.

#     Returns:
#         torch.Tensor: Reconstructed image.
#     """
#     # Get the shape of the PSF
#     # Create the inverse Gaussian kernel matching the PSF shape
#     device = psf.device  # Ensure the kernel is on the same device as the PSF
#     # Perform FFT on the inputs
#     blur_fft = torch.fft.rfft2(blur)
#     psf_fft = torch.fft.rfft2(psf)
#     psf_fft_shape = psf_fft.shape[-2:]  # Assuming last two dimensions are height and width
#     inv_gaussian_kernel = inv_gaussian_kernel_2d_non_square(psf_fft_shape, sigma, device)

#     # Adjust the PSF in frequency domain
#     # print("Shape of psf_fft: ", psf_fft.shape)
#     # print("Shape of inv_gaussian_kernel: ", inv_gaussian_kernel.shape)
#     # print("Shape of delta: ", delta.shape)
#     psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta * inv_gaussian_kernel)

#     # Perform inverse FFT to reconstruct the image
#     img = torch.fft.ifftshift(torch.fft.irfft2(psf_fft * blur_fft), (-2, -1))
#     return img.real

# def WieNer_Learnable_Reg(blur, psf, delta, reg_kernel):
#     blur_fft = torch.fft.rfft2(blur)
#     psf_fft = torch.fft.rfft2(psf)
#     # print("Shape of blur_fft: ", blur_fft.shape)
#     # print("Shape of psf_fft: ", psf_fft.shape)
#     # print("Shape of reg_kernel: ", reg_kernel.shape)
#     # print("Shape of delta: ", delta.shape)
#     psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta * reg_kernel)
#     img = torch.fft.ifftshift(torch.fft.irfft2(psf_fft * blur_fft), (-2, -1))
#     return img.real


# class WieNer_GlobalFilter(nn.Module):
#     def __init__(self, input_size):
#         # super().__init__()
#         super(WieNer_GlobalFilter, self).__init__()
#         dummy_input = torch.Tensor(*input_size)
#         dummy_output = torch.fft.rfft2(dummy_input, dim=(-2, -1), norm=None)

#         complex_size = list(dummy_output.size()) + [2]
#         self.raw_weights = nn.Parameter(torch.randn(complex_size, dtype=torch.float32) * 0.02)
#         self.psf_weights = nn.Parameter(torch.randn(complex_size, dtype=torch.float32) * 0.02)
#         # print("Shape of raw_weights: ", self.raw_weights.shape)
#         # print("Shape of psf_weights: ", self.psf_weights.shape)
#         # self.delta = nn.Parameter(torch.Tensor(dummy_output.size()))
#         # print("Delta size: ", self.delta.size())
#         self.sigmoid = nn.Sigmoid()
#         # self.delta.data.fill_(0)

#     def forward(self, raw, psf, epsilon = 1e-6):

#         raw_fft = torch.fft.rfft2(raw, dim = (-2,-1), norm = 'ortho') # norm = 'ortho' is used to normalize the FFT in original size
#         # raw_fft = raw_fft * torch.view_as_complex(self.raw_weights)
#         raw_weights = torch.view_as_complex(self.raw_weights)
#         raw_fft = raw_fft * (self.sigmoid(raw_weights) + 0.5) # 0.5 is added to make the range of weights from 0 to 1

#         psf_fft = torch.fft.rfft2(psf, dim = (-2, -1), norm = 'ortho')
#         psf_fft_conj = torch.conj(psf_fft)
#         # psf_fft = psf_fft_conj / (psf_fft * psf_fft_conj + self.delta)
#         psf_fft = psf_fft_conj / (psf_fft * psf_fft_conj + epsilon)
#         psf_weights = torch.view_as_complex(self.psf_weights)
#         # psf_fft = psf_fft * psf_weights
#         psf_fft = psf_fft * (self.sigmoid(psf_weights) + 0.5)

#         out = torch.fft.irfft2(raw_fft * psf_fft)
#         img = torch.fft.ifftshift(out, dim=(-2, -1))

#         return img.real

# class WieNer_GlobalFilter2X(nn.Module):
#     def __init__(self, input_size, pad_factor=1):
#         super(WieNer_GlobalFilter2X, self).__init__()
#         # Calculate the padded height and width based on the pad_factor
#         self.pad_factor = pad_factor
#         self.pad_height = (input_size[2] * self.pad_factor) // 2
#         self.pad_width = (input_size[3] * self.pad_factor) // 2
#         self.padded_height = input_size[2] + 2 * self.pad_height
#         self.padded_width = input_size[3] + 2 * self.pad_width

#         # Initialize dummy input with doubled spatial dimensions for weight initialization
#         dummy_input = torch.Tensor(input_size[0], input_size[1], self.padded_height, self.padded_width)
#         dummy_output = torch.fft.rfft2(dummy_input, dim=(-2, -1), norm=None)

#         # Set up weight matrices to match the FFT output's complex size
#         complex_size = list(dummy_output.size()) + [2]
#         self.raw_weights = nn.Parameter(torch.randn(complex_size, dtype=torch.float32) * 0.02)
#         self.psf_weights = nn.Parameter(torch.randn(complex_size, dtype=torch.float32) * 0.02)

#         # print("Shape of raw_weights: ", self.raw_weights.shape)
#         # print("Shape of psf_weights: ", self.psf_weights.shape)

#         self.sigmoid = nn.Sigmoid()
#         self.input_size = input_size  # Store the original input size for cropping

#     def forward(self, raw, psf, epsilon=1e-6):

#         # Apply replicate padding to raw and zero padding to psf based on pad parameters
#         raw_padded = F.pad(raw, (self.pad_width, self.pad_width, self.pad_height, self.pad_height), mode='replicate')


#         psf_padded = F.pad(psf, (self.pad_width, self.pad_width, self.pad_height, self.pad_height), mode='constant', value=0)

#         # Perform FFT on the padded raw input and psf
#         raw_fft = torch.fft.rfft2(raw_padded, dim=(-2, -1), norm='ortho')
#         psf_fft = torch.fft.rfft2(psf_padded, dim=(-2, -1), norm='ortho')

#         # Apply learned weights to raw FFT
#         raw_weights = torch.view_as_complex(self.raw_weights)
#         raw_fft = raw_fft * (self.sigmoid(raw_weights) + 0.5)

#         # Compute conjugate of psf_fft and apply Wiener filter formula
#         psf_fft_conj = torch.conj(psf_fft)
#         psf_fft = psf_fft_conj / (psf_fft * psf_fft_conj + epsilon)

#         # Apply learned weights to psf FFT
#         psf_weights = torch.view_as_complex(self.psf_weights)
#         psf_fft = psf_fft * (self.sigmoid(psf_weights) + 0.5)

#         # Perform inverse FFT and center shift
#         out = torch.fft.irfft2(raw_fft * psf_fft, s=(self.padded_height, self.padded_width))
#         img = torch.fft.ifftshift(out, dim=(-2, -1))

#         # Center crop the result to the original input size
#         crop_y1 = (img.size(-2) - self.input_size[2]) // 2
#         crop_x1 = (img.size(-1) - self.input_size[3]) // 2
#         img_cropped = img[..., crop_y1:crop_y1 + self.input_size[2], crop_x1:crop_x1 + self.input_size[3]]

#         return img_cropped.real

###


# =============================
#         ConvNeXt Blocks
# =============================
class GRN(nn.Module):
    """GRN (Global Response Normalization) layer."""

    def __init__(self, dim):
        """
        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.gamma = nn.Parameter(
            torch.zeros(1, dim, 1, 1)
        )  # Matches the channel dimension
        self.beta = nn.Parameter(
            torch.zeros(1, dim, 1, 1)
        )  # Matches the channel dimension

    def forward(self, x):
        """
        Forward pass of the GRN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute global L2 norm along spatial dimensions
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        # Normalize by mean L2 norm
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        # Apply GRN transformation
        return self.gamma * (x * Nx) + self.beta + x


class ConvNextBlockV2(nn.Module):
    """ConvNeXtV2 Block."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super(ConvNextBlockV2, self).__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
        self.ds_conv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, bias=False
        )
        self.net = nn.Sequential(
            (
                nn.GroupNorm(1, dim) if norm else nn.Identity()
            ),  # num_groups=1 is equivalent to LayerNorm
            nn.Conv2d(dim, dim_out * mult, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),  # num_groups=1 is equivalent to LayerNorm
            GRN(dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, kernel_size=3, padding=1, bias=False),
        )
        self.res_conv = (
            nn.Conv2d(dim, dim_out, kernel_size=1, bias=False)
            if dim != dim_out
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.net(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """ConvNeXt Block as described in https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super(ConvNextBlock, self).__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
        self.ds_conv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, bias=False
        )
        self.net = nn.Sequential(
            (
                nn.GroupNorm(1, dim) if norm else nn.Identity()
            ),  # num_groups=1 is equivalent to LayerNorm
            nn.Conv2d(dim, dim_out * mult, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),  # num_groups=1 is equivalent to LayerNorm
            nn.Conv2d(dim_out * mult, dim_out, kernel_size=3, padding=1, bias=False),
        )
        self.res_conv = (
            nn.Conv2d(dim, dim_out, kernel_size=1, bias=False)
            if dim != dim_out
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.net(h)
        return h + self.res_conv(x)


class ConvNextBlock3d(nn.Module):
    """ConvNeXt Block adapted for 3D inputs."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super(ConvNextBlock3d, self).__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
        self.ds_conv = nn.Conv3d(
            dim, dim, kernel_size=7, padding=3, groups=dim, bias=False
        )
        self.net = nn.Sequential(
            (
                nn.GroupNorm(1, dim) if norm else nn.Identity()
            ),  # num_groups=1 is equivalent to LayerNorm
            nn.Conv3d(dim, dim_out * mult, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),  # num_groups=1 is equivalent to LayerNorm
            nn.Conv3d(dim_out * mult, dim_out, kernel_size=3, padding=1, bias=False),
        )
        self.res_conv = (
            nn.Conv3d(dim, dim_out, kernel_size=1, bias=False)
            if dim != dim_out
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1 1")
        h = self.net(h)
        return h + self.res_conv(x)


# =============================
#          Convolution Blocks
# =============================


class DoubleConvL(nn.Module):
    """(convolution => [LayerNorm] => GeLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),  # same as LayerNorm
            nn.GELU(approximate="none"),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),  # same as LayerNorm
            nn.GELU(approximate="none"),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvG(nn.Module):
    """(convolution => [GN] => GeLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=32, num_channels=mid_channels),  # same as LayerNorm
            nn.GELU(approximate="none"),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=32, num_channels=mid_channels),  # same as LayerNorm
            nn.GELU(approximate="none"),
        )

    def forward(self, x):
        return self.double_conv(x)


### checkpoint 20240730-020933_nnDP_60K_GELU_0728dataset_LPIPS_SMOOTH_L1_BOTH_COLOR_AND_DEPTH_obj_5_20_with_image_cc_gss
###### GELU activation function 사용 + DoubleConv에서 BatchNorm2D 그대로 사용 ############################


class DoubleConvB_conv_1x1(nn.Module):
    """(convolution => [BN] => GeLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(approximate="none"),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(approximate="none"),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvB(nn.Module):
    """(convolution => [BN] => GeLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(approximate="none"),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(approximate="none"),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution."""

    def __init__(self, nin, nout, kernel_size=3, padding=1, stride=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin,
            nin,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=nin,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# =============================
#        U-Net Components
# =============================


class DownC_LA(nn.Module):
    """Downscaling with average pooling followed by ConvNextBlock and Linear Attention."""

    def __init__(self, in_channels, out_channels):
        super(DownC_LA, self).__init__()
        self.conv = ConvNextBlock(in_channels, out_channels)
        self.residual = Residual(PreNorm(out_channels, LinearAttention(out_channels)))
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        x = self.avgpool(x)
        return x


class DownC(nn.Module):
    """Downscaling with average pooling followed by ConvNextBlock."""

    def __init__(self, in_channels, out_channels):
        super(DownC, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2), ConvNextBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class DownC_V2(nn.Module):
    """Downscaling with average pooling followed by ConvNextBlockV2."""

    def __init__(self, in_channels, out_channels):
        super(DownC_V2, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2), ConvNextBlockV2(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class DownC(nn.Module):
    """Downscaling with average pooling followed by ConvNextBlock."""

    def __init__(self, in_channels, out_channels):
        super(DownC, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2), ConvNextBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class DownD(nn.Module):
    """Downscaling with avgpool then DepthwiseSeparableConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2), DepthwiseSeparableConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class DownB(nn.Module):
    """Downscaling with avgpool then DoubleConvB."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2), DoubleConvB(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class DownL(nn.Module):
    """Downscaling with average pooling followed by DoubleConvL."""

    def __init__(self, in_channels, out_channels):
        super(DownL, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2), DoubleConvL(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class DownG(nn.Module):
    """Downscaling with average pooling followed by DoubleConvG."""

    def __init__(self, in_channels, out_channels):
        super(DownG, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2), DoubleConvG(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class DownG_MaxPool(nn.Module):
    """Downscaling with average pooling followed by DoubleConvG."""

    def __init__(self, in_channels, out_channels):
        super(DownG_MaxPool, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvG(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpC_V2(nn.Module):
    """Upscaling then ConvNextBlockV2."""

    def __init__(self, in_channels, out_channels):
        super(UpC_V2, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvNextBlockV2(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpC(nn.Module):
    """Upscaling then ConvNextBlock."""

    def __init__(self, in_channels, out_channels):
        super(UpC, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvNextBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpC_LA(nn.Module):
    """Upscaling then ConvNextBlock and Linear Attention."""

    def __init__(self, in_channels, out_channels):
        super(UpC_LA, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvNextBlock(in_channels, out_channels)
        self.residual = Residual(PreNorm(out_channels, LinearAttention(out_channels)))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.residual(x)
        return x


class UpD(nn.Module):
    """Upscaling then DepthwiseSeparableConv."""

    def __init__(self, in_channels, out_channels):
        super(UpD, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DepthwiseSeparableConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpB(nn.Module):
    """Upscaling then ConvNextBlock."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvB(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ensure the sizes match for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpL(nn.Module):
    """Upscaling then DoubleConvL."""

    def __init__(self, in_channels, out_channels):
        super(UpL, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvL(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpG_ConvT(nn.Module):
    """Upscaling then DoubleConvG."""

    def __init__(self, in_channels, out_channels):
        super(UpG_ConvT, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(
            in_channels // 2, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConvG(in_channels, out_channels)
        # self.conv = DoubleConvG(in_channels, out_channels, in_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpG(nn.Module):
    """Upscaling then DoubleConvG."""

    def __init__(self, in_channels, out_channels):
        super(UpG, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvG(in_channels, out_channels)
        # self.conv = DoubleConvG(in_channels, out_channels, in_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class MidC_LA(nn.Module):
    """Middle block with ConvNextBlock and Self Attention."""

    def __init__(self, in_channels, out_channels):
        super(MidC, self).__init__()
        self.conv1 = ConvNextBlock(in_channels, out_channels)
        self.residual = Residual(PreNorm(out_channels, Self_Attention(out_channels)))
        self.conv2 = ConvNextBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual(x)
        x = self.conv2(x)
        return x


# =============================
#        DeepLIR Linear and Self-Attention Blocks
# =============================


def exists(x):
    return x is not None


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    # a module that applies normalization before the function
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# for downsample block
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5  # scale factor
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(num_groups=1, num_channels=dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  # split the tensor into 3 parts
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum(
            "b h d n, b h e n -> b h d e", k, v
        )  # matrix multiplication

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# for bottle neck block
class Self_Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# =============================
#        Attention Blocks from Attention U-Net
# =============================


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters."""

    """PyTorch implementation of Attention U-Net: Learning Where to Look for the Pancreas. by Oktay et al"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        Args:
            F_g: Number of feature maps in gating signal.
            F_l: Number of feature maps in skip connection.
            n_coefficients: Number of learnable attention coefficients.
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(
                F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(n_coefficients),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(n_coefficients),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        Args:
            gate: Gating signal from previous layer.
            skip_connection: Features from the encoder path.
        Returns:
            Output activations after applying attention.
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


# =============================
#        Grid Attention Blocks from FlatNet3D
# =============================
class GridAttention_LayNorm(nn.Module):
    """Based on https://github.com/ozan-oktay/Attention-Gated-Networks

    Published in https://arxiv.org/abs/1804.03999"""

    def __init__(
        self,
        in_channels,
        gating_channels,
        inter_channels=None,
        dim=2,
        sub_sample_factor=2,
    ):
        super().__init__()

        assert dim in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dim

        # Default parameter set
        self.dim = dim
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dim == 3:
            conv_nd = nn.Conv3d
            layernorm = nn.GroupNorm
            self.upsample_mode = "trilinear"
        elif dim == 2:
            conv_nd = nn.Conv2d
            layernorm = nn.GroupNorm
            self.upsample_mode = "bilinear"
        else:
            raise NotImplementedError

        # Output transform
        self.w = nn.Sequential(
            conv_nd(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
            ),
            layernorm(num_groups=1, num_channels=self.in_channels),
        )
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=self.sub_sample_kernel_size,
            stride=self.sub_sample_factor,
            bias=False,
        )
        self.phi = conv_nd(
            in_channels=self.gating_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.psi = conv_nd(
            in_channels=self.inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        # self.init_weights()

    def forward(self, x, g):
        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(
            self.phi(g),
            size=theta_x.shape[2:],
            mode=self.upsample_mode,
            align_corners=False,
        )
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=x.shape[2:], mode=self.upsample_mode, align_corners=False
        )
        y = sigm_psi_f.expand_as(x) * x
        wy = self.w(y)

        return wy, sigm_psi_f


class GridAttention(nn.Module):
    """Based on https://github.com/ozan-oktay/Attention-Gated-Networks

    Published in https://arxiv.org/abs/1804.03999"""

    def __init__(
        self,
        in_channels,
        gating_channels,
        inter_channels=None,
        dim=2,
        sub_sample_factor=2,
    ):
        super().__init__()

        assert dim in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dim

        # Default parameter set
        self.dim = dim
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dim == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = "trilinear"
        elif dim == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = "bilinear"
        else:
            raise NotImplementedError

        # Output transform
        self.w = nn.Sequential(
            conv_nd(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
            ),
            bn(self.in_channels),
        )
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=self.sub_sample_kernel_size,
            stride=self.sub_sample_factor,
            bias=False,
        )
        self.phi = conv_nd(
            in_channels=self.gating_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.psi = conv_nd(
            in_channels=self.inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        # self.init_weights()

    def forward(self, x, g):
        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(
            self.phi(g),
            size=theta_x.shape[2:],
            mode=self.upsample_mode,
            align_corners=False,
        )
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=x.shape[2:], mode=self.upsample_mode, align_corners=False
        )
        y = sigm_psi_f.expand_as(x) * x
        wy = self.w(y)

        return wy, sigm_psi_f


# =============================
#        Dual Attention Blocks
# =============================


class _EncoderBlock(nn.Module):
    """
    Encoder block for Semantic Attention Module
    """

    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """
    Decoder Block for Semantic Attention Module
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class semanticModule(nn.Module):
    """
    Semantic attention module
    """

    def __init__(self, in_dim):
        super(semanticModule, self).__init__()
        self.chanel_in = in_dim

        self.enc1 = _EncoderBlock(in_dim, in_dim * 2)
        self.enc2 = _EncoderBlock(in_dim * 2, in_dim * 4)
        self.dec2 = _DecoderBlock(in_dim * 4, in_dim * 2, in_dim * 2)
        self.dec1 = _DecoderBlock(in_dim * 2, in_dim, in_dim)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec2 = self.dec2(enc2)
        dec1 = self.dec1(F.upsample(dec2, enc1.size()[2:], mode="bilinear"))

        # return enc2.view(-1), dec1 # for two refinement steps
        return dec1


class PAM_Module(nn.Module):
    """Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        print("proj_query shape: ", proj_query.shape)  # too large
        print("proj_key shape: ", proj_key.shape)  # too large (352 x 512 = 180224)
        energy = torch.bmm(proj_query, proj_key)  # cannot calculate

        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):
    """
    Helper Function for PAM and CAM attention

    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """

    def __init__(self, in_ch, use_pam=True):
        super(PAM_CAM_Layer, self).__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(),
            PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.attn(x)


# =============================
#        Unused Customized Components
# =============================

# class Double_ConvNextBlock_G(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.convnextblock = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=(7, 7), padding=3),
#             nn.GroupNorm(num_groups=mid_channels, num_channels=mid_channels), # same as LayerNorm
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 1), padding=0),
#             nn.GELU(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), padding=0),
#         )
#         self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0)

#     def forward(self, x):
#         return self.convnextblock(x) + self.conv_1x1(x)
