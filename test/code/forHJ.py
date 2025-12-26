import sys
import torch
import numpy as np
import torchvision
from torch import nn
import torch.nn.functional as F
import sys
import torchvision
from einops import rearrange

def exists(x):
    return x is not None

class MWDNet_CPSF_depth(nn.Module):
    def __init__(self, n_channels, n_classes, psf, channels=64, height=336, width=336, depth=3, dim=32):
        super(MWDNet_CPSF_depth, self).__init__()
        self.in_channels = n_channels
        self.out_channels = n_classes
        self.psf = psf
        _, psf_c, height_p, width_p = psf.size()
        channels = [dim * (2 ** i) for i in range(depth+1)] # list [32, 64, 128, 256]
        h = [height // (2 ** i) for i in range(depth+1)]
        w = [width // (2 ** i) for i in range(depth+1)]
        h_p = [height_p // (2 ** i) for i in range(depth+1)]
        w_p = [width_p // (2 ** i) for i in range(depth+1)]
        
        self.inc0 = DoubleConvG(psf_c, channels[0])
        self.down11 = DownG(channels[0], channels[1])
        self.down22 = DownG(channels[1], channels[2])
        self.down33 = DownG(channels[2], channels[3])

# 디코더 (Up-sampling) 브랜치
        # U-Net의 forward 로직(cat)에 따라 in_channels를 (아래층 채널 + 스킵연결 채널)로 계산
        self.up1 = Upconvnext(channels[3] + channels[3], channels[2]) # x5(256) + x4(256) -> 128
        self.up2 = Upconvnext(channels[2] + channels[2], channels[1]) # x(128) + x3(128) -> 64
        self.up3 = Upconvnext(channels[1] + channels[1], channels[0]) # x(64) + x2(64) -> 32
        self.up4 = Upconvnext(channels[0] + channels[0], channels[0]) # x(32) + x1(32) -> 32
        
        # forward에서 3(intensity)과 n_classes(depth)로 분리하므로, n_classes + 3이 맞습니다.
        self.outc = OutConv(channels[0], n_classes + 3)

# 메인 인코더 (Down-sampling) 브랜치
        self.inc = ConvNextBlock(n_channels, channels[0])
        self.down1 = DownConvNext(channels[0], channels[1])
        self.down2 = DownConvNext(channels[1], channels[2])
        self.down3 = DownConvNext(channels[2], channels[3])
        self.down4 = DownConvNext(channels[3], channels[3]) # Bottleneck
        
        # self.delta = nn.Parameter(torch.tensor(np.ones(5)*0.01, dtype=torch.float32))
        # self.w = nn.Parameter(torch.tensor(np.ones(2)*0.01, dtype=torch.float32))    
        self.w1 = W2(channels[0], h[0], w[0], h_p[0], w_p[0], k=1) # 64, 270, 480
        self.w2 = W2(channels[1], h[1], w[1], h_p[1], w_p[1], k=1) # 128, 135, 240
        self.w3 = W2(channels[2], h[2], w[2], h_p[2], w_p[2], k=1) # 256, 67, 120
        self.w4 = W2(channels[3], h[3], w[3], h_p[3], w_p[3], k=1) # 512, 33, 60
        
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        current_psf = self.psf.to(x.device)
        x1 = self.inc(x)         
        x2 = self.down1(x1)  
        x3 = self.down2(x2)     
        x4 = self.down3(x3)       
        x5 = self.down4(x4) 

        psf1 = self.inc0(current_psf)
        psf2 = self.down11(psf1)
        psf3 = self.down22(psf2)
        psf4 = self.down33(psf3)

        x1 = self.w1(x1, psf1)
        x2 = self.w2(x2, psf2)
        x3 = self.w3(x3, psf3)
        x4 = self.w4(x4, psf4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        intensity, x = torch.split(x, [3, self.out_channels], dim=1)
        intensity = self.sig(intensity)
        x = self.softmax(x)
        return intensity, x

class W2(nn.Module):
    """
    'W' 클래스를 기반으로, WieNer_SV(멀티 커널, k 차원, kernel_weights 등) 기능을 결합하되
    추가적인 패딩 없이 FFT를 수행하는 예시.
    
    Args:
        channels (int): 입력/출력 채널 수
        height (int): 이미지 높이(H)
        width (int):  이미지 너비(W)
        k (int): 추가로 사용할 PSF(또는 ROI) 개수
    """
    def __init__(self, channels, height, width, height_p, width_p, k=1):
        super(W2, self).__init__()
        self.height_freq = height + height_p
        self.width_freq = (width + width_p)// 2 + 1

        self.psf_weights = nn.Parameter(
            torch.ones(k, channels, self.height_freq, self.width_freq) * 0.01
        )
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=channels)

        self.alpha = nn.Parameter(torch.ones(k, 1, 1, 1) * 0.1)
        
        self.relu = nn.ReLU()
        self.k = k

    def forward(self, raw: torch.Tensor, psf: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        """
        Args:
            raw (torch.Tensor): (B, C, H, W) 형태의 입력
            psf (torch.Tensor): (B, C, H_p, W_p) 형태의 PSF (혹은 동일 크기 H, W일 수도 있음)
            epsilon (float): Regularization 파라미터
        
        Returns:
            torch.Tensor: (B, C, H, W) 형태의 복원 결과
        """
        B, C, H, W = raw.shape
        _, _, H_p, W_p = psf.shape
        
        psf = psf.reshape(self.k,-1,psf.size(-2),psf.size(-1))
        psf_sum = psf.sum(dim=(-2, -1), keepdim=True)          # (B, C, 1, 1)
        psf_normalized = psf / (psf_sum.abs() * self.alpha + 1e-12)
        
        # Apply symmetric padding to raw input
        raw_padded = F.pad(
            raw,
            (W_p // 2, W_p - W_p // 2, H_p // 2, H_p - H_p // 2),
            mode='replicate'
        )
        raw_padded = gaus_t(raw_padded, fwhm=2)
        psf_padded = F.pad(
            psf_normalized,
            (W // 2, W - W // 2, H // 2, H - H // 2),
            mode='constant'
        )
        
        # Compute FFT with 'ortho' normalization to maintain energy
        raw_fft = torch.fft.rfft2(raw_padded, dim=(-2, -1))  # Shape: (B, C, H_freq, W_freq)
        psf_fft = torch.fft.rfft2(psf_padded, s=(raw_padded.size(-2), raw_padded.size(-1)), dim=(-2, -1))  # Shape: (B, C, H_freq, W_freq)
        
        pw = self.relu(self.psf_weights)
        wiener_filter = psf_fft.conj() / (psf_fft.abs()**2 + epsilon + pw)
        out_fft = raw_fft * wiener_filter  # (k, B, C, H, W//2+1)

        out_spatial = torch.fft.irfft2(out_fft, dim=(-2, -1))
        out_spatial = torch.fft.ifftshift(out_spatial, dim=(-2, -1))
        start_H = H_p // 2
        start_W = W_p // 2
        out_cropped = out_spatial[..., start_H:start_H + H, start_W:start_W + W]  # Shape: (N, B, C, H, W)

        return self.group_norm(out_cropped.real)

def gaussian_window(size, fwhm):
    with torch.no_grad():
        sigma = size / fwhm
        x = torch.arange(size) - (size - 1) / 2
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    return gauss.detach()

def gaus_t(x, fwhm=3):
    b, c, w, h = x.size()
    device = x.device
    dtype = x.dtype

    ga_w = gaussian_window(w, fwhm)
    ga_h = gaussian_window(h, fwhm)

    # 외적을 이용하여 2D 가우시안 윈도우 생성
    ga = ga_w.unsqueeze(1) * ga_h.unsqueeze(0)
    ga = ga.unsqueeze(0).unsqueeze(0)  # (1, 1, w, h)
    ga = ga.expand(b, c, w, h)  # 입력 텐서와 크기 맞추기

    return x * ga.to(x.device)

class DoubleConvG(nn.Module):
    """(Convolution => [GroupNorm] => GELU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=1):
        super(DoubleConvG, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConvNext(nn.Module):
    """Downscaling with average pooling followed by ConvNeXt."""

    def __init__(self, in_channels, out_channels):
        super(DownConvNext, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            ConvNextBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)

class Upconvnext(nn.Module):
    """Upscaling then DoubleConvG with CAWBlock."""
    def __init__(self, in_channels, out_channels):
        super(Upconvnext, self).__init__()
        self.upw = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.caw_block = ConvNextBlock(in_channels//2, out_channels)
        self.convw1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, bias=False)

    def forward(self, w1, w2):
        w1 = self.upw(w1)
        diffY = w2.size()[2] - w1.size()[2]
        diffX = w2.size()[3] - w1.size()[3]
        w1 = F.pad(w1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        w = torch.cat([w2, w1], dim=1)
        w = F.gelu(self.convw1(w))
        w = self.caw_block(w)
        return w


class ConvNextBlock(nn.Module):
    """ConvNeXt Block as described in https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super(ConvNextBlock, self).__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
        self.ds_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, kernel_size=3, padding=1, bias=False),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1, bias=False) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.net(h)
        return h + self.res_conv(x)

class OutConv(nn.Module):
    """Final 1x1 Convolution to map to desired output channels."""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DownG(nn.Module):
    """Downscaling with average pooling followed by DoubleConvG."""

    def __init__(self, in_channels, out_channels):
        super(DownG, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConvG(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)