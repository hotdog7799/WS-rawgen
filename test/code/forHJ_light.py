import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.utils.checkpoint import checkpoint

def exists(x):
    return x is not None

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: Depthwise Conv -> Pointwise Conv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConvDS(nn.Module):
    """(DepthwiseSeparableConv => [GroupNorm] => GELU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=1):
        super(DoubleConvDS, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Determine number of groups for GroupNorm
        # GroupNorm requires num_channels to be divisible by num_groups.
        # Ensure num_groups divides mid_channels and out_channels.
        gn_groups_mid = min(num_groups, mid_channels) if mid_channels % num_groups == 0 else 1
        gn_groups_out = min(num_groups, out_channels) if out_channels % num_groups == 0 else 1

        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups_mid, num_channels=mid_channels),
            nn.GELU(),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups_out, num_channels=out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

class DownG_DS(nn.Module):
    """Downscaling with average pooling followed by DoubleConvDS."""
    def __init__(self, in_channels, out_channels):
        super(DownG_DS, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConvDS(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)

# Re-using ConvNextBlock as it's already quite efficient, but we can reduce 'mult'
class ConvNextBlock(nn.Module):
    """ConvNeXt Block"""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True): # mult default 2
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

class DownConvNext(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvNext, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            ConvNextBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.avgpool_conv(x)

class Upconvnext(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upconvnext, self).__init__()
        self.upw = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Reduced mult to 1 for efficiency in Up path
        self.caw_block = ConvNextBlock(in_channels//2, out_channels, mult=1) 
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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class W2(nn.Module):
    def __init__(self, channels, height, width, height_p, width_p, k=1):
        super(W2, self).__init__()
        self.h, self.w = height, width
        self.h_p, self.w_p = height_p, width_p
        
        self.height_freq = height + height_p
        self.width_freq = (width + width_p) // 2 + 1

        self.psf_weights = nn.Parameter(torch.ones(k, channels, 1, 1) * 0.01)
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.alpha = nn.Parameter(torch.ones(k, 1, 1, 1) * 0.1)
        self.k = k

    def forward(self, raw: torch.Tensor, psf: torch.Tensor, epsilon=1e-4) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            raw, psf = raw.float(), psf.float()
            B, C, H, W = raw.shape
            D = psf.shape[0] 
            
            raw_padded = F.pad(raw, (self.w_p // 2, self.w_p - self.w_p // 2, 
                                    self.h_p // 2, self.h_p - self.h_p // 2), mode='replicate')
            raw_fft = torch.fft.rfft2(raw_padded, dim=(-2, -1))
            target_fft_size = (raw_padded.size(-2), raw_padded.size(-1))
            
            accumulated_filter = torch.zeros_like(raw_fft[0]).unsqueeze(0)
            pw_all = F.softplus(self.psf_weights.float())

            for i in range(D):
                psf_d = psf[i:i+1] 
                psf_padded = F.pad(psf_d, (W // 2, W - W // 2, H // 2, H - H // 2), mode='constant', value=0)
                psf_fft = torch.fft.rfft2(psf_padded, s=target_fft_size, dim=(-2, -1))
                
                denom = psf_fft.abs()**2 + pw_all + epsilon
                wiener_filter = psf_fft.conj() / (denom + 1e-8)
                accumulated_filter += wiener_filter 
                
            out_fft = raw_fft * (accumulated_filter / D)
            out_spatial = torch.fft.ifftshift(torch.fft.irfft2(out_fft, dim=(-2, -1)), dim=(-2, -1))
            
            start_H, start_W = self.h_p // 2, self.w_p // 2
            out_cropped = out_spatial[..., start_H:start_H + H, start_W:start_W + W]
            out_cropped = torch.clamp(out_cropped.real, min=-10.0, max=10.0)

        return self.group_norm(out_cropped)

class DoubleConvG(nn.Module):
    """Origina DoubleConvG for reference if needed, but using DoubleConvDS now"""
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


class MWDNet_CPSF_depth_light(nn.Module):
    # CHANGED default dim to 16 (from 32)
    def __init__(self, n_channels, n_classes, psf, channels=64, height=256, width=256, depth=3, dim=16):
        super(MWDNet_CPSF_depth_light, self).__init__()
        self.in_channels = n_channels
        self.out_channels = n_classes
        self.psf = psf
        _, psf_c, height_p, width_p = psf.size()
        
        # Channels scaling: if dim=16, then [16, 32, 64, 128]
        channels_list = [dim * (2 ** i) for i in range(depth+1)] 
        
        self.channels_list = channels_list # save for inspection
        
        h = [height // (2 ** i) for i in range(depth+1)]
        w = [width // (2 ** i) for i in range(depth+1)]
        h_p = [height_p // (2 ** i) for i in range(depth+1)]
        w_p = [width_p // (2 ** i) for i in range(depth+1)]
        
        # CHANGED: Use DoubleConvDS/DownG_DS for PSF branch
        self.inc0 = DoubleConvDS(psf_c, channels_list[0])
        self.down11 = DownG_DS(channels_list[0], channels_list[1])
        self.down22 = DownG_DS(channels_list[1], channels_list[2])
        self.down33 = DownG_DS(channels_list[2], channels_list[3])

        # Decoder
        self.up1 = Upconvnext(channels_list[3] + channels_list[3], channels_list[2]) 
        self.up2 = Upconvnext(channels_list[2] + channels_list[2], channels_list[1]) 
        self.up3 = Upconvnext(channels_list[1] + channels_list[1], channels_list[0]) 
        self.up4 = Upconvnext(channels_list[0] + channels_list[0], channels_list[0]) 
        
        self.outc = OutConv(channels_list[0], n_classes + 3)

        # Main Encoder
        self.inc = ConvNextBlock(n_channels, channels_list[0])
        self.down1 = DownConvNext(channels_list[0], channels_list[1])
        self.down2 = DownConvNext(channels_list[1], channels_list[2])
        self.down3 = DownConvNext(channels_list[2], channels_list[3])
        self.down4 = DownConvNext(channels_list[3], channels_list[3]) # Bottleneck
        
        self.w1 = W2(channels_list[0], h[0], w[0], h_p[0], w_p[0], k=1) 
        self.w2 = W2(channels_list[1], h[1], w[1], h_p[1], w_p[1], k=1) 
        self.w3 = W2(channels_list[2], h[2], w[2], h_p[2], w_p[2], k=1) 
        self.w4 = W2(channels_list[3], h[3], w[3], h_p[3], w_p[3], k=1) 
        
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

        x1 = checkpoint(self.w1, x1, psf1, use_reentrant=False)
        x2 = checkpoint(self.w2, x2, psf2, use_reentrant=False)
        x3 = checkpoint(self.w3, x3, psf3, use_reentrant=False)
        x4 = checkpoint(self.w4, x4, psf4, use_reentrant=False)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        intensity, x = torch.split(x, [3, self.out_channels], dim=1)
        intensity = self.sig(intensity)
        x = self.softmax(x)
        return intensity, x
