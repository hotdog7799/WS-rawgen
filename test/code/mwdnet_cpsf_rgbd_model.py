import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from block import DoubleConvG, DownG, UpG_ConvT, OutConv
from block import WieNerPad  # Make sure you have this implemented somewhere


def create_trainable_gaussian_delta(shape, alpha=1e-3, sigma=1e-3):
    C, H, W = shape
    # fy = torch.fft.fftfreq(H, d=1.0).unsqueeze(1)  # (H, 1)
    # fx = torch.fft.fftfreq(W, d=1.0).unsqueeze(0)  # (1, W_rfft)
    # gaussian = torch.exp(- (fx + fy) / (2 * sigma**2))  # (H, W_rfft)
    gaussian = torch.rand((H, W // 2+1))
    delta = alpha * gaussian
    return nn.Parameter(delta.clone().detach(), requires_grad=True)


class MWDNet_CPSF_RGBD_large_w_softmax_change_wiener_reg(nn.Module):
    def __init__(self, in_channels, out_channels, psf):
        super(MWDNet_CPSF_RGBD_large_w_softmax_change_wiener_reg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psf = psf

        self.inc = DoubleConvG(in_channels, 64)
        self.down1 = DownG(64, 128)
        self.down2 = DownG(128, 256)
        self.down3 = DownG(256, 512)
        self.down4 = DownG(512, 512)

        self.up1 = UpG_ConvT(1024, 256)
        self.up2 = UpG_ConvT(512, 128)
        self.up3 = UpG_ConvT(256, 64)
        self.up4 = UpG_ConvT(128, 64)
        self.outc = OutConv(64, out_channels + 3)

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.w = nn.Parameter(torch.tensor(np.ones((1, psf.shape[1], psf.shape[2], psf.shape[3])), dtype=torch.float32))

        sized = 2700
        self.delta1 = create_trainable_gaussian_delta(shape=(64, sized, sized), alpha=1e+3, sigma=1e+4)
        self.delta2 = create_trainable_gaussian_delta(shape=(128, sized // 2, sized // 2), alpha=1e+3, sigma=1e+4)
        self.delta3 = create_trainable_gaussian_delta(shape=(256, sized // 4, sized // 4), alpha=1e+3, sigma=1e+4)
        self.delta4 = create_trainable_gaussian_delta(shape=(512, sized // 8, sized // 8), alpha=1e+3, sigma=1e+4)

        self.inc0 = DoubleConvG(psf.shape[1], 64)
        self.down11 = DownG(64, 128)
        self.down22 = DownG(128, 256)
        self.down33 = DownG(256, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        psf1 = self.inc0(self.w * self.psf.to(x.device))
        psf2 = self.down11(psf1)
        psf3 = self.down22(psf2)
        psf4 = self.down33(psf3)
        print(psf4.shape)
        print(self.delta4.shape)

        x4 = WieNerPad(x4, psf4, self.delta4.repeat(1, psf4.shape[1], 1, 1))
        x3 = WieNerPad(x3, psf3, self.delta3.repeat(1, psf3.shape[1], 1, 1))
        x2 = WieNerPad(x2, psf2, self.delta2.repeat(1, psf2.shape[1], 1, 1))
        x1 = WieNerPad(x1, psf1, self.delta1.repeat(1, psf1.shape[1], 1, 1))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        intensity, x = torch.split(x, [3, self.out_channels], dim=1)
        intensity = self.sig(intensity)
        x = self.softmax(x)

        return intensity, x
