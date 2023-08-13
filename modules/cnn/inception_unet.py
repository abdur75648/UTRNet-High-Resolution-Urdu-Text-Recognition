"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch.nn.functional as F
import torch.nn as nn
import torch

'''
Source - https://github.com/mribrahim/Pytorch-UNet-and-Inception/blob/e627658ee84e26ef3befd1ded4904048997e84f8/unet/inception.py
An implementation of this paper - https://dl.acm.org/doi/abs/10.1145/3376922
'''

class InceptionConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.double_conv4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outputs = [self.double_conv1(x), self.double_conv2(x), self.double_conv3(x), self.double_conv4(x)]
        return torch.cat(outputs, 1)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpInception(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = self.up(x3)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class InceptionUNet(nn.Module):
    def __init__(self, n_channels=1, out_channels=512, bilinear=True):
        super(InceptionUNet, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.block1 = InceptionConv(64, 32)
        self.block2 = InceptionConv(128, 64)
        self.block3 = InceptionConv(256, 128)
        self.block4 = InceptionConv(512, 128)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = UpInception(1024+512, 256 // factor, bilinear)
        self.up2 = UpInception(896, 128 // factor, bilinear)
        self.up3 = UpInception(448, 32 // factor, bilinear)
        self.up4 = UpInception(208, 16, bilinear)
        self.outc = OutConv(16, out_channels)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        block1 = self.block1(x1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        x = self.up1(x5, x4, block4)
        # x = torch.cat(x, block4)      
        x = self.up2(x, x3, block3)
        # x = torch.cat(x, block3)
        x = self.up3(x, x2, block2)
        # x = torch.cat(x, block2)
        x = self.up4(x, x1, block1)
        # x = torch.cat(x, block1)
        logits = self.outc(x)
        return logits

# x = torch.randn(1, 1, 32, 400)
# net = InceptionUNet()
# out = net(x)
# print(out.shape)