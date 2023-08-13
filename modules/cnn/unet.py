"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code For UNet Feature Extractor - Source - https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=512):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # print(x.shape) # torch.Size([1, 1, 32, 400])
        x1 = self.inc(x)
        # print(x1.shape) # torch.Size([1, 32, 32, 400])
        x2 = self.down1(x1)
        # print(x2.shape) # torch.Size([1, 64, 16, 200])
        x3 = self.down2(x2)
        # print(x3.shape) # torch.Size([1, 128, 8, 100])
        x4 = self.down3(x3)
        # print(x4.shape) # torch.Size([1, 256, 4, 50])
        x5 = self.down4(x4)
        # print(x5.shape) # torch.Size([1, 512, 2, 25])
        
        # print("Upscaling...")
        x = self.up1(x5, x4)
        # print(x.shape) # torch.Size([1, 256, 4, 50])
        x = self.up2(x, x3)
        # print(x.shape) # torch.Size([1, 128, 8, 100])
        x = self.up3(x, x2)
        # print(x.shape) # torch.Size([1, 64, 16, 200])
        x = self.up4(x, x1)
        # print(x.shape) # torch.Size([1, 32, 32, 400])
        logits = self.outc(x)
        # print(logits.shape) # torch.Size([1, 512, 32, 400])
        return logits

# x = torch.randn(1, 1, 32, 400)
# net = UNet()
# out = net(x)
# print(out.shape)