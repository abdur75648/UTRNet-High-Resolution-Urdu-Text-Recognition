"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

"""
Source - https://github.com/sfczekalski/attention_unet
Article - https://towardsdatascience.com/biomedical-image-segmentation-attention-u-net-29b6f0827405
"""

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class AttnUNet(nn.Module):

    def __init__(self, img_ch=1, output_ch=512):
        super(AttnUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 32)
        self.Conv2 = ConvBlock(32, 64)
        self.Conv3 = ConvBlock(64, 128)
        self.Conv4 = ConvBlock(128, 256)
        self.Conv5 = ConvBlock(256, 512)
        
        self.Up5 = UpConv(512, 256)
        self.Att5 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv5 = ConvBlock(512, 256)
        
        self.Up4 = UpConv(256, 128)
        self.Att4 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv4 = ConvBlock(256, 128)
        
        self.Up3 = UpConv(128, 64)
        self.Att3 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv3 = ConvBlock(128, 64)
        
        self.Up2 = UpConv(64, 32)
        self.Att2 = AttentionBlock(F_g=32, F_l=32, n_coefficients=16)
        self.UpConv2 = ConvBlock(64, 32)

        self.Conv = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        # print("="*20,"Feeding to Encoder","="*20)
        # print ("Size 0", x.shape)
        e1 = self.Conv1(x)
        # print ("Size 1", e1.shape)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        # print ("Size 2", e2.shape)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        # print ("Size 3", e3.shape)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        # print ("Size 4", e4.shape)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        # print ("Size 5 (Final Encoder Output) : ", e5.shape)
        
        # print("\n","="*20,"Feeding to Decoder now","="*20)

        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)
        # print ("d5 ", d5.shape)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)
        # print ("d4 ", d4.shape)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)
        # print ("d3 ", d3.shape)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)
        # print ("d2 ", d2.shape)

        out = self.Conv(d2)
        # print("out (Final Decoder Output) : ", out.shape)

        return out
                
# x = torch.randn(1, 1, 32, 400)
# net = AttnUNet(1,512)
# out = net(x)
# print(out.shape)