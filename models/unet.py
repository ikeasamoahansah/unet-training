import torch
import torch.nn as nn
from .unet_helper import *


class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, bilinear=False) -> None:
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.input_c = ConvBlock(n_channels, 64)
        self.down_1 = Downscaling(64, 128)
        self.down_2 = Downscaling(128, 256)
        self.down_3 = Downscaling(256, 512)
        factor = 2 if bilinear else 1
        self.down_4 = Downscaling(512, 1024 // factor)
        self.up_1 = Upscaler(1024, 512 // factor, bilinear)
        self.up_2 = Upscaler(512, 256 // factor, bilinear)
        self.up_3 = Upscaler(256, 128 // factor, bilinear)
        self.up_4 = Upscaler(128, 64, bilinear)
        self.output_c = nn.Conv2d(input_channels=64, output_channels=n_classes, kernel_size=1)


    def forward(self, x):
        x1 = self.input_c(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        out = self.output_c(x)

        return out


