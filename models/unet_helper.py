import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    """ConvBlock with BatchNorm and ReLU"""
    
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        stride: int=1,
        padding: int=1,
    ) -> None:
        super().__init__()

        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.tensor) -> torch.tensor:
        return self.layers(x)


class Downscaling(nn.Module):

    """Downscaling with maxpool and 2 convblocks"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.down(x)


class Upscaler(nn.Module):

    """Upscaler with conv 2x2 and 2 convblocks"""

    def __init__(self, in_channels, out_channels, bilinear=True) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    # taken from : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


