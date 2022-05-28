import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size = 3, dropout=False, drop_prob=0.5):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=1)
        )
        
        self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size,kernel_size), padding=(kernel_size//2,kernel_size//2)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
        )

    def forward(self, x1):

        up = self.up(x1)
        out = self.conv(up) 
        
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)),
                nn.Sigmoid()
        ) 

    def forward(self, x):
        return self.conv(x)
