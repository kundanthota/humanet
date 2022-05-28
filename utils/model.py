from lib2to3.pgen2.pgen import NFAState
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.layers import *

class Deep2DEncoder(nn.Module):
    def __init__(self, image_size=512, kernel_size=3, n_filters=32, dropout=False, drop_prob=0.2):
        super(Deep2DEncoder, self).__init__()
        self.image_size = image_size
        self.n_filters = n_filters

        self.conv1 = nn.Sequential(
                nn.Conv2d(1, n_filters, kernel_size=(kernel_size,kernel_size), padding=(kernel_size//2,kernel_size//2)),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace = True),
        )
        self.pool = nn.MaxPool2d((2,2),(2,2))
        self.conv2 = nn.Sequential(
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size,kernel_size), padding=(kernel_size//2,kernel_size//2)),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace = True),
        )
        self.fc = nn.Sequential(
                nn.Linear(math.ceil(image_size/32 * image_size/32 * n_filters), 256),
        )
    def forward(self, x):   #512 x 512
        x1 = self.conv1(x)  
        x1 = self.pool(x1)  #256 x 256
        x2 = self.conv2(x1)
        x2 = self.pool(x2)  #128 x 128
        x3 = self.conv2(x2)
        x3 = self.pool(x3)  # 64 x 64
        x4 = self.conv2(x3)
        x4 = self.pool(x4)  # 32 x 32
        x5 = self.conv2(x4)
        x5 = self.pool(x5)  # 16 x 16

        flatten = x5.view(-1, math.ceil(self.image_size/32 * self.image_size/32 * self.n_filters))
        x6 = self.fc(flatten)

        
        return x6

class Deep2DDecoder(nn.Module):
    def __init__(self, image_size, kernel_size=3, bilinear=True, n_filters=32, dropout=False, drop_prob=0.2):
        super(Deep2DDecoder, self).__init__()
        
        self.image_size = image_size
        self.n_filters = n_filters

        self.fc = nn.Sequential(
                nn.Linear(256, math.ceil(image_size/32 * image_size/32 * n_filters)),
                nn.LeakyReLU(0.5)
        )

        #self.up1 = Up(1, n_filters, bilinear, kernel_size=kernel_size, dropout=dropout, drop_prob=drop_prob)
        self.up= Up(n_filters, n_filters, bilinear, kernel_size=kernel_size, dropout=dropout, drop_prob=drop_prob)
        self.outc = OutConv(n_filters, 1)

    def forward(self, x): #16 x 16
        x = self.fc(x)
        x = torch.reshape(x, (-1, self.n_filters, math.ceil(self.image_size/32), math.ceil(self.image_size/32)))
        x = self.up(x)
        x = self.up(x) # 64 x 64
        x = self.up(x) # 128 x 128
        x = self.up(x) # 256 x 256
        x = self.up(x) # 512 x 512
        
        out = self.outc(x)

        return out