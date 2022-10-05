import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *
from DenseBlocks import *

class ParallelFuse(nn.Sequential):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([model(x) for model in self], dim=0).sum(dim=0, keepdim=True)

class ParallelFuseBn(ParallelFuse):
    def __init__(self, channels_out: int, *args) -> None:
        super().__init__(*args)
        self.bn = nn.BatchNorm2d(channels_out)
    
    def forward(self, x) -> Tensor:
        return self.bn(super().forward(x))

class ResidualBlock1(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dense_block_count: int = 4) -> None:
        super().__init__()
        
        self.dense_blocks = ParallelFuseBn(channels_out, *[DenseBlock1(channels_in, channels_in, channels_out) for i in range(0, dense_block_count)])
        
        if channels_in != channels_out:
            self.shortcut = ConvBn(channels_in, channels_out, 3, 1, padding = 1)
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(self.dense_blocks.forward(x) + self.shortcut(x), inplace=True)

class ResidualBlock2(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.dense = DenseBlock2(channels_in, channels_out)
        if channels_in != channels_out:
            self.shortcut = ConvBn(channels_in, channels_out, 3, 1, padding = 1)
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(self.dense(x) + self.shortcut(x), inplace=True)
