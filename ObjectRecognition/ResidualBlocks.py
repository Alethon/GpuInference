from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *
from Blocks import *

class Residual(nn.Module):
    def __init__(self, block) -> None:
        super().__init__()
        self.block = block
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + x

class Darknet3ResidualBlock(nn.Sequential):
    def __init__(self, channels_in: int, channel_neck: int) -> None:
        super().__init__(Residual(Darknet3Block(channels_in, channel_neck)), nn.LeakyReLU(0.1))
