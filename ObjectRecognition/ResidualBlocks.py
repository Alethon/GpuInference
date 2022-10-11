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

class Darknet3ResidualBlock(Residual):
    def __init__(self, channels: int, channel_neck: int) -> None:
        super().__init__(Darknet3Block(channels, channel_neck))
