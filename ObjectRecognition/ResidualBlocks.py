import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *
from Blocks import *

class Darknet3ResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channel_neck: int) -> None:
        super().__init__()
        self.dense = Darknet3Block(channels_in, channel_neck)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(self.dense.forward(x) + x, inplace=True)
