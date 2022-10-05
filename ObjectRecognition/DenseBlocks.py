import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *

class DenseBlock1(nn.Module):
    def __init__(self, channels_in: int, channels: int, channels_out: int) -> None:
        super().__init__()
        self.cbd1 = ConvBnDropout(channels_in, channels, 1, 1)
        self.cbd2 = ConvBnDropout(channels, channels, 3, 1, padding = 1)
        self.cb3 = ConvBn(channels, channels_out, 1, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.cb3(self.cbd2(self.cbd1(x)))

class DenseBlock2(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.cbd1 = ConvBnDropout(channels_in, channels_out, 1, 1)
        self.cb2 = ConvBn(channels_out, channels_out, 3, 1, padding = 1)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.cb2(self.cbd1(x))
