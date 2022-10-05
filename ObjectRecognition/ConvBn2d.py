import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class ConvBn(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int, padding: int = 0, bias: bool = False) -> None:
        super().__init__()
        self.conv: nn.Module = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=bias)
        self.bn: nn.Module   = nn.BatchNorm2d(channels_out)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))

class ConvBnDropout(ConvBn):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int, padding: int = 0, bias: bool = False) -> None:
        super().__init__(channels_in, channels_out, kernel_size, stride, padding, bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(super().forward(x), inplace=True)
