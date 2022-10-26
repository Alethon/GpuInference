import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class ConvBn(nn.Sequential):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int, padding: int = 0, bias: bool = False, prefix = [], suffix = []) -> None:
        super().__init__(*prefix,
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=padding, bias=bias),
            nn.BatchNorm2d(channels_out),
            *suffix)

class ConvBnLeaky(ConvBn):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int, padding: int = 0, bias: bool = False, prefix = [], suffix = []) -> None:
        super().__init__(channels_in, channels_out, kernel_size, stride, padding=padding, bias=bias, prefix=prefix, suffix=[nn.LeakyReLU(0.1), *suffix])
