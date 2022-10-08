from typing import List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *

class DarknetTiny3Block(ConvBnLeaky):
    def __init__(self, channels_in: int, channels_out: int, args: List = []) -> None:
        super().__init__(channels_in, channels_out, 3, 1, padding = 1, args=args)

class Darknet3Block(nn.Module):
    def __init__(self, channels_in: int, channel_neck: int) -> None:
        super().__init__()
        self.cbd1 = ConvBnLeaky(channels_in, channel_neck, 1, 1)
        self.cbd2 = ConvBnLeaky(channel_neck, channels_in, 3, 1, padding = 1)
        self.cb3 = ConvBn(channels_in, channels_in, 1, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.cb3(self.cbd2(self.cbd1(x)))
