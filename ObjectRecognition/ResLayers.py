import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ResBlocks import *

class ResidualLayer1(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, block_count: int) -> None:
        super().__init__()

        layers = []
        for i in range(0, block_count - 1):
            layers.append(ResidualBlock1(channels_in, channels_in))
        
        self.layers = nn.Sequential(*[*layers, ResidualBlock1(channels_in, channels_out)])
        
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
