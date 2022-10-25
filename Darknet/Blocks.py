from typing import List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from Darknet.ConvBn2d import *

class DarknetTiny3Block(ConvBnLeaky):
    def __init__(self, channels_in: int, channels_out: int, prefix = [], suffix = []) -> None:
        super().__init__(channels_in, channels_out, 3, 1, padding = 1, prefix=prefix, suffix=suffix)

class Darknet3Block(nn.Sequential):
    def __init__(self, channels: int, channel_neck: int, prefix = [], suffix = []) -> None:
        super().__init__(*prefix,
            ConvBnLeaky(channels, channel_neck, 1, 1),
            ConvBnLeaky(channel_neck, channels, 3, 1, padding = 1),
            *suffix)
