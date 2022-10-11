import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from Blocks import *
from ResidualBlocks import *

class DarknetTiny3Layer(nn.Sequential):
    def __init__(self, channels_in: int, channels_out: int, prefix = [], suffix = []) -> None:
        super().__init__(*prefix, nn.MaxPool2d(2, 2), DarknetTiny3Block(channels_in, channels_out), *suffix)

class Darknet3Layer(nn.Sequential):
    def __init__(self, channels_in: int, channel_neck: int, block_count: int, prefix = [], suffix = []) -> None:
        super().__init__(*prefix, *[Darknet3Block(channels_in, channel_neck) for _ in range(0, block_count)], *suffix)

class Darknet3ResidualLayer(nn.Sequential):
    def __init__(self, channels_in: int, channel_neck: int, block_count: int, prefix = [], suffix = []) -> None:
        super().__init__(*prefix, *[Darknet3ResidualBlock(channels_in, channel_neck) for _ in range(0, block_count)], *suffix)
