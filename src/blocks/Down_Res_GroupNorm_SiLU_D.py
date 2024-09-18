import torch.nn as nn
from Res_GroupNorm_SiLU_D import ResBlockGroupNorm


class Down_Res_GroupNorm_SiLU_D(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        self.module = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlockGroupNorm(in_channels, out_channels, *args, **kwargs),
            ResBlockGroupNorm(out_channels, out_channels, *args, **kwargs),
        )

    def forward(self, x):
        return self.module(x)
