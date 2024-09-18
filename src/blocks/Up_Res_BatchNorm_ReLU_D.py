import torch
import torch.nn as nn

from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm


class Up_Res_GroupNorm_SiLU_D(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.module = nn.Sequential(
            ResBlockGroupNorm(out_channels * 2, out_channels, *args, **kwargs),
            ResBlockGroupNorm(out_channels, out_channels, *args, **kwargs),
        )

    def forward(self, x, g):
        x = self.up(x)
        x = torch.cat([x, g], dim=1)
        x = self.module(x)
        return x

