import torch
import torch.nn as nn

from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm


class Up_Res_GroupNorm_SiLU_D(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=32, dropout_rate=0.1):
        super(Up_Res_GroupNorm_SiLU_D, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.module = nn.Sequential(
            ResBlockGroupNorm(
                out_channels * 2,  # For the concatenation
                out_channels,
                n_groups=n_groups,
                dropout_rate=dropout_rate,
            ),
            ResBlockGroupNorm(
                out_channels, out_channels, n_groups=n_groups, dropout_rate=dropout_rate
            ),
        )

    def forward(self, x, g):
        g = self.up(g)
        g = torch.cat([g, x], dim=1)
        g = self.module(g)
        return g
