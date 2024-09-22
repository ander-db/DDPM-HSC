import torch.nn as nn
from .Res_GroupNorm_SiLU_D import ResBlockGroupNorm


class Down_Res_GroupNorm_SiLU_D(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=32, dropout_rate=0.1):
        super(Down_Res_GroupNorm_SiLU_D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate

        self.module = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlockGroupNorm(in_channels, out_channels, n_groups, dropout_rate),
            ResBlockGroupNorm(out_channels, out_channels, n_groups, dropout_rate),
        )

    def forward(self, x):
        return self.module(x)
