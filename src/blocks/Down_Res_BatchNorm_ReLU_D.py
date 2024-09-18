import torch.nn as nn

from Res_GroupNorm_SiLU_D import ResBlockGroupNorm


class Encoder_Down_Res_GroupNorm_SiLU_D(nn.Module):
    def __init__(self, channel_layers: list[int] = [64, 128, 256, 512]):
        super().__init__()
        self.channel_layers = channel_layers
        self.encoder = self._build_encoder()

    def _build_encoder(self):
        encoder = nn.ModuleList()
        for i in range(len(self.channel_layers) - 1):
            encoder.append(
                Down_Res_GroupNorm_SiLU_D(
                    self.channel_layers[i],
                    self.channel_layers[i + 1],
                )
            )
        return encoder


class Down_Res_GroupNorm_SiLU_D(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        self.module = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlockGroupNorm(in_channels, out_channels, *args, **kwargs),
            ResBlockGroupNorm(out_channels, out_channels, *args, **kwargs),
        )

    def forward(self, x):
        return self.module(x)

