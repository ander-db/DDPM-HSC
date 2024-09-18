from .Res_Base import ResBlock

import torch.nn as nn


class ResBlockBatchNorm(ResBlock):
    """
    Residual block with Batch Normalization and ReLU activation.

    This block uses traditional Batch Normalization and ReLU activation functions,
    which are common in many convolutional neural network architectures.

    Structure:
    Input
      |
      ├─── Main Path ──────────────────────────────────────────────┐
      │    BNorm -> ReLU -> Conv -> Dropout -> BNorm -> ReLU -> Conv -> Dropout
      │                                                            │
      └─── Residual Connection (Optional 1x1 Conv) ────────────────┘
                                                                   |
                                                                Output
    Attributes:
        dropout_prob (float): Probability of an element to be zeroed in the dropout layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
    ):
        self.dropout_rate = dropout_rate
        super(ResBlockBatchNorm, self).__init__(in_channels, out_channels)

    def _build_main_path(self) -> nn.Sequential:

        return nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
        )
