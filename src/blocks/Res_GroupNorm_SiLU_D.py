from .Res_Base import ResBlock

import torch.nn as nn
import warnings


class ResBlockGroupNorm(ResBlock):
    """
    Residual block with Group Normalization and SiLU activation.

    This block uses Group Normalization instead of Batch Normalization, which can be
    beneficial for smaller batch sizes or when batch statistics are unreliable.

    Structure:
    Input
      |
      ├─── Main Path ──────────────────────────────────────────────┐
      │    GNorm -> SiLU -> Conv -> Dropout -> GNorm -> SiLU -> Conv -> Dropout
      │                                                            │
      └─── Residual Connection (Optional 1x1 Conv) ────────────────┘
                                                                   |
                                                                Output

    Attributes:
        n_groups (int): Number of groups for Group Normalization.
        dropout_prob (float): Probability of an element to be zeroed in the dropout layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int = 32,
        dropout_rate: float = 0.1,
    ):
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        super().__init__(in_channels, out_channels)

    def _adjust_groups(self, channels: int) -> int:
        if self.n_groups > channels:
            warnings.warn(
                f"Number of groups ({self.n_groups}) is greater than the number of channels ({channels}). "
                f"Setting number of groups to {channels}.",
                UserWarning,
            )
            return channels
        return self.n_groups

    def _build_main_path(self) -> nn.Sequential:
        n_groups_first = self._adjust_groups(self.in_channels)
        n_groups_second = self._adjust_groups(self.out_channels)

        return nn.Sequential(
            nn.GroupNorm(n_groups_first, self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
            nn.GroupNorm(n_groups_second, self.out_channels),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
        )
