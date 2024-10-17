from typing import Optional
import torch
import torch.nn as nn
import warnings


class ResBlockGroupNorm(nn.Module):
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
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_groups (int): Number of groups for Group Normalization.
        dropout_rate (float): Probability of an element to be zeroed in the dropout layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int = 32,
        dropout_rate: float = 0.1,
        initialization_method: str | None = None,
    ):
        super().__init__()

        # Save attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate

        # Build main path and residual connection
        self.main_path = self._build_main_path()
        self.residual_connection = self._build_residual_connection()

        # Initialize weights
        if initialization_method:
            self._init_weights(initialization_method)

    def _adjust_groups(self, channels: int) -> int:
        if self.n_groups > channels:
            warnings.warn(
                f"Number of groups ({self.n_groups}) is greater than the number of channels ({channels}). "
                f"Setting number of groups to {channels}.",
                UserWarning,
            )
            return channels

        if channels % self.n_groups != 0:
            # Encontrar el mayor divisor de channels que sea menor o igual a self.n_groups
            new_n_groups = self.n_groups
            while new_n_groups > 1:
                if channels % new_n_groups == 0:
                    break
                new_n_groups -= 1

            warnings.warn(
                f"Number of channels ({channels}) is not divisible by the number of groups ({self.n_groups}). "
                f"Setting number of groups to {new_n_groups}.",
                UserWarning,
            )
            return new_n_groups

        return self.n_groups

    def _build_main_path(self) -> nn.Sequential:
        n_groups_first = self._adjust_groups(self.in_channels)
        n_groups_second = self._adjust_groups(self.out_channels)

        return nn.Sequential(
           nn.GroupNorm(n_groups_first, self.in_channels, eps=1e-3),
           nn.SiLU(),
           nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
           nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
           nn.GroupNorm(n_groups_second, self.out_channels, eps=1e-3),
           nn.SiLU(),
           nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
           nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
        )

        # Instead of group, silu, conv, dropout, group, silu, conv, dropout
        # We'll use conv, group, silu, dropout, conv, group

        #return nn.Sequential(
        #    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
        #    nn.GroupNorm(n_groups_second, self.out_channels, eps=1e-3),
        #    nn.SiLU(),
        #    nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
        #    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        #    nn.GroupNorm(n_groups_second, self.out_channels, eps=1e-3),
        #)

    def _build_residual_connection(self) -> Optional[nn.Module]:
        if self.in_channels != self.out_channels:
            return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        return None

    def _init_weights(self, initialization_method: str):
        """Initialize weights using the specified method."""

        if initialization_method == "xavier_uniform":

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        elif initialization_method == "kaiming_normal":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlockGroupNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying the residual block.
        """
        main_output = self.main_path(x)
        residual = self.residual_connection(x) if self.residual_connection else x
        return main_output + residual
