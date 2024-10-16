import torch
import torch.nn as nn
from typing import Optional


class ResBlockBatchNorm(nn.Module):
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
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dropout_rate (float): Probability of an element to be zeroed in the dropout layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
        initialization_method: str = "xavier_uniform",
    ):
        super().__init__()

        # Save attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        # Build main path and residual connection
        self.main_path = self._build_main_path()
        self.residual_connection = self._build_residual_connection()

        # Initialize weights
        self._init_weights(initialization_method)

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
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlockBatchNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying the residual block.
        """
        main_output = self.main_path(x)
        residual = self.residual_connection(x) if self.residual_connection else x
        return main_output + residual
