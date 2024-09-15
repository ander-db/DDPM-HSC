from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.res_blocks import ResBlock, ResBlockBatchNorm

from typing import Type, Dict, Any


class BaseDownBlock(nn.Module, ABC):
    """
    Abstract base class for down blocks in the Residual Attention U-Net model.

    This class defines the interface for all down blocks.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(BaseDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DownBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after downsampling and processing.
        """
        pass


class DownsampleDoubleResBlock(BaseDownBlock):
    """
    Down block with downsampling and two residual blocks.

    Structure:
    Input
      |
      v
    MaxPool2d (2x2, stride 2)
      |
      v
    ResBlock (in_channels -> out_channels)
      |
      v
    ResBlock (out_channels -> out_channels)
      |
      v
    Output

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        res_block_type (Type[ResBlock]): Type of ResBlock to use (e.g., ResBlockBatchNorm, ResBlockGroupNorm).
        res_block_params (Dict[str, Any]): Additional parameters for the ResBlock constructor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_block_type: Type[ResBlock] = ResBlockBatchNorm,
        res_block_params: Dict[str, Any] = {},
    ):
        super(DownsampleDoubleResBlock, self).__init__(in_channels, out_channels)

        self.down_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_type(
                in_channels=in_channels, out_channels=out_channels, **res_block_params
            ),
            res_block_type(
                in_channels=out_channels, out_channels=out_channels, **res_block_params
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_block(x)


# Example of another possible implementation
class SingleConvDownBlock(BaseDownBlock):
    """
    Down block with downsampling and a single convolutional layer.

    Structure:
    Input
      |
      v
    MaxPool2d (2x2, stride 2)
      |
      v
    Conv2d (in_channels -> out_channels)
      |
      v
    Output
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SingleConvDownBlock, self).__init__(in_channels, out_channels)

        self.down_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_block(x)
