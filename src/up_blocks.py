from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Type, Dict, Any
from src.res_blocks import ResBlock, ResBlockBatchNorm

class BaseUpBlock(nn.Module, ABC):
    """
    Abstract base class for up blocks in the Residual Attention U-Net model.

    This class defines the interface for all up blocks.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(BaseUpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpBlock.

        Args:
            x (torch.Tensor): Input tensor from the previous layer.
            skip (torch.Tensor): Skip connection input from the encoder.

        Returns:
            torch.Tensor: Output after upsampling and processing.
        """
        pass


class UpSampleDoubleResBlock(BaseUpBlock):
    """
    Up block with upsampling and two residual blocks.

    Structure:
    Input
      |
      v
    ConvTranspose2d (2x2, stride 2)
      |
      v
    Concatenate with skip connection
      |
      v
    ResBlock (in_channels*2 -> out_channels)
      |
      v
    ResBlock (out_channels -> out_channels)
      |
      v
    Output

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        res_block_type (Type[ResBlock]): Type of ResBlock to use.
        res_block_params (Dict[str, Any]): Additional parameters for the ResBlock constructor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_block_type: Type[ResBlock] = ResBlockBatchNorm,
        res_block_params: Dict[str, Any] = {},
    ):
        super(UpSampleDoubleResBlock, self).__init__(in_channels, out_channels)

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv_blocks = nn.Sequential(
            res_block_type(
                in_channels=out_channels
                * 2,  # *2 because of concatenation with skip connection
                out_channels=out_channels,
                **res_block_params
            ),
            res_block_type(
                in_channels=out_channels, out_channels=out_channels, **res_block_params
            ),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_blocks(x)


class SingleConvUpBlock(BaseUpBlock):
    """
    Up block with upsampling and a single convolutional layer.

    Structure:
    Input
      |
      v
    ConvTranspose2d (2x2, stride 2)
      |
      v
    Concatenate with skip connection
      |
      v
    Conv2d (in_channels*2 -> out_channels)
      |
      v
    Output
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SingleConvUpBlock, self).__init__(in_channels, out_channels)

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
