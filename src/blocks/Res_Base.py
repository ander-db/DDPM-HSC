import torch
import torch.nn as nn
from typing import Optional


class ResBlock(nn.Module):
    """
    Base class for residual blocks in a neural network.

    This class defines the structure for residual blocks, which are fundamental
    components in many modern neural network architectures. Residual blocks allow
    for easier training of deep networks by providing skip connections.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        main_path (nn.Sequential): The main convolutional path of the block.
        residual_connection (Optional[nn.Module]): The residual connection, if needed.

    Note:
        Subclasses must implement the _build_main_path method.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the ResBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.main_path = self._build_main_path()
        self.residual_connection = self._build_residual_connection()

    def _build_main_path(self) -> nn.Sequential:
        """
        Build the main convolutional path of the residual block.

        Returns:
            nn.Sequential: The main path of the block.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _build_residual_connection(self) -> Optional[nn.Module]:
        """
        Build the residual connection if the input and output channels differ.

        Returns:
            Optional[nn.Module]: A 1x1 convolution if channels change, None otherwise.
        """
        if self.in_channels != self.out_channels:
            return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying the residual block.
        """
        main_output = self.main_path(x)
        residual = self.residual_connection(x) if self.residual_connection else x
        return main_output + residual
