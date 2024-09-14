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
        """
        Initialize the ResBlockGroupNorm.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_groups (int, optional): Number of groups for Group Normalization. Defaults to 32.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
        """
        self.n_groups = n_groups
        self.dropout_rate = dropout_rate
        super(ResBlockGroupNorm, self).__init__(in_channels, out_channels)

    def _build_main_path(self) -> nn.Sequential:
        """
        Build the main path of the residual block with Group Normalization.

        Returns:
            nn.Sequential: The main convolutional path of the block.
        """

        return nn.Sequential(
            nn.GroupNorm(self.n_groups, self.in_channels),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
            nn.GroupNorm(self.n_groups, self.out_channels),
            nn.SiLU(),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
        )


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
