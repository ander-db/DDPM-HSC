import torch
import torch.jit
import torch.nn as nn

import lightning as l

from typing import List


class ResidualAttentionUNet(l.LightningModule):
    """
    Residual Attention U-Net model for imager to image and image segmentation tasks.

    # Architecture Draw
    TODO

    Args:
    TODO

    Returns:
    TODO
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        down_channels: List[int] = [64, 128, 256],
        mid_channels: List[int] = [512, 512],
        up_channels: List[int] = [256, 128, 64],
        n_group_norm: int = 32,
        n_res_block: int = 3,
        lr: float = 1e-4,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(reduction="mean"),
    ):
        super(ResidualAttentionUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.up_channels = up_channels
        self.n_group_norm = n_group_norm
        self.n_res_block = n_res_block
        self.lr = lr
        self.loss_fn = loss_fn

        self._check_architecture()
        self._build_model()

    def _build_model(self):
        """
        Build the model architecture.
        """
        pass

    def _check_architecture(self):
        """
        Check the model architecture.
        """

        assert self.in_channels > 0, "Input channels must be greater than 0."
        assert self.out_channels > 0, "Output channels must be greater than 0."
        assert len(self.down_channels) > 0, "Down channels must be greater than 0."
        assert len(self.mid_channels) > 0, "Mid channels must be greater than 0."
        assert len(self.up_channels) > 0, "Up channels must be greater than 0."
        assert self.n_group_norm > 0, "Group normalization must be greater than 0."
        assert self.n_res_block > 0, "Residual blocks must be greater than 0."
        assert self.lr > 0, "Learning rate must be greater than 0."
        assert self.loss_fn is not None, "Loss function must be defined."
        assert isinstance(
            self.loss_fn, torch.nn.Module
        ), "Loss function must be a torch.nn.Module."

        assert len(set(self.mid_channels)) == 1, "All mid channels must be the same."

        assert (
            self.down_channels[-1] * 2 == self.mid_channels[0]
        ), "Last down channels must be half of the first mid channels."

        assert (
            self.up_channels[0] * 2 == self.mid_channels[-1]
        ), "First up channels must be half of the last mid channels."


class AttentionBlock(nn.Module):
    """
    Optimized Attention block for the Residual Attention U-Net model.

    This block implements a spatial attention mechanism that allows the network
    to focus on relevant features in the input tensor. It's designed to be efficient
    and clear in its implementation.

    Args:
        base_channels (int): Base number of channels. The input tensor 'x' will have 2 * base_channels, while the gating signal 'g' will have base_channels.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
    """

    def __init__(self, base_channels: int, dropout_rate: float = 0.1):
        super(AttentionBlock, self).__init__()

        self.base_channels = base_channels
        self.dropout_rate = dropout_rate

        self._build_model()

    def _build_model(self):
        """Build the optimized model architecture."""
        # Process input from skip connection
        self.W_x = nn.Conv2d(
            self.base_channels * 2, self.base_channels, kernel_size=1, stride=2
        )

        # Process gating signal
        self.W_g = nn.Conv2d(self.base_channels, self.base_channels, kernel_size=1)

        # Generate attention map
        self.psi = nn.Sequential(
            nn.Conv2d(self.base_channels, 1, kernel_size=1), nn.Sigmoid()
        )

        # Use ConvTranspose2d for upsampling (can be faster than Upsample in some cases)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # Combine final convolution and dropout into a sequential block
        self.final_block = nn.Sequential(
            nn.Conv2d(self.base_channels * 2, self.base_channels, kernel_size=1),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
        )

        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention block.

        Args:
            x (torch.Tensor): Input feature map from skip connection,
                              shape (batch_size, base_channels*2, height, width)
            g (torch.Tensor): Gating signal from lower layer,
                              shape (batch_size, base_channels, height/2, width/2)

        Returns:
            torch.Tensor: Attended feature map, shape (batch_size, base_channels, height, width)
        """
        # Process skip connection input
        theta_x = self.W_x(x)

        # Process gating signal
        phi_g = self.W_g(g)

        # Combine and apply ReLU
        f = self.relu(theta_x + phi_g)

        # Generate attention map
        f = self.psi(f)

        # Upsample attention map
        f = self.upsample(f)

        # Apply attention
        y = x * f

        # Final processing
        return self.final_block(y)
