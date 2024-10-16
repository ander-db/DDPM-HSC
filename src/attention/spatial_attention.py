from typing import Tuple
import torch
import torch.nn as nn


class SpatialAttentionUNet(nn.Module):
    """
    Optimized Attention block for the Residual Attention U-Net model.

    This block implements a spatial attention mechanism that allows the network
    to focus on relevant features in the input tensor. It's designed to be efficient
    and clear in its implementation.

    Structure:
    - TODO: Add structure diagram

    Args:
        base_channels (int): Base number of channels. The input tensor 'x' will have 2 * base_channels, while the gating signal 'g' will have base_channels.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
    """

    def __init__(
        self,
        g_channels: int,
        dropout_rate: float = 0.1,
        initialization_method: str = "xavier_uniform",
    ):
        super(SpatialAttentionUNet, self).__init__()

        self.g_channels = g_channels
        self.dropout_rate = dropout_rate

        self._build_model()
        self._init_weights(initialization_method)

    def _build_model(self):
        """Build the optimized model architecture."""
        # Process input from skip connection
        self.W_x = nn.Conv2d(
            self.g_channels // 2, self.g_channels, kernel_size=1, stride=2
        )

        # Process gating signal
        self.W_g = nn.Conv2d(self.g_channels, self.g_channels, kernel_size=1, stride=1)

        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)

        # Generate attention map
        self.psi = nn.Conv2d(self.g_channels, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        # Use ConvTranspose2d for upsampling (can be faster than Upsample in some cases)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, stride=2, padding=1, output_padding=0
            ),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
        )

    def _init_weights(self, initialization_method: str):
        """Initialize weights using the specified method."""

        if initialization_method == "xavier_uniform":

            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        elif initialization_method == "kaiming_normal":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self, g: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        theta_x = self.W_x(x)
        phi_g = self.W_g(g)

        f = self.relu(theta_x + phi_g)
        f = self.psi(f)
        f = self.sigmoid(f)

        attention_map = self.upsample(f)

        y = x * attention_map

        return y, attention_map
