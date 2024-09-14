import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
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
