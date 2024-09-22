import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding
    Description:
    - This module is used to create the positional encoding.
    Architecture:
    - Sinusoidal Positional encoding
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_sequence_length: int = 1_000,
        frequency_scale: int = 10_000,
    ):
        """
        Args:
            embedding_dim (int): The dimension of the embedding.
            max_sequence_length (int): The maximum length of the input sequence.
            frequency_scale (float): The scale factor for frequency calculations.
        """
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.frequency_scale = frequency_scale
        self.register_buffer(
            "positional_encoding",
            self._get_positional_encoding(max_sequence_length, embedding_dim),
        )

    def _get_positional_encoding(self, max_sequence_length, embedding_dim):
        positional_encoding = torch.zeros(max_sequence_length, embedding_dim)
        position_indices = torch.arange(0, max_sequence_length).unsqueeze(1).float()
        angular_frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(self.frequency_scale)) / embedding_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position_indices * angular_frequencies)
        positional_encoding[:, 1::2] = torch.cos(position_indices * angular_frequencies)
        return positional_encoding

    def forward(self, time_steps):
        """
        Args:
            time_steps: torch.Tensor of shape (B,) representing time steps or positions
        """
        return self.positional_encoding[time_steps]
