import torch.nn as nn


class TimeEmbeddingProjectionLinearSiLU(nn.Module):
    """
    TimeEmbeddingProjectionLinearSiLU is a class that projects
    the time encoding tensor to a linear layer with SiLU activation.
    """

    def __init__(self, te_dim, out_channels):
        super(TimeEmbeddingProjectionLinearSiLU, self).__init__()
        self.te_projection = nn.Sequential(
            nn.Linear(te_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, t):
        return self.te_projection(t)
