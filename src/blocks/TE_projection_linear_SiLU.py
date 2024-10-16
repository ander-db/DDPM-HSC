import torch.nn as nn


class TimeEmbeddingProjectionLinearSiLU(nn.Module):
    """
    TimeEmbeddingProjectionLinearSiLU is a class that projects
    the time encoding tensor to a linear layer with SiLU activation.
    """

    def __init__(self, te_dim, out_channels, initialization_method="xavier_uniform"):
        super(TimeEmbeddingProjectionLinearSiLU, self).__init__()

        # Attributes
        self.te_dim = te_dim
        self.out_channels = out_channels

        # Build module
        self._build_module()

        # Initialize weights
        self._init_weights(initialization_method)

    def _build_module(self):

        self.te_projection = nn.Sequential(
            nn.Linear(self.te_dim, self.out_channels),
            nn.SiLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )

    def _init_weights(self, initialization_method: str):
        """Initialize weights using the specified method."""

        if initialization_method == "xavier_uniform":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        if initialization_method == "kaiming_normal":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, t):
        return self.te_projection(t)
