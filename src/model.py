import torch
import torch.jit

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
