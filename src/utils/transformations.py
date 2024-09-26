import torch
import random
from typing import Tuple, Optional


class RandomFlipRotateTransform:
    """
    Randomly flips and rotates the input tensor based on the given probabilities.

    Args:
        flip_prob (float): Probability of applying a flip transformation
        rotate_prob (float): Probability of applying a rotation transformation
    """

    def __init__(self, flip_prob: float = 0.5, rotate_prob: float = 0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.flip_options = ["no_flip", "flip_x", "flip_y", "flip_xy"]
        self.rotate_options = [0, 90, 180, 270]

    def __call__(
        self, input_tensor: torch.Tensor, target_tensor: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply random flip and rotation transformations to the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor to be transformed
            target_tensor (Optional[torch.Tensor]): Target tensor to be transformed

        Returns:
            Tuple[torch.Tensor, Optional
        """

        if input_tensor.dim() != 2:
            raise ValueError("Input tensor must have 2 dimensions (height, width)")

        # Apply flip transformation
        if random.random() < self.flip_prob:
            flip_type = random.choice(self.flip_options)
            input_tensor = self._apply_flip(input_tensor, flip_type)
            if target_tensor is not None:
                target_tensor = self._apply_flip(target_tensor, flip_type)

        # Apply rotation transformation
        if random.random() < self.rotate_prob:
            angle = random.choice(self.rotate_options)
            input_tensor = self._apply_rotation(input_tensor, angle)
            if target_tensor is not None:
                target_tensor = self._apply_rotation(target_tensor, angle)

        return input_tensor, target_tensor

    def _apply_flip(self, tensor: torch.Tensor, flip_type: str) -> torch.Tensor:
        if flip_type == "no_flip":
            return tensor
        elif flip_type == "flip_x":
            return torch.flip(tensor, [1])
        elif flip_type == "flip_y":
            return torch.flip(tensor, [0])
        elif flip_type == "flip_xy":
            return torch.flip(tensor, [0, 1])
        else:
            raise ValueError(f"Unknown flip type: {flip_type}")

    def _apply_rotation(self, tensor: torch.Tensor, angle: int) -> torch.Tensor:
        k = -(angle // 90) % 4
        return torch.rot90(tensor, k)
