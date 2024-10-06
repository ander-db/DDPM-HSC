import torch


class NormalizeTransform:
    """
    Normalize the input and target tensors between -1 and 1.
    """

    def __call__(self, input_tensor, target_tensor):

        input_tensor = self.normalize_tensor_between_minus1_1(input_tensor)
        target_tensor = self.normalize_tensor_between_minus1_1(target_tensor)

        return input_tensor, target_tensor

    def normalize_tensor_between_minus1_1(self, tensor):
        """
        Normalize the tensor between -1 and 1.

        Args:
        - tensor (torch.Tensor): Input tensor. Shape (C=1, H, W).
        """
        tensor_min = torch.min(tensor)
        tensor_max = torch.max(tensor)
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        tensor = (tensor - 0.5) / 0.5

        return tensor
