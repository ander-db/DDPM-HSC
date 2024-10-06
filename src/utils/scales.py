import torch
import random
from typing import Tuple, Optional


class LogScaleTransform:
    """
    Applies the logarithmic scale to the input and target tensors.

    Args:
    - input_tensor (torch.Tensor): Input tensor. Shape (C, H, W).
    - target_tensor (torch.Tensor): Target tensor. Shape (C, H, W).
    """

    def __call__(self, input_tensor, target_tensor):
        input_tensor = log_scale_tensor(input_tensor.squeeze()).unsqueeze(0)
        target_tensor = log_scale_tensor(target_tensor.squeeze()).unsqueeze(0)

        return input_tensor, target_tensor


def log_scale_tensor(tensor, a=1000, epsilon=1e-5) -> torch.Tensor:
    """
    Aplica la escala logarítmica especificada en la documentación de SAOImage a los datos de la imagen.

    Args:
        tensor (torch.Tensor): Datos de la imagen.
        a (int): Parámetro de la escala logarítmica.
        epsilon (float): Valor pequeño para evitar errores en la operación.

    Returns:
        torch.Tensor: Datos de la imagen escalados logarítmicamente.
    """

    # Validación de los parámetros
    assert a > 0, "El parámetro 'a' debe ser mayor que cero."
    assert epsilon > 0, "El parámetro 'epsilon' debe ser mayor que cero."
    assert isinstance(tensor, torch.Tensor), "Los datos deben ser un tensor de PyTorch."
    assert len(tensor.shape) == 2, "Los datos deben tener dos dimensiones."

    # Normalize the tensor between 0 and 1
    tensor = norm_tensor_between_0_1(tensor)

    data_adjusted = tensor + torch.abs(torch.min(tensor)) + epsilon
    return torch.log(a * data_adjusted + 1) / torch.log(
        torch.tensor(a).type(tensor.dtype)
    )


def log_scale_batch_tensor(tensor, a=1000, epsilon=1e-5) -> torch.Tensor:
    """
    Aplica la escala logarítmica especificada en la documentación de SAOImage a los datos de la imagen.

    Args:
        tensor (torch.Tensor): Datos de la imagen.
        a (int): Parámetro de la escala logarítmica.
        epsilon (float): Valor pequeño para evitar errores en la operación.

    Returns:
        torch.Tensor: Datos de la imagen escalados logarítmicamente.
    """

    # Validación de los parámetros
    assert a > 0, "El parámetro 'a' debe ser mayor que cero."
    assert epsilon > 0, "El parámetro 'epsilon' debe ser mayor que cero."
    assert isinstance(tensor, torch.Tensor), "Los datos deben ser un tensor de PyTorch."
    assert len(tensor.shape) == 4, "Los datos deben tener cuatro dimensiones."

    # Aplica el escalado logarítmico a cada imagen del batch de datos
    return torch.stack(
        [log_scale_tensor(image.squeeze(), a, epsilon).unsqueeze(0) for image in tensor]
    )


def norm_tensor_between_0_1(tensor) -> torch.Tensor:
    """
    Normaliza un tensor de PyTorch entre 0 y 1.

    Args:
        tensor (torch.Tensor): Tensor de PyTorch. (B, C, H, W)
        min_value (float): Valor mínimo de la normalización.
        max_value (float): Valor máximo de la normalización.

    Returns:
        torch.Tensor: Tensor normalizado.
    """

    z = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

    return z
