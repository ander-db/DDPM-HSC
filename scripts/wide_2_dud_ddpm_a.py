import torch
import torch.nn as nn

from src.ddpm.ddpm_A import DDPM_2D


if __name__ == "__main__":

    model = DDPM_2D()

    t_0 = torch.tensor([0])
    sample_image = torch.randn(1, 1, 64, 64)

    res = model(sample_image, t_0)
    print(res)
