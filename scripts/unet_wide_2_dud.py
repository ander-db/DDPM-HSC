import torch
import torch.utils.data
import torch.nn as nn

import lightning as l


from src.unet.UNet_Res_GroupNorm_SiLU_D import UNet_Res_GroupNorm_SiLU_D


if __name__ == "__main__":
    # Dataset

    # Model
    model = UNet_Res_GroupNorm_SiLU_D()

    input_tensor = torch.randn(100, 3, 64, 64)
    target_tensor = torch.randn(100, 3, 64, 64)

    dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Trainer

    trainer = l.Trainer(max_epochs=10)
    trainer.fit(model, dataloader)
