# image2image-RAUNet-PytorchLightning
RAUnet for image to image tasks implemeted with Pytorch Lightning


## Mappings generator

```python
python -m scripts.generate_mapping_files --output_dir ./data/mappings/ --split_name seed_42_full  --seed 42  --train_ratio 0.7 --val_ratio 0.15 --cross_validation 5
```
