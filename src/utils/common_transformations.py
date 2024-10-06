from src.utils.normalizations import NormalizeTransform
from src.utils.scales import LogScaleTransform
from src.utils.transformations import RandomFlipRotateTransform


class Compose:
    """
    Composes several transforms together.

    Args:
    - transforms (list of callables): List of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_tensor, target_tensor):
        for transform in self.transforms:
            input_tensor, target_tensor = transform(input_tensor, target_tensor)
        return input_tensor, target_tensor


TRANSFORM_LOG_NORM_FLIP = Compose(
    [LogScaleTransform(), NormalizeTransform(), RandomFlipRotateTransform()]
)
