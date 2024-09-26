import pytest
import torch
from src.utils.transformations import RandomFlipRotateTransform


@pytest.fixture
def sample_tensor():
    return torch.tensor([[1, 2], [3, 4]]).float()


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0, torch.tensor([[1, 2], [3, 4]])),
        (90, torch.tensor([[3, 1], [4, 2]])),
        (180, torch.tensor([[4, 3], [2, 1]])),
        (270, torch.tensor([[2, 4], [1, 3]])),
    ],
)
def test_all_rotations(sample_tensor, angle, expected):
    transform = RandomFlipRotateTransform(flip_prob=0, rotate_prob=1)
    transform.rotate_options = [angle]  # Force specific rotation

    print(f"\nTesting rotation by {angle} degrees")
    print("Input tensor:")
    print(sample_tensor)

    result, _ = transform(sample_tensor)

    print(f"Result after {angle} degree rotation:")
    print(result)

    print("Expected result:")
    print(expected)

    assert torch.all(
        result.eq(expected)
    ), f"\nRotation by {angle} degrees failed.\nExpected:\n{expected}\nGot:\n{result}"
    print(f"Rotation by {angle} degrees test passed successfully!")


@pytest.mark.parametrize(
    "flip_type, expected",
    [
        ("no_flip", torch.tensor([[1, 2], [3, 4]])),
        ("flip_x", torch.tensor([[2, 1], [4, 3]])),
        ("flip_y", torch.tensor([[3, 4], [1, 2]])),
        ("flip_xy", torch.tensor([[4, 3], [2, 1]])),
    ],
)
def test_all_flips(sample_tensor, flip_type, expected):
    transform = RandomFlipRotateTransform(flip_prob=1, rotate_prob=0)
    transform.flip_options = [flip_type]  # Force specific flip

    print(f"\nTesting {flip_type}")
    print("Input tensor:")
    print(sample_tensor)

    result, _ = transform(sample_tensor)

    print(f"Result after {flip_type}:")
    print(result)

    print("Expected result:")
    print(expected)

    assert torch.all(
        result.eq(expected)
    ), f"\n{flip_type} failed.\nExpected:\n{expected}\nGot:\n{result}"
    print(f"{flip_type} test passed successfully!")


def test_invalid_dimensions():
    transform = RandomFlipRotateTransform()
    invalid_tensor = torch.randn(3, 32, 32)  # 3D tensor instead of 2D
    with pytest.raises(ValueError, match="Input tensor must have 2 dimensions"):
        transform(invalid_tensor)


if __name__ == "__main__":
    pytest.main(["-v"])
