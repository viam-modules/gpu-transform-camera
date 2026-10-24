import os
import sys

import pytest
from PIL import Image
from viam.media.video import ViamImage

# Add the project root to Python path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io

import matplotlib.pyplot as plt
from viam.media.utils.pil import pil_to_viam_image
from viam.media.video import CameraMimeType

from src.transform_pipeline import GPUTransformPipeline
from src.utils.gpu_utils import viam_to_tensor


# unit tests
@pytest.fixture
def sample_image():
    """Creates a sample image object for testing."""
    width, height = 128, 128
    pil_image = Image.new("RGB", (width, height), color="white")
    return pil_to_viam_image(pil_image, mime_type=CameraMimeType.JPEG)


@pytest.fixture
def transform_config():
    """Returns a sample transform configuration."""
    return [{"type": "resize", "attributes": {"width_px": 64, "height_px": 64}}]


def test_config_validation(transform_config):
    """Tests if invalid transform raise errors"""
    pipeline = GPUTransformPipeline(transform_config)
    assert pipeline is not None

    # Test invalid transform type
    invalid_config = [{"type": "invalid_transform", "attributes": {}}]
    with pytest.raises(ValueError):
        GPUTransformPipeline(invalid_config)


def test_missing_attributes(sample_image):
    """Tests if missing attributes raise errors"""
    invalid_attrs = [{"type": "resize", "attributes": {"width_px": 50}}]
    with pytest.raises(KeyError):
        GPUTransformPipeline(invalid_attrs)


def test_resize_transform(sample_image, transform_config):
    pipeline = GPUTransformPipeline(transform_config)
    result = pipeline.transform(sample_image)
    assert isinstance(result, ViamImage)
    result_pil = Image.open(io.BytesIO(result.data))
    assert result_pil.size == (64, 64)


def test_device_handling(sample_image, transform_config):
    pipeline = GPUTransformPipeline(transform_config)
    tensor = viam_to_tensor(sample_image).to(pipeline.device)
    assert tensor.device == pipeline.device


def test_multiple_transforms(sample_image):
    """test multiple transforms on the same image"""
    config = [
        {"type": "resize", "attributes": {"width_px": 50, "height_px": 50}},
        {"type": "rotate", "attributes": {"angle_degs": 90}},
    ]
    pipeline = GPUTransformPipeline(config)
    result = pipeline.transform(sample_image)
    assert isinstance(result, ViamImage)


def test_normalize_transform(sample_image):
    """Test normalize transform"""
    config = [{"type": "normalize", "attributes": {"mean": [0.5], "std": [0.5]}}]
    pipeline = GPUTransformPipeline(config)
    result = pipeline.transform(sample_image)
    assert isinstance(result, ViamImage)


def test_crop_transform(sample_image):
    """Test crop transform"""
    config = [
        {
            "type": "crop",
            "attributes": {
                "x_min_px": 25,
                "y_min_px": 25,
                "x_max_px": 75,
                "y_max_px": 75,
                "overlay_crop_box": False,
            },
        }
    ]
    pipeline = GPUTransformPipeline(config)
    result = pipeline.transform(sample_image)

    # assert type and dims of output
    assert isinstance(result, ViamImage)
    result_pil = Image.open(io.BytesIO(result.data))
    assert result_pil.size == (50, 50)


def test_visual_transforms():
    """display transforms on test image"""

    original = Image.open(
        "/Users/isha.yerramilli-rao/gpu-transform-module/gpu-transform-camera/tests/test_images/test_image.jpg"
    )
    viam_image = pil_to_viam_image(original, mime_type=CameraMimeType.JPEG)
    config = [
        {
            "type": "crop",
            "attributes": {
                "x_min_px": 50,
                "y_min_px": 50,
                "x_max_px": 700,
                "y_max_px": 700,
                "overlay_crop_box": False,
            },
        },
        {"type": "rotate", "attributes": {"angle_degs": 45}},
    ]

    pipeline = GPUTransformPipeline(config)
    result = pipeline.transform(viam_image)

    # Convert result back to PIL for display
    result_pil = Image.open(io.BytesIO(result.data))

    # Display original and transformed
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result_pil)
    plt.title("Transformed")
    plt.axis("off")

    plt.show()
