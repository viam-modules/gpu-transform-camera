from typing import Dict

import pytest
import pytest_asyncio
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig
from viam.services.vision import Vision
from src.models.gpu_transform import GPUTransformCamera
from fake_camera import FakeCamera

# to test without viam server camera
CAMERA_NAME = "fake-camera"

PASSING_PROPERTIES = Vision.Properties(
    classifications_supported=True,
    detections_supported=True,
    object_point_clouds_supported=False,
)

MIN_CONFIDENCE_PASSING = 0.8

WORKING_CONFIG_DICT = {
    "source": CAMERA_NAME,
    "pipeline": [{"type": "resize", "attributes": {"width_px": 224, "height_px": 224}}],
}  # can test with multiple transforms


IMG_PATH = "/Users/isha.yerramilli-rao/gpu-transform-module/src/tests/test_images"


def get_config(config_dict: Dict, name: str) -> ComponentConfig:
    """returns a config populated with picture_directory and camera_name
    attributes.X

    Returns:``
        ComponentConfig: _description_
    """
    struct = Struct()
    struct.update(dictionary=config_dict)
    config = ComponentConfig(name=name, attributes=struct)
    return config


def get_transform_component(config_dict: Dict, reconfigure=True):
    service = GPUTransformCamera("test")
    cam = FakeCamera(CAMERA_NAME, img_path=IMG_PATH, use_ring_buffer=True)
    camera_name = cam.get_resource_name(CAMERA_NAME)  # returns as ResourceName type
    cfg = get_config(config_dict, CAMERA_NAME)
    service.validate_config(cfg)
    if reconfigure:
        service.reconfigure(cfg, dependencies={camera_name: cam})
    return service


class TestGPUTransformCamera:
    @pytest_asyncio.fixture(autouse=True)
    async def setup_service(self):
        """Setup the transform camera service for testing"""
        self.service = get_transform_component(WORKING_CONFIG_DICT, reconfigure=True)
        yield
        # Clean up after tests
        await self.service.close()

    @pytest.mark.asyncio
    async def test_camera_initialization(self):
        """Test that camera initializes correctly with valid config"""
        assert self.service is not None
        assert self.service.source_camera is not None
        assert self.service.pipeline is not None
        assert self.service.source == CAMERA_NAME

    @pytest.mark.asyncio
    async def test_get_image_returns_viam_image(self):
        """Test that get_image returns a ViamImage"""
        image = await self.service.get_image()
        assert image is not None
        # Check it's a ViamImage type
        from viam.media.video import ViamImage

        assert isinstance(image, ViamImage)

    @pytest.mark.asyncio
    async def test_get_image_with_mime_type(self):
        """Test get_image with specific mime type"""
        image = await self.service.get_image(mime_type="image/jpeg")
        assert image is not None

    @pytest.mark.asyncio
    async def test_transform_pipeline_applied(self):
        """Test that transforms are actually applied"""
        # Get original image from source
        original_image = await self.service.source_camera.get_image()

        # Get transformed image
        transformed_image = await self.service.get_image()

        # They should be different objects (transform was applied)
        assert original_image != transformed_image

    @pytest.mark.asyncio
    async def test_multiple_get_image_calls(self):
        """Test multiple consecutive get_image calls work"""
        image1 = await self.service.get_image()
        image2 = await self.service.get_image()
        image3 = await self.service.get_image()

        assert image1 is not None
        assert image2 is not None
        assert image3 is not None

    def test_validate_config_with_valid_config(self):
        """Test config validation passes with valid config"""
        cfg = get_config(WORKING_CONFIG_DICT, "test")
        deps, _ = GPUTransformCamera.validate_config(cfg)
        assert CAMERA_NAME in deps

    def test_validate_config_missing_source(self):
        """Test config validation fails with missing source"""
        bad_config = {
            "pipeline": [
                {"type": "resize", "attributes": {"width_px": 100, "height_px": 100}}
            ]
        }
        cfg = get_config(bad_config, "test")
        with pytest.raises(Exception, match="missing required source"):
            GPUTransformCamera.validate_config(cfg)

    def test_validate_config_empty_source(self):
        """Test config validation fails with empty source"""
        bad_config = {
            "source": "",
            "pipeline": [
                {"type": "resize", "attributes": {"width_px": 100, "height_px": 100}}
            ],
        }
        cfg = get_config(bad_config, "test")
        with pytest.raises(ValueError, match="source cannot be empty"):
            GPUTransformCamera.validate_config(cfg)

    def test_invalid_transform_type(self):
        """Test that invalid transform types raise errors during initialization"""
        bad_config = {
            "source": CAMERA_NAME,
            "pipeline": [{"type": "invalid_transform", "attributes": {}}],
        }
        with pytest.raises(ValueError, match="Unknown transform"):
            get_transform_component(bad_config)

    def test_missing_transform_attributes(self):
        """Test that missing required attributes raise errors"""
        bad_config = {
            "source": CAMERA_NAME,
            "pipeline": [
                {"type": "resize", "attributes": {}}
            ],  # Missing width_px, height_px
        }
        with pytest.raises(KeyError):
            get_transform_component(bad_config)

    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_transforms(self):
        """Test pipeline with multiple transforms works"""
        multi_transform_config = {
            "source": CAMERA_NAME,
            "pipeline": [
                {"type": "resize", "attributes": {"width_px": 224, "height_px": 224}},
                {"type": "grayscale", "attributes": {}},
            ],
        }
        service = get_transform_component(multi_transform_config)
        try:
            image = await service.get_image()
            assert image is not None
        finally:
            await service.close()
