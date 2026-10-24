from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from google.protobuf.json_format import MessageToDict
from viam.components.camera import Camera, DistortionParameters, IntrinsicParameters
from viam.logging import getLogger
from viam.media.video import NamedImage, ViamImage
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.proto.component.camera import GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from src.transform_pipeline import GPUTransformPipeline

logger = getLogger(__name__)


# GPU transform camera class
class GPUTransformCamera(Camera, Reconfigurable):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "camera"), "gpu-transform")

    class Properties(NamedTuple):
        intrinsic_parameters: IntrinsicParameters
        """The properties of the camera"""
        distortion_parameters: DistortionParameters
        """The distortion parameters of the camera"""

    def __init__(self, name: str):
        super().__init__(name)
        self.source_camera: Camera
        self.pipeline: Optional[GPUTransformPipeline] = None

    # constructor
    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        camera = cls(config.name)
        camera.reconfigure(config, dependencies)
        return camera

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        req_deps = []
        fields = config.attributes.fields
        if "source" not in fields:
            raise Exception(
                "missing required source to provide feed for transform camera"
            )
        elif not fields["source"].HasField("string_value"):
            raise Exception("source must be a string")
        source = fields["source"].string_value
        if not source:
            raise ValueError("source cannot be empty")

        req_deps.append(source)
        return req_deps, []

    async def get_image(
        self,
        mime_type: str = "",
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ViamImage:
        if self.source_camera is None:
            raise RuntimeError("Source camera not configured")
        if self.pipeline is None:
            raise RuntimeError("Transform pipeline not configured")

        source_image = await self.source_camera.get_image(mime_type, timeout=timeout)

        transformed_image = self.pipeline.transform(source_image, mime_type)
        logger.info(f"Returning transformed image: {type(transformed_image)}")
        return transformed_image

    async def get_images(
        self, *, timeout: Optional[float] = None
    ) -> Tuple[List[NamedImage], ResponseMetadata]:
        raise NotImplementedError

    async def get_point_cloud(
        self, *, timeout: Optional[float] = None
    ) -> Tuple[bytes, str]:
        raise NotImplementedError

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> GetPropertiesResponse:
        if self.source_camera is None:
            raise RuntimeError("Source camera not configured")

        return await self.source_camera.get_properties(timeout=timeout, **kwargs)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        source = config.attributes.fields["source"].string_value
        camera_resource = dependencies[Camera.get_resource_name(source)]
        logger.info(f"camera name: {source}")
        self.source_camera = cast(Camera, camera_resource)
        self.source = source

        # Parse pipeline configuration with proper protobuf conversion
        if "pipeline" in config.attributes.fields:
            pipeline_config = [
                MessageToDict(item.struct_value)
                for item in config.attributes.fields["pipeline"].list_value.values
            ]

            logger.info(f"pipeline config: {pipeline_config}")
            self.pipeline = GPUTransformPipeline(pipeline_config)

    async def close(self):
        if self.source_camera is not None:
            await self.source_camera.close()
        self.pipeline = None
