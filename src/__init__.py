from viam.resource.registry import Registry, ResourceCreatorRegistration
from src.models.gpu_transform import GPUTransformCamera
from src.transform_pipeline import GPUTransformPipeline
from viam.components.camera import Camera

Registry.register_resource_creator(
    Camera.API,
    GPUTransformCamera.MODEL,
    ResourceCreatorRegistration(GPUTransformCamera.new),
)

__all__ = ["GPUTransformCamera", "GPUTransformPipeline"]
