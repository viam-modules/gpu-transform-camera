import asyncio

from viam.module.module import Module

# Importing the src package registers the model with Viam's resource registry.
import src.models.gpu_transform as gpu_transform  # noqa: F401
import src.transform_pipeline as transform_pipeline  # noqa: F401

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
