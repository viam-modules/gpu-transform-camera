from src.utils.gpu_utils import get_device, viam_to_tensor, tensor_to_viam
import torch
from viam.media.video import ViamImage, CameraMimeType
from typing import List, Dict, Any, Optional
import torchvision.transforms as T
import torchvision.transforms.functional as F
from viam.logging import getLogger

logger = getLogger(__name__)


# classes to use functional transform as align with current transform registry on viam
class TransformCrop(torch.nn.Module):
    def __init__(
        self, x_min_px, y_min_px, x_max_px, y_max_px, overlay_crop_box=False
    ):  # store overlay_crop_box for display but not used in transform
        super().__init__()
        self.top = y_min_px
        self.left = x_min_px
        self.height = y_max_px - y_min_px
        self.width = x_max_px - x_min_px

    def forward(self, img):
        return F.crop(img, self.top, self.left, self.height, self.width)


class TransformRotate(torch.nn.Module):
    def __init__(self, angle_degs):
        super().__init__()
        self.angle = angle_degs

    def forward(self, img):
        return F.rotate(img, self.angle)


TRANSFORM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resize": {
        "transform": T.Resize,
        "parser": lambda attrs: {"size": (attrs["width_px"], attrs["height_px"])},
    },
    "to_tensor": {"transform": T.ToTensor, "parser": lambda attrs: {}},
    "normalize": {
        "transform": T.Normalize,
        "parser": lambda attrs: {"mean": attrs["mean"], "std": attrs["std"]},
    },
    "grayscale": {"transform": T.Grayscale, "parser": lambda attrs: {}},
    "rotate": {
        "transform": TransformRotate,
        "parser": lambda attrs: {"angle_degs": attrs["angle_degs"]},
    },
    "crop": {
        "transform": TransformCrop,
        "parser": lambda attrs: {
            "x_min_px": attrs["x_min_px"],
            "y_min_px": attrs["y_min_px"],
            "x_max_px": attrs["x_max_px"],
            "y_max_px": attrs["y_max_px"],
            "overlay_crop_box": attrs["overlay_crop_box"],
        },
    },
}


class GPUTransformPipeline:
    def __init__(self, transform_config: List[Dict[str, Any]]):
        """
        Initialize GPU transform pipeline with configuration.

        Expected format:
        [
            {"type": "resize", "attributes": {"width_px": 224, "height_px": 224}},
            {"type": "normalize", "attributes": {"mean": [0.485], "std": [0.229]}},
        ]
        """
        logger.info(
            f"Initializing GPU transform pipeline with {len(transform_config)} transforms"
        )
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        if len(transform_config) == 0:
            raise ValueError("Transform config cannot be empty")

        # build transform registry
        # parse config into GPU transforms
        self.transforms = self._build_pipeline(transform_config)

    def _build_pipeline(self, transform_config: List[Dict[str, Any]]) -> T.Compose:
        transform_list = []

        for transform_entry in transform_config:
            transform = transform_entry["type"]
            attrs = transform_entry["attributes"]
            if transform == "resize":
                if (
                    not float(attrs["width_px"]).is_integer()
                    or not float(attrs["height_px"]).is_integer()
                ):
                    raise ValueError("resize requires integer width_px and height_px")

                attrs["width_px"] = int(attrs["width_px"])
                attrs["height_px"] = int(attrs["height_px"])
            elif transform == "normalize":
                if not (
                    isinstance(attrs.get("mean"), list)
                    and isinstance(attrs.get("std"), list)
                ):
                    raise ValueError("normalize requires mean and std lists")
            elif transform == "rotate":
                if not isinstance(attrs.get("angle_degs"), (int, float)):
                    raise ValueError("rotate requires angle_degs as a number")
            elif transform == "crop":
                for key in ["x_min_px", "y_min_px", "x_max_px", "y_max_px"]:
                    if not isinstance(attrs.get(key), float):
                        raise ValueError(f"crop requires integer {key}")
                if "overlay_crop_box" in attrs and not isinstance(
                    attrs["overlay_crop_box"], bool
                ):
                    raise ValueError("crop requires overlay_crop_box as a bool")

            if transform not in TRANSFORM_REGISTRY:
                raise ValueError(f"Unknown transform: {transform}")

            info = TRANSFORM_REGISTRY[transform]
            transform_class = info["transform"]
            params = info["parser"](transform_entry["attributes"])

            transform_fn = transform_class(**params)  # initialize transform
            transform_list.append(transform_fn)

        transform_pipeline = T.Compose(
            transform_list
        )  # compose transforms into a single pipeline

        return transform_pipeline

    def transform(self, image: ViamImage, mime_type: Optional[str] = None) -> ViamImage:
        if mime_type is None:
            mime_type = CameraMimeType.JPEG  # default to jpeg

        tensor = viam_to_tensor(image, device=self.device)  # ViamImage to tensor
        transformed = self.transforms(tensor)  # apply transforms
        return tensor_to_viam(transformed, mime_type)  # return converted viam image
