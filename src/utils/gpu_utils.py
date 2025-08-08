import torch
from viam.media.video import ViamImage, CameraMimeType
from viam.logging import getLogger
import torchvision.transforms as T
from PIL import Image
import io

# from torchvision.io import decode_jpeg, decode_png, encode_jpeg, encode_png
from viam.media.utils.pil import pil_to_viam_image

logger = getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_viam(tensor: torch.Tensor, mime_type: str) -> ViamImage:
    # implementation with torchvision.io
    # if mime_type == CameraMimeType.JPEG:
    #     encoded_tensor = encode_jpeg(tensor)  # can do on gpu
    # elif mime_type == CameraMimeType.PNG:
    #     # move tensor to cpu to use encode_png (encode only supports cpu tensors)
    #     encoded_tensor = encode_png(tensor.cpu())
    # encoded_bytes = encoded_tensor.cpu().numpy().tobytes()
    # viam_image = ViamImage(data=encoded_bytes, mime_type=mime_type)

    # convert tensor to PIL first
    pil_image = T.ToPILImage()(tensor)
    viam_image = pil_to_viam_image(
        pil_image, mime_type=CameraMimeType.JPEG
    )  # TODO: add mime type to avoid always using jpeg
    return viam_image


def viam_to_tensor(viam_image: ViamImage, device: torch.device = None) -> torch.Tensor:
    # implementation with torchvision.io
    # data_tensor = torch.frombuffer(viam_image.data, dtype=torch.uint8)
    # if viam_image.mime_type == CameraMimeType.JPEG:
    #     tensor = decode_jpeg(data_tensor, device=device)  # can do directly on gpu
    # elif viam_image.mime_type == CameraMimeType.PNG:
    #     tensor = decode_png(data_tensor).to(device)  # move to gpu after
    # else:
    #     raise ValueError(f"Unsupported mime type: {viam_image.mime_type}")
    # return tensor

    pil_image = Image.open(io.BytesIO(viam_image.data))
    return T.ToTensor()(pil_image)
