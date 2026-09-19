"""Image preprocessing utilities.

Provide image format conversion and geometric preprocessing for the client/server of
websocket remote inference:
- ``convert_to_uint8``: convert float images to uint8 (shrinks the payload before
  network transfer, 4 bytes/pixel -> 1 byte/pixel);
- ``resize_with_pad`` / ``_resize_with_pad_pil``: aspect-preserving resize + zero
  padding to a target size (replicates tf.image.resize_with_pad, avoiding distortion
  from non-uniform stretching).
"""
import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.

    Notes (added): if the input is a floating-point image (assumed to be in [0,1]),
    it is multiplied by 255 and cast to uint8; integer inputs are returned unchanged.
    Used by clients to shrink observation payloads before sending (4 bytes/pixel ->
    1 byte/pixel).

    Args:
        img (np.ndarray): input image of any shape (usually [..., H, W, C] or [..., C, H, W]).

    Returns:
        np.ndarray: uint8 image; non-float inputs are returned as-is.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray,
                    height: int,
                    width: int,
                    method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].

    Notes (added): PIL-based "aspect-preserving resize + zero padding" for a batch of
    images (replicates tf.image.resize_with_pad): first scale by the long-side ratio so
    the image fits into (height, width), then paste it centered onto an all-zero
    background, so the picture is never stretched. Supports arbitrary leading batch
    dims (flattened to [-1, H, W, C] internally, processed per image, then reshaped back).
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([
        _resize_with_pad_pil(Image.fromarray(im), height, width, method=method)
        for im in images
    ])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int,
                         method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].

    Notes (added): single-image version of resize_with_pad (internal helper). Steps:
    1) scale by ratio = max(cur_w/w, cur_h/h) so the result just fits into the target box;
    2) create a (width, height) all-zero background and paste the scaled image centered on it.

    Args:
        image (Image.Image): a single PIL image.
        height (int): target height in pixels.
        width (int): target width in pixels.
        method (int): PIL interpolation mode (e.g. Image.BILINEAR).

    Returns:
        Image.Image: padded image of size (width, height); returned unchanged if the
        size already matches.
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height),
                                 resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
