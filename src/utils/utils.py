import math
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.io import write_jpeg
from torchvision.utils import draw_bounding_boxes, make_grid


def resize_transform(img, size=(256, 128)):
    """
    Resize a torch image to the specified size.
    Used before passing the image to reid model.
    """
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
        ]
    )
    return transform(img)


def compute_centers(boxes, bottom=True, box_projection_centers=None):
    """
    Compute the 2D centers of a torch tensor of bounding boxes.
    """
    if bottom is True and box_projection_centers is not None:
        raise ValueError("Cannot project boxes to bottom and use box_projection_centers simultaneously.")
    centers = torch.zeros((boxes.shape[0], 2))
    centers[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
    if box_projection_centers is not None:
        alpha_w, alpha_h = box_projection_centers
        centers[:, 1] = boxes[:, 1] + alpha_h * boxes[:, 3]
    elif bottom:
        centers[:, 1] = boxes[:, 1] + boxes[:, 3]
    else:
        centers[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
    return centers


def tlwh_to_xyah(tlwh):
    """
    Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = tlwh.clone()
    if ret.dim() == 1:
        ret = ret.unsqueeze(0)
    ret[:, :2] += ret[:, 2:] / 2
    ret[:, 2] /= ret[:, 3]
    return ret


def xyah_to_tlwh(xyah):
    """Get current position in bounding box format `(top left x, top left y,
    width, height)`.
    """
    ret = xyah.clone()
    if ret.dim() == 1:
        ret = ret.unsqueeze(0)
    ret[:, 2] *= ret[:, 3]
    ret[:, :2] -= ret[:, 2:] / 2
    return ret


def tlwh_to_tlbr(tlwh):
    """Convert bounding box to format `(top left x, top left y, bottom right
    x, bottom right y)`.
    """
    ret = tlwh.clone()
    if ret.dim() == 1:
        ret = ret.unsqueeze(0)
    ret[:, 2:] += ret[:, :2]
    return ret


def expand_boxes(in_boxes, factor):
    boxes = in_boxes.clone()
    cx, cy = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
    w, h = boxes[:, 2] * factor, boxes[:, 3] * factor
    boxes[:, 0] = cx - w / 2
    boxes[:, 1] = cy - h / 2
    boxes[:, 2] = w
    boxes[:, 3] = h
    return boxes


def remove_border_boxes(boxes, border):
    xy1x2y2 = tlwh_to_tlbr(boxes)
    keep = (
        (xy1x2y2[:, 0] > border)
        & (xy1x2y2[:, 1] > border)
        & (xy1x2y2[:, 2] < (1920 - border))
        & (xy1x2y2[:, 3] < (1080 - border))
    )
    return keep


def size_filter(boxes, size_min, size_max):
    sizes = boxes[:, 2] * boxes[:, 3]
    keep = (sizes >= size_min) & (sizes <= size_max)
    return keep


def mpl_cmap_to_rgb(cmap_name: str, seed: int = 0) -> List[Tuple[int, int, int]]:
    """Returns a list of RGB values from a matplotlib colormap."""
    cmap = plt.get_cmap(cmap_name)
    colors = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        colors.append(tuple(int(255 * c) for c in rgb))
    random.seed(seed)
    random.shuffle(colors)
    return colors


def render_image_grid(images: List[torch.Tensor], *args, **kwargs) -> torch.Tensor:
    """Renders a grid of images.

    Args:
        images (List[torch.Tensor]): List of N images of shape (C, H, W).
        *args: Additional arguments to pass to the make_grid function.
        **kwargs: Additional keyword arguments to pass to the make_grid function.

    Returns:
        torch.Tensor: Image grid of shape (C, H, W).
    """
    images = torch.stack(images)
    nrow = math.ceil(math.sqrt(len(images)))
    return make_grid(images, nrow=nrow, *args, **kwargs)


def render_images_with_boxes(
    image: torch.Tensor,
    boxes: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    confs: Optional[torch.Tensor] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    *args,
    **kwargs,
) -> List[torch.Tensor]:
    """Render image with bounding boxes. Colors correspond to the label index. Boxes are
    expected to be in MOT-format, i.e., (bb_left, bb_top, bb_widht, bb_height).

    Args:
        images (torch.Tensor): Image of shape (C, H, W).
        boxes (torch.Tensor): Boxes of shape (K, 4).
        labels (torch.Tensor): Label of shape (K,).
        colors (Optional[List[Tuple[int, int, int]]]): List of RGB colors. Defaults to None.
        *args: Additional arguments to pass to the draw_bounding_boxes function.
        **kwargs: Additional keyword arguments to pass to the draw_bounding_boxes function.

    Returns:
        torch.Tensor: Image with bounding boxes.
    """
    if boxes is None:
        return image

    if colors is None:
        colors = mpl_cmap_to_rgb("rainbow")

    if labels is None:
        labels = torch.zeros(boxes.size(0))

    color_palette = [colors[label % len(colors)] for label in labels]

    _labels = [str(label.item()) for i, label in enumerate(labels)]

    if confs is not None:
        _labels = [f"{label} ({conf.item():.2f})" for label, conf in zip(_labels, confs)]

    img = image.clone()
    bxs = boxes.clone()
    bxs[:, 2:] += bxs[:, :2]

    img = draw_bounding_boxes(
        img,
        bxs,
        labels=_labels,
        colors=color_palette,
        *args,
        **kwargs,
    )
    return img


def normalize_features(x):
    # shape of x: (C, N, F)
    # normalize features per channelg
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-8
    return (x - mean) / std


def nanmax(x, dim=None):
    """Function like torch.nanmean for max."""
    mask = torch.isnan(x)
    x_masked = torch.where(mask, torch.tensor(float("-inf")).to(x.device), x)
    max_vals, _ = torch.max(x_masked, dim=dim)

    # Restore NaN values if max is -inf (because all were NaN along dimension)
    max_vals = torch.where(max_vals == float("-inf"), torch.tensor(float("nan")).to(x.device), max_vals)
    return max_vals
