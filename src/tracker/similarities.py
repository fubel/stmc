import torch
from torchvision.ops import box_iou


def cosine_similarity(a, b, eps=1e-8):
    """
    Compute pairwise appearance distance between features.
    from https://stackoverflow.com/a/58144658
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def batch_cosine_similarity(a, b, eps=1e-8):
    """Compute batched pairwise appearance distance between features.

    Args:
        a (torch.Tensor): (B, N, feature_dim) tensor.
        b (torch.Tensor): (B, N, feature_dim) tensor.
        eps (float, optional): Epsilon to prevent division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: (B, N, N) tensor of pairwise similarities.
    """
    # Compute norms along feature dimension and add new dimensions needed for broadcasting
    a_n = a.norm(dim=2)[:, :, None]
    b_n = b.norm(dim=2)[:, :, None]

    # Perform normalization and prevent division by zero.
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

    # Compute similarity matrix using batch matrix multiplication.
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


def batched_box_iou(boxes):
    """Compute batched pairwise IoU between boxes.

    Args:
        boxes (torch.Tensor): (B, N, 4) tensor of boxes.

    Returns:
        torch.Tensor: (B, N, N) tensor of pairwise IoU.
    """
    ious = []
    for sub_boxes in boxes:
        ious.append(box_iou(sub_boxes, sub_boxes))
    return torch.stack(ious)


def bev_distance(bev_positions):
    """Compute distance between positions on ground plane.

    Args:
        bev_positions (torch.Tensor): (N, 2) tensor of positions.

    Returns:
        torch.Tensor: (N, N) tensor of pairwise similarities.
    """
    return torch.norm(bev_positions[:, None] - bev_positions[None, :], dim=2)


def batch_bev_distance(bev_positions):
    """Compute batched distance similarity between positions on ground plane.

    Args:
        bev_positions (torch.Tensor): (B, N, 2) tensor of positions.

    Returns:
        torch.Tensor: (B, N, N) tensor of pairwise similarities.
    """
    # Subtract positions across the batch, adding extra dimensions for broadcasting
    diff = bev_positions[:, :, None] - bev_positions[:, None, :]

    # Compute norm along the last dimension (x and y coordinates)
    norm = torch.norm(diff, dim=-1)

    # Return similarity
    return norm
