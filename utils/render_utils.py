"""
renderer.py - Volume rendering utilities for NeRF.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def render(W: int, H: int, K: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """
    Render a scene.

    Args:
    - W: Width of viewport.
    - H: Height of viewport.
    - K: Tensor of shape (B, 4, 4) representing camera intrinsic (i.e., projection matrix).
    - E: Tensor of shape (B, 4, 4) representing eye matrix (i.e, camera-to-world matrix).

    Returns:
    - img: Tensor of shape (B, W, H, C) representing batch of images
    """
    rays_orig, rays_dir = generate_rays(W, H, K, E)


def generate_rays(W: int, H: int, K: int, E: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    TODO: Batchfy this function!

    Generate rays given viewport dimensions, camera intrinsic, and camera extrinsic.

    Args:
    - W: Width of viewport.
    - H: Height of viewport.
    - K: Tensor of shape (B, 4, 4) representing camera intrinsic (i.e., projection matrix).
    - E: Tensor of shape (B, 4, 4) representing eye matrix (i.e, camera-to-world matrix).

    Returns:
    - rays_orig: Tensor of shape (B, W, H, 3) representing the origin of rays.
    - rays_dir: Tensor of shape (B, W, H, 3) representing the direction of rays.
    """
    B = K.shape[0]

    y, x = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # for reason behind weird ordering, see https://pytorch.org/docs/stable/generated/torch.meshgrid.html

    # batchfy x, y
    x = x.unsqueeze(0).repeat((B, 1, 1))
    y = y.unsqueeze(0).repeat((B, 1, 1))

    # batchfy camera parameters
    viewport_W = K[:, 0, 2].unsqueeze(1).unsqueeze(2).repeat(1, W, H)
    viewport_H = K[:, 1, 2].unsqueeze(1).unsqueeze(2).repeat(1, W, H)
    focal_x = K[:, 0, 0].unsqueeze(1).unsqueeze(2).repeat(1, W, H)
    focal_y = K[:, 1, 1].unsqueeze(1).unsqueeze(2).repeat(1, W, H)

    # compute ray directions as if the camera is located at origin, looking at -z direction.
    directions = torch.stack(
        [
            torch.div(
                x - viewport_W,
                focal_x,
            ),
            -torch.div(
                y - viewport_H,
                focal_y,
            ),
            -torch.ones_like(x),
        ],
        dim=-1,
    )

    # multiply the inverse of camera-to-world matrix to get the correct ray orientations.
    directions = directions.reshape(-1, W * H, 3)  # (B, W * H, 3)
    rays_dir = torch.bmm(
        torch.inverse(E[:, :3, :3]), directions.transpose(1, 2)
    )  # (3, 3) x (3, H*W)
    rays_dir = F.normalize(rays_dir.transpose(1, 2), dim=-1)
    rays_dir = rays_dir.reshape(B, W, H, 3)

    # translate the camera position.
    rays_orig = E[:, :3, -1].unsqueeze(1).unsqueeze(2).repeat(1, W, H, 1)

    assert rays_dir.shape == rays_orig.shape, "[!] The ray origin-direction pair must match"

    return rays_orig, rays_dir


def sample_points_along_rays(
    rays_orig: torch.Tensor, rays_dir: torch.Tensor, num_samples: int, near: float, far: float
) -> torch.Tensor:
    """
    Sample coordinates where the radiance field intensities are to be evaluated.

    Args:
    - rays_orig: Tensor of shape (W, H, 3) representing the origin of rays.
    - rays_dir: Tensor of shape (W, H, 3) representing the direction of rays.

    Returns:
    -
    """
    pass
