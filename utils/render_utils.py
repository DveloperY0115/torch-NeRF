"""
renderer.py - Volume rendering utilities for NeRF.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def generate_rays(H: int, W: int, K: int, E: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Generate rays given viewport dimensions, camera intrinsic, and camera extrinsic.

    Args:
    - H: Height of viewport.
    - W: Width of viewport.
    - K: Tensor of shape (4, 4) representing camera intrinsic (i.e., projection matrix).
    - E: Tensor of shape (4, 4) representing eye matrix (i.e, camera-to-world matrix).

    Returns:
    - rays_origin: Tensor of shape (800, 800, 3) representing the origin of rays.
    - rays_direction: Tensor of shape (800, 800, 3) representing the direction of rays.
    """
    y, x = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # for reason behind weird ordering, see https://pytorch.org/docs/stable/generated/torch.meshgrid.html

    # compute ray directions as if the camera is located at origin, looking at -z direction.
    directions = torch.stack(
        [(x - K[0][2]) / K[0][0], -(y - K[1][2]) / K[1][1], -torch.ones_like(x)], dim=-1
    )

    # multiply the inverse of camera-to-world matrix to get the correct ray orientations.
    directions = directions.reshape(-1, 3)  # (W * H, 3)
    rays_direction = torch.mm(torch.inverse(E[:3, :3]), directions.t())  # (3, 3) x (3, H*W)
    rays_direction = F.normalize(rays_direction.t(), dim=-1)
    rays_direction = rays_direction.reshape(W, H, 3)

    # translate the camera position.
    rays_origin = E[:3, -1].expand(rays_direction.shape)

    return rays_origin, rays_direction


def render(H: int, W: int, K: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """
    Render a scene.

    Args:
    - H: Height of viewport.
    - W: Width of viewport.
    - K: Tensor of shape (B, 4, 4) representing camera intrinsic (i.e., projection matrix).
    - E: Tensor of shape (B, 4, 4) representing eye matrix (i.e, camera-to-world matrix).
    """
    rays_origin, rays_direction = generate_rays(H, W, K, E)
