"""
Ray samplers for sampling rays used for volume rendering.
"""

import typing

import torch


class RaySamplerBase(object):
    """
    Base class for ray samplers.
    """

    def __init__(self):
        pass

    def _get_ray_directions(
        self,
        pixel_coords: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        cam_extrinsic: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Computes view direction vectors represented in the world frame.

        Args:
            pixel_coords (torch.Tensor): Tensor of shape (N, 2).
                A flattened array of pixel coordinates.
            cam_intrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera intrinsic matrix.
            cam_extrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera extrinsic matrix.
            normalize (bool). A flag for normalizing resulting vectors.
                If true, normalize ray direction vectors to make them unit vectors.

        Returns:
            An instance of torch.Tensor of shape (N, 2) containing
            view direction vectors represented in the world frame.
        """
        # (u, v) -> (x, y)
        pixel_coords = pixel_coords.float()
        pixel_coords[:, 0] = (pixel_coords[:, 0] - cam_intrinsic[0, 2]) / cam_intrinsic[0, 0]
        pixel_coords[:, 1] = (pixel_coords[:, 1] - cam_intrinsic[1, 2]) / cam_intrinsic[1, 1]

        # (x, y) -> (x, y, -1)
        ray_dir = torch.cat([pixel_coords, -torch.ones(pixel_coords.shape[0], 1)], dim=-1)

        # rotate vectors in (local) camera frame to the (global) world frame
        ray_dir = ray_dir @ cam_extrinsic[:3, :3]

        if normalize:
            ray_dir /= torch.linalg.vector_norm(
                ray_dir,
                ord=2,
                dim=-1,
                keepdim=True,
            )

        return ray_dir

    def _get_ray_origin(
        self,
        cam_extrinsic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes ray origin coordinate in the world frame.

        Args:
            cam_extrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera extrinsic matrix.

        Returns:
            An instance of torch.Tensor of shape (3,) representing
            the origin of the camera in the world frame.
        """
        return cam_extrinsic[:3, -1]

    def generate_rays(
        self,
        pixel_coords: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        cam_extrinsic: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays by computing:
            (1) the coordinate of rays' origin;
            (2) the directions of rays;

        Args:
            pixel_coords (torch.Tensor): Tensor of shape (N, 2).
                A flattened array of pixel coordinates.
            cam_intrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera intrinsic matrix.
            cam_extrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera extrinsic matrix.

        Returns:
            ray_origin (torch.Tensor): Tensor of shape (N, 3).
                Coordinates of ray origins in the world frame.
            ray_dir (torch.Tensor): Tensor of shape (N, 3).

        """
        ray_dir = self._get_ray_directions(pixel_coords, cam_intrinsic, cam_extrinsic)
        ray_origin = self._get_ray_origin(cam_extrinsic).expand(ray_dir.shape)

        return ray_origin, ray_dir

