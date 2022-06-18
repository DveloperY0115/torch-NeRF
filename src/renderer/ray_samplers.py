"""
Ray samplers for sampling rays used for volume rendering.
"""

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
    ) -> torch.Tensor:
        """
        Compute view direction vectors represented in the camera frame.

        Args:
            pixel_coords (torch.Tensor): Tensor of shape (N, 2).
                A flattened array of pixel coordinates.
            cam_intrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera intrinsic matrix.
            cam_extrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera extrinsic matrix.

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

        return ray_dir

        """
        Sample rays.
        """
        raise NotImplementedError()
