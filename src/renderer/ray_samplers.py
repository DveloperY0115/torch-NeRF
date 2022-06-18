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
                Ray direction vectors in the world frame.
        """
        ray_dir = self._get_ray_directions(pixel_coords, cam_intrinsic, cam_extrinsic)
        ray_origin = self._get_ray_origin(cam_extrinsic).expand(ray_dir.shape)

        return ray_origin, ray_dir

    def map_rays_to_ndc(
        self,
        focal_length: float,
        z_near: float,
        img_height: int,
        img_width: int,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects ray origin, directions in the world frame to NDC.

        For details regarding mathematical derivation of the projection,
        please refer to the supplementary material of 'NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper
        honorable mention)'.

        Args:
            focal_length (float): A focal length of the camera.
            z_near (float): The nearest depth of the view frustum.
            img_height (int): The height of the rendered image.
            img_width (int): The width of the rendered image.
            ray_origin (torch.Tensor): Tensor of shape (N, 3).
                Coordinates of ray origins in the world frame.
            ray_dir (torch.Tensor): Tensor of shape (N, 3).
                Ray direction vectors in the world frame.

        Returns:
            projected_origin (torch.Tensor): Tensor of shape (N, 3).
                Coordinates of ray origins in NDC.
            projected_dir (torch.Tensor): Tensor of shape (N, 3).
                Ray direction vectors in NDC.
        """
        if z_near <= 0:
            raise ValueError(f"Expected a positive real number. Got {z_near}.")

        # shift ray origins to the near plane at z = -z_near
        t_near = -(z_near + ray_origin[:, 2]) / ray_dir[:, 2]
        ray_origin = ray_origin + t_near * ray_dir

        # project the ray origin
        origin_x = -(2 * focal_length / img_width) * (ray_origin[:, 0] / ray_origin[:, 2])
        origin_y = -(2 * focal_length / img_height) * (ray_origin[:, 1] / ray_origin[:, 2])
        origin_z = 1 + (2 * z_near / ray_origin[:, 2])
        projected_origin = torch.stack(
            [origin_x, origin_y, origin_z],
            dim=-1,
        )

        # project the ray directions
        dir_x = -(2 * focal_length / img_width) * (
            (ray_dir[:, 0] / ray_dir[:, 2]) - (ray_origin[:, 0] / ray_origin[:, 2])
        )
        dir_y = -(2 * focal_length / img_height) * (
            (ray_dir[:, 1] / ray_dir[:, 2]) - (ray_origin[:, 1] / ray_origin[:, 2])
        )
        dir_z = -(2 * z_near / ray_origin[:, 2])
        projected_dir = torch.stack(
            [dir_x, dir_y, dir_z],
            dim=-1,
        )

        return projected_origin, projected_dir

