"""
Ray samplers for sampling rays used for volume rendering.
"""

import typing

import torch
import src.renderer.cameras as cameras


class RayBundle(object):
    """
    A data structure for holding data, metadata for rays.

    Attributes:
        ray_origin (torch.Tensor): Tensor of shape (N, 3) representing ray origins.
        ray_dir (torch.Tensor): Tensor of shape (N, 3) representing ray directions.
        frame_type (str): A string indicating the type of the frame where ray origin
            and direction lie in.
    """

    def __init__(
        self,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
        t_near: float,
        t_far: float,
        frame_type: str,
    ):
        self._ray_origin = ray_origin
        self._ray_dir = ray_dir
        self._t_near = t_near
        self._t_far = t_far
        self._frame_type = frame_type

    @property
    def ray_origin(self) -> torch.Tensor:
        """Returns an instance of torch.Tensor representing ray origins."""
        return self._ray_origin

    @property
    def ray_dir(self) -> torch.Tensor:
        """Returns an instance of torch.Tensor reprsenting ray directions."""
        return self._ray_dir

    @property
    def t_near(self) -> float:
        """Returns the nearest ray distance."""
        return self._t_near

    @property
    def t_far(self) -> float:
        """Returns the farthest ray distance."""
        return self._t_far

    @property
    def frame_type(self) -> torch.Tensor:
        """Returns a string indicating with respect to which frame ray origins
        and directions are represented."""
        return self._frame_type


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
    ) -> torch.Tensor:
        """
        Computes view direction vectors represented in the camera frame.

        Args:
            pixel_coords (torch.Tensor): Tensor of shape (N, 2).
                A flattened array of pixel coordinates.
            cam_intrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera intrinsic matrix.

        Returns:
            An instance of torch.Tensor of shape (N, 3) containing
            view direction vectors represented in the camera frame.
        """
        # (u, v) -> (x, y)
        pixel_coords = pixel_coords.float()
        pixel_coords[:, 0] = (pixel_coords[:, 0] - cam_intrinsic[0, 2]) / cam_intrinsic[0, 0]
        pixel_coords[:, 1] = (pixel_coords[:, 1] - cam_intrinsic[1, 2]) / cam_intrinsic[1, 1]

        # (x, y) -> (x, y, -1)
        ray_dir = torch.cat(
            [
                pixel_coords,
                -torch.ones(pixel_coords.shape[0], 1),
            ],
            dim=-1,
        )

        return ray_dir

    def _get_ray_origin(
        self,
        z_near: float,
        ray_dir: torch.Tensor,
        translate_to_pixel: bool,
    ) -> torch.Tensor:
        """
        Computes ray origin coordinate in the world frame.

        Args:
            cam_extrinsic (torch.Tensor): Tensor of shape (4, 4).
                A camera extrinsic matrix.

        Returns:
            An instance of torch.Tensor of shape (3,) representing
            the origin of the camera in the camera frame.
        """
        if translate_to_pixel:
            t_near = (-(z_near) / ray_dir[:, 2]).unsqueeze(-1)
            ray_origin = t_near * ray_dir
        else:
            ray_origin = torch.zeros_like(ray_dir)

        return ray_origin

    def generate_rays(
        self,
        pixel_coords: torch.Tensor,
        camera: cameras.CameraBase,
        project_to_ndc: bool,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays by computing:
            (1) the coordinate of rays' origin;
            (2) the directions of rays;

        Args:
            pixel_coords (torch.Tensor): Tensor of shape (N, 2).
                A flattened array of pixel coordinates.

        Returns:
            ray_origin (torch.Tensor): Tensor of shape (N, 3).
                Coordinates of ray origins in the world frame.
            ray_dir (torch.Tensor): Tensor of shape (N, 3).
                Ray direction vectors in the world frame.
        """
        # generate ray direction vectors, origin coordinates in the camera frame.
        ray_dir = self._get_ray_directions(pixel_coords, camera.intrinsic)
        ray_origin = self._get_ray_origin(
            camera.z_near,
            ray_dir,
            translate_to_pixel=True,
        )
        ray_dir /= torch.linalg.vector_norm(
            ray_dir,
            ord=2,
            dim=-1,
            keepdim=True,
        )

        # project rays to NDC if requested
        #  NOTE: Although the supplementary material of the paper explains that
        #  NDC transformation is applied to coordinates & vectors lying in the "camera" frame,
        #  the official implementation applies this to the vectors in the "world" frame.
        if project_to_ndc:
            focal_lengths = camera.focal_lengths
            if focal_lengths[0] != focal_lengths[1]:
                raise ValueError(
                    "Focal length used for computing NDC is ambiguous."
                    f"Two different focal lengths ({focal_lengths[0]}, {focal_lengths[1]}) "
                    "exists but only one can be used."
                )
            ray_origin, ray_dir = self.map_rays_to_ndc(
                focal_lengths[0],
                camera.z_near,
                camera.img_height,
                camera.img_width,
                ray_origin,
                ray_dir,
            )

        # transform the coordinates and vectors into the world frame
        ray_dir = ray_dir @ camera.extrinsic[:3, :3]
        ray_origin = ray_origin + camera.extrinsic[:3, -1]

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

