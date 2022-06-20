"""
Volume renderer implemented using Pytorch.
"""

import random

import torch
import src.query_struct as query_struct
import src.renderer.cameras as cameras
import src.renderer.integrators as integrators
import src.renderer.ray_samplers as ray_samplers


class VolumeRenderer(object):
    """
    Volume renderer.

    Attributes:
        camera (Camera): An instance of class derived from 'CameraBase'.
            Defines camera intrinsic / extrinsics.
        integrator (Integrator): An instance of class derived from 'IntegratorBase'.
            Computes numerical integrations to determine pixel colors in a differentiable manner.
        sampler (RaySampler): An instance of class derived from 'RaySamplerBase'.
            Samples the points in 3D space to evaluate neural scene representations.
    """

    def __init__(
        self,
        camera: cameras.CameraBase,
        integrator: integrators.IntegratorBase,
        sampler: ray_samplers.RaySamplerBase,
    ):
        """
        Constructor of class 'VolumeRenderer'.

        Args:
            camera (Camera): An instance of class derived from 'CameraBase'.
                Defines camera intrinsic / extrinsics.
            integrator (Integrator): An instance of class derived from 'IntegratorBase'.
                Computes numerical integrations to determine pixel colors in differentiable manner.
            sampler (RaySampler): An instance of class derived from 'RaySamplerBase'.
                Samples the points in 3D space to evaluate neural scene representations.
        """
        # initialize fundamental components
        self._camera = camera
        self._integrator = integrator
        self._sampler = sampler

        # precompute pixel coordinates
        self._screen_coords = self._generate_screen_coords()

    def render_scene(
        self,
        scene: query_struct.QueryStructBase,
        num_pixels: int,
        num_samples: int,
        project_to_ndc: bool,
    ):
        """
        Renders the scene by querying underlying 3D inductive bias.

        Args:
            scene (QueryStruct): An instance of class derived from 'QueryStructBase'.
            num_samples (int): Number of samples drawn along each ray.
            num_pixels (int): Number of pixels to render.
                If smaller than the total number of pixels in the current resolution,
                sample pixels randomly.
            project_to_ndc (bool):

        Returns:
            pixel_rgb (torch.Tensor): An instance of torch.Tensor of shape (num_pixels, 3).
                The final pixel colors of rendered image lying in RGB color space.
        """
        if not isinstance(num_pixels, int):
            raise ValueError(f"Expected variable of type int. Got {type(num_pixels)}.")

        # sample pixels to render
        if num_pixels < self.camera.img_height * self.camera.img_width:
            pixel_to_render = torch.tensor(
                random.sample(
                    list(range(self.camera.img_height * self.camera.img_width)),
                    num_pixels,
                )
            )
        else:
            pixel_to_render = torch.arange(
                0, self.camera.img_height * self.camera.img_width
            )  # render the entire image

        # generate sample points along rays
        coords = self.screen_coords.clone()
        coords = coords[pixel_to_render, :]
        ray_bundle = self.sampler.generate_rays(
            coords,
            self.camera,
            project_to_ndc=project_to_ndc,
        )

        # sample points along rays
        sample_pts, ray_dir, delta = self.sampler.sample_along_rays(
            ray_bundle,
            num_samples,
        )

        # query the scene to get density and radiance
        sigma, radiance = scene.query_points(sample_pts, ray_dir)

        # compute pixel colors by evaluating the volume rendering equation
        pixel_rgb = self.integrator.integrate_along_rays(sigma, radiance, delta)

        return pixel_rgb

    def _generate_screen_coords(self) -> torch.Tensor:
        """
        Generates screen space coordinates.

        Returns:
            An instance of torch.Tensor of shape (N, 2) containing
            pixel coordinates of image whose resolution is (self.img_height, self.img_width).
        """
        ys = torch.arange(0, self.camera.img_height)
        xs = torch.arange(0, self.camera.img_width)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y = (self.camera.img_height - 1) - grid_y  # [0, H-1] -> [H-1, 0]

        coords = torch.cat(
            [grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)],
            dim=-1,
        ).reshape(self.camera.img_height * self.camera.img_width, -1)

        return coords

    @property
    def camera(self) -> cameras.CameraBase:
        """Returns the current camera configuration."""
        return self._camera

    @property
    def integrator(self) -> integrators.IntegratorBase:
        """Returns the current integrator in-use."""
        return self._integrator

    @property
    def sampler(self) -> ray_samplers.RaySamplerBase:
        """Returns the current ray sampler in-use."""
        return self._sampler

    @property
    def screen_coords(self) -> torch.Tensor:
        """Returns the tensor of screen space coordinates."""
        return self._screen_coords
