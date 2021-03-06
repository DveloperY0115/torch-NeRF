"""
Volume renderer implemented using Pytorch.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch_nerf.src.scene as scene
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.src.renderer.integrators.quadrature_integrator as integrators
import torch_nerf.src.renderer.ray_samplers as ray_samplers


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
        integrator: integrators.IntegratorBase,
        sampler: ray_samplers.RaySamplerBase,
        camera: Optional[cameras.PerspectiveCamera] = None,
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
        self._integrator = integrator
        self._sampler = sampler
        self._camera = camera

        # precompute pixel coordinates
        if self._camera:
            self._screen_coords = self._generate_screen_coords()
        else:
            self._screen_coords = None
            print(
                "Warning: Camera parameters are not initialized."
            )  # TODO: replace this with logger

    def render_scene(
        self,
        target_scene: scene.PrimitiveBase,
        num_pixels: int,
        num_samples: Union[int, Tuple[int, int]],
        project_to_ndc: bool,
        device: int,
        pixel_indices: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        num_ray_batch: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Renders the scene by querying underlying 3D inductive bias.

        Args:
            target_scene (QueryStruct): An instance of class derived from 'QueryStructBase'.
            num_pixels (int): Number of pixels to render.
                If smaller than the total number of pixels in the current resolution,
                sample pixels randomly.
            num_samples (int | Tuple[int, int]): Number of samples drawn along each ray.
                (1) a single integer: the number of coarse samples.
                (2) a tuple of integers: the number of coarse and fine samples, respectively.
            project_to_ndc (bool): A flag to toggle NDC projection.
            device (int): The index to the CUDA device.
            pixel_indices (torch.Tensor): An instance of torch.Tensor of shape (num_pixels,).
                The indices of pixels from where rays will be casted.
            weights (torch.Tensor): An instance of torch.Tensor of shape (num_pixels, num_samples[0]).
                The weights obtained by evaluating 'coarse' network representing a scene. This
                weight is normalized and used during inverse-CDF sampling. Please refer to
                Section 5.2 - Hierarchical sampling of 'NeRF: Representing Scenes as Neural
                Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'
                for details.
            num_ray_batch (int): The number of ray batches.
                The entire set of rays are divided into 'num_ray_batch'-batches and processed
                individually. Ray batching is necessary to avoid out-of-memory.

        Returns:
            pixel_rgb (torch.Tensor): An instance of torch.Tensor of shape (num_pixels, 3).
                The final pixel colors of rendered image lying in RGB color space.
            pixel_to_render (torch.Tensor): An instance of torch.Tensor of shape (num_pixels, ).
                The array holding index of pixels rendered.
            weights (torch.Tensor): An instance of torch.Tensor of shape (num_pixels, num_samples).
                Weight of each sample point along rays.
        """
        if not isinstance(num_pixels, int):
            raise ValueError(f"Expected variable of type int. Got {type(num_pixels)}.")
        if isinstance(num_samples, (tuple, list)):
            if len(num_samples) != 2:
                raise ValueError(
                    "Expected a tuple of length 2 for num_samples of type tuple. "
                    f"Got a tuple of length {len(num_samples)}."
                )
            if pixel_indices is None:
                raise ValueError(
                    "Expected a predefined set of pixels to render in hierarchical sampling. "
                    "Pixel indices are not provided."
                )

        # sample pixels to render
        if not pixel_indices is None:
            pixel_to_render = pixel_indices
        else:
            if num_pixels < self.camera.img_height * self.camera.img_width:
                pixel_to_render = torch.tensor(
                    np.random.choice(
                        self.camera.img_height * self.camera.img_width,
                        size=[num_pixels],
                        replace=False,
                    ),
                )
            else:  # render the entire image
                pixel_to_render = torch.arange(
                    0,
                    self.camera.img_height * self.camera.img_width,
                )

        # generate sample points along rays
        coords = self.screen_coords.clone()
        coords = coords[pixel_to_render, :]
        ray_bundle = self.sampler.generate_rays(
            coords,
            self.camera,
            project_to_ndc=project_to_ndc,
        )

        # =====================================================================
        # Memory-bandwidth intensive operations (must be done directly on GPUs)
        # =====================================================================

        # sample points along rays
        sample_pts, ray_dir, delta_t = self.sampler.sample_along_rays(
            ray_bundle,
            num_samples,
            device=device,
            weights=weights,
        )

        # render rays
        pixel_rgb, weights, sigma, radiance = self._render_ray_batches(
            target_scene,
            sample_pts,
            ray_dir,
            delta_t,
            num_batch=1 if num_ray_batch is None else num_ray_batch,
        )

        # =====================================================================
        # Memory-bandwidth intensive operations (must be done directly on GPUs)
        # =====================================================================

        return pixel_rgb, pixel_to_render, weights

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

    def _render_ray_batches(
        self,
        target_scene,
        sample_pts: torch.Tensor,
        ray_dir: torch.Tensor,
        delta_t: torch.Tensor,
        num_batch: int,
    ) -> torch.Tensor:
        """
        Renders an image by dividing its pixels into small batches.

        Args:
            target_scene (QueryStruct): An instance of class derived from 'QueryStructBase'.
            sample_pts (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                3D-coordinate of sample points sampled along rays. Here, N is the number of rays
                in a ray bundle and S is the number of sample points along each ray.
            ray_dir (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                3D-vectors of viewing (ray) directions.
            delta_t (torch.Tensor): An instance of torch.Tensor of shape (N, S) representing the
                difference between adjacent t's.
            num_batch (int): Number of ray batches to be processed.

        Returns:
            pixel_rgb (torch.Tensor): An instance of torch.Tensor of shape (N, 3).
                The pixel intensities determined by integrating radiance values along each ray.
            weights (torch.Tensor): An instance of torch.Tensor of shape (N, S).
                The weight obtained by multiplying transmittance and alpha during numerical integration.
            sigma (torch.Tensor): An instance of torch.Tensor of shape (N, S).
                Estimated densities at sample points.
            radiance (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                Estimated radiance values at sample points.
        """
        rgb = []
        weights = []
        sigma = []
        radiance = []

        partitions = torch.linspace(
            0,
            sample_pts.shape[0],
            num_batch + 1,
            dtype=torch.long,
        )
        partitions[-1] = sample_pts.shape[0]  # accomodate numerical issue

        for start, end in zip(partitions[0::1], partitions[1::1]):
            pts_batch = sample_pts[start:end, ...]
            dir_batch = ray_dir[start:end, ...]
            delta_batch = delta_t[start:end, ...]

            # query the scene to get density and radiance
            sigma_batch, radiance_batch = target_scene.query_points(pts_batch, dir_batch)

            # compute pixel colors by evaluating the volume rendering equation
            rgb_batch, weights_batch = self.integrator.integrate_along_rays(
                sigma_batch, radiance_batch, delta_batch
            )

            # collect rendering outputs
            rgb.append(rgb_batch)
            weights.append(weights_batch)
            sigma.append(sigma_batch)
            radiance.append(radiance_batch)

        pixel_rgb = torch.cat(rgb, dim=0)
        weights = torch.cat(weights, dim=0)
        sigma = torch.cat(sigma, dim=0)
        radiance = torch.cat(radiance, dim=0)

        return pixel_rgb, weights, sigma, radiance

    @property
    def camera(self) -> cameras.PerspectiveCamera:
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
        assert (
            not self._screen_coords is None
        ), "Screen coordinates must not be None at rendering time."
        return self._screen_coords

    @camera.setter
    def camera(self, new_camera: cameras.PerspectiveCamera) -> None:
        self._camera = new_camera
        self._screen_coords = self._generate_screen_coords()  # update screen coordinate
