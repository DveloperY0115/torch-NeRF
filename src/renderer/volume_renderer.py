"""
Volume renderer implemented using Pytorch.
"""

import random
import typing

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
        img_res: typing.Tuple[int, int],
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
            img_res (Tuple): 2-Tuple containing height and width of rendered images.
        """
        # initialize fundamental components
        self._camera = camera
        self._integrator = integrator
        self._sampler = sampler

        # configure output image resolution
        self._img_res = img_res
        self._img_height = img_res[0]
        self._img_width = img_res[1]
        self._screen_coords = self._generate_screen_coords()

    def render_scene(
        self,
        scene: query_struct.QueryStructBase,
        num_pixels: int,
    ):
        """
        Renders the scene by querying underlying 3D inductive bias.

        Args:
            scene (QueryStruct): An instance of class derived from 'QueryStructBase'.
            num_pixels (int): Number of pixels to render.
                If smaller than the total number of pixels in the current resolution,
                sample pixels randomly.
        """
        if not isinstance(num_pixels, int):
            raise ValueError(f"Expected variable of type int. Got {type(num_pixels)}.")

        # create NDCs
        coords = self.screen_coords.clone()
        coords = self._convert_screen_to_ndc(coords)

        # sample pixels to render
        if num_pixels < self.img_height * self.img_width:
            coord_indices = torch.tensor(
                random.sample(
                    list(range(self.img_height * self.img_width)),
                    num_pixels,
                )
            )
        else:
            coord_indices = torch.arange(
                0, self.img_height * self.img_width
            )  # render the entire image

        # TODO: sample rays

    def _generate_screen_coords(self) -> torch.Tensor:
        """
        Generates screen space coordinates.

        Returns:
            An instance of torch.Tensor of shape (N, 2) containing
            pixel coordinates of image whose resolution is (self.img_height, self.img_width).
        """
        ys = torch.arange(0, self.img_height)
        xs = torch.arange(0, self.img_width)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y = (self.img_height - 1) - grid_y  # [0, H-1] -> [H-1, 0]

        coords = torch.cat(
            [grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)],
            dim=-1,
        ).reshape(self.img_height * self.img_width, -1)

        return coords

    def _convert_screen_to_ndc(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts pixel coordinates to normalized device coordinate (NDC).

        Args:
            coords (torch.Tensor): Tensor of shape (N, 2).
                A flattened array of pixel coordinates.

        Returns:
            An instance of torch.Tensor of shape (N, 2) containing
            normalized device coordinates (NDCs).
        """
        coords = coords.float()
        coords[:, 0] /= self.img_width - 1
        coords[:, 1] /= self.img_height - 1
        coords = (coords - 0.5) * 2.0

        return coords

    def _convert_ndc_to_screen(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts normalized device coordinates (NDCs) to pixel coordinates.

        Args:
            coords (torch.Tensor): Tensor of shape (N, 2).
                A flattened array of normalized device coordinates.

        Returns:
            An instance of torch.Tensor of shape (N, 2) containing
            pixel coordinates.
        """
        coords = 0.5 * coords + 0.5
        coords[:, 0] *= self.img_width - 1
        coords[:, 1] *= self.img_height - 1
        coords = coords.long()

        return coords

    # =============================================
    # getters
    # =============================================
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

    @property
    def img_res(self) -> typing.Tuple[int, int]:
        """Returns the current image resolution setting."""
        return self._img_res

    @property
    def img_height(self) -> int:
        """Returns the current height of rendered images."""
        return self._img_height

    @property
    def img_width(self) -> int:
        """Returns the current width of rendered images."""
        return self._img_width

    # =============================================
    # setters
    # =============================================
    @img_res.setter
    def img_res(
        self,
        new_res: typing.Union[typing.Tuple, typing.List],
    ) -> None:
        if not isinstance(new_res, tuple, list):
            raise ValueError(f"Expected tuple. Got {type(new_res)}.")
        if len(new_res) != 2:
            raise ValueError(f"Expected tuple of length 2. Got {len(new_res)}-tuple.")
        if not isinstance(new_res[0], int):
            raise ValueError(f"Expected tuple of integers. Got tuple of {type(new_res[0])}(s).")
        self._img_res = new_res
        self._screen_coords = self._generate_screen_coords()  # update screen space coordinates

    @img_height.setter
    def img_height(
        self,
        new_height: int,
    ) -> None:
        if not isinstance(new_height, int):
            raise ValueError(f"Expected integer as argument. Got {type(new_height)}.")
        self._img_height = new_height
        self._screen_coords = self._generate_screen_coords()  # update screen space coordinates

    @img_width.setter
    def img_width(
        self,
        new_width: int,
    ) -> None:
        if not isinstance(new_width, int):
            raise ValueError(f"Expected integer as argument. Got {type(new_width)}.")
        self._img_width = new_width
        self._screen_coords = self._generate_screen_coords()  # update screen space coordinates
