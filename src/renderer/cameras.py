"""
Camera classes used inside renderer(s).
"""

import typing

import torch


class CameraBase(object):
    """
    Basic camera class.

    Attributes:
        intrinsic (torch.Tensor | Dict): Camera intrinsic. Can be one of:
            1. Tensor of shape (4, 4) representing an intrinsic matrix.
            2. Dictionary of camera intrinsic parameters whose keys are:
                ['f_x', 'f_y', 'img_width', 'img_height']
        extrinsic (torch.Tensor): Tensor of shape (4, 4) representing an extrinsic matrix.
        z_near (float): A floating point number representing the nearest depth rendered.
        z_far (float): A floating point number representing the farthest depth rendered.
    """

    def __init__(
        self,
        intrinsic: typing.Union[torch.Tensor, typing.Dict[str, float]],
        extrinsic: torch.Tensor,
        z_near: float,
        z_far: float,
    ):
        """
        Constructor of class 'CameraBase'.

        Args:
            intrinsic (torch.Tensor | Dict): Camera intrinsic. Can be one of:
                1. Tensor of shape (4, 4) representing an intrinsic matrix.
                2. Dictionary of camera intrinsic parameters whose keys are:
                    ['f_x', 'f_y', 'img_width', 'img_height']
            extrinsic (torch.Tensor): Tensor of shape (4, 4) representing an extrinsic matrix.
            z_near (float): A floating point number representing the nearest depth rendered.
            z_far (float): A floating point number representing the farthest depth rendered.
        """
        if not isinstance(intrinsic, torch.Tensor, dict):
            raise ValueError(
                "Expected torch.Tensor of Python Dict as a camera intrinsic. "
                f"Got {type(intrinsic)}."
            )

        self._extrinsic = extrinsic
        self._z_near = z_near
        self._z_far = z_far

        if isinstance(intrinsic, torch.Tensor):
            if intrinsic.shape != torch.Size((4, 4)):
                raise ValueError(f"Expected a tensor of shape (4, 4). Got {intrinsic.shape}.")
            self._intrinsic = intrinsic

            self._focal_x = intrinsic[0, 0]
            self._focal_y = intrinsic[1, 1]
            self._img_width = int(2 * intrinsic[0, 2])
            self._img_height = int(2 * intrinsic[1, 2])
        else:
            # construct camera intrinsic matrix on the fly
            focal_x = float(intrinsic["f_x"])
            focal_y = float(intrinsic["f_y"])
            img_width = float(intrinsic["img_width"])
            img_height = float(intrinsic["img_height"])
            self._intrinsic = self._construct_intrinsic_from_params(
                focal_x,
                focal_y,
                img_width,
                img_height,
                z_near,
                z_far,
            )

            self._focal_x = focal_x
            self._focal_y = focal_y
            self._img_width = img_width
            self._img_height = img_height

    def _construct_intrinsic_from_params(
        self,
        focal_x: float,
        focal_y: float,
        img_width: float,
        img_height: float,
        z_near: float,
        z_far: float,
    ) -> torch.Tensor:
        """
        Constructs the camera intrinsic matrix from given camera parameters.

        Note that the third and fourth row of the resulting matrix is NOT
        significant in the neural rendering pipeline. The matrix is only used
        for computing the ray origin and directions. We derive the matrix for
        notational uniformity.

        Args:
            focal_x (float): Focal length of the camera along the horizontal axis.
            focal_y (float): Focal length of the camera along the vertical axis.
            img_width (float): Width of the image.
            img_height (float): Height of the image.
            z_near (float): A floating point number representing the nearest depth rendered.
            z_far (float): A floating point number representing the farthest depth rendered.

        Returns:
            An instance of torch.Tensor of shape (4, 4) representing the intrinsic
            matrix of the camera.
        """
        intrinsic = torch.tensor(
            [
                [focal_x, 0.0, img_width / 2.0, 0.0],
                [0.0, focal_y, img_height / 2.0, 0.0],
                [
                    0.0,
                    0.0,
                    -(z_near + z_far) / (z_far - z_near),
                    -2 * z_near * z_far / (z_far - z_near),
                ],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return intrinsic

    @property
    def intrinsic(self) -> torch.Tensor:
        """Returns the intrinsic matrix of the camera."""
        return self._intrinsic

    @property
    def extrinsic(self) -> torch.Tensor:
        """Returns the extrinsic matrix of the camera."""
        return self._extrinsic

    @property
    def z_near(self) -> float:
        """Returns the nearest depth rendered."""
        return self._z_near

    @property
    def z_far(self) -> float:
        """Returns the farthest depth rendered."""
        return self._z_far

    @property
    def focal_lengths(self) -> typing.Tuple[float, float]:
        """Returns the focal lengths of the camera."""
        return self._focal_lengths

    @intrinsic.setter
    def intrinsic(
        self,
        new_intrinsic: torch.Tensor,
    ) -> None:
        if not isinstance(new_intrinsic, torch.Tensor):
            raise ValueError(f"Expected variable of type torch.Tensor. Got {type(new_intrinsic)}.")
        if new_intrinsic.shape != torch.Size((4, 4)):
            raise ValueError(f"Expected tensor of shape (4, 4). Got {new_intrinsic.shape}.")
        self._intrinsic = new_intrinsic

    @extrinsic.setter
    def extrinsic(
        self,
        new_extrinsic: torch.Tensor,
    ) -> None:
        if not isinstance(new_extrinsic, torch.Tensor):
            raise ValueError(f"Expected variable of type torch.Tensor. Got {type(new_extrinsic)}.")
        if new_extrinsic.shape != torch.Size((4, 4)):
            raise ValueError(f"Expected tensor of shape (4, 4). Got {new_extrinsic.shape}.")
        self._extrinsic = new_extrinsic

    @z_near.setter
    def z_near(
        self,
        new_z_near: float,
    ) -> None:
        if not isinstance(new_z_near, int, float):
            raise ValueError(f"Expected variable of numeric type. Got {type(new_z_near)}.")
        self._z_near = float(new_z_near)

    @z_far.setter
    def z_far(
        self,
        new_z_far: float,
    ) -> None:
        if not isinstance(new_z_far, int, float):
            raise ValueError(f"Expected variable of numeric type. Got {type(new_z_far)}.")
        self._z_far = float(new_z_far)

    @focal_lengths.setter
    def focal_lengths(
        self,
        new_focal_lengths: typing.Tuple[float, float],
    ) -> None:
        if not isinstance(new_focal_lengths, tuple, list):
            raise ValueError(
                f"Expected a tuple or list as a parameter. Got {type(new_focal_lengths)}."
            )
        if not isinstance(new_focal_lengths[0], float):
            raise ValueError(
                f"Expected variable of numeric type. Got {type(new_focal_lengths[0])}."
            )
        self._focal_lengths = (float(f) for f in new_focal_lengths)
