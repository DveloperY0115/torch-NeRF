"""
Camera classes used inside renderer(s).
"""

import typing

import torch


class PerspectiveCamera(object):
    """
    Basic camera class.

    Attributes:
        intrinsic (torch.Tensor | Dict): Camera intrinsic. Can be one of:
            1. Tensor of shape (4, 4) representing an intrinsic matrix.
            2. Dictionary of camera intrinsic parameters whose keys are:
                ['f_x', 'f_y', 'img_width', 'img_height']
        extrinsic (torch.Tensor): Tensor of shape (4, 4) representing an extrinsic matrix.
        t_near (float): The nearest distance rays can reach. (cf. x = o + t_near * d).
        t_far (float): The farthest distance rays can reach. (cf. x = o + t_far * d).
        focal_lengths (tuple): A 2-tuple of floating point numbers representing the
            focal length of the horizontal axis, vertical axis, respectively.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
    """

    def __init__(
        self,
        intrinsic: typing.Union[torch.Tensor, typing.Dict[str, float]],
        extrinsic: torch.Tensor,
        t_near: float,
        t_far: float,
    ):
        """
        Constructor of class 'CameraBase'.

        Args:
            intrinsic (torch.Tensor | Dict): Camera intrinsic. Can be one of:
                1. Tensor of shape (4, 4) representing an intrinsic matrix.
                2. Dictionary of camera intrinsic parameters whose keys are:
                    ['f_x', 'f_y', 'img_width', 'img_height']
            extrinsic (torch.Tensor): Tensor of shape (4, 4) representing an extrinsic matrix.
            t_near (float): The nearest distance rays can reach. (cf. x = o + t_near * d).
            t_far (float): The farthest distance rays can reach. (cf. x = o + t_far * d).
        """
        if not isinstance(intrinsic, (torch.Tensor, dict)):
            raise ValueError(
                "Expected torch.Tensor of Python Dict as a camera intrinsic. "
                f"Got {type(intrinsic)}."
            )

        self._extrinsic = extrinsic
        self._t_near = t_near
        self._t_far = t_far

        if isinstance(intrinsic, torch.Tensor):
            if intrinsic.shape != torch.Size((4, 4)):
                raise ValueError(f"Expected a tensor of shape (4, 4). Got {intrinsic.shape}.")
            self._intrinsic = intrinsic

            self._focal_x = float(intrinsic[0, 0])
            self._focal_y = float(intrinsic[1, 1])
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
            )

            self._focal_x = float(focal_x)
            self._focal_y = float(focal_y)
            self._img_width = int(img_width)
            self._img_height = int(img_height)

    def _construct_intrinsic_from_params(
        self,
        focal_x: float,
        focal_y: float,
        img_width: float,
        img_height: float,
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

        Returns:
            An instance of torch.Tensor of shape (4, 4) representing the intrinsic
            matrix of the camera.
        """
        intrinsic = torch.tensor(
            [
                [focal_x, 0.0, img_width / 2.0, 0.0],
                [0.0, focal_y, img_height / 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],  # dummy row
                [0.0, 0.0, -1.0, 0.0],  # dummy row
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
    def t_near(self) -> float:
        """Returns the nearest ray distance rendered."""
        return self._t_near

    @property
    def t_far(self) -> float:
        """Returns the farthest ray distance rendered."""
        return self._t_far

    @property
    def img_width(self) -> int:
        """Returns the width of the image."""
        return self._img_width

    @property
    def img_height(self) -> int:
        """Returns the height of the image."""
        return self._img_height

    @property
    def focal_lengths(self) -> typing.Tuple[float, float]:
        """Returns the focal lengths of the camera."""
        return (self._focal_x, self._focal_y)

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

    @t_near.setter
    def t_near(
        self,
        new_t_near: float,
    ) -> None:
        if not isinstance(new_t_near, int, float):
            raise ValueError(f"Expected variable of numeric type. Got {type(new_t_near)}.")
        self._t_near = float(new_t_near)

    @t_far.setter
    def t_far(
        self,
        new_t_far: float,
    ) -> None:
        if not isinstance(new_t_far, int, float):
            raise ValueError(f"Expected variable of numeric type. Got {type(new_t_far)}.")
        self._t_far = float(new_t_far)
