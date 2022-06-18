"""
Camera classes used inside renderer(s).
"""

import torch


class CameraBase(object):
    """
    Basic camera class.

    Attributes:
        intrinsic (torch.Tensor): Tensor of shape (4, 4) representing an intrinsic matrix.
        extrinsic (torch.Tensor): Tensor of shape (4, 4) representing an extrinsic matrix.
        z_near (float): A floating point number representing the nearest depth rendered.
        z_far (float): A floating point number representing the farthest depth rendered.
    """

    def __init__(
        self,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        z_near: float,
        z_far: float,
    ):
        """
        Constructor of class 'CameraBase'.

        Args:
            intrinsic (torch.Tensor): Tensor of shape (4, 4) representing an intrinsic matrix.
            extrinsic (torch.Tensor): Tensor of shape (4, 4) representing an extrinsic matrix.
            z_near (float): A floating point number representing the nearest depth rendered.
            z_far (float): A floating point number representing the farthest depth rendered.
        """
        self._intrinsic = intrinsic
        self._extrinsic = extrinsic
        self._z_near = z_near
        self._z_far = z_far

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
        self._z_near = new_z_near

    @z_far.setter
    def z_far(
        self,
        new_z_far: float,
    ) -> None:
        if not isinstance(new_z_far, int, float):
            raise ValueError(f"Expected variable of numeric type. Got {type(new_z_far)}.")
        self._z_far = new_z_far
