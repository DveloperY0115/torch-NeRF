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
        t_near (float): A floating point number representing the nearest depth rendered.
        t_far (float): A floating point number representing the farthest depth rendered.
    """

    def __init__(
        self,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        t_near: float,
        t_far: float,
    ):
        """
        Constructor of class 'CameraBase'.

        Args:
            intrinsic (torch.Tensor): Tensor of shape (4, 4) representing an intrinsic matrix.
            extrinsic (torch.Tensor): Tensor of shape (4, 4) representing an extrinsic matrix.
            t_near (float): A floating point number representing the nearest depth rendered.
            t_far (float): A floating point number representing the farthest depth rendered.
        """
        self._intrinsic = intrinsic
        self._extrinsic = extrinsic
        self._t_near = t_near
        self._t_far = t_far

    # =============================================
    # getters
    # =============================================
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
        """Returns the nearest depth rendered."""
        return self._t_near

    @property
    def t_far(self) -> float:
        """Returns the farthest depth rendered."""
        return self._t_far

    # =============================================
    # setters
    # =============================================
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
        self._t_near = new_t_near

    @t_far.setter
    def t_far(
        self,
        new_t_far: float,
    ) -> None:
        if not isinstance(new_t_far, int, float):
            raise ValueError(f"Expected variable of numeric type. Got {type(new_t_far)}.")
        self._t_far = new_t_far
