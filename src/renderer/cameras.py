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

    Args:
        intrinsic (torch.Tensor): Tensor of shape (4, 4) representing an intrinsic matrix.
        extrinsic (torch.Tensor): Tensor of shape (4, 4) representing an extrinsic matrix.
    """

    def __init__(
        self,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
    ):
        self._intrinsic = intrinsic
        self._extrinsic = extrinsic

    @property
    def intrinsic(self) -> torch.Tensor:
        """Returns the intrinsic matrix of the camera."""
        return self._intrinsic

    @property
    def extrinsic(self) -> torch.Tensor:
        """Returns the extrinsic matrix of the camera."""
        return self._extrinsic
