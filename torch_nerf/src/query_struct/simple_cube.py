"""
A simple cubic query structure suitable for bounded scenes.
"""

import typing

import torch
from src.query_struct.query_struct_base import QueryStructBase


class QSCube(QueryStructBase):
    """
    A simple cubic query structure.

    Attributes:
        radiance_field (torch.nn.Module): A network representing the scene.
    """

    def __init__(
        self,
        radiance_field: torch.nn.Module,
    ):
        """
        Constructor for QSCube.

        Args:
            radiance_field (torch.nn.Module): A network representing the scene.
        """
        super().__init__()

        if not isinstance(radiance_field, torch.nn.Module):
            raise ValueError(
                f"Expected a parameter of type torch.nn.Module. Got {type(radiance_field)}."
            )
        self._radiance_field = radiance_field

    def query_points(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Query 3D scene to retrieve radiance and density values.

        Args:
            pos (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                3D coordinates of sample points.
            view_dir (torch.Tensor): An instance of torch.Tensor of shape (N, 3).
                View direction vectors associated with sample points.

        Returns:
            sigma (torch.Tensor): An instance of torch.Tensor of shape (N, S).
                The density at each sample point.
            radiance (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                The radiance at each sample point.
        """
        return self._radiance_field(pos, view_dir)
