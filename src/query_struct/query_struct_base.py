"""
Base class for query structures.
"""

import typing

import torch


class QueryStructBase(object):
    """
    Query structure base class.
    """

    def __init__(self, *args, **kwargs):
        pass

    def query_points(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Query 3D scene to retrieve radiance and density values.

        Args:
            pos (torch.Tensor): 3D coordinates of sample points.
            view_dir (torch.Tensor): View direction vectors associated with sample points.

        Returns:
            sigma (torch.Tensor): An instance of torch.Tensor of shape (N, S).
                The density at each sample point.
            radiance (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                The radiance at each sample point.
        """
        raise NotImplementedError()
