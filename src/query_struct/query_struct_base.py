"""
Base class for query structures.
"""

import torch


class QueryStructBase(object):
    """
    Query structure base class.
    """

    def __init__(self):
        pass

    def query_points(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query 3D scene to retrieve radiance and density values.

        Args:
            pos (torch.Tensor): 3D coordinates of sample points.
            view_dir (torch.Tensor): View direction vectors associated with sample points.

        Returns:

        """
        raise NotImplementedError()
