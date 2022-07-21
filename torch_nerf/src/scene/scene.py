from typing import Sequence, Tuple

import torch
from torch_nerf.src.scene.primitives import PrimitiveBase


class Scene:
    """
    Scene object representing an renderable scene.

    Attributes:

    """

    def __init__(self, primitives: Sequence[PrimitiveBase]):
        """
        Constructor for 'Scene'.

        Args:
            primitives (Sequence[PrimitiveBase]): A collection of scene primitives.
        """
        self._primitives = primitives

    def query_points(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query 3D scene to retrieve radiance and density values.

        TODO: Extend the implementation to support
        primitive hierarchy, KD-tree like spatial structure, etc.

        Args:
            pos (torch.Tensor): 3D coordinates of sample points.
            view_dir (torch.Tensor): View direction vectors associated with sample points.

        Returns:
            sigma (torch.Tensor): An instance of torch.Tensor of shape (N, S).
                The density at each sample point.
            radiance (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                The radiance at each sample point.
        """
        return self._primitives.query_points(pos, view_dir)
