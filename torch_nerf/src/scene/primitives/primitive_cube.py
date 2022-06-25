"""
A simple cubic primitive suitable for forward-facing, bounded scenes.
"""

from typing import Dict, Optional, Tuple

import torch
from torch_nerf.src.scene.primitives.primitive_base import PrimitiveBase
from torch_nerf.src.signal_encoder.signal_encoder_base import SignalEncoderBase


class PrimitiveCube(PrimitiveBase):
    """
    A simple cubic scene primitive.

    Attributes:
        radiance_field (torch.nn.Module): A network representing the scene.
    """

    def __init__(
        self,
        radiance_field: torch.nn.Module,
        encoders: Optional[Dict[str, SignalEncoderBase]] = None,
    ):
        """
        Constructor for QSCube.

        Args:
            radiance_field (torch.nn.Module): A network representing the scene.
        """
        super().__init__(encoders=encoders)

        if not isinstance(radiance_field, torch.nn.Module):
            raise ValueError(
                f"Expected a parameter of type torch.nn.Module. Got {type(radiance_field)}."
            )
        self._radiance_field = radiance_field

    def query_points(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the volume bounded by the cube to retrieve radiance and density values.

        Args:
            pos (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                3D coordinates of sample points.
            view_dir (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                View direction vectors associated with sample points.

        Returns:
            sigma (torch.Tensor): An instance of torch.Tensor of shape (N, S).
                The density at each sample point.
            radiance (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                The radiance at each sample point.
        """
        if pos.shape != view_dir.shape:
            raise ValueError(
                "Expected tensors of same shape. "
                f"Got {pos.shape} and {view_dir.shape}, respectively."
            )
        num_ray, num_sample, _ = pos.shape

        if not self.encoders is None:  # encode input signals
            pos = self.encoders["coord_enc"].encode(pos)
            view_dir = self.encoders["dir_enc"].encode(view_dir)

        sigma, radiance = self._radiance_field(
            pos.reshape(num_ray * num_sample, -1),
            view_dir.reshape(num_ray * num_sample, -1),
        )

        return sigma.reshape(num_ray, num_sample), radiance.reshape(num_ray, num_sample, -1)

    @property
    def radiance_field(self) -> torch.nn.Module:
        """Returns the network queried through this query structure."""
        return self._radiance_field
