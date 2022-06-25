"""
Base class for scene primitives.
"""

from typing import Dict, Optional, Tuple

import torch
from torch_nerf.src.signal_encoder.signal_encoder_base import SignalEncoderBase


class PrimitiveBase(object):
    """
    Scene primitive base class.
    """

    def __init__(
        self,
        encoders: Optional[Dict[str, SignalEncoderBase]] = None,
    ):
        if not isinstance(encoders, dict):
            raise ValueError(f"Expected a parameter of type Dict. Got {type(encoders)}")
        if not "coord_enc" in encoders.keys():
            raise ValueError(f"Missing required encoder type 'coord_enc'. Got {encoders.keys()}.")
        if not "dir_enc" in encoders.keys():
            raise ValueError(f"Missing required encoder type 'dir_enc'. Got {encoders.keys()}.")
        self._encoders = encoders

    def query_points(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @property
    def encoders(self) -> Optional[Dict[str, SignalEncoderBase]]:
        """
        Returns the signal encoders that process signals before
        querying the neural radiance field(s).
        """
        return self._encoders

    @encoders.setter
    def encoders(self, new_encoders) -> None:
        if not isinstance(new_encoders, dict):
            raise ValueError(f"Expected a parameter of type Dict. Got {type(new_encoders)}")
        if not "coord_enc" in new_encoders.keys():
            raise ValueError(
                f"Missing required encoder type 'coord_enc'. Got {new_encoders.keys()}."
            )
        if not "dir_enc" in new_encoders.keys():
            raise ValueError(f"Missing required encoder type 'dir_enc'. Got {new_encoders.keys()}.")
        self._encoders = new_encoders
