"""
An implementation of multi-resolution hash table presented in 'InstantNGP'.
"""

from typing import Tuple

import torch
from torch_nerf.src.scene.primitives.primitive_base import PrimitiveBase


class MultiResHashTable:
    """
    A multi-resolution hash table implemented using Pytorch.

    Attributes:
        num_level (int): Number of grid resolution levels.
        max_entry_per_level (int): Number of entries in the hash table for each resolution level.
        feat_dim (int): Dimensionality of feature vectors.
        min_res (int): The coarest voxel grid resolution.
        max_res (int): The finest voxel grid resolution.
    """

    def __init__(
        self,
        num_level: int,
        max_entry_per_level: int,
        feat_dim: int,
        min_res: int,
        max_res: int,
    ):
        """
        Constructor for 'MultiResHashTable'.

        Args:
            num_level (int): Number of grid resolution levels.
            max_entry_per_level (int): Number of entries in the hash table for each resolution level.
            feat_dim (int): Dimensionality of feature vectors.
            min_res (int): The coarest voxel grid resolution.
            max_res (int): The finest voxel grid resolution.
        """
        pass


class PrimitiveHashEncoding(PrimitiveBase):
    """
    A scene primitive representing a scene as a combination of
    a multi-resolution hash table and the accompanying MLP.

    Attributes:
        radiance_field (torch.nn.Module):
        hash_table (MultiResHashTable):
    """

    def __init__(
        self,
        radiance_field: torch.nn.Module,
        num_level: int,
        max_entry_per_level: int,
        feat_dim: int,
        min_res: int,
        max_res: int,
    ):
        """
        Constructor for 'PrimitiveHashTable'.

        Args:
            radiance_field (torch.nn.Module):
            num_level (int): Number of grid resolution levels.
            max_entry_per_level (int): Number of entries in the hash table for each resolution level.
            feat_dim (int): Dimensionality of feature vectors.
            min_res (int): The coarest voxel grid resolution.
            max_res (int): The finest voxel grid resolution.
        """
        super().__init__()

        if not isinstance(radiance_field, torch.nn.Module):
            raise ValueError(
                f"Expected a parameter of type torch.nn.Module. Got {type(radiance_field)}."
            )
        self._radiance_field = radiance_field

        # construct multi-resolution hash table
        self._hash_talbe = MultiResHashTable(
            num_level,
            max_entry_per_level,
            feat_dim,
            min_res,
            max_res,
        )

    def query_points(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        return super().query_points(pos, view_dir)

    @property
    def radiance_field(self) -> torch.nn.Module:
        """Returns the network queried through this query structure."""
        return self._radiance_field

    @property
    def hash_table(self) -> MultiResHashTable:
        """Returns the hash table used for spatial encoding."""
        return self._hash_talbe
