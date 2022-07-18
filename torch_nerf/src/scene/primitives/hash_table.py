"""
An implementation of multi-resolution hash encoding presented in
"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (SIGGRAPH 2022)".
"""

from typing import Tuple

import torch
from torch_nerf.src.scene.primitives.primitive_base import PrimitiveBase


def spatial_hash_func(vert_coords: torch.Tensor) -> torch.Tensor:
    """
    Hashes the given integer vertex coordinates.

    The input coordinate (x, y, z) is first scaled by the level's grid resolution
    and rounded down and up yielding the two integer vertices spanning a voxel.

    This function computes the hashed values of the coordinates of integer vertices
    following the definition of a spatial hash function presented in [Teschner et al., 2003].

    Args:
        vert_coords (torch.Tensor): Tensor of shape (N, 3).
            The coordinates of integer vertcies being hashed.

    Returns:
        hashes (torch.Tensor): Tensor of shape (N, ).
            The outputs of the spatial hash function.
    """
    raise NotImplementedError()


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
            max_entry_per_level (int): Number of entries in the hash table at each resolution.
            feat_dim (int): Dimensionality of feature vectors.
            min_res (int): The coarest voxel grid resolution.
            max_res (int): The finest voxel grid resolution.
        """
        self._num_level = num_level
        self._max_entry_per_level = max_entry_per_level
        self._feat_dim = feat_dim
        self._min_res = min_res
        self._max_res = max_res

        # initialize the table entries
        # the entry values lie in U[-10^{-4}, 10^{-4})
        self._tables = 2 * (10**-4) * torch.rand(
            (
                self._num_level,
                self._max_entry_per_level,
                self._feat_dim,
            ),
            requires_grad=True,
        ) - (10**-4)

        # initialize the voxel grid resolutions
        coeff = torch.tensor((self._max_res / self._min_res) ** (1 / (self._num_level - 1)))
        coeffs = coeff ** torch.arange(self._num_level)
        self._resolutions = torch.floor(self._min_res * coeffs)

        # register the hash function
        self._hash_func = spatial_hash_func

    @property
    def num_level(self) -> int:
        """Returns the number of grid resolution levels."""
        return self._num_level

    @property
    def max_entry_per_level(self) -> int:
        """Returns the number of entries in the hash table for each resolution level."""
        return self._max_entry_per_level

    @property
    def feat_dim(self) -> int:
        """Returns the dimensionality of feature vectors."""
        return self._feat_dim

    @property
    def min_res(self) -> int:
        """Returns the coarest voxel grid resolution."""
        return self._min_res

    @property
    def max_res(self) -> int:
        """Returns the finest voxel grid resolution."""
        return self._max_res


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
