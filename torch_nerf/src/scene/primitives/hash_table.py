"""
An implementation of multi-resolution hash encoding presented in
"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (SIGGRAPH 2022)".
"""

from typing import Optional, Tuple

import torch

from torch_nerf.src.scene.primitives.primitive_base import PrimitiveBase
from torch_nerf.src.signal_encoder.signal_encoder_base import SignalEncoderBase


def spatial_hash_func(
    vert_coords: torch.Tensor,
    num_table_entry: int,
) -> torch.Tensor:
    """
    Hashes the given integer vertex coordinates.

    The input coordinate (x, y, z) is first scaled by the level's grid resolution
    and rounded down and up yielding the two integer vertices spanning a voxel.

    This function computes the hashed values of the coordinates of integer vertices
    following the definition of a spatial hash function presented in [Teschner et al., 2003].

    Args:
        vert_coords (torch.Tensor): Tensor of shape (N, 3).
            The coordinates of integer vertcies being hashed.
        num_table_entry (int): Number of entries in the hash table.

    Returns:
        indices (torch.Tensor): Tensor of shape (N, ).
            The indices specifying entries in the hash table at the level.
    """
    if vert_coords.dtype != torch.int32:
        raise ValueError(
            f"Expected integer coordinates as input. Got a tensor of type {vert_coords.type}."
        )
    if vert_coords.ndim != 2:
        raise ValueError(
            "Expected 2D tensor. "
            f"Got {vert_coords.ndim}-dimensional tensor of shape {vert_coords.shape}."
        )

    coeffs = torch.tensor(
        [[1, 2654435761, 805459861]],
        dtype=torch.int32,
        device=vert_coords.get_device(),
    )

    # hash the integer coordinates
    x = coeffs * vert_coords
    indices = torch.bitwise_xor(x[..., 0:1], x[..., 1:2])
    indices = torch.bitwise_xor(indices, x[..., 2:])
    indices = indices % num_table_entry

    return indices


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
        self._tables = 2 * (10**-4) * torch.rand(
            (
                self._num_level,
                self._max_entry_per_level,
                self._feat_dim,
            ),
            requires_grad=True,
        ) - (10**-4)

        # initialize the voxel grid resolutions
        coeff = torch.tensor(
            (self._max_res / self._min_res) ** (1 / (self._num_level - 1))
        )
        coeffs = coeff ** torch.arange(self._num_level)
        self._resolutions = torch.floor(self._min_res * coeffs)

        # register the hash function
        self._hash_func = spatial_hash_func

    def query_table(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Queries multiple levels of the hash tables and retrieves the feature vectors.

        Args:
            coords (torch.Tensor): Tensor of shape (N, 3).
                3D coordinates of sample points.

        Returns:
            features (torch.Tensor): Tensor of shape (N, F).
                Concatenated feature vectors each retrieved from each level of hash tables.
        """
        # scale input coordinates
        scaled_coords = self._scale_coordinates(coords)

        # query hash tables to compute final feature vector
        features = []
        for scaled_coord, table in zip(scaled_coords, self._tables):
            floor = torch.floor(scaled_coord)
            ceil = torch.ceil(scaled_coord)

            # identify 8 corners of the voxels enclosing queried points
            coord_fff = floor
            coord_cff = torch.cat([ceil[:, 0:1], floor[:, 1:2], floor[:, 2:]], dim=-1)
            coord_fcf = torch.cat([floor[:, 0:1], ceil[:, 1:2], floor[:, 2:]], dim=-1)
            coord_ffc = torch.cat([floor[:, 0:1], floor[:, 1:2], ceil[:, 2:]], dim=-1)
            coord_ccf = torch.cat([ceil[:, 0:1], ceil[:, 1:2], floor[:, 2:]], dim=-1)
            coord_cfc = torch.cat([ceil[:, 0:1], floor[:, 1:2], ceil[:, 2:]], dim=-1)
            coord_fcc = torch.cat([floor[:, 0:1], ceil[:, 1:2], ceil[:, 2:]], dim=-1)
            coord_ccc = ceil

            # hash the coordinates to derived hash table indices
            num_coords = coord_fff.shape[0]
            vert_coords = torch.cat(
                [
                    coord_fff,
                    coord_cff,
                    coord_fcf,
                    coord_ffc,
                    coord_ccf,
                    coord_cfc,
                    coord_fcc,
                    coord_ccc,
                ],
                dim=0,
            )
            indices = self._hash_func(vert_coords, self._max_entry_per_level)

            # retrieve feature vectors from the table
            vert_feature = table[indices]
            (
                feature_fff,
                feature_cff,
                feature_fcf,
                feature_ffc,
                feature_ccf,
                feature_cfc,
                feature_fcc,
                feature_ccc,
            ) = torch.split(vert_feature, num_coords, dim=0)

            # trilinear interpolation
            weight_fff = torch.cumprod(
                torch.abs(coord_ccc.float() - scaled_coord), dim=-1
            )
            weight_cff = torch.cumprod(
                torch.abs(coord_fcc.float() - scaled_coord), dim=-1
            )
            weight_fcf = torch.cumprod(
                torch.abs(coord_cfc.float() - scaled_coord), dim=-1
            )
            weight_ffc = torch.cumprod(
                torch.abs(coord_ccf.float() - scaled_coord), dim=-1
            )
            weight_ccf = torch.cumprod(
                torch.abs(coord_ffc.float() - scaled_coord), dim=-1
            )
            weight_cfc = torch.cumprod(
                torch.abs(coord_fcf.float() - scaled_coord), dim=-1
            )
            weight_fcc = torch.cumprod(
                torch.abs(coord_cff.float() - scaled_coord), dim=-1
            )
            weight_ccc = torch.cumprod(
                torch.abs(coord_fff.float() - scaled_coord), dim=-1
            )
            features.append(
                feature_fff * weight_fff
                + feature_cff * weight_cff
                + feature_fcf * weight_fcf
                + feature_ffc * weight_ffc
                + feature_ccf * weight_ccf
                + feature_cfc * weight_cfc
                + feature_fcc * weight_fcc
                + feature_ccc * weight_ccc
            )

        features = torch.cat(features, dim=-1)
        return features

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

    def _scale_coordinates(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scales the given 3D coordinates according to the resolution of hash grid being queried.

        Args:
            coords (torch.Tensor): Tensor of shape (N, 3).
                3D (real-valued) coordinates of sample points.

        Returns:
            scaled_coords (torch.Tensor): Tensor of shape (L, N, 3).
                A set of 3D (real-valued) coordinates each scaled according
                to the resolution of the hash grid.
        """
        scaled_coords = coords.unsqueeze(0).repeat(self._num_level, 1, 1)
        scaled_coords = self._resolutions.float() * scaled_coords
        return scaled_coords


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
        view_dir_encoder: Optional[SignalEncoderBase] = None,
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
        self._view_dir_encoder = view_dir_encoder

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
        """
        Queries the volume bounded by the cube to retrieve radiance and density values.

        Args:
            pos (torch.Tensor): Tensor of shape (N, S, 3).
                3D coordinates of sample points.
            view_dir (torch.Tensor): Tensor of shape (N, S, 3).
                View direction vectors associated with sample points.

        Returns:
            sigma (torch.Tensor): Tensor of shape (N, S).
                The density at each sample point.
            radiance (torch.Tensor): Tensor of shape (N, S, 3).
                The radiance at each sample point.
        """
        num_ray, num_sample = super().query_points(pos, view_dir)

        if not self._view_dir_encoder is None:  # encode view direction vectors
            view_dir = self._view_dir_encoder.encode(view_dir)

        features = self._hash_talbe.query_table(pos.reshape(num_ray * num_sample, -1))

        sigma, radiance = self._radiance_field(
            features,
            view_dir.reshape(num_ray * num_sample, -1),
        )

        return sigma.reshape(num_ray, num_sample), radiance.reshape(
            num_ray, num_sample, -1
        )

    @property
    def radiance_field(self) -> torch.nn.Module:
        """Returns the network queried through this query structure."""
        return self._radiance_field

    @property
    def hash_table(self) -> MultiResHashTable:
        """Returns the hash table used for spatial encoding."""
        return self._hash_talbe
