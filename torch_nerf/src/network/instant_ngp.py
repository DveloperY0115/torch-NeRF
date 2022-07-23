"""
Pytorch implementation of MLP used in Instant Neural Graphics Primitives (SIGGRAPH 2022).
"""

from typing import Tuple

import torch
import torch.nn as nn


class InstantNeRF(nn.Module):
    """
    A neural network that approximates neural radiance fields.

    This class implements the NeRF model described in the paper
    'Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
    (SIGGRAPH 2022, Best Paper)'. For architecture details, please refer to the
    Section 5.4 of the paper.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        num_level (int): Number of grid resolution levels.
        max_entry_per_level (int): Number of entries in the hash table.
        table_min_res (int): The coarest voxel grid resolution.
        table_max_res (int): The finest voxel grid resolution.
        density_feat_dim (int): Dimensionality of feature vector within density network.
        color_feat_dim (int): Dimensionality of feature vector within color network.
        table_feat_dim (int): Dimensionality of feature vectors stored as entries of the hash table.
        is_hdr (int): A flag for switching output activation of the color MLP.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        num_level: int,
        log_max_entry_per_level: int,
        table_min_res: int,
        table_max_res: int,
        density_feat_dim: int = 64,
        color_feat_dim: int = 64,
        table_feat_dim: int = 2,
        is_hdr: bool = False,
    ) -> None:
        """
        Constructor of class 'InstantNeRF'.

        Args:
            pos_dim (int): Dimensionality of coordinate vectors of sample points.
            view_dir_dim (int): Dimensionality of view direction vectors.
            num_level (int): Number of grid resolution levels.
            log_max_entry_per_level (int): Number of entries in the hash table.
            for each resolution level in log (base 2) scale.
            table_min_res (int): The coarest voxel grid resolution.
            table_max_res (int): The finest voxel grid resolution.
            density_feat_dim (int): Dimensionality of feature vector within density network.
                Set to 64 by default following the paper.
            color_feat_dim (int): Dimensionality of feature vector within color network.
                Set to 64 by default following the paper.
            table_feat_dim (int): Dimensionality of feature vectors stored as entries of
                the hash table. Set to 2 by default following the paper.
            is_hdr (bool): A flag for switching output activation of the color MLP.
                If True, the network is assumed to be trained on high dynamic range (HDR)
                training images and the exponential activation is used.
                Otherwise, the network is assumed to be trained on low dynamic range (i.e., sRGB)
                training images and the sigmoid activation for limiting the output range to
                [0.0, 1.0] is used.
        """
        super().__init__()

        density_mlp_out_dim = 16
        color_mlp_out_dim = 3

        self._pos_dim = pos_dim
        self._view_dir_dim = view_dir_dim
        self._density_mlp_in_dim = num_level * table_feat_dim
        self._density_feat_dim = density_feat_dim
        self._color_feat_dim = color_feat_dim
        self._is_hdr = is_hdr

        # MLPs
        self.density_mlp = InstantNeRFMLP(
            in_dim=self._density_mlp_in_dim,
            out_dim=density_mlp_out_dim,
            feat_dim=self._density_feat_dim,
            num_hidden_layer=1,
        )
        self.color_mlp = InstantNeRFMLP(
            in_dim=density_mlp_out_dim + self._view_dir_dim,
            out_dim=color_mlp_out_dim,
            feat_dim=self._color_feat_dim,
            num_hidden_layer=2,
        )

        # activation layer
        self.density_actvn = nn.ReLU()
        self.color_actvn = torch.exp if self._is_hdr else nn.Sigmoid()

        # Multi-resolution hash table
        self.hash_table = MultiResHashTable(
            num_level,
            log_max_entry_per_level,
            table_feat_dim,
            table_min_res,
            table_max_res,
        )

    def forward(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predicts the corresponding radiance (RGB) and density (sigma).

        Args:
            pos (torch.Tensor): Tensor of shape (N, self.pos_dim).
                Coordinates of sample points along rays.
            view_dir (torch.Tensor): Tensor of shape (N, self.dir_dim).
                View direction vectors.

        Returns:
            A tuple containing predicted radiance (RGB) and density (sigma) at sample points.
        """
        # check input tensors
        if (pos.ndim != 2) or (view_dir.ndim != 2):
            raise ValueError(f"Expected 2D tensors. Got {pos.ndim}, {view_dir.ndim}-D tensors.")
        if pos.shape[0] != view_dir.shape[0]:
            raise ValueError(
                f"The number of samples must match. Got {pos.shape[0]} and {view_dir.shape[0]}."
            )
        if pos.shape[-1] != self._pos_dim:
            raise ValueError(f"Expected {self._pos_dim}-D position vector. Got {pos.shape[-1]}.")
        if view_dir.shape[-1] != self._view_dir_dim:
            raise ValueError(
                f"Expected {self._view_dir_dim}-D view direction vector. Got {view_dir.shape[-1]}."
            )

        # query the hash table
        table_features = self.hash_table(pos)

        # predict density (sigma)
        density_out = self.density_mlp(table_features)
        density_out[..., 0] = self.density_actvn(density_out[..., 0])
        density = density_out[..., 0].clone()

        # predict radiance (RGB)
        color = self.color_mlp(torch.cat([density_out, view_dir], dim=-1))
        color = self.color_actvn(color)

        return density, color

    @property
    def pos_dim(self) -> int:
        """Returns the acceptable dimensionality of coordinate vectors."""
        return self._pos_dim

    @property
    def view_dir_dim(self) -> int:
        """Returns the acceptable dimensionality of view direction vectors."""
        return self._view_dir_dim

    @property
    def num_level(self) -> int:
        """Returns the number of grid resolution levels"""
        return self.hash_table.num_level

    @property
    def max_entry_per_level(self) -> int:
        """Returns the number of entries in each level of hash tables."""
        return self.hash_table.max_entry_per_level

    @property
    def table_min_res(self) -> int:
        """Returns the lowest resolution of a voxel grid."""
        return self.hash_table.min_res

    @property
    def table_max_res(self) -> int:
        """Returns the highest resolution of a voxel grid.`"""
        return self.hash_table.max_res

    @property
    def density_feat_dim(self) -> int:
        """Returns the dimensionality of density network hidden layer feature vectors."""
        return self._density_feat_dim

    @property
    def color_feat_dim(self) -> int:
        """Returns the dimensionality of color network hidden layer feature vectors."""
        return self._color_feat_dim

    @property
    def is_hdr(self) -> bool:
        """Returns the flag indicating which type of output activation is used."""
        return self._is_hdr


class InstantNeRFMLP(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    This class implements the shallow, light-weight MLP used in the paper
    'Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
    (SIGGRAPH 2022, Best Paper)'.

    All the neural primitives presented in the paper, such as gigapixel images
    and signed distance functions (SDFs), are parameterized by this MLP except for NeRF
    using the cascade of two MLPs. For architecture details, please refer to the Section 4
    of the paper.

    Attributes:
        in_dim (int): Dimensionality of input features.
        out_dim (int): Dimensionality of output features.
        feat_dim (int): Dimensionality of hidden layer features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        feat_dim: int,
        num_hidden_layer: int = 2,
    ) -> None:
        """
        Constructor of class 'InstantNeRFMLP'.

        Args:
            in_dim (int): Dimensionality of input features.
            out_dim (int): Dimensionality of output features.
            feat_dim (int): Dimensionality of hidden layer features.
            num_hidden_layer (int): Number of hidden layers involved in the forward propagation.
                Set to 2 by default.
        """
        super().__init__()

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._feat_dim = feat_dim
        self._num_hidden_layer = num_hidden_layer

        # fully-connected layers
        self.fc_in = nn.Linear(self._in_dim, self._feat_dim)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(self._feat_dim, self._feat_dim) for _ in range(self._num_hidden_layer)]
        )
        self.fc_out = nn.Linear(self._feat_dim, self._out_dim)

        # activation layer
        self.relu_actvn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Tensor of shape (N, self.in_dim).
                A batch of input feature vectors.

        Returns:
            out (torch.Tensor): Tensor of shape (N, self.out_dim).
                A batch of output feature vectors.
        """
        # check input tensors
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D tensor. Got {x.ndim}-D tensor.")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected {self.in_dim}-D position vector. Got {x.shape[-1]}.")

        out = self.relu_actvn(self.fc_in(x))

        for hidden_layer in self.fc_hidden:
            out = self.relu_actvn(hidden_layer(out))

        out = self.fc_out(out)

        return out

    @property
    def in_dim(self) -> int:
        """Returns the acceptable dimensionality of input vectors."""
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """Returns the acceptable dimensionality of output vectors."""
        return self._out_dim

    @property
    def feat_dim(self) -> int:
        """Returns the acceptable dimensionality of hidden layer feature vectors."""
        return self._feat_dim

    @property
    def num_hidden_layer(self) -> int:
        """Returns the number of hidden layers included in the MLP."""
        return self._num_hidden_layer


class MultiResHashTable(nn.Module):
    """
    A multi-resolution hash table implemented using Pytorch.

    Attributes:
        num_level (int): Number of grid resolution levels.
        log_max_entry_per_level (int): Number of entries in the hash table
            for each resolution level in log (base 2) scale.
        feat_dim (int): Dimensionality of feature vectors.
        min_res (int): The coarest voxel grid resolution.
        max_res (int): The finest voxel grid resolution.
    """

    def __init__(
        self,
        num_level: int,
        log_max_entry_per_level: int,
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
        super().__init__()

        self._num_level = int(num_level)
        self._max_entry_per_level = int(2**log_max_entry_per_level)
        self._feat_dim = int(feat_dim)
        self._min_res = int(min_res)
        self._max_res = int(max_res)

        # initialize the table entries
        tables = 2 * (10**-4) * torch.rand(
            (
                self._num_level,
                self._max_entry_per_level,
                self._feat_dim,
            ),
            dtype=torch.half,
            requires_grad=True,
        ) - (10**-4)
        self.register_parameter("tables", nn.Parameter(tables))

        # initialize the voxel grid resolutions
        coeff = torch.tensor((self._max_res / self._min_res) ** (1 / (self._num_level - 1)))
        coeffs = coeff ** torch.arange(self._num_level)
        resolutions = torch.floor(self._min_res * coeffs)

        # register the hash function
        self._hash_func = spatial_hash_func

        # register tensors to the buffer
        self.register_buffer("coeffs", coeffs)
        self.register_buffer("resolutions", resolutions)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            coords (torch.Tensor): Tensor of shape (N, 3).
                3D coordinates of sample points.

        Returns:
            features (torch.Tensor): Tensor of shape (N, F).
                Concatenated feature vectors each retrieved from each level of hash tables.
        """
        return self.query_table(coords)

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
        for scaled_coord, table in zip(scaled_coords, self.tables):
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
            ).int()
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
            weight_fff = torch.prod(
                torch.abs(coord_ccc.float() - scaled_coord), dim=-1, keepdim=True
            )
            weight_cff = torch.prod(
                torch.abs(coord_fcc.float() - scaled_coord), dim=-1, keepdim=True
            )
            weight_fcf = torch.prod(
                torch.abs(coord_cfc.float() - scaled_coord), dim=-1, keepdim=True
            )
            weight_ffc = torch.prod(
                torch.abs(coord_ccf.float() - scaled_coord), dim=-1, keepdim=True
            )
            weight_ccf = torch.prod(
                torch.abs(coord_ffc.float() - scaled_coord), dim=-1, keepdim=True
            )
            weight_cfc = torch.prod(
                torch.abs(coord_fcf.float() - scaled_coord), dim=-1, keepdim=True
            )
            weight_fcc = torch.prod(
                torch.abs(coord_cff.float() - scaled_coord), dim=-1, keepdim=True
            )
            weight_ccc = torch.prod(
                torch.abs(coord_fff.float() - scaled_coord), dim=-1, keepdim=True
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
        scaled_coords = self.resolutions.float().unsqueeze(-1).unsqueeze(-1) * scaled_coords
        return scaled_coords


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
            f"Expected integer coordinates as input. Got a tensor of type {vert_coords.dtype}."
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
    indices = torch.bitwise_xor(x[..., 0], x[..., 1])
    indices = torch.bitwise_xor(indices, x[..., 2])
    indices = indices % num_table_entry

    return indices.long()
