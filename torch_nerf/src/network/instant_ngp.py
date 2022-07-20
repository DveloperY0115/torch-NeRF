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
        density_feat_dim (int): Dimensionality of feature vector within density network.
        color_feat_dim (int): Dimensionality of feature vector within color network.
        is_hdr (int): A flag for switching output activation of the color MLP.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        density_feat_dim: int = 64,
        color_feat_dim: int = 64,
        is_hdr: bool = False,
    ) -> None:
        """
        Constructor of class 'InstantNeRF'.

        Args:
            pos_dim (int): Dimensionality of coordinate vectors of sample points.
            view_dir_dim (int): Dimensionality of view direction vectors.
            density_feat_dim (int): Dimensionality of feature vector within density network.
                Set to 64 by default following the paper.
            color_feat_dim (int): Dimensionality of feature vector within color network.
                Set to 64 by default following the paper.
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
        self._density_feat_dim = density_feat_dim
        self._color_feat_dim = color_feat_dim
        self._is_hdr = is_hdr

        # MLPs
        self.density_mlp = InstantNeRFMLP(
            in_dim=self._pos_dim,
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

        # predict density (sigma)
        density_out = self.density_mlp(pos)
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
