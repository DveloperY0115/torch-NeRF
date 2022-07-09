"""
Pytorch implementation of MLP used in Instant Neural Graphics Primitives (SIGGRAPH 2022).
"""

from typing import Tuple

import torch
import torch.nn as nn


class InstantNGPNeRF(nn.Module):
    def __init__(self) -> None:
        super().__init__()

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
            pos (torch.Tensor): Tensor of shape (N, self.pos_dim). Coordinates of sample points along rays.
            view_dir (torch.Tensor): Tensor of shape (N, self.dir_dim). View direction vectors.

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


class InstantNGPMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        feat_dim: int,
    ) -> None:
        """
        Constructor of class 'InstantNGPMLP'.

        This class implements the shallow, light-weight MLP used in the paper
        "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
        (SIGGRAPH 2022, Best Paper)". All the neural primitives, such as gigapixel images
        and signed distance functions (SDFs), are parameterized by this MLP except for NeRF
        using the cascade of two MLPs. For architecture details, please refer to the Section 4
        of the paper.

        Args:
            in_dim (int): The dimensionality of input features.
            out_dim (int): The dimensionality of output features.
            feat_dim (int): The dimensionality of hidden layer features.
        """
        super().__init__()

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._feat_dim = feat_dim

        # fully-connected layers
        self.fc_in = nn.Linear(self._in_dim, self._feat_dim)
        self.fc_1 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_2 = nn.Linear(self._feat_dim, self._feat_dim)
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
        out = self.relu_actvn(self.fc_1(out))
        out = self.relu_actvn(self.fc_2(out))
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
