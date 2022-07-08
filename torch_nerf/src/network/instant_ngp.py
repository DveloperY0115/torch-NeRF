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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x (torch.Tensor):
        """
        pass
