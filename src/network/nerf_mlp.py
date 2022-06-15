"""Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

import typing

import torch
import torch.nn as nn


class NeRFMLP(nn.Module):
    """
    A simple MLP used for learning neural radiance fields.
    """

    def __init__(self, pos_dim: int, view_dir_dim: int, feat_dim: int = 256) -> None:
        super().__init__()

        rgb_dim = 3
        density_dim = 1

        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        self.feat_dim = feat_dim

        # fully-connected layers
        self.fc_in = nn.Linear(self.pos_dim, self.feat_dim)
        self.fc_1 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fc_2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fc_3 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fc_4 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fc_5 = nn.Linear(self.feat_dim + self.pos_dim, self.feat_dim)
        self.fc_6 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fc_7 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fc_8 = nn.Linear(self.feat_dim, self.feat_dim + density_dim)
        self.fc_9 = nn.Linear(self.feat_dim + self.view_dir_dim, self.feat_dim // 2)
        self.fc_out = nn.Linear(self.feat_dim // 2, rgb_dim)

        # activation layer
        self.relu_actvn = nn.ReLU()
        self.sigmoid_actvn = nn.Sigmoid()

    def forward(
        self, pos: torch.Tensor, view_dir: torch.Tensor
    ) -> typing.Dict[torch.Tensor, torch.Tensor]:
        """Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: torch.Tensor of shape (N, self.pos_dim). Coordinates of sample points along rays.
            view_dir: torch.Tensor of shape (N, self.dir_dim). View direction vectors.

        Returns:
            A dict containing predicted radiance (RGB) and density (sigma) at sample points.
        """
        # check input tensors
        if (pos.ndim != 2) or (view_dir.ndim != 2):
            raise ValueError(f"Expected 2D tensors. Got {pos.ndim}, {view_dir.ndim}-D tensors.")
        if pos.shape[0] != view_dir.shape[0]:
            raise ValueError(
                f"The number of samples must match. Got {pos.shape[0]} and {view_dir.shape[0]}."
            )
        if pos.shape[-1] != self.pos_dim:
            raise ValueError(f"Expected {self.pos_dim}-D position vector. Got {pos.shape[-1]}.")
        if view_dir.shape[-1] != self.view_dir_dim:
            raise ValueError(
                f"Expected {self.view_dir_dim}-D view direction vector. Got {view_dir.shape[-1]}."
            )

        x = self.relu_actvn(self.fc_in(pos))
        x = self.relu_actvn(self.fc_1(x))
        x = self.relu_actvn(self.fc_2(x))
        x = self.relu_actvn(self.fc_3(x))
        x = self.relu_actvn(self.fc_4(x))

        x = torch.cat([pos, x], dim=-1)

        x = self.relu_actvn(self.fc_5(x))
        x = self.relu_actvn(self.fc_6(x))
        x = self.relu_actvn(self.fc_7(x))
        x = self.fc_8(x)

        sigma = x[:, 0]
        x = torch.cat([x[:, 1:], view_dir], dim=-1)

        x = self.relu_actvn(self.fc_9(x))
        rgb = self.sigmoid_actvn(self.fc_out(x))

        return {"sigma": sigma, "rgb": rgb}
