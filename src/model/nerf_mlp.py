"""
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Ben Mildenhall et al. (ECCV 2020)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.signal_encoder.positional_encoder import NeRFPositionalEncoder


class NeRFMLP(nn.Module):
    def __init__(
        self,
        pos_dim: int = 3,
        view_dir_dim: int = 3,
        L_pos: int = 10,
        L_direction: int = 4,
    ):
        """
        Constructor for NeRF.

        Args:
        - pos_dim: Dimensionality of vector representing a point in space. Set to 3 by default.
        - view_dir_dim: Dimensionality of vector representing a view direction. Set to 3 by default.
        - L_position: Level of positional encoding for positional vectors. Set to 3 by default.
        - L_direction: Level of positional encoding for viewing (direction) vectors.
            Set to 3 by default (unit vector instead of spherical notaion).
        """
        super().__init__()

        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim

        self.L_pos = L_pos
        self.L_direction = L_direction

        # Positional encoders for each input of forward method
        self.coord_encoder = NeRFPositionalEncoder(in_dim=3, L=self.L_pos)
        self.direction_encoder = NeRFPositionalEncoder(in_dim=2, L=self.L_direction)

        # MLPs approximating radiance field
        self.fc_in = nn.Linear(self.pos_dim * 2 * self.L_pos, 256)

        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 256)
        self.fc_4 = nn.Linear(256, 256)

        self.fc_5 = nn.Linear(self.pos_dim * 2 * self.L_pos + 256, 256)

        self.fc_6 = nn.Linear(256, 256)
        self.fc_7 = nn.Linear(256, 256)

        self.fc_8 = nn.Linear(256, 256 + 1)
        self.fc_9 = nn.Linear(256 + 1 + self.view_dir_dim * 2 * self.L_direction, 128)

        self.fc_out = nn.Linear(128, 3)

    def forward(self, ray_bundle: RayBundle, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation of NeRF.

        Args:
        - ray_bundle: RayBundle object consists of:
            - origins: Tensor of shape (B, ..., 3) representing the origins of the rays.
            - directions: Tensor of shape (B, ..., 3) representing the directions of the rays.
            - lengths: Tensor of shape (B, ..., n_pts_per_ray) representing the parameters denoting
                at which the points on rays are sampled.
            - xys: Tensor of shape (B, ..., 2) representing the xy locations of each ray's pixel in the screen space.

        Returns:
        - sigma: Tensor of shape (B, N). Tensor of density at each input point.
        - rgb: Tensor of shape (B, N, 3). Tensor of radiance at each input point.
        """
        # TODO: Modify this function using Pytorch3D functionalities!!
        x = ray_bundle_to_ray_points(ray_bundle)
        n_pts_per_ray = x.shape[2]
        d = ray_bundle.directions.unsqueeze(2).repeat(1, 1, n_pts_per_ray, 1)

        # positional encoding for inputs
        x = self.coord_encoder.encode(x)  # (B, N, 3) -> (B, N, 2 * 3 * L_pos)
        d = self.direction_encoder.encode(d)  # (B, N, 3) -> (B, N, 2 * 3 * L_direction)

        skip = x.clone()

        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))

        x = torch.cat((x, skip), dim=3)
        x = F.relu(self.fc_5(x))
        x = F.relu(self.fc_6(x))
        x = F.relu(self.fc_7(x))

        x = self.fc_8(x)
        sigma = x[:, :, :, 0].unsqueeze(-1)  # sigma.shape: (B, n_rays, n_pts_per_ray, 1)
        x = torch.cat((x, d), dim=3)
        x = F.relu(self.fc_9(x))
        rgb = torch.sigmoid(self.fc_out(x))  # rgb.shape: (B, n_rays, n_pts_per_ray, 3)

        return sigma, rgb
