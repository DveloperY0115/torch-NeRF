"""
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Ben Mildenhall et al. (ECCV 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoder import NeRFPositionalEncoder


class NeRFCls(nn.Module):
    def __init__(self, L_coordinate=10, L_direction=4, type="coarse"):
        """
        Constructor for NeRF.

        Args:
        - L_position: Int. Level of positional encoding for positional vectors
        - L_direction: Int. Level of positional encoding for viewing (direction) vectors
        """

        super(NeRFCls, self).__init__()

        self.L_coordinate = L_coordinate
        self.L_direction = L_direction

        if type == "coarse":
            self.is_coarse = True  # coarse network
        else:
            self.is_coarse = False  # fine network

        # TODO: Implement positional encoding
        # Positional encoders for each input of forward method
        self.coord_encoder = NeRFPositionalEncoder(self.L_coordinate)
        self.direction_encoder = NeRFPositionalEncoder(self.L_direction)

        # MLPs
        self.fc_1 = nn.Linear(3 * 2 * self.L_coordinate, 256)

        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 256)
        self.fc_4 = nn.Linear(256, 256)
        self.fc_5 = nn.Linear(256, 256)

        self.fc_6 = nn.Linear(3 * 2 * self.L_coordinate + 256, 256)

        self.fc_7 = nn.Linear(256, 256)
        self.fc_8 = nn.Linear(256, 256)

        self.fc_9 = nn.Linear(256, 256 + 1)
        self.fc_10 = nn.Linear(256 + 1 + 3 * 2 * self.L_direction, 128)
        self.fc_11 = nn.Linear(128, 3)

    def forward(self, x, d):
        """
        Forward propagation of NeRF.

        Args:
        - x (torch.Tensor): Tensor of shape (B, N, 3). Tensor of sample point coordinates.
        - d (torch.Tensor): Tensor of shape (B, N, 3). Tensor of view direction vectors.

        Returns:
        - sigma (torch.Tensor): Tensor of shape (B, N). Tensor of density at each input point.
        - rgb (torch.Tensor): Tensor of shape (B, N, 3). Tensor of radiance at each input point.
        """

        # positional encoding for inputs
        x = self.coord_encoder(x)  # (B, N, 3) -> (B, N, 2 * 3 * L_coordinate)
        d = self.direction_encoder(d)  # (B, N, 3) -> (B, N, 2 * 3 * L_direction)

        skip = x.clone()

        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))
        x = F.relu(self.fc_5(x))

        x = torch.cat((x, skip), dim=2)
        x = F.relu(self.fc_6(x))
        x = F.relu(self.fc_7(x))
        x = F.relu(self.fc_8(x))

        x = self.fc_9(x)
        sigma = x[:, :, 0]
        x = torch.cat((x, d), dim=2)
        x = F.relu(self.fc_10(x))
        rgb = F.sigmoid(self.fc_11(x))

        return sigma, rgb
