"""
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Ben Mildenhall et al. (ECCV 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFCls(nn.Module):

    def __init__(self, L_position=10, L_direction=4, type='coarse'):
        """
        Constructor for NeRF.

        Args:
        - L_position: Int. Level of positional encoding for positional vectors
        - L_direction: Int. Level of positional encoding for viewing (direction) vectors
        """

        super(NeRFCls, self).__init__()

        self.L_position = L_position
        self.L_direction = L_direction

        if type == 'coarse':
            self.is_coarse = True    # coarse network
        else:
            self.is_coarse = False    # fine network

        # MLP
        self.mlps = nn.ModuleList(
            [nn.Linear(3 * 2 * L_position, 256)] + [nn.Linear(256, 256) for _ in range(4)] + 
            [nn.Linear(3 * 2 * L_position + 256, 256)] + [nn.Linear(256, 256) for _ in range(3)] +
            [nn.Linear(3 * 2 * L_direction + 256 + 1, 128), nn.Linear(128, 3)]
            )

    def forward(self, *args):
        """
        Forward propagation of NeRF.

        Args:
        - TBD

        Returns:
        - TBD
        """
        pass
         