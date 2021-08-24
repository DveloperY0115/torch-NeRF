"""
positional_encoder.py - Class for efficient positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement positional encoding


class NeRFPositionalEncoder(nn.Module):
    def __init__(self, L):
        """
        Constructor for NeRFPositionalEncoder.

        Args:
        - L (int): Level of positional encoding. Usually 4 (directional) or 10 (positional).
        """
        super(NeRFPositionalEncoder, self).__init__()
        self.L = L

        # construct matrix for positional encoding

    def forward(self, x):
        """
        Forward propagation.

        Note that gradient is not recorded within this code block.

        Args:
        - x (torch.Tensor): Tensor of shape (B, N, C) whose elements will be encoded.

        Returns:
        - Positional encoded 'x'
        """

        with torch.no_grad():
            raise NotImplementedError
