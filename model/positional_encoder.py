"""
positional_encoder.py - Class for efficient positional encoding.
"""

from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement positional encoding

class NeRFPositionalEncoder(object):
    def __init__(self, in_dim: int, L: int):
        """
        Constructor for NeRFPositionalEncoder.

        Args:
        - in_dim: Dimension of input data. (e.g., 3 for coordinate in 3D, 2 for view direction)
        - L: Level of positional encoding. Usually 4 (directional) or 10 (positional).
        """
        super().__init__()

        self.L = L
        self.in_dim = in_dim
        self.out_dim = 2 * self.L * self.in_dim

        # creating embedding function
        self.embed_fns = self.create_embedding_fn()

    def create_embedding_fn(self):
        """

        """
        embed_fns = []

        max_freq_level = self.L

        freq_bands = torch.linspace(2.0 ** 0, 2.0 ** max_freq_level, steps=self.L)

        for freq in freq_bands:
            embed_fns.append(lambda x, freq=freq: torch.sin(freq * pi * x))
            embed_fns.append(lambda x, freq=freq: torch.cos(freq * pi * x))

        return embed_fns

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
        - x: Tensor of shape (B, N, C) whose elements will be encoded.

        Returns:
        - Positional encoded 'x'
        """
        return torch.cat([fn(x) for fn in self.embed_fns], -1)
