"""
positional_encoder.py - Class for efficient positional encoding.

Implementation influenced by: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
"""

from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        Create embedding function from given number of frequency bands and dimension of data being encoded.

        The definition of positional encoding function is as follows:
        f(p) = (sin(2^0 * pi * p), cos(2^0 * pi * p), ..., sin(2^{L-1} * pi * p), cos(2^{L-1} * pi * p))
        and is computed on each (x, y, z) or (theta, phi) triplet (or tuple).

        Thus, the form of resulting tensor is:
        f(pos) = [
                sin(2^0 * pi * x), sin(2^0 * pi * y), sin(2^0 * pi * z),  
                cos(2^0 * pi * x), cos(2^0 * pi * y), cos(2^0 * pi * z), 
                    ...
                ]
        where pos = (x, y, z)

        For details, please refer to section 5.1 of NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Mildenhall et al. (ECCV 2020) 
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
        Compute positional encoding on the given tensor.

        Args:
        - x: Tensor of shape (B, N, C) whose elements will be encoded.

        Returns:
        - Positional encoded 'x'
        """
        return torch.cat([fn(x) for fn in self.embed_fns], -1)
