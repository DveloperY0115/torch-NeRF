"""
Implementation of positional encoder proposed in NeRF (ECCV 2020).

Source: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
"""

from math import pi

import torch


class PositionalEncoder(object):
    """
    Implementation of positional encoding.

    Attributes:
        in_dim (int): Dimensionality of the data.
        embed_level (int): Level of positional encoding.
        out_dim (int): Dimensionality of the encoded data.
    """

    def __init__(
        self,
        in_dim: int,
        embed_level: int,
    ):
        """
        Constructor for PositionalEncoder.

        Args:
            in_dim (int): Dimensionality of the data.
            embed_level (int): Level of positional encoding.
        """
        super().__init__()

        self._embed_level = embed_level
        self._in_dim = in_dim
        self._out_dim = 2 * self._embed_level * self._in_dim

        # creating embedding function
        self._embed_fns = self._create_embedding_fn()

    def _create_embedding_fn(self):
        """
        Creates embedding function from given
            (1) number of frequency bands;
            (2) dimension of data being encoded;

        The positional encoding is defined as:
        f(p) = [
                sin(2^0 * pi * p), cos(2^0 * pi * p),
                                ...,
                sin(2^{L-1} * pi * p), cos(2^{L-1} * pi * p)
            ],
        and is computed for all components of the input vector.

        Thus, the form of resulting tensor is:
        f(pos) = [
                sin(2^0 * pi * x), sin(2^0 * pi * y), sin(2^0 * pi * z),
                cos(2^0 * pi * x), cos(2^0 * pi * y), cos(2^0 * pi * z),
                    ...
            ],
        where pos = (x, y, z).

        For details, please refer to 'NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis, Mildenhall et al. (ECCV 2020)'.
        """
        embed_fns = []

        max_freq_level = self._embed_level

        freq_bands = 2 ** torch.arange(0.0, max_freq_level, dtype=torch.float32)

        for freq in freq_bands:
            embed_fns.append(lambda x, freq=freq: torch.sin(freq * pi * x))
            embed_fns.append(lambda x, freq=freq: torch.cos(freq * pi * x))

        return embed_fns

    def encode(self, in_signal: torch.Tensor) -> torch.Tensor:
        """
        Computes positional encoding of the given signal.

        Args:
            in_signal: An instance of torch.Tensor of shape (N, C).
                Input signal being encoded.

        Returns:
            An instance of torch.Tensor of shape (N, self.out_dim).
                The positional encoding of the input signal.
        """
        return torch.cat([fn(in_signal) for fn in self._embed_fns], -1)
