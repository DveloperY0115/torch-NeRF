"""
Implementation of positional encoder proposed in NeRF (ECCV 2020).

Source: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
"""

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
        include_input: bool,
    ):
        """
        Constructor for PositionalEncoder.

        Args:
            in_dim (int): Dimensionality of the data.
            embed_level (int): Level of positional encoding.
            include_input (bool): A flat that determines whether to include
                raw input in the encoding.
        """
        super().__init__()

        self._embed_level = embed_level
        self._include_input = include_input
        self._in_dim = in_dim
        self._out_dim = 2 * self._embed_level * self._in_dim
        if self._include_input:
            self._out_dim += self._in_dim

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

        NOTE: Following the official implementation, this code implements
        a slightly different encoding scheme:
            (1) the constant 'pi' in sinusoidals is dropped;
            (2) the encoding includes the original value 'x' as well;
        For details, please refer to https://github.com/bmild/nerf/issues/12.
        """
        embed_fns = []

        max_freq_level = self._embed_level

        freq_bands = 2 ** torch.arange(0.0, max_freq_level, dtype=torch.float32)

        if self._include_input:
            embed_fns.append(lambda x: x)

        for freq in freq_bands:
            embed_fns.append(lambda x, freq=freq: torch.sin(freq * x))
            embed_fns.append(lambda x, freq=freq: torch.cos(freq * x))

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

    @property
    def in_dim(self) -> int:
        """Returns the dimensionality of the input vector that the encoder takes."""
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """Returns the dimensionality of the output vector after encoding."""
        return self._out_dim
