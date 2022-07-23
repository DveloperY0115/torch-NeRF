"""
Implementation of positional encoder used in Instant-NGP (SIGGRAPH 2022).

Source: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py
"""

import torch

from torch_nerf.src.signal_encoder.signal_encoder_base import SignalEncoderBase


class SHEncoder(SignalEncoderBase):
    """
    Implementation of spherical harmonics encoding.

    Attributes:
        in_dim (int): Dimensionality of the data.
        degree (int): Degree of spherical harmonics.
        out_dim (int): Dimensionality of the encoded data.
    """

    def __init__(
        self,
        in_dim: int,
        degree: int,
    ):
        """
        Constructor for SHEncoder.

        Args:
            in_dim (int): Dimensionality of the data.
            degree (int): Degree of spherical harmonics.
        """
        super().__init__()

        self._in_dim = in_dim
        self._degree = degree
        self._out_dim = degree**2

        # coefficients for spherical harmonics
        self._coeff_0 = 0.28209479177387814
        self._coeff_1 = 0.4886025119029199
        self._coeff_2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        ]
        self._coeff_3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435,
        ]
        self._coeff_4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761,
        ]

    @property
    def in_dim(self) -> int:
        """Returns the dimensionality of the input data."""
        return self._in_dim

    @property
    def degree(self) -> int:
        """Returns the degree of spherical harmonics."""
        return self._degree

    @property
    def out_dim(self) -> int:
        """Returns the dimensionality of the encoded data."""
        return self._out_dim

    def encode(self, in_signal: torch.Tensor) -> torch.Tensor:
        """
        Embedds the input signal.

        Args:
            in_signal (torch.Tensor): Tensor of shape (N, C).
                Input signal being encoded.

        Returns:
            encoded_signal (torch.Tensor): Tensor of shape (N, self.out_dim).
                The embedding of the input signal.
        """
        encoded_signal = torch.empty(
            (in_signal.shape[0], self._out_dim), dtype=in_signal.get_device()
        )

        x, y, z = in_signal.unbind(-1)

        encoded_signal[..., 0] = self._coeff_0
        if self.degree > 1:
            encoded_signal[..., 1] = -self._coeff_1 * y
            encoded_signal[..., 2] = self._coeff_1 * z
            encoded_signal[..., 3] = -self._coeff_1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                encoded_signal[..., 4] = self._coeff_2[0] * xy
                encoded_signal[..., 5] = self._coeff_2[1] * yz
                encoded_signal[..., 6] = self._coeff_2[2] * (2.0 * zz - xx - yy)
                encoded_signal[..., 7] = self._coeff_2[3] * xz
                encoded_signal[..., 8] = self._coeff_2[4] * (xx - yy)
                if self.degree > 3:
                    encoded_signal[..., 9] = self._coeff_3[0] * y * (3 * xx - yy)
                    encoded_signal[..., 10] = self._coeff_3[1] * xy * z
                    encoded_signal[..., 11] = self._coeff_3[2] * y * (4 * zz - xx - yy)
                    encoded_signal[..., 12] = self._coeff_3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    encoded_signal[..., 13] = self._coeff_3[4] * x * (4 * zz - xx - yy)
                    encoded_signal[..., 14] = self._coeff_3[5] * z * (xx - yy)
                    encoded_signal[..., 15] = self._coeff_3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        encoded_signal[..., 16] = self._coeff_4[0] * xy * (xx - yy)
                        encoded_signal[..., 17] = self._coeff_4[1] * yz * (3 * xx - yy)
                        encoded_signal[..., 18] = self._coeff_4[2] * xy * (7 * zz - 1)
                        encoded_signal[..., 19] = self._coeff_4[3] * yz * (7 * zz - 3)
                        encoded_signal[..., 20] = self._coeff_4[4] * (zz * (35 * zz - 30) + 3)
                        encoded_signal[..., 21] = self._coeff_4[5] * xz * (7 * zz - 3)
                        encoded_signal[..., 22] = self._coeff_4[6] * (xx - yy) * (7 * zz - 1)
                        encoded_signal[..., 23] = self._coeff_4[7] * xz * (xx - 3 * yy)
                        encoded_signal[..., 24] = self._coeff_4[8] * (
                            xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                        )

        return encoded_signal
