"""
Integrators for computing pixel colors from radiance and density.
"""

import torch


class IntegratorBase(object):
    """
    Base class for integrators.
    """

    def __init__(self):
        pass

    def compute_pixel_color(
        self,
        sigma: torch.Tensor,
        delta: torch.Tensor,
        radiance: torch.Tensor,
    ):
        """
        Determine pixel colors given densities, interval length, and radiance values
        obtained along rays.
        """
        raise NotImplementedError()


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    def __init__(self):
        super().__init__()

    def compute_pixel_color(
        self,
        sigma: torch.Tensor,
        delta: torch.Tensor,
        radiance: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()
