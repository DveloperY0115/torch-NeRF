"""
Integrators for computing pixel colors from radiance and density.
"""

import torch


class IntegratorBase(object):
    """
    Base class for integrators.
    """

    def __init__(self, *arg, **kwargs):
        pass

    def integrate_along_rays(
        self,
        sigma: torch.Tensor,
        radiance: torch.Tensor,
        delta: torch.Tensor,
    ):
        """
        Determines pixel colors given densities, interval length, and radiance values
        obtained along rays.
        """
        raise NotImplementedError()


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    def integrate_along_rays(
        self,
        sigma: torch.Tensor,
        radiance: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Determines pixel colors given densities, interval length, and radiance values
        obtained along rays.

        This method implements the quadrature rule discussed in 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma (torch.Tensor): An instance of torch.Tensor of shape (N, S) representing
                the density at each sample point.
            radiance (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3) representing
                the radiance at each sample point.
            delta (torch.Tensor): An instance of torch.Tensor of shape (N, S) representing the
                difference between adjacent t's.

        Returns:
            An instance of torch.Tensor of shape (N, 3) representing the final pixel
            colors in RGB space.
        """
        sigma_delta = sigma * delta

        # compute transmittance: T_{i}
        transmittance = torch.exp(
            -torch.cumsum(
                torch.cat(
                    [torch.zeros((sigma.shape[0], 1), device=sigma_delta.device), sigma_delta],
                    dim=-1,
                ),
                dim=-1,
            )[..., :-1]
        )

        # compute alpha: (1 - exp (- sigma_{i} * delta_{i}))
        alpha = 1.0 - torch.exp(-sigma_delta)

        # compute numerical integral to determine pixel colors
        # C = sum_{i=1}^{S} T_{i} * alpha_{i} * c_{i}
        rgb = torch.sum(
            transmittance.unsqueeze(-1) * alpha.unsqueeze(-1) * radiance,
            dim=1,
        )

        return rgb
