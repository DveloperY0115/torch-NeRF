"""
Utilities related to sampling.
"""

import torch


def sample_pdf(
    bins: torch.Tensor,
    partition_size: float,
    weights: torch.Tensor,
    num_sample: int,
) -> torch.Tensor:
    """
    Draws samples from the probability density represented by given weights.

    Args:
        bins (torch.Tensor): An instance of torch.Tensor of shape (N, S).
            The start and end values of intervals on which probability masses
            are defined.
        partition_size (float): The length of each bin.
        partition_size (float):
        weights (torch.Tensor): An instance of torch.Tensor of shape (N, S).
            An unnormalized PDF is represented as a vector of shape (S,).
        num_sample (int): Number of samples to be drawn from the distribution.

    Returns:

    """
    # construct the PDFs
    weights += 1e-5
    normalizer = torch.sum(weights, dim=-1, keepdim=True)
    pdf = weights / normalizer

    # compute CDFs
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(
        [torch.zeros((cdf.shape[0], 1)), cdf[..., :-1]],
        dim=-1,
    )

    # sample from the uniform distribution: U[0, 1)
    cdf_ys = torch.rand((cdf.shape[0], num_sample))

    # heuristically sample from the distribution
    cdf_ys = cdf_ys.contiguous()
    t_indices = (
        torch.searchsorted(
            cdf,
            cdf_ys,
            right=True,
        )
        - 1
    )
    t_start = torch.gather(bins, 1, t_indices)
    t_samples = t_start + partition_size * torch.rand_like(t_start)

    return t_samples
