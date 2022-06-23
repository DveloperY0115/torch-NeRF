"""
Implementation of Stratified sampler.
"""

import torch
from torch_nerf.src.renderer.ray_samplers.sampler_base import *


class StratifiedSampler(RaySamplerBase):
    """
    Stratified sampler proposed in NeRF (ECCV 2020).
    """

    def sample_along_rays(
        self,
        ray_bundle: RayBundle,
        num_sample: int,
        cdf: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Samples points along rays.

        This method implements the stratified sampling method proposed in 'NeRF: Representing
        Scenes as Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable
        mention)'.

        Specifically, given the nearest and farthest scene bound t_near and t_far,
        the algorithm:
            (1) partitions the interval [t_near, t_far] into num_samples bins of equal length;
            (2) draws one sample from the uniform distribution within each bin;
        For details, please refer to the paper.

        Args:
            ray_bundle (RayBundle): An instance of RayBundle containing ray origin, ray direction,
                the nearest/farthest ray distances.
            num_sample (int): Number of samples sampled along each ray.
            cdf (torch.Tensor): An instance of torch.Tensor of shape (num_sample, 1).
                If provided, the samples are sampled from the distribution represented by it
                using the inverse sampling technique.

        Returns:
            sample_pts (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                3D-coordinate of sample points sampled along rays. Here, N is the number of rays
                in a ray bundle and S is the number of sample points along each ray.
            ray_dir (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                3D-vectors of viewing (ray) directions.
            delta (torch.Tensor): An instance of torch.Tensor of shape (N, S) representing the
                difference between adjacent t's.
        """
        if cdf:
            if not isinstance(cdf, torch.Tensor):
                raise ValueError(f"Expected an instance of torch.Tensor. Got {type(cdf)}.")
            if cdf.shape[0] != num_sample:
                raise ValueError(
                    "Expected the same number of bins from where each sample point is drawn. "
                    f"Got {num_sample}, {cdf.shape[0]}, respectively."
                )

        # equally partition the interval [t_near, t_far]
        t_bins = (
            torch.linspace(
                ray_bundle.t_near,
                ray_bundle.t_far,
                num_sample + 1,
            )[:-1]
            .unsqueeze(0)
            .expand(ray_bundle.ray_origin.shape[0], -1)
        )
        partition_size = (ray_bundle.t_far - ray_bundle.t_near) / num_sample

        if cdf:  # sample from the given distribution
            raise NotImplementedError("TODO: Necessary for implementing hierarchical sampling!")
        else:  # sample from the uniform distribution within each interval
            t_samples = t_bins + partition_size * torch.rand_like(t_bins)

        # compute delta: t_{i+1} - t_{i}
        delta = torch.diff(
            torch.cat([t_samples, 1e8 * torch.ones((t_samples.shape[0], 1))], dim=-1),
            n=1,
            dim=-1,
        )

        # derive coordinates of sample points
        ray_origin = ray_bundle.ray_origin
        ray_origin = ray_origin.unsqueeze(1).repeat((1, num_sample, 1))
        ray_dir = ray_bundle.ray_dir
        ray_dir = ray_dir.unsqueeze(1).repeat((1, num_sample, 1))
        sample_pts = ray_origin + t_samples.unsqueeze(-1) * ray_dir

        return sample_pts, ray_dir, delta
