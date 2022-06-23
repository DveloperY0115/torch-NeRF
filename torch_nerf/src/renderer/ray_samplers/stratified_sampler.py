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
        weights: torch.Tensor = None,
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
            weights (torch.Tensor): An instance of torch.Tensor of shape (num_ray, num_sample).
                If provided, the samples are sampled using the inverse sampling technique
                from the distribution represented by the CDF derived from it.

        Returns:
            sample_pts (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                3D-coordinate of sample points sampled along rays. Here, N is the number of rays
                in a ray bundle and S is the number of sample points along each ray.
            ray_dir (torch.Tensor): An instance of torch.Tensor of shape (N, S, 3).
                3D-vectors of viewing (ray) directions.
            delta (torch.Tensor): An instance of torch.Tensor of shape (N, S) representing the
                difference between adjacent t's.
        """
        if not weights is None:
            if not isinstance(weights, torch.Tensor):
                raise ValueError(f"Expected an instance of torch.Tensor. Got {type(weights)}.")

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

        if not weights is None:  # sample from the given distribution
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

    def _create_t_bins(
        self,
        t_near: float,
        t_far: float,
        num_samples: int,
        num_rays: int,
    ) -> typing.Tuple[torch.Tensor, float]:
        """
        Generates a partition of t's.

        Args:
            t_start (float):
            t_end (float):
            num_samples (int):
            num_rays (int):

        Returns:
            t_bins (torch.Tensor): An instance of torch.Tensor of shape (num_rays, num_samples).
                The equally subdivided intervals of t's.
            partition_size (float): The length of each interval.
        """
        t_bins = (
            torch.linspace(
                t_near,
                t_far,
                num_samples + 1,
            )[:-1]
            .unsqueeze(0)
            .expand(num_rays, -1)
        )
        partition_size = (t_far - t_near) / num_samples

        return t_bins, partition_size
