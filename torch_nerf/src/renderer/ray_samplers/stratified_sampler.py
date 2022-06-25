"""
Implementation of Stratified sampler.
"""

from typing import Tuple, Union

import torch
from torch_nerf.src.renderer.ray_samplers.sampler_base import *
from torch_nerf.src.renderer.ray_samplers.utils import sample_pdf


class StratifiedSampler(RaySamplerBase):
    """
    Stratified sampler proposed in NeRF (ECCV 2020).
    """

    def sample_along_rays(
        self,
        ray_bundle: RayBundle,
        num_samples: Union[int, Tuple[int, int]],
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
            num_samples (int | Tuple[int, int]): Number of samples drawn along each ray.
                (1) a single integer: the number of coarse samples.
                (2) a tuple of integers: the number of coarse and fine samples, respectively.
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
        if not weights is None:  # hierarchical sampling
            if not isinstance(weights, torch.Tensor):
                raise ValueError(f"Expected an instance of torch.Tensor. Got {type(weights)}.")
            if not isinstance(num_samples, (tuple, list)):
                raise ValueError(
                    "Expected a tuple for parameter 'num_samples' when hierarchical sampling is used. "
                    f"Got a parameter of type {type(num_samples)}."
                )

            num_sample_coarse, num_sample_fine = num_samples

            # draw coarse samples
            t_bins, partition_size = self._create_t_bins(
                ray_bundle.t_near,
                ray_bundle.t_far,
                num_sample_coarse,
            )
            t_bins = t_bins.unsqueeze(0)
            t_bins = t_bins.repeat((ray_bundle.ray_origin.shape[0], 1))
            t_samples_coarse = t_bins + partition_size * torch.rand_like(t_bins)

            # draw fine samples
            weights = weights.cpu()  # sampling is done on CPU
            t_samples_fine = sample_pdf(
                t_bins,
                partition_size,
                weights,
                num_sample_fine,
            )
            t_samples, _ = torch.sort(
                torch.cat([t_samples_coarse, t_samples_fine], dim=-1),
                dim=-1,
            )
        else:
            if not isinstance(num_samples, int):
                raise ValueError(
                    "Expected an integer for parameter 'num_samples' when hierarchical sampling is unused. "
                    f"Got a parameter of type {type(num_samples)}."
                )

            # equally partition the interval [t_near, t_far]
            t_bins, partition_size = self._create_t_bins(
                ray_bundle.t_near,
                ray_bundle.t_far,
                num_samples,
            )
            t_bins = t_bins.unsqueeze(0)
            t_bins = t_bins.repeat((ray_bundle.ray_origin.shape[0], 1))

            # sample from the uniform distribution within each interval
            t_samples = t_bins + partition_size * torch.rand_like(t_bins)

        # compute delta: t_{i+1} - t_{i}
        delta = torch.diff(
            torch.cat([t_samples, 1e8 * torch.ones((t_samples.shape[0], 1))], dim=-1),
            n=1,
            dim=-1,
        )

        # derive coordinates of sample points
        ray_origin = ray_bundle.ray_origin
        ray_origin = ray_origin.unsqueeze(1).repeat((1, t_samples.shape[1], 1))
        ray_dir = ray_bundle.ray_dir
        ray_dir = ray_dir.unsqueeze(1).repeat((1, t_samples.shape[1], 1))
        sample_pts = ray_origin + t_samples.unsqueeze(-1) * ray_dir

        return sample_pts, ray_dir, delta

    def _create_t_bins(
        self,
        t_start: float,
        t_end: float,
        num_partitions: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Generates a partition of t's by subdividing the interval [t_near, t_far].

        This method returns a 1D tensor whose elements are:
        [t_near, t_near + a, t_near + 2 * a, ..., t_near + (num_partitions - 1) * a]
        where num_partitions = (t_far - t_near) / num_partitions.

        Args:
            t_start (float): The left endpoint of the interval.
            t_end (float): The right endpoint of the interval.
            num_partitions (int): The number of partitions of equal size
                dividing the given interval.

        Returns:
            t_bins (torch.Tensor): An instance of torch.Tensor of shape (num_samples, ).
                The equally subdivided intervals of t's.
            partition_size (float): The length of each interval.
        """
        t_bins = torch.linspace(
            t_start,
            t_end,
            num_partitions + 1,
        )[:-1]
        partition_size = (t_end - t_start) / num_partitions

        return t_bins, partition_size
