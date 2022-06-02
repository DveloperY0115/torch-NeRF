# torch-NeRF

## Overview

Pytorch implementation of **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** (Mildenhall et al., ECCV 2020 Oral, Best Paper Honorable Mention)

| ![NeRF_Overview](./media/nerf_overview.png) |
|:--:|
|*NeRF Overview.* Figure from the [project page](https://www.matthewtancik.com/nerf) of NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Mildenhall et al., ECCV 2020. |

## TODOs

- [ ] Extend the implementation to [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields (ICCV 2021)](https://arxiv.org/abs/2103.13415).
- [ ] Extend the implementation to [Block-NeRF: Scalable Large Scene Neural View Synthesis (CVPR 2022)](https://waymo.com/research/block-nerf/).
- [ ] Extend the implementation to [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields (CVPR 2022)](https://jonbarron.info/mipnerf360/).
- [ ] Adapt techniques for accelerating training \& inference. The selected candidates are listed below (subject to change):
  - [SNeRG (ICCV 2021)](https://phog.github.io/snerg/)
  - [KiloNeRF (ICCV 2021)](https://arxiv.org/abs/2103.13744)
  - [Instant-NGP (SIGGRAPH 2022)](https://nvlabs.github.io/instant-ngp/)
  - [Point-NeRF (CVPR 2022)](https://xharlie.github.io/projects/project_sites/pointnerf/index.html)
  - [TensoRF (arXiv 2022)](https://arxiv.org/abs/2203.09517)
