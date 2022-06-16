"""A test script for 'VolumeRenderer'."""

import sys

sys.path.append(".")
sys.path.append("..")

import torch
import src.renderer.cameras as cameras
import src.renderer.integrators as integrators
import src.renderer.ray_samplers as ray_samplers
from src.renderer.volume_renderer import VolumeRenderer


def main():
    """The entry point of test."""

    # initialize renderer
    camera = cameras.CameraBase(
        torch.eye(4),  # dummy intrinsic
        torch.eye(4),  # dummy extrinsic
    )
    integrator = integrators.QuadratureIntegrator()
    sampler = ray_samplers.RaySamplerBase()
    img_res = (800, 800)
    renderer = VolumeRenderer(camera, integrator, sampler, img_res)

    print(renderer.screen_coords.shape)


if __name__ == "__main__":
    main()
