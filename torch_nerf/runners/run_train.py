"""A script for training."""

import sys

sys.path.append(".")
sys.path.append("..")

import torch
import torch.utils.data as data
import src.network as network
import src.renderer.cameras as cameras
import src.renderer.integrators as integrators
import src.renderer.ray_samplers as ray_samplers
from src.renderer.volume_renderer import VolumeRenderer
from src.utils.data.blender_dataset import NeRFBlenderDataset

def main():
    """The entry point of test."""

    # initialize dataset
    root_path = "data/nerf_synthetic/lego"
    dataset = NeRFBlenderDataset(root_path, "train")
    loader = iter(data.DataLoader(dataset, batch_size=1))

    focal_length = dataset.focal_length
    img_height = dataset.img_height
    img_width = dataset.img_width

    # initialize model & query structure
    nerf_mlp = network.nerf_mlp.NeRFMLP(pos_dim, view_dir_dim)

    # load image and camera extrinsic
    rgb_gt, extrinsic = next(loader)
    rgb_gt = rgb_gt.squeeze()  # [B, *] -> [*]
    extrinsic = extrinsic.squeeze()  # [B, *] -> [*]

    # initialize renderer
    camera = cameras.CameraBase(
        {
            "f_x": focal_length,
            "f_y": focal_length,
            "img_width": img_width,
            "img_height": img_height,
        },
        extrinsic,
        2.0,
        6.0,
    )
    integrator = integrators.QuadratureIntegrator()
    sampler = ray_samplers.StratifiedSampler()
    renderer = VolumeRenderer(camera, integrator, sampler)

    renderer.render_scene(
        ,
        num_pixels=1024,
        num_samples=128,
        project_to_ndc=False,
    )


if __name__ == "__main__":
    main()
