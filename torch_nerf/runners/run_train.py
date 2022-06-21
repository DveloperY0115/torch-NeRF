"""A script for training."""

import sys

sys.path.append(".")
sys.path.append("..")

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.utils.data as data
import torch_nerf.src.network as network
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.src.renderer.integrators as integrators
import torch_nerf.src.renderer.ray_samplers as ray_samplers
from torch_nerf.src.renderer.volume_renderer import VolumeRenderer
from torch_nerf.src.utils.data.blender_dataset import NeRFBlenderDataset


@hydra.main(config_path="torch_nerf/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """The entry point of train."""
    print(OmegaConf.to_yaml((cfg)))
    return

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
        None,
        num_pixels=1024,
        num_samples=128,
        project_to_ndc=False,
    )


if __name__ == "__main__":
    main()
