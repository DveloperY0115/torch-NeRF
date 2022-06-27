"""A script for scene rendering."""

import os
import sys

sys.path.append(".")
sys.path.append("..")

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as tvu
from tqdm import tqdm
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.runners.runner_utils as runner_utils


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """The entry point of rendering code."""
    log_dir = os.path.join("render_out", cfg.data.dataset_type, cfg.data.scene_name)

    # configure device
    runner_utils.init_cuda(cfg)

    # initialize data, renderer, and scene
    dataset, _ = runner_utils.init_dataset_and_loader(cfg)
    renderer = runner_utils.init_renderer(cfg)
    scenes = runner_utils.init_scene_repr(cfg)

    if cfg.train_params.ckpt.path is None:
        raise ValueError("Checkpoint file must be provided for rendering.")
    if not os.path.exists(cfg.train_params.ckpt.path):
        raise ValueError("Checkpoint file does not exist.")

    # load scene representation
    _ = runner_utils.load_ckpt(
        cfg.train_params.ckpt.path,
        scenes,
        optimizer=None,
        scheduler=None,
    )

    # render
    runner_utils.visualize_scene(
        cfg,
        scenes,
        renderer,
        intrinsics={
            "f_x": dataset.focal_length,
            "f_y": dataset.focal_length,
            "img_width": dataset.img_width,
            "img_height": dataset.img_height,
        },
        extrinsics=dataset.render_poses,
        img_res=(dataset.img_height, dataset.img_width),
        save_dir=log_dir,
    )

    print("Rendering done.")


if __name__ == "__main__":
    main()
