"""A script for training."""

import os
import sys

sys.path.append(".")
sys.path.append("..")

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torchvision.utils as tvu
from tqdm import tqdm
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.runners.runner_utils as runner_utils


def train_one_epoch(
    cfg,
    scene,
    renderer,
    dataset,
    loader,
    loss_func,
    optimizer,
    scheduler,
) -> float:
    """
    Training routine for one epoch.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.
        scene (QueryStruct): Neural scene representation to be optimized.
        renderer (VolumeRenderer): Volume renderer used to render the scene.
        loader (torch.utils.data.DataLoader): Loader for training data.
        loss_func (torch.nn.Module): Objective function to be optimized.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler.ExponentialLR): Learning rate scheduler.

    Returns:
        total_loss (float): The average of losses computed over an epoch.
    """
    total_loss = 0.0

    for batch in loader:
        pixel_gt, extrinsic = batch
        pixel_gt = pixel_gt.squeeze()
        pixel_gt = torch.reshape(pixel_gt, (-1, 3))  # (H, W, 3) -> (H * W, 3)
        extrinsic = extrinsic.squeeze()

        # initialize gradients
        optimizer.zero_grad()

        # set the camera
        renderer.camera = cameras.PerspectiveCamera(
            {
                "f_x": dataset.focal_length,
                "f_y": dataset.focal_length,
                "img_width": dataset.img_width,
                "img_height": dataset.img_height,
            },
            extrinsic,
            cfg.renderer.t_near,
            cfg.renderer.t_far,
        )

        pixel_pred, pixel_indices = renderer.render_scene(
            scene,
            num_pixels=cfg.renderer.num_pixels,
            num_samples=cfg.renderer.num_samples,
            project_to_ndc=cfg.renderer.project_to_ndc,
            device=torch.cuda.current_device(),
        )

        # compute L2 loss
        loss = loss_func(pixel_gt[pixel_indices, ...].cuda(), pixel_pred)
        total_loss += loss.item()

        # step
        loss.backward()
        optimizer.step()
        if not scheduler is None:
            scheduler.step()

    total_loss /= len(loader)

    return total_loss


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """The entry point of train."""
    # configure device
    runner_utils.init_cuda(cfg)

    # initialize data, renderer, and scene
    dataset, loader = runner_utils.init_dataset_and_loader(cfg)
    renderer = runner_utils.init_renderer(cfg)
    scene = runner_utils.init_scene_repr(cfg)
    optimizer, scheduler = runner_utils.init_optimizer_and_scheduler(cfg, scene)
    loss_func = runner_utils.init_objective_func(cfg)

    # train the model
    for epoch in tqdm(range(cfg.train_params.optim.num_iter // len(dataset))):
        epoch_loss = train_one_epoch(
            cfg, scene, renderer, dataset, loader, loss_func, optimizer, scheduler
        )

        print(f"Loss {epoch}: {epoch_loss}")


if __name__ == "__main__":
    main()
