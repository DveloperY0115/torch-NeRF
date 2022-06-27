"""A script for training."""

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


def train_one_epoch(
    cfg,
    scenes,
    renderer,
    dataset,
    loader,
    loss_func,
    optimizer,
    scheduler=None,
) -> float:
    """
    Trains the scene for one epoch.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.
        scene (QueryStruct): Neural scene representation to be optimized.
        renderer (VolumeRenderer): Volume renderer used to render the scene.
        dataset (torch.utils.data.Dataset): Dataset for training data.
        loader (torch.utils.data.DataLoader): Loader for training data.
        loss_func (torch.nn.Module): Objective function to be optimized.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler.ExponentialLR): Learning rate scheduler.
            Set to None by default.

    Returns:
        total_loss (float): The average of losses computed over an epoch.
    """
    if not "coarse" in scenes.keys():
        raise ValueError(
            "At least a coarse representation the scene is required for training. "
            f"Got a dictionary whose keys are {scenes.keys()}."
        )

    total_loss = 0.0
    total_coarse_loss = 0.0
    total_fine_loss = 0.0

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

        # forward prop. coarse network
        coarse_pred, coarse_indices, coarse_weights = renderer.render_scene(
            scenes["coarse"],
            num_pixels=cfg.renderer.num_pixels,
            num_samples=cfg.renderer.num_samples_coarse,
            project_to_ndc=cfg.renderer.project_to_ndc,
            device=torch.cuda.current_device(),
        )
        loss = loss_func(pixel_gt[coarse_indices, ...].cuda(), coarse_pred)
        total_coarse_loss += loss.item()

        # forward prop. fine network
        if "fine" in scenes.keys():
            fine_pred, fine_indices, _ = renderer.render_scene(
                scenes["fine"],
                num_pixels=cfg.renderer.num_pixels,
                num_samples=(cfg.renderer.num_samples_coarse, cfg.renderer.num_samples_fine),
                project_to_ndc=cfg.renderer.project_to_ndc,
                pixel_indices=coarse_indices,  # sample the ray from the same pixels
                weights=coarse_weights,
                device=torch.cuda.current_device(),
            )
            fine_loss = loss_func(pixel_gt[fine_indices, ...].cuda(), fine_pred)
            total_fine_loss += fine_loss.item()
            loss += fine_loss

        total_loss += loss.item()

        # step
        loss.backward()
        optimizer.step()
        if not scheduler is None:
            scheduler.step()

    # compute average loss
    total_loss /= len(loader)
    total_coarse_loss /= len(loader)
    total_fine_loss /= len(loader)

    return {
        "total_loss": total_loss,
        "total_coarse_loss": total_coarse_loss,
        "total_fine_loss": total_fine_loss,
    }


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """The entry point of training code."""
    # identify log directory
    log_dir = HydraConfig.get().runtime.output_dir

    # configure device
    runner_utils.init_cuda(cfg)

    # initialize data, renderer, and scene
    dataset, loader = runner_utils.init_dataset_and_loader(cfg)
    renderer = runner_utils.init_renderer(cfg)
    scenes = runner_utils.init_scene_repr(cfg)
    optimizer, scheduler = runner_utils.init_optimizer_and_scheduler(cfg, scenes)
    loss_func = runner_utils.init_objective_func(cfg)

    # load if checkpoint exists
    start_epoch = runner_utils.load_ckpt(
        cfg.train_params.ckpt.path,
        scenes,
        optimizer,
        scheduler,
    )

    # initialize writer
    tb_log_dir = os.path.join(log_dir, "tensorboard")
    if not os.path.exists(tb_log_dir):
        os.mkdir(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # train the model
    for epoch in tqdm(range(start_epoch, cfg.train_params.optim.num_iter // len(dataset))):
        # train
        losses = train_one_epoch(
            cfg, scenes, renderer, dataset, loader, loss_func, optimizer, scheduler
        )
        for loss_name, value in losses.items():
            writer.add_scalar(f"Loss/{loss_name}", value, epoch)

        # save checkpoint
        if (epoch + 1) % cfg.train_params.log.epoch_btw_ckpt == 0:
            ckpt_dir = os.path.join(log_dir, "ckpt")

            runner_utils.save_ckpt(
                ckpt_dir,
                epoch,
                scenes,
                optimizer,
                scheduler,
            )

        # visualize
        if (epoch + 1) % cfg.train_params.log.epoch_btw_vis == 0:
            save_dir = os.path.join(
                log_dir,
                f"vis/epoch_{epoch}",
            )

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
                save_dir=save_dir,
                num_imgs=1,
            )

    writer.flush()


if __name__ == "__main__":
    main()
