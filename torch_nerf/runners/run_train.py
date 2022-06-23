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


def load_ckpt(
    ckpt_file,
    scene,
    optimizer,
    scheduler,
) -> int:
    """
    Loads the checkpoint.

    Args:
        scene ():
        optimizer ():
        scheduler ():

    Returns:
        epoch: The epoch from where training continues.
    """
    epoch = 0

    if ckpt_file is None or not os.path.exists(ckpt_file):
        print("Checkpoint file not found.")
        return epoch

    # TODO: Update the code after writing code for checkpointing
    ckpt = torch.load(ckpt_file, map_location="cpu")
    scene.radiance_field.load_state_dict(ckpt)
    print("Radiance field weight loaded.")
    print("TODO: Update codes for checkpointing")
    return epoch


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


def visualize_train_scene(
    cfg,
    scene,
    renderer,
    dataset,
    loader,
    save_dir: str,
):
    """
    Visualizes the scene being trained.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.
        scene (QueryStruct): Neural scene representation to be optimized.
        renderer (VolumeRenderer): Volume renderer used to render the scene.
        dataset (torch.utils.data.Dataset): Dataset for training data.
        save_dir (str): Directory to store render outputs.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    pred_img_dir = os.path.join(save_dir, "pred_imgs")
    if not os.path.exists(pred_img_dir):
        os.mkdir(pred_img_dir)

    render_poses = dataset.render_poses

    with torch.no_grad():
        # for view_idx, extrinsic in tqdm(enumerate(render_poses)):
        for view_idx, batch in tqdm(enumerate(loader)):
            _, extrinsic = batch
            extrinsic = extrinsic.squeeze()

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

            num_total_pixel = dataset.img_width * dataset.img_height
            pixel_pred, _ = renderer.render_scene(
                scene,
                num_pixels=num_total_pixel,
                num_samples=cfg.renderer.num_samples,
                project_to_ndc=cfg.renderer.project_to_ndc,
                device=torch.cuda.current_device(),
                num_ray_batch=num_total_pixel // cfg.renderer.num_pixels,
            )

            # (H * W, C) -> (C, H, W)
            pixel_pred = pixel_pred.reshape(dataset.img_height, dataset.img_width, -1)
            pixel_pred = pixel_pred.permute(2, 0, 1)

            tvu.save_image(
                pixel_pred,
                os.path.join(pred_img_dir, f"{str(view_idx).zfill(5)}.png"),
            )


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """The entry point of train."""
    # identify log directory
    log_dir = HydraConfig.get().runtime.output_dir

    # configure device
    runner_utils.init_cuda(cfg)

    # initialize data, renderer, and scene
    dataset, loader = runner_utils.init_dataset_and_loader(cfg)
    renderer = runner_utils.init_renderer(cfg)
    scene = runner_utils.init_scene_repr(cfg)
    optimizer, scheduler = runner_utils.init_optimizer_and_scheduler(cfg, scene)
    loss_func = runner_utils.init_objective_func(cfg)

    # load if checkpoint exists
    load_ckpt(
        cfg.train_params.ckpt.path,
        scene,
        optimizer,
        scheduler,
    )

    # initialize writer
    writer = SummaryWriter(log_dir=log_dir)

    # train the model
    for epoch in tqdm(range(cfg.train_params.optim.num_iter // len(dataset))):
        # train
        epoch_loss = train_one_epoch(
            cfg, scene, renderer, dataset, loader, loss_func, optimizer, scheduler
        )
        writer.add_scalar("Loss/Train", epoch_loss)

        # log
        if (epoch + 1) % cfg.train_params.log.visualize_every == 0.0:
            save_dir = os.path.join(
                log_dir,
                f"vis/epoch_{epoch}",
            )

            visualize_train_scene(
                cfg,
                scene,
                renderer,
                dataset,
                loader,
                save_dir,
            )

    writer.flush()


if __name__ == "__main__":
    main()
