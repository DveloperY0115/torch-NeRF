"""
train.py

A script for training.
"""

from pathlib import Path
from typing import Dict

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm


from torch_nerf.runners.runner_utils import (
    _init_cuda,
    _init_dataset_and_loader,
    _init_optimizer_and_scheduler,
    _init_renderer,
    _init_scene_repr,
    _init_tensorboard,
    _init_loss_func,
    _load_ckpt,
    _save_ckpt,
    _visualize_scene,
)
import torch_nerf.src.renderer.cameras as cameras


def train_one_epoch(
    cfg,
    default_scene,
    renderer,
    dataset,
    loader,
    loss_func,
    optimizer,
    scheduler,
    fine_scene=None,
) -> Dict[str, torch.Tensor]:
    """
    Routine for one training epoch.

    Args:
        cfg (DictConfig): Configuration.
        default_scene (torch_nerf.src.renderer.scenes.Scene): Default scene.
        renderer (torch_nerf.src.renderer.renderer.Renderer): Renderer.
        dataset (torch_nerf.src.data.dataset.Dataset): Dataset.
        loader (torch.utils.data.DataLoader): Data loader.
        loss_func (torch.nn.modules.loss._Loss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        fine_scene (torch_nerf.src.renderer.scenes.Scene, optional): Fine scene. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: Loss dictionary.
    """

    loss_dict = {}

    for batch in loader:

        loss = 0.0

        # parse batch
        pixel_gt, extrinsic = batch
        pixel_gt = pixel_gt.squeeze()
        pixel_gt = pixel_gt.reshape(-1, 3)  # (H, W, 3) -> (H * W, 3)
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

        # forward prop. default (coarse) network
        coarse_pred, coarse_indices, coarse_weights = renderer.render_scene(
            default_scene,
            num_pixels=cfg.renderer.num_pixels,
            num_samples=cfg.renderer.num_samples_coarse,
            project_to_ndc=cfg.renderer.project_to_ndc,
            device=torch.cuda.current_device(),
        )
        coarse_loss = loss_func(pixel_gt[coarse_indices, ...].cuda(), coarse_pred)
        loss += coarse_loss
        if "coarse_loss" not in loss_dict:
            loss_dict["coarse_loss"] = coarse_loss.item()
        else:
            loss_dict["coarse_loss"] += coarse_loss.item()

        if not fine_scene is None:

            # forward prop. fine network
            fine_pred, fine_indices, _ = renderer.render_scene(
                fine_scene,
                num_pixels=cfg.renderer.num_pixels,
                num_samples=(
                    cfg.renderer.num_samples_coarse,
                    cfg.renderer.num_samples_fine,
                ),
                project_to_ndc=cfg.renderer.project_to_ndc,
                pixel_indices=coarse_indices,  # sample the ray from the same pixels
                weights=coarse_weights,
                device=torch.cuda.current_device(),
            )
            fine_loss = loss_func(pixel_gt[fine_indices, ...].cuda(), fine_pred)
            loss += fine_loss
            if "fine_loss" not in loss_dict:
                loss_dict["fine_loss"] = fine_loss.item()
            else:
                loss_dict["fine_loss"] += fine_loss.item()

        if "loss" not in loss_dict:
            loss_dict["loss"] = loss.item()
        else:
            loss_dict["loss"] += loss.item()

        # back prop.
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # compute average loss over all batches
    for key in loss_dict.keys():
        loss_dict[key] /= len(loader)

    return loss_dict


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    """The entry point of training code."""

    log_dir = Path(HydraConfig.get().runtime.output_dir)
    if "log_dir" in cfg.keys():

        # identify existing log directory
        log_dir = Path(cfg.log_dir)
        assert log_dir.exists(), f"Provided log directory {str(log_dir)} does not exist."

        # override the current config with the existing one
        config_dir = log_dir / ".hydra"
        assert config_dir.exists(), "Provided log directory does not contain config directory."
        cfg = OmegaConf.load(config_dir / "config.yaml")

    # initialize Tensorboard writer
    tb_log_dir = log_dir / "tensorboard"
    writer = _init_tensorboard(tb_log_dir)

    # initialize CUDA device
    _init_cuda(cfg)

    # initialize renderer, data
    renderer = _init_renderer(cfg)
    dataset, loader = _init_dataset_and_loader(cfg)

    # initialize scene and network parameters
    default_scene, fine_scene = _init_scene_repr(cfg)

    # initialize optimizer and learning rate scheduler
    optimizer, scheduler = _init_optimizer_and_scheduler(
        cfg,
        default_scene,
        fine_scene=fine_scene,
    )

    # initialize objective function
    loss_func = _init_loss_func(cfg)

    # load if checkpoint exists
    start_epoch = _load_ckpt(
        log_dir / "ckpt",
        default_scene,
        fine_scene,
        optimizer,
        scheduler,
    )

    for epoch in tqdm(range(start_epoch, cfg.train_params.optim.num_iter // len(dataset))):
        # train
        train_loss_dict = train_one_epoch(
            cfg,
            default_scene,
            renderer,
            dataset,
            loader,
            loss_func,
            optimizer,
            scheduler,
            fine_scene=fine_scene,
        )
        for loss_name, value in train_loss_dict.items():
            writer.add_scalar(f"train/{loss_name}", value, epoch)

        # save checkpoint
        if (epoch + 1) % cfg.train_params.log.epoch_btw_ckpt == 0:
            ckpt_dir = log_dir / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_ckpt(
                ckpt_dir,
                epoch,
                default_scene,
                fine_scene,
                optimizer,
                scheduler,
            )

        # visualize
        if (epoch + 1) % cfg.train_params.log.epoch_btw_vis == 0:
            save_dir = log_dir / f"vis/epoch_{epoch}"
            save_dir.mkdir(parents=True, exist_ok=True)
            _visualize_scene(
                cfg,
                default_scene,
                fine_scene,
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


if __name__ == "__main__":
    main()
