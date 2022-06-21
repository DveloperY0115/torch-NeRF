"""A script for training."""

import sys

sys.path.append(".")
sys.path.append("..")

import cv2
import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.runners.runner_utils as runner_utils


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """The entry point of train."""

    # initialize data, renderer, and scene
    dataset, loader = runner_utils.init_dataset_and_loader(cfg)
    renderer = runner_utils.init_renderer(cfg)
    scene = runner_utils.init_scene_repr(cfg)
    optimizer, scheduler = runner_utils.init_optimizer_and_scheduler(cfg, scene)
    loss_func = runner_utils.init_objective_func(cfg)

    # train the model
    for batch in tqdm(loader):
        pixel_gt, extrinsic = batch
        pixel_gt = pixel_gt.squeeze()
        pixel_gt = torch.reshape(pixel_gt, (-1, 3))  # (H, W, 3) -> (H * W, 3)
        extrinsic = extrinsic.squeeze()

        # initialize gradients to zero
        optimizer.zero_grad()

        # set the camera
        camera = cameras.CameraBase(
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
        renderer.camera = camera

        pixel_pred, pixel_indices = renderer.render_scene(
            scene,
            num_pixels=1024,
            num_samples=128,
            project_to_ndc=False,
        )

        # compute L2 loss
        loss = loss_func(pixel_gt[pixel_indices, ...], pixel_pred)

        # step
        loss.backward()
        optimizer.step()
        if not scheduler is None:
            scheduler.step()


if __name__ == "__main__":
    main()
