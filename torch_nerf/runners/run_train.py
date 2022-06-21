"""A script for training."""

import os
import sys

sys.path.append(".")
sys.path.append("..")

import hydra
from omegaconf import DictConfig
import torch
import torchvision.utils as tvu
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
        for batch_idx, batch in tqdm(enumerate(loader)):
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
                num_pixels=cfg.renderer.num_pixels,
                num_samples=cfg.renderer.num_samples,
                project_to_ndc=cfg.renderer.project_to_ndc,
                device=torch.cuda.current_device(),
            )

            pixel_gt = pixel_gt.cuda()

            # compute L2 loss
            loss = loss_func(pixel_gt[pixel_indices, ...], pixel_pred)

            # step
            loss.backward()
            optimizer.step()
            if not scheduler is None:
                scheduler.step()
            print(f"Iter {epoch * len(dataset) + batch_idx} Loss: {loss}.")

            # render full image at the end of each epoch
            if batch_idx == len(dataset) - 1:
                render_dir = "renders"
                if not os.path.exists(render_dir):
                    os.mkdir(render_dir)

                with torch.no_grad():
                    pixel_pred, _ = renderer.render_scene(
                        scene,
                        num_pixels=dataset.img_width * dataset.img_height,
                        num_samples=cfg.renderer.num_samples,
                        project_to_ndc=cfg.renderer.project_to_ndc,
                        device=torch.cuda.current_device(),
                    )

                    pixel_pred = torch.reshape(
                        pixel_pred, (dataset.img_height, dataset.img_width, -1)
                    )

                    tvu.save_image(pixel_pred, os.path.join(render_dir, f"epoch_{epoch}.png"))



if __name__ == "__main__":
    main()
