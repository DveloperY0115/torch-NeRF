"""
render.py

A script for rendering.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.utils.data as data
import torchvision.utils as tvu
from tqdm import tqdm

from torch_nerf.runners.runner_utils import (
    _init_cuda,
    _init_dataset_and_loader,
    _init_optimizer_and_scheduler,
    _init_renderer,
    _init_scene_repr,
    _load_ckpt,
)

import torch_nerf.src.scene as scene
import torch_nerf.src.renderer.cameras as cameras
from torch_nerf.src.renderer.volume_renderer import VolumeRenderer
from torch_nerf.src.utils.data.blender_dataset import NeRFBlenderDataset
from torch_nerf.src.utils.data.llff_dataset import LLFFDataset


def init_dataset(cfg: DictConfig) -> data.Dataset:
    """
    Initializes the dataset.
    """
    dataset_type = str(cfg.data.dataset_type)

    if dataset_type == "nerf_synthetic":
        dataset = NeRFBlenderDataset(
            cfg.data.data_root,
            scene_name=cfg.data.scene_name,
            data_type="test",
            half_res=cfg.data.half_res,
            white_bg=cfg.data.white_bg,
        )
    elif dataset_type == "nerf_llff":
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")
    elif dataset_type == "nerf_deepvoxels":
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")

    return dataset

@torch.no_grad()
def render_scene(
    cfg: DictConfig,
    default_scene: scene.Scene,
    fine_scene: scene.Scene,
    renderer: VolumeRenderer,
    intrinsic: Union[Dict, torch.Tensor],
    extrinsic: torch.Tensor,
    img_res: Tuple[int, int],
):
    """
    Renders the scene from the specified viewpoint.
    """
    renderer.camera = cameras.PerspectiveCamera(
        intrinsic,
        extrinsic,
        cfg.renderer.t_near,
        cfg.renderer.t_far,
    )

    img_height, img_width = img_res
    num_total_pixel = img_height * img_width

    rendered_img, default_indices, default_weights = renderer.render_scene(
        default_scene,
        num_pixels=num_total_pixel,
        num_samples=cfg.renderer.num_samples_coarse,
        project_to_ndc=cfg.renderer.project_to_ndc,
        device=torch.cuda.current_device(),
        num_ray_batch=num_total_pixel // cfg.renderer.num_pixels,
    )
    if not fine_scene is None:
        rendered_img, _, _ = renderer.render_scene(
            fine_scene,
            num_pixels=num_total_pixel,
            num_samples=(cfg.renderer.num_samples_coarse, cfg.renderer.num_samples_fine),
            project_to_ndc=cfg.renderer.project_to_ndc,
            pixel_indices=default_indices,
            weights=default_weights,
            device=torch.cuda.current_device(),
            num_ray_batch=num_total_pixel // cfg.renderer.num_pixels,
        )

    # (H * W, C) -> (C, H, W)
    rendered_img = rendered_img.reshape(img_height, img_width, -1)
    rendered_img = rendered_img.permute(2, 0, 1)

    rendered_img = torch.clamp(rendered_img, 0.0, 1.0)

    return rendered_img


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    """The entry point of rendering code."""
    
    assert "log_dir" in cfg.keys(), "'log_dir' must be specified in config file."
    log_dir = Path(cfg.log_dir)
    assert log_dir.exists(), f"Provided log directory {str(log_dir)} does not exist."

    # check if the current run is for rendering testing (evaluation)
    render_test_views = cfg.get("render_test_views", False)

    # override the current config with the existing one
    config_dir = log_dir / ".hydra"
    assert config_dir.exists(), "Provided log directory does not contain config directory."
    cfg = OmegaConf.load(config_dir / "config.yaml")

    # initialize CUDA device
    _init_cuda(cfg)

    # initialize renderer, data
    renderer = _init_renderer(cfg)

    # initialize dataset
    dataset = init_dataset(cfg)

    # initialize scene and network parameters
    default_scene, fine_scene = _init_scene_repr(cfg)

    # initialize optimizer and learning rate scheduler
    optimizer, scheduler = _init_optimizer_and_scheduler(
        cfg,
        default_scene,
        fine_scene=fine_scene,
    )

    # find the latest checkpoint
    ckpt_dir = log_dir / "ckpt"
    assert ckpt_dir.exists(), (
        f"Checkpoint directory {str(ckpt_dir)} does not exist."
    )

    # load checkpoint
    _ = _load_ckpt(
        ckpt_dir,
        default_scene,
        fine_scene,
        optimizer,
        scheduler,
    )

    # create directory to save rendering outputs
    render_dir = log_dir / "render"
    render_dir.mkdir(exist_ok=True, parents=True)
    save_dir = render_dir / "video"
    if render_test_views:
        save_dir = render_dir / "test_views"
    save_dir.mkdir(exist_ok=True, parents=True)
    print(
        f"Rendering outputs will be saved under: {str(save_dir)}"
    )

    poses = dataset._render_poses
    image_fnames = [
        str(i).zfill(6) for i in range(len(poses))
    ]
    if render_test_views:
        poses = dataset._poses
        image_fnames = dataset._image_fnames
    for view_idx, pose in tqdm(enumerate(poses)):

        # render
        rendered_img = render_scene(
          cfg,
          default_scene,
          fine_scene,
          renderer,
          intrinsic={
            "f_x": dataset.focal_length,
            "f_y": dataset.focal_length,
            "img_width": dataset.img_width,
            "img_height": dataset.img_height,  
          },
          extrinsic=pose,
          img_res=(dataset.img_height, dataset.img_width),
        )

        # save
        tvu.save_image(
            rendered_img,
            str(save_dir / f"{image_fnames[view_idx]}.png"),
        )


if __name__ == "__main__":
    main()
