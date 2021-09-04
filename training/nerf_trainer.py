from argparse import Namespace
from typing import Tuple
from utils.render_utils import render
import wandb

import torch
import torch.optim as optim
import torch.utils.data as data

from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCGridRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)

from model.nerf_cls import NeRFCls
from utils.dataset import NeRFDataset
from utils.load_blender import load_blender_data
from .base_trainer import BaseTrainer


class NeRFTrainer(BaseTrainer):
    def __init__(self, opts: Namespace, checkpoint: str = None):
        super().__init__(opts=opts)

        # construct neural radiance field
        self.model = NeRFCls(pos_dim=3, view_dir_dim=3, L_pos=10, L_direction=4)
        self.model = self.model.to(self.device)

        # optimizer & learning rate scheduler
        self.optimizer = self.configure_optimizer()
        self.lr_scheduler = self.configure_lr_scheduler()

        # dataset & data loader
        self.train_dataset, self.valid_dataset, self.test_dataset = self.configure_dataset()
        self.train_loader, self.valid_loader, self.test_loader = self.configure_dataloader()

        # load checkpoint if available
        if checkpoint is not None:
            if self.load_checkpoint(checkpoint):
                print("[!] Successfully loaded checkpoint at {}".format(checkpoint))
            else:
                print("[!] Failed to load checkpoint at {}".format(checkpoint))
        else:
            print("[!] Start without checkpoint")

        # initialize W&B if requested
        if self.opts.log_wandb:
            wandb.init(project="NeRF-{}-{}".format(str(type(self.model)), self.opts.dataset_type))

    def train(self):
        for self.epoch in range(self.initial_epoch, self.opts.num_epoch):
            _ = self.train_one_epoch()
            _, self.validate_one_epoch()

            print("=======================================")
            print("Epoch {}".format(self.epoch))
            # print("Training loss: {}".format(train_loss))
            # print("Test loss: {}".format(test_loss))
            print("=======================================")
        
            if self.opts.log_wandb:
                # wandb.log(something)
                pass

            if (self.epcoh + 1) % self.opts.save_period == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        pass

    def validate_one_epoch(self):
        pass

    def initialize_renderer(
        self,
    ) -> Tuple[pytorch3d.renderer.ImplicitRenderer, pytorch3d.renderer.ImplicitRenderer]:
        """
        Create renderer used for retrieving images from neural radiance fields.

        Returns:
        - renderer_grid: Renderer using NDCGridRaysampler as ray sampler.
        - renderer_mc: Renderer using MonteCarloRaysampler as ray sampler.
        """
        # Number of pixels of rendered images along each dimension.
        # For better quality, the renderer will first render double resolution image
        # and then resize the image to fit target size (supersampling).
        render_size = self.train_dataset.get_camera_params()["H"] * 2

        # The object is assumed to lie at the world origin, (0, 0, 0).
        # Therefore, the radiance field will only be defined within finite volume.
        volume_extent_world = 3.0

        # NDCGridRaysampler generates a rectangular image grid
        # of rays whose coordinates follow the convention of Pytorch3D.
        raysampler_grid = NDCGridRaysampler(
            image_height=render_size,
            image_width=render_size,
            n_pts_per_ray=128,
            min_depth=0.1,
            max_depth=volume_extent_world,
        )

        # MonteCarloRaysampler generates a random subset of 'n_rays_per_image'
        # rays emitted from the image plane.
        raysampler_mc = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=750,
            n_pts_per_ray=128,
            min_depth=0.1,
            max_depth=volume_extent_world,
        )

        # EmissionAbsorptionRaymarcher is used to render
        # the rays into a single 3D color vector in RGB space
        # and an opacity scalar (corresponds to density).
        raymarcher = EmissionAbsorptionRaymarcher()

        # Create renderers
        renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)
        renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)

        return renderer_grid, renderer_mc

    def configure_optimizer(self) -> torch.optim.Optimizer:
        optimizer = optim.Adam(self.model.parameters(), lr=self.opts.lr)
        return optimizer

    def configure_lr_scheduler(self) -> torch.optim.lr_scheduler:
        return None

    def configure_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

        if self.opts.dataset_type == "Blender":
            imgs, poses, render_poses, (H, W, focal), idx_split = load_blender_data(
                self.opts.dataset_dir
            )
            idx_train, idx_val, idx_test = idx_split

            # set z_near, z_far
            z_near = 2.0
            z_far = 6.0

            # split data
            train_imgs, valid_imgs, test_imgs = imgs[idx_train], imgs[idx_val], imgs[idx_test]
            train_poses, valid_poses, test_poses = poses[idx_train], poses[idx_val], poses[idx_test]

            train_dataset = NeRFDataset(
                train_imgs, train_poses, camera_params=[int(H), int(W), focal, z_near, z_far]
            )
            valid_dataset = NeRFDataset(
                valid_imgs, valid_poses, camera_params=[int(H), int(W), focal, z_near, z_far]
            )
            test_dataset = NeRFDataset(
                test_imgs, test_poses, camera_params=[int(H), int(W), focal, z_near, z_far]
            )
        else:
            # TODO: Support other kinds of datasets
            pass

        print("[!] Successfully loaded dataset at: {}".format(self.opts.dataset_dir))
        print("[!] Dataset used: {}".format(self.opts.dataset_type))

        return train_dataset, valid_dataset, test_dataset

    def configure_dataloader(
        self,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_loader = data.DataLoader(
            self.train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=self.opts.num_workers,
        )

        valid_loader = data.DataLoader(
            self.valid_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=self.opts.num_workers,
        )

        test_loader = data.DataLoader(
            self.test_dataset, batch_size=self.opts.batch_size, shuffle=False
        )

        return train_loader, valid_loader, test_loader
