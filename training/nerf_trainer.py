from argparse import Namespace
from typing import Tuple
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
        self.train_dataset, self.test_dataset = self.configure_dataset()
        self.train_loader, self.test_loader = self.configure_dataloader()

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
        pass

    def train_one_epoch(self):
        pass

    def test_one_epoch(self):
        pass

    def configure_optimizer(self) -> torch.optim.Optimizer:
        optimizer = optim.Adam(self.model.parameters(), lr=self.opts.lr)
        return optimizer

    def configure_lr_scheduler(self) -> torch.optim.lr_scheduler:
        return None

    def configure_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

        if self.opts.dataset_type == "Blender":
            images, poses, render_poses, (H, W, focal), idx_split = load_blender_data(
                self.opts.dataset_dir
            )
            idx_train, idx_val, idx_test = idx_split

            near = 2.0
            far = 6.0
        else:
            # TODO: Support other kinds of datasets
            pass

        print("[!] Successfully loaded dataset at: {}".format(self.opts.dataset_dir))
        print("[!] Dataset used: {}".format(self.opts.dataset_type))

        return images, poses, render_poses, (H, W, focal), idx_train, idx_val, idx_test

    def configure_dataloader(
        self,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_loader = data.DataLoader(
            self.train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=self.opts.num_workers,
        )

        test_loader = data.DataLoader(
            self.test_dataset, batch_size=self.opts.test_dataset_size, shuffle=False
        )

        return train_loader, test_loader
