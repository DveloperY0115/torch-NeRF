"""
blender_dataset.py - Abstraction on Pytorch dataset.
"""

from json import load
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .load_blender import load_blender_data


class BlenderDataset(data.Dataset):
    def __init__(self, root_dir: str) -> None:
        """
        Abstract dataset object for 'synthetic blender' dataset.

        Args:
        - root_dir: Root directory of dataset to be loaded.
        """
        super().__init__()

        (
            self.imgs,
            self.poses,
            self.render_poses,
            self.camera_params,
            self.i_split,
        ) = load_blender_data(root_dir)

        self.img_height = self.camera_params[0]
        self.img_width = self.camera_params[1]
        self.focal_length = self.camera_params[2]

        assert self.imgs.shape[0] == self.poses.shape[0], "[!] Dataset sizes do not match."

        print("==============================")
        print("[!] Loaded data successfully.")
        print("==============================")

    def __len__(self):
        return self.imgs[0]

    def __getitem__(self, index):
        return self.imgs[index], self.poses[index]
