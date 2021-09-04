"""
dataset.py - Abstraction of NeRF dataset
"""

from typing import Tuple

import torch
import torch.utils.data as data


class NeRFDataset(data.Dataset):
    def __init__(
        self,
        imgs: torch.Tensor,
        poses: torch.Tensor,
        camera_params: Tuple[float, float, float, float, float],
    ) -> None:
        """
        Train dataset for NeRF.

        Args:
        - imgs: Tensor of shape () representing images.
        - poses: Tensor of shape () representing camera poses associated with each image in 'imgs'.
        - camera_params: Tuple containing
            - height of image (or viewport)
            - width of image (or viewport)
            - focal length of the camera
            - z_near value
            - z_far value
        """
        super().__init__()

        self.imgs = imgs
        self.poses = poses

        self.camera_params = {}
        self.camera_params["H"] = camera_params[0]
        self.camera_params["W"] = camera_params[1]
        self.camera_params["f"] = camera_params[2]
        self.camera_params["z_near"] = camera_params[3]
        self.camera_params["z_far"] = camera_params[4]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        return self.imgs[index], self.poses[index]

    def get_camera_params(self):
        """
        Get camera parameters used in the current scene.

        Returns:
        - camera_params: Dict containing
            - height of image (or viewport)
            - width of image (or viewport)
            - focal length of the camera
            - z_near value
            - z_far value
        """
        return self.camera_params
