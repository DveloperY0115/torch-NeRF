"""
llff_dataset.py - Abstraction of 'LLFF' dataset.
"""

import os
from typing import Tuple

import torch
import torch.utils.data as data
from torch_nerf.src.utils.data.load_llff import load_llff_data


class LLFFDataset(data.Dataset):
    """
    Dataset object for loading 'LLFF' dataset.

    Attributes:
        root_dir (str): A string indicating the root directory of the dataset.
        dataset_type (str): A string indicating the type of the dataset.
    """

    def __init__(
        self,
        root_dir: str,
        scene_name: str,
        factor: int,
        recenter: bool,
        bd_factor: float,
        spherify: bool,
    ) -> None:
        """
        Constructor of 'LLFFDataset'.

        Args:
            root_dir (str): A string indicating the root directory of the dataset.
            scene_name (str): A string indicating the name of the Blender scene.
            factor (int): A downsample factor for LLFF images.
            recenter (bool): A flag for recentering camera poses around the "central" pose.
            bd_factor (float):
            spherify (bool):
        """
        # check arguments
        scene_names = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
        if not scene_name in scene_names:
            raise ValueError(
                f"Unsupported scene type. Expected one of {scene_names}. Got {scene_name}."
            )
        if not os.path.exists(root_dir):
            raise ValueError(f"The directory {root_dir} does not exist.")

        super().__init__()

        self._root_dir = str(os.path.join(root_dir, scene_name))

        (
            self._imgs,
            self._poses,
            self._camera_params,
            self._z_bounds,
            self._render_poses,
            self._idx_test,
        ) = load_llff_data(
            self._root_dir,
            factor=factor,
            recenter=recenter,
            bd_factor=bd_factor,
            spherify=spherify,
        )

        # np.ndarray -> torch.Tensor
        self._imgs = torch.tensor(self._imgs)
        self._poses = torch.tensor(self._poses)
        self._camera_params = torch.tensor(self._camera_params)
        self._z_bounds = torch.tensor(self._z_bounds)
        self._render_poses = torch.tensor(self._render_poses)

        self._img_height = int(self._camera_params[0, 0])
        self._img_width = int(self._camera_params[0, 1])
        self._focal_length = float(self._camera_params[0, 2])

        if not self._imgs.shape[0] == self._poses.shape[0]:
            raise AssertionError(
                (
                    "Dataset sizes do not match. Got "
                    f"{self._imgs.shape[0]} images and {self._poses.shape[0]} camera poses.",
                )
            )

    def __len__(self) -> int:
        """Returns the total number of data in the dataset."""
        return self._imgs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data corresponding to the given index.

        Args:
            index (int): Index of the data to be retrieved.

        Returns:
            img (torch.Tensor): An instance of torch.Tensor of shape (C, H, W).
                A posed RGB image.
            pose (torch.Tensor): An instance of torch.Tensor of shape (3, 4).
                The camera extrinsics associated with 'img'.
        """
        img = self._imgs[index]
        pose = self._poses[index]

        return img, pose

    @property
    def img_height(self) -> int:
        """Returns the height of images in the dataset."""
        return self._img_height

    @property
    def img_width(self) -> int:
        """Returns the width of images in the dataset."""
        return self._img_width

    @property
    def focal_length(self) -> float:
        """Returns the focal length used for rendering images in the dataset."""
        return self._focal_length

    @property
    def render_poses(self) -> torch.Tensor:
        """Returns the predefined poses to render the scene."""
        return self._render_poses

    @property
    def z_bounds(self) -> torch.Tensor:
        """Returns the depth bounds of the images."""
        return self._z_bounds
