"""
blender_dataset.py - Abstraction on Pytorch dataset.
"""

from typing import Tuple

import torch
import torch.utils.data as data
from torch_nerf.src.utils.data.load_blender import load_blender_data


class NeRFBlenderDataset(data.Dataset):
    """
    Dataset object for loading 'synthetic blender' dataset.

    Attributes:
        root_dir (str): A string indicating the root directory of the dataset.
        dataset_type (str): A string indicating the type of the dataset.
    """

    def __init__(
        self,
        root_dir: str,
        data_type: str,
        half_res: bool,
        white_bg: bool = True,
    ):
        """
        Constructor of 'NeRFBlenderDataset'.

        Args:
            root_dir (str): A string indicating the root directory of the dataset.
            data_type (str): A string indicating the type of the dataset.
            half_res (bool): A flag that determines whether to half the image resolution.
            white_bg (bool): A flag that determines whether to make background of images white.
        """
        # check arguments
        data_types = ["train", "val", "test"]
        if not data_type in data_types:
            raise ValueError(
                f"Unsupported dataset type. Expected one of {data_types}. Got {data_type}"
            )

        super().__init__()

        self._root_dir = root_dir
        self._data_type = data_type
        self._white_bg = white_bg

        (
            self._imgs,
            self._poses,
            self._camera_params,
            self._render_poses,
        ) = load_blender_data(root_dir, data_type, half_res=half_res)

        self._img_height = self._camera_params[0]
        self._img_width = self._camera_params[1]
        self._focal_length = self._camera_params[2]

        if not self._imgs.shape[0] == self._poses.shape[0]:
            raise AssertionError(
                (
                    "Dataset sizes do not match. Got "
                    f"{self._imgs.shape[0]} images and {self._poses.shape[0]} camera poses.",
                )
            )

        print("==============================")
        print("Loaded dataset successfully.")
        print(f"Number of training data: {self._imgs.shape[0]}")
        print(f"Image resolution: ({self._img_height}, {self.img_width})")
        print(f"Focal length(s): ({self._focal_length}, {self._focal_length})")
        print("==============================")

    def __len__(self):
        """Returns the total number of data in the dataset."""
        return self._imgs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data corresponding to the given index.

        Args:
            index (int): Index of the data to be retrieved.

        Returns:
            A tuple of torch.Tensor instances each representing input RGB images
                and camera extrinsic matrices.
        """
        img = torch.tensor(self._imgs[index])
        pose = torch.tensor(self._poses[index])

        if self._white_bg:
            alpha = img[..., -1]
            img[alpha == 0.0, :] = 1.0
        return img[..., :-1], pose

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
