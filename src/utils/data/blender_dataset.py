"""
blender_dataset.py - Abstraction on Pytorch dataset.
"""

import typing

import torch
import torch.utils.data as data
from src.utils.data.load_blender import load_blender_data


class NeRFBlenderDataset(data.Dataset):
    """
    Dataset object for loading 'synthetic blender' dataset.

    Args:
        root_dir (str): A string indicating the root directory of the dataset.
    """

    def __init__(self, root_dir: str):
        super().__init__()

        (
            self._imgs,
            self._poses,
            self._render_poses,
            self._camera_params,
            self._i_split,
        ) = load_blender_data(root_dir)

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
        print("Loaded data successfully.")
        print("==============================")

    def __len__(self):
        """Returns the total number of data in the dataset."""
        return self._imgs.shape[0]

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data corresponding to the given index.

        Args:
            index (int): Index of the data to be retrieved.

        Returns:
            A tuple of torch.Tensor instances each representing input images
                and camera extrinsic matrices.
        """
        return self._imgs[index], self._poses[index]

    @property
    def img_height(self):
        """Returns the height of images in the dataset."""
        return self._img_height

    @property
    def img_width(self):
        """Returns the width of images in the dataset."""
        return self._img_width

    @property
    def focal_length(self):
        """Returns the focal length used for rendering images in the dataset."""
        return self._focal_length
