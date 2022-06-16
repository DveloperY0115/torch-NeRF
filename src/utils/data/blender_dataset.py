"""
blender_dataset.py - Abstraction on Pytorch dataset.
"""

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
        """Returns the total number of data in the dataset."""
        return self.imgs.shape[0]

    def __getitem__(self, index):
        """Returns the data corresponding to the given index."""
        return self.imgs[index], self.poses[index]
