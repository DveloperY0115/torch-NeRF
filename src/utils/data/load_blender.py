"""
load_blender.py - Utility for loading blender scenes.
"""

import json
import os
import typing

import cv2
import imageio
import numpy as np
import torch


def translate_along_z_by(trans: float) -> torch.Tensor:
    """
    Creates the Affine transformation that translates a point along z-axis.

    Args:
        trans (float): Translation offset along z-axis.

    Returns:
        A torch.Tensor instance of shape (4, 4) representing an Affine matrix.
    """
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, trans],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rotate_around_x_by(phi: float) -> torch.Tensor:
    """
    Creates the Affine transformation that rotates a point around x-axis.

    Args:
        phi (float): Rotation angle in degree.

    Returns:
        A torch.Tensor instance of shape (4, 4) representing an Affine matrix.
    """
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rotate_around_y_by(theta: float):
    """
    Returns the Affine transformation that rotates a point around y-axis.

    Args:
        theta (float): Rotation angle in degree.

    Returns:
        A torch.Tensor instance of shape (4, 4) representing an Affine matrix.
    """
    return torch.Tensor(
        [
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(
    theta: float,
    phi: float,
    radius: float,
) -> torch.Tensor:
    """
    Creates the camera extrinsic matrix from the given spherical coordinate.

    Args:
        theta (float): Rotation angle in degree.
        phi (float): Rotation angle in degree.
        radius (float):

    Returns:
        A torch.Tensor instance of shape (4, 4) representing a camera extrinsic matrix.
    """
    camera_to_world = translate_along_z_by(radius)
    camera_to_world = rotate_around_x_by(phi / 180.0 * np.pi) @ camera_to_world
    camera_to_world = rotate_around_y_by(theta / 180.0 * np.pi) @ camera_to_world
    camera_to_world = (
        torch.Tensor(
            [
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
        )
        @ camera_to_world
    )
    return camera_to_world


def load_blender_data(
    base_dir: str,
    half_res: bool = False,
    test_idx_skip: int = 1,
) -> typing.Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    typing.List[float],
    typing.Dict[typing.List],
]:
    """
    Load 'synthetic blender' data.

    Args:
        base_dir (str): Root directory of dataset to be loaded.
        half_res (bool): Determines whether to halve the size of images or not.
            Set to 'False' by default.
        test_idx_skip (int): Step size used for skipping some test data.

    Returns:
        imgs (torch.Tensor): Tensor of shape (B, W, H, 4).
            Dataset of RGBA images.
        poses (torch.Tensor): Tensor of shape (B, 4, 4).
            Camera extrinsic matrices in camera-to-world format.
        render_poses (torch.Tensor):  Tensor of shape (40, 4, 4).
            Camera extrinsics used for rendering.
        intrinsic_params (List of int): List containing image height, width, and focal length.
        i_split (Dict of List): Dictionary of lists each containing indices
            of training, validation, and test data.
    """
    splits = ["train", "val", "test"]
    metas = {}

    for split in splits:
        with open(os.path.join(base_dir, f"transforms_{split}.json"), "r") as pose_file:
            metas[split] = json.load(pose_file)

    all_imgs = []
    all_poses = []
    counts = [0]

    for split in splits:
        meta = metas[split]
        imgs = []
        poses = []
        if split == "train" or test_idx_skip == 0:
            skip = 1  # do not skip any test
        else:
            skip = test_idx_skip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(base_dir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])  # accumulate the number of images read
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    # camera intrinsics
    img_height, img_width = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])  # horizontal field of view (FOV)
    focal = 0.5 * img_width / np.tan(0.5 * camera_angle_x)
    intrinsic_params = [img_height, img_width, focal]

    # camera extrinsics for image rendering
    render_poses = torch.stack(
        [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0
    )

    if half_res:
        img_height = img_height // 2
        img_width = img_width // 2
        focal = focal / 2.0

        imgs_half_res = np.zeros((imgs.shape[0], img_height, img_width, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(
                img, (img_width, img_height), interpolation=cv2.INTER_AREA
            )
        imgs = imgs_half_res
\
    return imgs, poses, render_poses, intrinsic_params, i_split
