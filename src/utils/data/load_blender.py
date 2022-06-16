"""
load_blender.py - Utility for loading blender scenes.
"""

import json
import os
from typing import Tuple, List

import cv2
import imageio
import numpy as np
import torch


trans_t = lambda t: torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]]
).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(
    basedir: str, half_res: bool = False, testskip: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float], List[List]]:
    """
    Load 'synthetic blender' data.

    Args:
    - basedir: Root directory of dataset to be loaded.
    - half_res: Determines whether to halve the size of images or not. Set to 'False' by default.
    - testskip: ??

    Returns:
    - imgs: Tensor of shape (B, W, H, 4) representing dataset of RGBA images.
    - poses: Tensor of shape (B, 4, 4) representing camera-to-world transformation
        (i.e., matrix describing the camera's orientation and translation).
    - render_poses:  Tensor of shape (40, 4, 4) representing camera poses used for rendering.
    - [H, W, focal]: Image height, image width, and focal length, respectively.
    - i_split: List of lists each containing indices of training, validation, and test data, respectively.
    """
    splits = ["train", "val", "test"]
    metas = {}

    for s in splits:
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
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

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])  # horizontal field of view (FOV)
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack(
        [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0
    )

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split
