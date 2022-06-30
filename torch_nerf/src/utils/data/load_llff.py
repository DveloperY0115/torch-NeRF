"""
Utility functions for loading LLFF data.

The code was brought from the official implementation of NeRF (ECCV 2020)
(https://github.com/bmild/nerf/blob/master/load_llff.py) and slightly modified 
in terms of naming conventions, documentations, etc.
"""

import os
from typing import List, Optional, Tuple

import imageio
import numpy as np
from shutil import copy
from subprocess import check_output


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(base_dir, "images")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
            resizearg = "{}%".format(100.0 / r)
        else:
            name = "images_{}x{}".format(r[1], r[0])
            resizearg = "{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print("Minifying", r, basedir)

        os.makedirs(imgdir)
        check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split(".")[-1]
        args = " ".join(["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != "png":
            check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
            print("Removed duplicates")
        print("Done")


def _load_data(
    base_dir: str,
    factor: Optional[int] = None,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads camera parameters, scene bounds, and images.

    Args:
        base_dir (str): A string indicating the base directory of the dataset.
        factor (int):
        img_width (int): The desired width of the output images.
            Set to None by default.
        img_height (int): The desired height of the output images.
            Set to None by default.

    Returns:
        imgs (np.ndarray): An instance of np.ndarray of shape ().

        extrinsics (np.ndarray): An instance of np.ndarray of shape ().

        intrinsics (np.ndarray): An instance of np.ndarray of shape ().

        z_bounds (np.ndarray): An instance of np.ndarray of shape ().

    """
    # load the camera parameters and scene z-bounds
    poses_raw = np.load(os.path.join(base_dir, "poses_bounds.npy"))
    camera_params = (
        poses_raw[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    )  # (N, 15) -> (3, 5, N)
    z_bounds = poses_raw[:, -2:].transpose([1, 0])  # (N, 2) -> (2, N)

    # parse extrinsics and intrinsics
    extrinsics = camera_params[:, :-1, :]  # (3, 4, N)
    intrinsics = camera_params[:, -1, :]  # (3, N)

    img0 = [
        os.path.join(base_dir, "images", f)
        for f in sorted(os.listdir(os.path.join(base_dir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    img_shape = imageio.imread(img0).shape

    suffix = ""

    # resize the images if requested
    if factor is not None:
        suffix = f"_{factor}"
        _minify(base_dir, factors=[factor])
        factor = factor
    elif img_height is not None:
        factor = img_shape[0] / float(img_height)
        img_width = int(img_shape[1] / factor)
        _minify(base_dir, resolutions=[[img_height, img_width]])
        suffix = f"_{img_width}x{img_height}"
    elif img_width is not None:
        factor = img_shape[1] / float(img_width)
        img_height = int(img_shape[0] / factor)
        _minify(base_dir, resolutions=[[img_height, img_width]])
        suffix = f"_{img_width}x{img_height}"
    else:
        factor = 1

    # check whether the image directory exists
    img_dir = os.path.join(base_dir, "images" + suffix)
    if not os.path.exists(img_dir):
        raise ValueError(f"The base directory of dataset {img_dir} does not exist.")

    # identify files to be read
    img_files = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    if camera_params.shape[-1] != len(img_files):
        raise ValueError(
            f"Mismatch between imgs {len(img_files)} and poses {camera_params.shape[-1]}."
        )

    # update the intrinsics (the size of images has probably changed)
    img_shape = imageio.imread(img_files[0]).shape
    intrinsics[:2, :] = np.array(img_shape[:2]).reshape([2, 1])  # update height, width
    intrinsics[2, :] *= 1.0 / factor  # update focal length

    # Correct rotation matrix ordering and move variable dim to axis 0
    # Please refer to the issue for details: https://github.com/bmild/nerf/issues/34
    extrinsics = np.concatenate(
        [extrinsics[:, 1:2, :], -extrinsics[:, 0:1, :], extrinsics[:, 2:, :]],
        axis=1,
    )

    # load images
    imgs = [imread(file)[..., :3] / 255.0 for file in img_files]
    imgs = np.stack(imgs, axis=-1)

    # swap the ordering of axes - (*, N) -> (N, *)
    imgs = np.moveaxis(imgs, source=-1, destination=0).astype(np.float32)
    extrinsics = np.moveaxis(extrinsics, source=-1, destination=0).astype(np.float32)
    intrinsics = np.moveaxis(intrinsics, source=-1, destination=0).astype(np.float32)
    z_bounds = np.moveaxis(z_bounds, source=-1, destination=0).astype(np.float32)

    return imgs, extrinsics, intrinsics, z_bounds


def imread(img_file: str) -> np.ndarray:
    """
    A simple wrapper around imageio.imread.

    Args:
        img_file (str): A name of the image file to be loaded.

    Returns:
        An instance of np.ndarray of shape (H, W, C).
            The array representing the loaded image.
    """
    if img_file.endswith("png"):
        return imageio.imread(img_file, ignoregamma=True)
    else:
        return imageio.imread(img_file)


def normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normalizes the given vector.

    Args:
        vec (np.ndarray): An instance of np.ndarray of shape ().

    Returns:
        normalized (np.ndarray): An instance of np.ndarray of shape ().
            The unit vector whose direction is same as the input vector but
            with its L2 norm 1.
    """
    normalized = vec / np.linalg.norm(vec)
    return normalized


def build_extrinsic(
    z_vec: np.ndarray,
    up_vec: np.ndarray,
    camera_position: np.ndarray,
) -> np.ndarray:
    """
    Constructs the camera extrinsic matrix given the z-axis basis,
    up vector, and the coordinate of the camera in the world frame.

    Args:
        z_vec (np.ndarray): An instance of np.ndarray of shape ().

        up_vec (np.ndarray): An instance of np.ndarray of shape ().

        camera_position (np.ndarray): An instance of np.ndarray of shape ().

    Returns:
        extrinsic (np.ndarray): An instance of np.ndarray of shape ().

    """
    z_vec = normalize(z_vec)
    x_vec = normalize(np.cross(up_vec, z_vec))
    y_vec = normalize(np.cross(z_vec, x_vec))
    extrinsic = np.stack([x_vec, y_vec, z_vec, camera_position], 1)
    return extrinsic


def world_to_camera(coord_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """
    Computes the camera frame coordinates of 3D points given their coordinates
    in the world frame.

    Args:
        coord_world (np.ndarray): An instance of np.ndarray of shape ().
        camera_to_world (np.ndarray): An instance of np.ndarray of shape ().

    Returns:
        coord_camera (np.ndarray): An instance of np.ndarray of shape ().

    """
    coord_camera = np.matmul(
        camera_to_world[:3, :3].T,  # orthonormal, inverse of 'camera_to_world'
        (coord_world - camera_to_world[:3, 3])[..., np.newaxis],
    )[..., 0]

    return coord_camera


def poses_avg(poses: np.ndarray) -> np.ndarray:
    """
    Computes the "central" pose of the given dataset.

    For detailed motivation behind this design decision, please
    refer to the following issues:
        (1) https://github.com/bmild/nerf/issues/18
        (2) https://github.com/bmild/nerf/issues/34

    Args:
        poses (np.ndarray): An instance of np.ndarray of shape (*, 3, 4).
            The camera poses associated with the images of a scene.

    Returns:
        avg_camera_to_world (np.ndarray): An instance of np.ndarray of shape (3, 4).
            The array holding the average camera pose matrix and additional data.
    """
    mean_position = poses[:, :3, 3].mean(axis=0)
    mean_z = normalize(poses[:, :3, 2].sum(axis=0))
    mean_y = poses[:, :3, 1].sum(axis=0)  # regard mean y-axis as "up" vector
    avg_camera_to_world = build_extrinsic(
        mean_z,
        mean_y,
        mean_position,
    )

    return avg_camera_to_world


def render_path_spiral(
    camera_to_world: np.ndarray,
    up_vec: np.ndarray,
    radiuses: np.ndarray,
    focal: float,
    z_rate: float,
    rots: int,
    N: int,
) -> np.ndarray:
    """
    Computes the series of camera poses that consititutes the spiral-like
    trajectory. The poses are used for rendering novel views.

    Args:
        camera_to_world (np.ndarray): An instance of np.ndarray of shape ().
        up_vec (np.ndarray): An instance of np.ndarray of shape ().
        rads (np.ndarray): An instance of np.ndarray of shape ().
        focal (float):
        z_rate (float): The rate of change of displacement along z-axis.
        rots (int): Number of rotations around the spiral axis.
        N (int): Number of key frame positions.

    Returns:
        render_poses (np.ndarray): An instance of np.ndarray of shape ().
            The consecutive camera poses constituting the spiral trajectory.
    """
    render_poses = []
    radiuses = np.array(list(radiuses) + [1.0])
    hwf = camera_to_world[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        camera_position = np.dot(
            camera_to_world[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1.0]) * radiuses,
        )
        z_vec = normalize(
            camera_position - np.dot(camera_to_world[:3, :4], np.array([0, 0, -focal, 1.0]))
        )
        render_poses.append(
            np.concatenate([build_extrinsic(z_vec, up_vec, camera_position), hwf], 1)
        )
    return render_poses


def recenter_poses(poses: np.ndarray) -> np.ndarray:
    """
    Recenter poses with respect to their "central" pose.

    Args:
        poses (np.ndarray): An instance of np.ndarray of shape (N, 3, 4).
            Camera extrinsic matrices represented in the form of Affine matrices.
    Returns:
        poses (np.ndarray): An instance of np.ndarray of shape (N, 3, 4).
            The camera poses adjusted according to their statistics (i.e., the central pose).
    """
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    camera_to_world = poses_avg(poses)
    camera_to_world = np.concatenate([camera_to_world, bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(camera_to_world) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):

    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)],
        -1,
    )

    return poses_reset, new_poses, bds


def load_llff_data(
    basedir, factor=8, recenter=True, bd_factor=0.75, spherify=False, path_zflat=False
):

    poses, bds, imgs = _load_data(
        basedir, factor=factor
    )  # factor=8 downsamples original imgs by 8x
    print("Loaded", basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        print("recentered", c2w.shape)
        print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = 0.8
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(
            c2w_path,
            up,
            rads,
            focal,
            z_rate=0.5,
            rots=N_rots,
            N=N_views,
        )

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print("Data:")
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print("HOLDOUT view is", i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test
