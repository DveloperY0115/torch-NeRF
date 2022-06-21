"""A set of utility functions commonly used in training/testing scripts."""

import typing

from omegaconf import DictConfig
import torch.utils.data as data
import torch_nerf.src.network as network
import torch_nerf.src.query_struct as qs
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.src.renderer.integrators as integrators
import torch_nerf.src.renderer.ray_samplers as ray_samplers
from torch_nerf.src.renderer.volume_renderer import VolumeRenderer
from torch_nerf.src.utils.data.blender_dataset import NeRFBlenderDataset


def init_dataset_and_loader(
    cfg: DictConfig,
) -> typing.Tuple[data.Dataset, data.DataLoader]:
    """
    Initialize the dataset and loader.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup dataset and loader.

    Returns:
        dataset (torch.utils.data.Dataset): Dataset object.
        loader (torch.utils.data.DataLoader): DataLoader object.
    """
    # identify dataset type
    if cfg.data.dataset_type == "nerf_synthetic":
        dataset = NeRFBlenderDataset(cfg.data.data_root, cfg.data.data_type)
    else:
        raise ValueError("Unsupported dataset.")

    # create a loader
    loader = data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
    )

    return dataset, loader


def init_renderer(config):
    """
    Initialize the renderer for rendering scene representations.

    Args:

    """
    return


def init_scene_repr(cfg: DictConfig) -> qs.QueryStructBase:
    """
    Initialize the scene representation to be trained / tested.

    Load pretrained scene if weights are provided as an argument.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.

    Returns:
        scene (QueryStruct): An instance of derived class of QueryStructBase.
            The scene representation.
    """
    # initialize query structure & radiance fields
    if cfg.query_struct.type == "cube":
        radiance_field = network.nerf_mlp.NeRFMLP(
            cfg.network.coord_encode_level,
            cfg.network.dir_encode_level,
        )
        scene = qs.simple_cube.QSCube(radiance_field)
    else:
        raise ValueError("Unsupported scene representation.")

    return scene
