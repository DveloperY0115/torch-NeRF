"""A set of utility functions commonly used in training/testing scripts."""

import typing

from omegaconf import DictConfig
import torch
import torch.utils.data as data
import torch_nerf.src.network as network
import torch_nerf.src.query_struct as qs
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.src.renderer.integrators as integrators
import torch_nerf.src.renderer.ray_samplers as ray_samplers
from torch_nerf.src.renderer.volume_renderer import VolumeRenderer
import torch_nerf.src.signal_encoder.positional_encoder as pe
from torch_nerf.src.utils.data.blender_dataset import NeRFBlenderDataset


def init_dataset_and_loader(
    cfg: DictConfig,
) -> typing.Tuple[data.Dataset, data.DataLoader]:
    """
    Initializes the dataset and loader.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup dataset and loader.

    Returns:
        dataset (torch.utils.data.Dataset): Dataset object.
        loader (torch.utils.data.DataLoader): DataLoader object.
    """
    if cfg.data.dataset_type == "nerf_synthetic":
        dataset = NeRFBlenderDataset(
            cfg.data.data_root,
            cfg.data.data_type,
            cfg.data.white_bg,
        )
    else:
        raise ValueError("Unsupported dataset.")

    loader = data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
    )

    return dataset, loader


def init_renderer(cfg: DictConfig):
    """
    Initializes the renderer for rendering scene representations.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup renderer.
    """
    integrator = None
    sampler = None

    # TODO: Separate below lines to functions
    if cfg.renderer.integrator_type == "quadrature":
        integrator = integrators.QuadratureIntegrator()
    else:
        raise ValueError("Unsupported integrator type.")

    if cfg.renderer.sampler_type == "stratified":
        sampler = ray_samplers.StratifiedSampler()
    else:
        raise ValueError("Unsupported ray sampler type.")

    renderer = VolumeRenderer(integrator, sampler)

    return renderer


def init_scene_repr(cfg: DictConfig) -> qs.QueryStructBase:
    """
    Initializes the scene representation to be trained / tested.

    Load pretrained scene if weights are provided as an argument.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.

    Returns:
        scene (QueryStruct): An instance of derived class of QueryStructBase.
            The scene representation.
    """
    if cfg.query_struct.type == "cube":
        radiance_field = network.NeRFMLP(
            2 * cfg.signal_encoder.coord_encode_level * cfg.network.pos_dim,
            2 * cfg.signal_encoder.dir_encode_level * cfg.network.view_dir_dim,
        )
        coord_enc = pe.PositionalEncoder(
            cfg.network.pos_dim,
            cfg.signal_encoder.coord_encode_level,
        )
        dir_enc = pe.PositionalEncoder(
            cfg.network.view_dir_dim,
            cfg.signal_encoder.dir_encode_level,
        )
        scene = qs.QSCube(
            radiance_field,
            {"coord_enc": coord_enc, "dir_enc": dir_enc},
        )

        return scene
    else:
        raise ValueError("Unsupported scene representation.")


def init_optimizer_and_scheduler(cfg: DictConfig, scene):
    """
    Initializes the optimizer and learning rate scheduler used for training.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup optimizer and learning rate scheduler.
        scene (QueryStruct): A scene representation holding the neural network(s)
            to be optimized.

    Returns:

    """
    optimizer = None
    scheduler = None

    if cfg.train_params.optim.optim_type == "adam":
        optimizer = torch.optim.Adam(
            scene.radiance_field.parameters(),
            lr=cfg.train_params.optim.init_lr,
        )  # TODO: A scene may contain two or more networks!

    if cfg.train_params.optim.scheduler_type == "exp":
        # compute decay rate
        init_lr = cfg.train_params.optim.init_lr
        end_lr = cfg.train_params.optim.end_lr
        num_iter = cfg.train_params.optim.num_iter
        gamma = pow(end_lr / init_lr, 1 / num_iter)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma,
        )

    return optimizer, scheduler


def init_objective_func(cfg: DictConfig) -> torch.nn.Module:
    """
    Initializes objective functions used to train neural radiance fields.

    Args:
        cfg (DictConfig): A config object holding flags required to construct
            objective functions.

    Returns:
        loss (torch.nn.Module): An instance of torch.nn.Module.
            Module that evaluates the value of objective function.
    """
    if cfg.objective.loss_type == "nerf_default":
        return torch.nn.MSELoss()
    else:
        raise ValueError("Unsupported loss configuration.")
