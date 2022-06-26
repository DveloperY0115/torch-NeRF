"""A set of utility functions commonly used in training/testing scripts."""

from typing import Tuple

from omegaconf import DictConfig
import torch
import torch.utils.data as data
import torch_nerf.src.network as network
import torch_nerf.src.scene as scene
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.src.renderer.integrators as integrators
import torch_nerf.src.renderer.ray_samplers as ray_samplers
from torch_nerf.src.renderer.volume_renderer import VolumeRenderer
import torch_nerf.src.signal_encoder.positional_encoder as pe
from torch_nerf.src.utils.data.blender_dataset import NeRFBlenderDataset


def init_cuda(cfg: DictConfig) -> None:
    """
    Checks availability of CUDA devices in the system and set the default device.
    """
    if torch.cuda.is_available():
        device_id = cfg.cuda.device_id

        if device_id > torch.cuda.device_count() - 1:
            print(
                "Invalid device ID. "
                f"There are {torch.cuda.device_count()} devices but got index {device_id}."
            )
            device_id = 0
            cfg.cuda.device_id = device_id  # overwrite config
            print(f"Set device ID to {cfg.cuda.device_id} by default.")
        torch.cuda.set_device(cfg.cuda.device_id)
        print(f"CUDA device detected. Using device {torch.cuda.current_device()}.")
    else:
        print("CUDA is not supported on this system. Using CPU by default.")


def init_dataset_and_loader(
    cfg: DictConfig,
) -> Tuple[data.Dataset, data.DataLoader]:
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
            data_type=cfg.data.data_type,
            half_res=cfg.data.half_res,
            white_bg=cfg.data.white_bg,
        )
    else:
        raise ValueError("Unsupported dataset.")

    loader = data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=4,  # TODO: Adjust dynamically according to cfg.cuda.device_id
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


def init_scene_repr(cfg: DictConfig) -> scene.PrimitiveBase:
    """
    Initializes the scene representation to be trained / tested.

    Load pretrained scene if weights are provided as an argument.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.

    Returns:
        scenes (Dict): A dictionary containing instances of subclasses of QueryStructBase.
            It contains two separate scene representations each associated with
            a key 'coarse' and 'fine', respectively.
    """
    if cfg.scene.type == "cube":
        scene_dict = {}

        # =========================================================
        # initialize 'coarse' scene
        # =========================================================
        coord_enc = pe.PositionalEncoder(
            cfg.network.pos_dim,
            cfg.signal_encoder.coord_encode_level,
            cfg.signal_encoder.include_input,
        )
        dir_enc = pe.PositionalEncoder(
            cfg.network.view_dir_dim,
            cfg.signal_encoder.dir_encode_level,
            cfg.signal_encoder.include_input,
        )

        coarse_network = network.NeRFMLP(
            coord_enc.out_dim,
            dir_enc.out_dim,
        ).to(cfg.cuda.device_id)

        coarse_scene = scene.PrimitiveCube(
            coarse_network,
            {"coord_enc": coord_enc, "dir_enc": dir_enc},
        )

        scene_dict["coarse"] = coarse_scene
        print("Initialized 'coarse' scene.")

        # =========================================================
        # initialize 'fine' scene
        # =========================================================
        if cfg.renderer.num_samples_fine > 0:
            fine_network = network.NeRFMLP(
                coord_enc.out_dim,
                dir_enc.out_dim,
            ).to(cfg.cuda.device_id)

            fine_scene = scene.PrimitiveCube(
                fine_network,
                {"coord_enc": coord_enc, "dir_enc": dir_enc},
            )

            scene_dict["fine"] = fine_scene
            print("Initialized 'fine' scene.")
        else:
            print("Hierarchical sampling disabled. Only 'coarse' scene will be used.")

        return scene_dict
    else:
        raise ValueError("Unsupported scene representation.")


def init_optimizer_and_scheduler(cfg: DictConfig, scenes):
    """
    Initializes the optimizer and learning rate scheduler used for training.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup optimizer and learning rate scheduler.
        scenes (Dict): A dictionary containing neural scene representation(s).

    Returns:
        optimizer ():
        scheduler ():
    """
    if not "coarse" in scenes.keys():
        raise ValueError(
            "At least a coarse representation the scene is required for training. "
            f"Got a dictionary whose keys are {scenes.keys()}."
        )

    optimizer = None
    scheduler = None

    # identify parameters to be optimized
    params = list(scenes["coarse"].radiance_field.parameters())
    if "fine" in scenes.keys():
        params += list(scenes["fine"].radiance_field.parameters())

    # ==============================================================================
    # configure optimizer
    if cfg.train_params.optim.optim_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.train_params.optim.init_lr,
        )  # TODO: A scene may contain two or more networks!
    else:
        raise NotImplementedError()

    # ==============================================================================

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
    else:
        raise NotImplementedError()

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
