"""A set of utility functions commonly used in training/testing scripts."""

import functools
import os
from typing import Callable, Dict, Optional, Tuple, Union

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as tvu
from tqdm import tqdm
import torch_nerf.src.network as network
import torch_nerf.src.scene as scene
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.src.renderer.integrators.quadrature_integrator as integrators
import torch_nerf.src.renderer.ray_samplers as ray_samplers
from torch_nerf.src.renderer.volume_renderer import VolumeRenderer
from torch_nerf.src.signal_encoder import PositionalEncoder, SHEncoder
from torch_nerf.src.utils.data.blender_dataset import NeRFBlenderDataset
from torch_nerf.src.utils.data.llff_dataset import LLFFDataset


def init_session(cfg: DictConfig, mode: str) -> Callable:
    """
    Initializes the current session and returns its entry point.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup the session.
        mode (str): A string indicating the type of current session.
            Can be one of "train", "render".

    Returns:
        run_session (Callable): A function that serves as the entry point for
            the current (training, validation, or visualization) session.
    """
    if not mode in ("train", "render"):
        raise ValueError(f"Unsupported mode. Expected one of 'train', 'render'. Got {mode}.")

    # identify log directories
    log_dir = HydraConfig.get().runtime.output_dir
    tb_log_dir = os.path.join(log_dir, "tensorboard")

    # initialize Tensorboard writer
    writer = _init_tensorboard(tb_log_dir)

    # initialize CUDA device
    _init_cuda(cfg)

    # initialize renderer
    renderer = _init_renderer(cfg)

    # initialize dataset and loaders
    dataset, loader = _init_dataset_and_loader(cfg)

    # initialize scene
    default_scene, fine_scene = _init_scene_repr(cfg)

    # initialize optimizer and learning rate scheduler
    optimizer, scheduler = _init_optimizer_and_scheduler(
        cfg,
        default_scene,
        fine_scene=fine_scene,
    )

    # initialize objective function
    loss_func = _init_loss_func(cfg)

    # load if checkpoint exists
    start_epoch = _load_ckpt(
        cfg.train_params.ckpt.path,
        default_scene,
        fine_scene,
        optimizer,
        scheduler,
    )

    # build train, validation, and visualization routine
    # with their parameters binded
    train_one_epoch = _build_train_routine(
        cfg,
        default_scene,
        fine_scene,
        renderer,
        dataset,
        loader,
        loss_func,
        optimizer,
        scheduler,
    )
    validate_one_epoch = _build_validation_routine(cfg)
    visualize = _build_visualization_routine(
        cfg,
        default_scene,
        fine_scene,
        renderer,
    )

    if mode == "train":

        def run_session():
            for epoch in tqdm(range(start_epoch, cfg.train_params.optim.num_iter // len(dataset))):
                # train
                train_losses = train_one_epoch()
                for loss_name, value in train_losses.items():
                    writer.add_scalar(f"Train_Loss/{loss_name}", value, epoch)

                # validate
                if not validate_one_epoch is None:
                    valid_losses = validate_one_epoch()
                    for loss_name, value in valid_losses.items():
                        writer.add_scalar(f"Validation_Loss/{loss_name}", value, epoch)

                # save checkpoint
                if (epoch + 1) % cfg.train_params.log.epoch_btw_ckpt == 0:
                    ckpt_dir = os.path.join(log_dir, "ckpt")
                    _save_ckpt(
                        ckpt_dir,
                        epoch,
                        default_scene,
                        fine_scene,
                        optimizer,
                        scheduler,
                    )

                # visualize
                if (epoch + 1) % cfg.train_params.log.epoch_btw_vis == 0:
                    save_dir = os.path.join(
                        log_dir,
                        f"vis/epoch_{epoch}",
                    )
                    visualize(
                        intrinsics={
                            "f_x": dataset.focal_length,
                            "f_y": dataset.focal_length,
                            "img_width": dataset.img_width,
                            "img_height": dataset.img_height,
                        },
                        extrinsics=dataset.render_poses,
                        img_res=(dataset.img_height, dataset.img_width),
                        save_dir=save_dir,
                        num_imgs=1,
                    )

    else:  # render

        def run_session():
            save_dir = os.path.join(
                "render_out",
                cfg.data.dataset_type,
                cfg.data.scene_name,
            )
            visualize(
                intrinsics={
                    "f_x": dataset.focal_length,
                    "f_y": dataset.focal_length,
                    "img_width": dataset.img_width,
                    "img_height": dataset.img_height,
                },
                extrinsics=dataset.render_poses,
                img_res=(dataset.img_height, dataset.img_width),
                save_dir=save_dir,
            )

    return run_session


def _build_train_routine(
    cfg: DictConfig,
    default_scene: scene.Scene,
    fine_scene: scene.Scene,
    renderer: VolumeRenderer,
    dataset: data.Dataset,
    loader: data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object = None,
) -> Callable:
    """
    Builds per epoch training routine.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.
        default_scene (scene.scene): A default scene representation to be optimized.
        fine_scene (scene.scene): A fine scene representation to be optimized.
            This representation is only used when hierarchical sampling is used.
        renderer (VolumeRenderer): Volume renderer used to render the scene.
        dataset (torch.utils.data.Dataset): Dataset for training data.
        loader (torch.utils.data.DataLoader): Loader for training data.
        loss_func (torch.nn.Module): Objective function to be optimized.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler.ExponentialLR): Learning rate scheduler.
            Set to None by default.

    Returns:
        train_one_epoch (functools.partial): A function that trains a neural scene representation
            for one epoch.
    """
    # resolve training configuration
    use_hierarchical_sampling = not fine_scene is None

    # TODO: Any more sophisticated way of modularizing this?

    if not use_hierarchical_sampling:

        def train_one_epoch(
            cfg,
            default_scene,
            renderer,
            dataset,
            loader,
            loss_func,
            optimizer,
            scheduler,
        ) -> Dict[str, torch.Tensor]:
            total_loss = 0.0

            for batch in loader:
                # parse batch
                pixel_gt, extrinsic = batch
                pixel_gt = pixel_gt.squeeze()
                pixel_gt = torch.reshape(pixel_gt, (-1, 3))  # (H, W, 3) -> (H * W, 3)
                extrinsic = extrinsic.squeeze()

                # initialize gradients
                optimizer.zero_grad()

                # set the camera
                renderer.camera = cameras.PerspectiveCamera(
                    {
                        "f_x": dataset.focal_length,
                        "f_y": dataset.focal_length,
                        "img_width": dataset.img_width,
                        "img_height": dataset.img_height,
                    },
                    extrinsic,
                    cfg.renderer.t_near,
                    cfg.renderer.t_far,
                )

                # forward prop.
                pred, indices, _ = renderer.render_scene(
                    default_scene,
                    num_pixels=cfg.renderer.num_pixels,
                    num_samples=cfg.renderer.num_samples_coarse,
                    project_to_ndc=cfg.renderer.project_to_ndc,
                    device=torch.cuda.current_device(),
                )

                loss = loss_func(pixel_gt[indices, ...].cuda(), pred)
                total_loss += loss.item()

                # step
                loss.backward()
                optimizer.step()
                if not scheduler is None:
                    scheduler.step()

            # compute average loss
            total_loss /= len(loader)

            return {
                "total_loss": total_loss,
            }

        return functools.partial(
            train_one_epoch,
            cfg,
            default_scene,
            renderer,
            dataset,
            loader,
            loss_func,
            optimizer,
            scheduler,
        )
    else:

        def train_one_epoch(
            cfg,
            default_scene,
            fine_scene,
            renderer,
            dataset,
            loader,
            loss_func,
            optimizer,
            scheduler,
        ) -> Dict[str, torch.Tensor]:
            total_loss = 0.0
            total_default_loss = 0.0
            total_fine_loss = 0.0

            for batch in loader:
                # parse batch
                pixel_gt, extrinsic = batch
                pixel_gt = pixel_gt.squeeze()
                pixel_gt = torch.reshape(pixel_gt, (-1, 3))  # (H, W, 3) -> (H * W, 3)
                extrinsic = extrinsic.squeeze()

                # initialize gradients
                optimizer.zero_grad()

                # set the camera
                renderer.camera = cameras.PerspectiveCamera(
                    {
                        "f_x": dataset.focal_length,
                        "f_y": dataset.focal_length,
                        "img_width": dataset.img_width,
                        "img_height": dataset.img_height,
                    },
                    extrinsic,
                    cfg.renderer.t_near,
                    cfg.renderer.t_far,
                )

                # forward prop. default (coarse) network
                default_pred, default_indices, default_weights = renderer.render_scene(
                    default_scene,
                    num_pixels=cfg.renderer.num_pixels,
                    num_samples=cfg.renderer.num_samples_coarse,
                    project_to_ndc=cfg.renderer.project_to_ndc,
                    device=torch.cuda.current_device(),
                )
                loss = loss_func(pixel_gt[default_indices, ...].cuda(), default_pred)
                total_default_loss += loss.item()

                # forward prop. fine network
                if not fine_scene is None:
                    fine_pred, fine_indices, _ = renderer.render_scene(
                        fine_scene,
                        num_pixels=cfg.renderer.num_pixels,
                        num_samples=(
                            cfg.renderer.num_samples_coarse,
                            cfg.renderer.num_samples_fine,
                        ),
                        project_to_ndc=cfg.renderer.project_to_ndc,
                        pixel_indices=default_indices,  # sample the ray from the same pixels
                        weights=default_weights,
                        device=torch.cuda.current_device(),
                    )
                    fine_loss = loss_func(pixel_gt[fine_indices, ...].cuda(), fine_pred)
                    total_fine_loss += fine_loss.item()
                    loss += fine_loss

                total_loss += loss.item()

                # step
                loss.backward()
                optimizer.step()
                if not scheduler is None:
                    scheduler.step()

            # compute average loss
            total_loss /= len(loader)
            total_default_loss /= len(loader)
            total_fine_loss /= len(loader)

            return {
                "total_loss": total_loss,
                "total_default_loss": total_default_loss,
                "total_fine_loss": total_fine_loss,
            }

        return functools.partial(
            train_one_epoch,
            cfg,
            default_scene,
            fine_scene,
            renderer,
            dataset,
            loader,
            loss_func,
            optimizer,
            scheduler,
        )


def _build_validation_routine(cfg) -> Callable:
    """ """
    return None


def _build_visualization_routine(
    cfg,
    default_scene: scene.Scene,
    fine_scene: scene.Scene,
    renderer: VolumeRenderer,
) -> Callable:
    """
    Builds per epoch visualization routine.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.
        default_scene (scene.scene): A default scene representation to be optimized.
        fine_scene (scene.scene): A fine scene representation to be optimized.
            This representation is only used when hierarchical sampling is used.
        renderer (VolumeRenderer): Volume renderer used to render the scene.

    Returns:
        visualize_scene (functools.partial): A function that visualizes a neural scene representation.
    """
    visualize_scene = functools.partial(
        _visualize_scene,
        cfg,
        default_scene,
        fine_scene,
        renderer,
    )

    return visualize_scene


def _init_cuda(cfg: DictConfig) -> None:
    """
    Checks availability of CUDA devices in the system and set the default device.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to configure CUDA devices.
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


def _init_dataset_and_loader(
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
            scene_name=cfg.data.scene_name,
            data_type=cfg.data.data_type,
            half_res=cfg.data.half_res,
            white_bg=cfg.data.white_bg,
        )
    elif cfg.data.dataset_type == "nerf_llff":
        dataset = LLFFDataset(
            cfg.data.data_root,
            scene_name=cfg.data.scene_name,
            factor=cfg.data.factor,
            recenter=cfg.data.recenter,
            bd_factor=cfg.data.bd_factor,
            spherify=cfg.data.spherify,
        )

        # update the near and far bounds
        if cfg.renderer.project_to_ndc:
            cfg.renderer.t_near = 0.0
            cfg.renderer.t_far = 1.0
            print(
                "Using NDC projection for LLFF scene. "
                f"Set (t_near, t_far) to ({cfg.renderer.t_near}, {cfg.renderer.t_far})."
            )
        else:
            cfg.renderer.t_near = float(torch.min(dataset.z_bounds) * 0.9)
            cfg.renderer.t_far = float(torch.max(dataset.z_bounds) * 1.0)
            print(
                "Proceeding without NDC projection. "
                f"Set (t_near, t_far) to ({cfg.renderer.t_near}, {cfg.renderer.t_far})."
            )
    elif cfg.data.dataset_type == "nerf_deepvoxels":
        raise NotImplementedError()
    else:
        raise ValueError("Unsupported dataset.")

    print("===========================================")
    print("Loaded dataset successfully.")
    print(f"Dataset type / Scene name: {cfg.data.dataset_type} / {cfg.data.scene_name}")
    print(f"Number of training data: {len(dataset)}")
    print(f"Image resolution: ({dataset.img_height}, {dataset.img_width})")
    print(f"Focal length(s): ({dataset.focal_length}, {dataset.focal_length})")
    print("===========================================")

    loader = data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=4,  # TODO: Adjust dynamically according to cfg.cuda.device_id
    )

    return dataset, loader


def _init_renderer(cfg: DictConfig) -> VolumeRenderer:
    """
    Initializes the renderer for rendering scene representations.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup renderer.

    Returns:
        renderer (VolumeRenderer): A differentiable volume renderer with
            the specified ray sampler and numerical integrator.
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


def _init_tensorboard(tb_log_dir: str) -> SummaryWriter:
    """
    Initializes tensorboard writer.

    Args:
        tb_log_dir (str): A directory where Tensorboard logs will be saved.

    Returns:
        writer (SummaryWriter): A writer (handle) for logging data.
    """
    if not os.path.exists(tb_log_dir):
        os.mkdir(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)
    return writer


def _init_scene_repr(cfg: DictConfig) -> Tuple[scene.Scene, Optional[scene.Scene]]:
    """
    Initializes the scene representation to be trained / tested.

    Load pretrained scene if weights are provided as an argument.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.

    Returns:
        default_scene (scene.Scene): A scene representation used by default.
        fine_scene (scene.Scene): An additional scene representation used with
            hierarchical sampling strategy.
    """
    if cfg.signal_encoder.type == "pe":
        coord_enc = PositionalEncoder(
            cfg.network.pos_dim,
            cfg.signal_encoder.coord_encode_level,
            cfg.signal_encoder.include_input,
        )
        dir_enc = PositionalEncoder(
            cfg.network.view_dir_dim,
            cfg.signal_encoder.dir_encode_level,
            cfg.signal_encoder.include_input,
        )
    elif cfg.signal_encoder.type == "sh":
        coord_enc = SHEncoder(
            cfg.network.pos_dim,
            cfg.signal_encoder.degree,
        )
        dir_enc = SHEncoder(
            cfg.network.view_dir_dim,
            cfg.signal_encoder.degree,
        )
    else:
        raise NotImplementedError()
    encoders = {
        "coord_enc": coord_enc,
        "dir_enc": dir_enc,
    }
    if cfg.scene.type == "cube":
        if cfg.network.type == "nerf":
            default_network = network.NeRF(
                coord_enc.out_dim,
                dir_enc.out_dim,
                cfg.network.use_softplus_actvn,
            ).to(cfg.cuda.device_id)
        elif cfg.network.type == "instant_nerf":
            default_network = network.InstantNeRF(
                cfg.network.pos_dim,
                dir_enc.out_dim,
                cfg.network.num_level,
                cfg.network.log_max_entry_per_level,
                cfg.network.min_res,
                cfg.network.max_res,
                table_feat_dim=cfg.network.table_feat_dim,
            ).to(cfg.cuda.device_id)
            encoders.pop("coord_enc", None)
        else:
            raise NotImplementedError()

        default_scene = scene.PrimitiveCube(
            default_network,
            encoders,
        )

        fine_scene = None
        if cfg.renderer.num_samples_fine > 0:  # initialize fine scene
            if cfg.network.type == "nerf":
                fine_network = network.NeRF(
                    coord_enc.out_dim,
                    dir_enc.out_dim,
                ).to(cfg.cuda.device_id)
            elif cfg.network.type == "instant_nerf":
                fine_network = network.InstantNeRF(
                    cfg.network.pos_dim,
                    dir_enc.out_dim,
                    cfg.network.num_level,
                    cfg.network.log_max_entry_per_level,
                    cfg.network.min_res,
                    cfg.network.max_res,
                    table_feat_dim=cfg.network.table_feat_dim,
                ).to(cfg.cuda.device_id)
            else:
                raise NotImplementedError()

            fine_scene = scene.PrimitiveCube(
                fine_network,
                encoders,
            )

        return default_scene, fine_scene
    else:
        raise ValueError("Unsupported scene representation.")


def _init_optimizer_and_scheduler(
    cfg: DictConfig,
    default_scene: scene.scene,
    fine_scene: scene.scene = None,
) -> Tuple[torch.optim.Optimizer, Optional[object]]:
    """
    Initializes the optimizer and learning rate scheduler used for training.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup optimizer and learning rate scheduler.
        scenes (Dict): A dictionary containing neural scene representation(s).

    Returns:
        optimizer (torch.optim.Optimizer):
        scheduler ():
    """
    optimizer = None
    scheduler = None

    # identify parameters to be optimized
    params = list(default_scene.radiance_field.parameters())
    if not fine_scene is None:
        params += list(fine_scene.radiance_field.parameters())

    # ==============================================================================
    # configure optimizer
    if cfg.train_params.optim.optim_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.train_params.optim.init_lr,
            eps=cfg.train_params.optim.eps,
        )
    else:
        raise NotImplementedError()

    # ==============================================================================
    # configure learning rate scheduler
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


def _init_loss_func(cfg: DictConfig) -> torch.nn.Module:
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


def _save_ckpt(
    ckpt_dir: str,
    epoch: int,
    default_scene: scene.scene,
    fine_scene: scene.scene,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> None:
    """
    Saves the checkpoint.

    Args:
        epoch (int):
        default_scene (scene.scene):
        fine_scene (scene.scene):
        optimizer ():
        scheduler ():
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = os.path.join(ckpt_dir, f"ckpt_{str(epoch).zfill(6)}.pth")

    ckpt = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # save scheduler state
    if not scheduler is None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    # save scene(s)
    ckpt["scene_default"] = default_scene.radiance_field.state_dict()
    if not fine_scene is None:
        ckpt["scene_fine"] = fine_scene.radiance_field.state_dict()

    torch.save(
        ckpt,
        ckpt_file,
    )


def _load_ckpt(
    ckpt_file,
    default_scene: scene.scene,
    fine_scene: scene.scene,
    optimizer: torch.optim.Optimizer,
    scheduler: object = None,
) -> int:
    """
    Loads the checkpoint.

    Args:
        ckpt_file (str): A path to the checkpoint file.
        default_scene (scene.scene):
        fine_scene (scene.scene):
        optimizer (torch.optim.Optimizer):
        scheduler ():

    Returns:
        epoch: The epoch from where training continues.
    """
    epoch = 0

    if ckpt_file is None or not os.path.exists(ckpt_file):
        print("Checkpoint file not found.")
        return epoch

    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load epoch
    epoch = ckpt["epoch"]

    # load scene(s)
    default_scene.radiance_field.load_state_dict(ckpt["scene_default"])
    default_scene.radiance_field.to(torch.cuda.current_device())
    if not fine_scene is None:
        fine_scene.radiance_field.load_state_dict(ckpt["scene_fine"])
        fine_scene.radiance_field.to(torch.cuda.current_device())

    # load optimizer and scheduler states
    if not optimizer is None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if not scheduler is None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print("Checkpoint loaded.")
    return epoch


def _visualize_scene(
    cfg,
    default_scene: scene.Scene,
    fine_scene: scene.Scene,
    renderer: VolumeRenderer,
    intrinsics: Union[Dict, torch.Tensor],
    extrinsics: torch.Tensor,
    img_res: Tuple[int, int],
    save_dir: str,
    num_imgs: int = None,
):
    """
    Visualizes the given scenes.

    If multiple scene representations are provided (e.g., coarse and fine scenes in hierarchical
    sampling), render the one that outputs the highest quality image.

    Args:
        cfg (DictConfig): A config object holding parameters required
            to setup scene representation.
        scenes (Dict): A dictionary of neural scene representation(s).
        renderer (VolumeRenderer): Volume renderer used to render the scene.
        intrinsics (Dict | torch.Tensor): A dictionary containing camera intrinsic parameters or
            the intrinsic matrix.
        extrinsics (torch.Tensor): An instance of torch.Tensor of shape (N, 4, 4)
            where N is the number of camera poses included in the trajectory.
        img_res (Tuple): A 2-tuple of form (img_height, img_width).
        save_dir (str): Directory to store render outputs.
        num_imgs (int): The number of images to be rendered. If not explicitly set,
            images will be rendered from all camera poses provided.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    pred_img_dir = os.path.join(save_dir, "pred_imgs")
    if not os.path.exists(pred_img_dir):
        os.mkdir(pred_img_dir)

    with torch.no_grad():
        for pose_idx, extrinsic in tqdm(enumerate((extrinsics))):
            if not num_imgs is None:
                if pose_idx >= num_imgs:
                    break

            # set the camera
            renderer.camera = cameras.PerspectiveCamera(
                intrinsics,
                extrinsic,
                cfg.renderer.t_near,
                cfg.renderer.t_far,
            )

            img_height, img_width = img_res
            num_total_pixel = img_height * img_width

            # render coarse scene first
            pixel_pred, default_indices, default_weights = renderer.render_scene(
                default_scene,
                num_pixels=num_total_pixel,
                num_samples=cfg.renderer.num_samples_coarse,
                project_to_ndc=cfg.renderer.project_to_ndc,
                device=torch.cuda.current_device(),
                num_ray_batch=num_total_pixel // cfg.renderer.num_pixels,
            )
            if not fine_scene is None:  # visualize "fine" scene
                pixel_pred, _, _ = renderer.render_scene(
                    fine_scene,
                    num_pixels=num_total_pixel,
                    num_samples=(cfg.renderer.num_samples_coarse, cfg.renderer.num_samples_fine),
                    project_to_ndc=cfg.renderer.project_to_ndc,
                    pixel_indices=default_indices,
                    weights=default_weights,
                    device=torch.cuda.current_device(),
                    num_ray_batch=num_total_pixel // cfg.renderer.num_pixels,
                )

            # (H * W, C) -> (C, H, W)
            pixel_pred = pixel_pred.reshape(img_height, img_width, -1)
            pixel_pred = pixel_pred.permute(2, 0, 1)

            # save the image
            tvu.save_image(
                pixel_pred,
                os.path.join(pred_img_dir, f"{str(pose_idx).zfill(5)}.png"),
            )
