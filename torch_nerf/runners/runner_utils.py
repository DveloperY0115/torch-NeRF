"""A set of utility functions commonly used in training/testing scripts."""

import os
from typing import Dict, Tuple, Union

from omegaconf import DictConfig
import torch
import torch.utils.data as data
import torchvision.utils as tvu
from tqdm import tqdm
import torch_nerf.src.network as network
import torch_nerf.src.scene as scene
import torch_nerf.src.renderer.cameras as cameras
import torch_nerf.src.renderer.integrators.quadrature_integrator as integrators
import torch_nerf.src.renderer.ray_samplers as ray_samplers
from torch_nerf.src.renderer.volume_renderer import VolumeRenderer
import torch_nerf.src.signal_encoder.positional_encoder as pe
from torch_nerf.src.utils.data.blender_dataset import NeRFBlenderDataset
from torch_nerf.src.utils.data.llff_dataset import LLFFDataset


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
            scene_name=cfg.data.scene_name,
            data_type=cfg.data.data_type,
            half_res=cfg.data.half_res,
            white_bg=cfg.data.white_bg,
        )
    elif cfg.data.dataset_type == "nerf_llff":
        dataset = LLFFDataset(
            cfg.data.data_root,
            scene_name=cfg.data.scene_name,
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
            cfg.renderer.t_near = torch.min(dataset.z_bounds) * 0.9
            cfg.renderer.t_far = torch.max(dataset.z_bounds) * 1.0
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
    print(f"Dataset type: {cfg.data.dataset_type}")
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


def save_ckpt(
    ckpt_dir: str,
    epoch: int,
    scenes,
    optimizer,
    scheduler,
) -> None:
    """
    Saves the checkpoint.

    Args:
        epoch (int):
        scenes (Dict):
        optimizer ():
        scheduler ():
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = os.path.join(ckpt_dir, f"ckpt_{str(epoch).zfill(6)}.pth")

    ckpt = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    for scene_type, scene in scenes.items():
        ckpt[f"scene_{scene_type}"] = scene.radiance_field.state_dict()

    torch.save(
        ckpt,
        ckpt_file,
    )


def load_ckpt(
    ckpt_file,
    scenes,
    optimizer,
    scheduler=None,
) -> int:
    """
    Loads the checkpoint.

    Args:
        ckpt_file (str): A path to the checkpoint file.
        scenes ():
        optimizer ():
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

    # load scene
    for scene_type, scene in scenes.items():
        scene.radiance_field.load_state_dict(ckpt[f"scene_{scene_type}"])
        scene.radiance_field.to(torch.cuda.current_device())

    # load optimizer and scheduler states
    if not optimizer is None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if not scheduler is None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print("Checkpoint loaded.")
    return epoch


def visualize_scene(
    cfg,
    scenes,
    renderer,
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
            pixel_pred, coarse_indices, coarse_weights = renderer.render_scene(
                scenes["coarse"],
                num_pixels=num_total_pixel,
                num_samples=cfg.renderer.num_samples_coarse,
                project_to_ndc=cfg.renderer.project_to_ndc,
                device=torch.cuda.current_device(),
                num_ray_batch=num_total_pixel // cfg.renderer.num_pixels,
            )
            if "fine" in scenes.keys():  # visualize "fine" scene
                pixel_pred, _, _ = renderer.render_scene(
                    scenes["fine"],
                    num_pixels=num_total_pixel,
                    num_samples=(cfg.renderer.num_samples_coarse, cfg.renderer.num_samples_fine),
                    project_to_ndc=cfg.renderer.project_to_ndc,
                    pixel_indices=coarse_indices,
                    weights=coarse_weights,
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
