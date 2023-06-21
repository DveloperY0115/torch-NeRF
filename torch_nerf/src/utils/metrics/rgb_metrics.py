"""
__init__.py

A simple wrapper for computing evaluation metrics.
"""

import math
from PIL import Image
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@torch.no_grad()
def compute_lpips_between_directories(pred_dir: Path, target_dir: Path) -> float:
    """
    Computes LPIPS between the image pairs under two directories.
    """
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='alex',
        normalize=True,
    )
    return compute_metric_between_directories(pred_dir, target_dir, lpips)

@torch.no_grad()
def compute_psnr_between_directories(pred_dir: Path, target_dir: Path) -> float:
    """
    Computes PSNR between the image pairs under two directories.
    """
    psnr = torchmetrics.PeakSignalNoiseRatio()
    return compute_metric_between_directories(pred_dir, target_dir, psnr)

@torch.no_grad()
def compute_ssim_between_directories(pred_dir: Path, target_dir: Path) -> float:
    """
    Computes SSIM between the image pairs under two directories.
    """
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    return compute_metric_between_directories(pred_dir, target_dir, ssim)

@torch.no_grad()
def compute_metric_between_directories(
    pred_dir: Path, target_dir: Path, metric_func: Callable, batch_size: int = 128,
) -> float:
    """
    Evaluates the given metric between the image pairs under two directories.
    """

    # check arguments
    assert pred_dir.exists(), f"The directory {pred_dir} does not exist."
    assert target_dir.exists(), f"The directory {target_dir} does not exist."

    metric = 0.0

    pred_set = []
    target_set = []

    # load images
    for file1 in pred_dir.iterdir():

        file2 = target_dir / file1.name
        assert file2.exists(), f"Expected a file with the same name under {target_dir}"

        # load images
        pred = Image.open(file1)
        target = Image.open(file2)

        # match size
        pred_height, pred_width = pred.size
        target_height, target_width = target.size

        height = min(pred_height, target_height)
        width = min(pred_width, target_width)        
        pred = pred.resize((width, height))
        target = target.resize((width, height))

        # convert PIL images to Numpy arrays
        pred = np.array(pred, dtype=np.float32) / 255.0
        target = np.array(target, dtype=np.float32) / 255.0

        # collect arrays
        pred_set.append(pred[np.newaxis])
        target_set.append(target[np.newaxis])
    pred_set = np.concatenate(pred_set, axis=0)
    target_set = np.concatenate(target_set, axis=0)

    # make background white
    if pred_set.shape[-1] == 4:
        alpha = pred_set[..., -1]
        pred_set[alpha == 0.0, ...] = 1.0
        pred_set = pred_set[..., :3]
    if target_set.shape[-1] == 4:
        alpha = target_set[..., -1]
        target_set[alpha == 0.0, ...] = 1.0
        target_set = target_set[..., :3]

    # convert to torch.Tensor
    pred_set = torch.from_numpy(pred_set).permute(0, 3, 1, 2)
    target_set = torch.from_numpy(target_set).permute(0, 3, 1, 2)
    assert len(pred_set) == len(target_set), (
        f"Expected two datasets to have the same size. Got {len(pred_set)} and {len(target_set)} images."
    )

    # split datasets into batches
    num_batch = 1
    if len(pred_set) > batch_size:
        num_batch = math.ceil(len(pred_set) / batch_size)
    pred_set = torch.chunk(pred_set, num_batch)
    target_set = torch.chunk(target_set, num_batch)

    # accumulate metric over batches
    for pred_batch, target_batch in zip(pred_set, target_set):
        metric += metric_func(pred_batch, target_batch).item()

    metric /= len(pred_set)

    return metric
