"""
evaluate.py

A script for evaluating a trained model.
"""

import argparse
from pathlib import Path

from torch_nerf.src.utils.metrics.rgb_metrics import (
    compute_lpips_between_directories,
    compute_psnr_between_directories,
    compute_ssim_between_directories,
)


def main():
    """
    The entry point of the evaluation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", type=str, help="Path to the first directory.")
    parser.add_argument("dir2", type=str, help="Path to the second directory.")
    args = parser.parse_args()

    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    assert dir1.exists(), f"The directory {dir1} does not exist."
    assert dir2.exists(), f"The directory {dir2} does not exist."

    lpips = compute_lpips_between_directories(dir1, dir2)
    psnr = compute_psnr_between_directories(dir1, dir2)
    ssim = compute_ssim_between_directories(dir1, dir2)

    print(f"LPIPS: {lpips:.4f}")
    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
