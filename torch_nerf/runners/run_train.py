"""A script for training."""

import sys

import hydra
from omegaconf import DictConfig

sys.path.append(".")
sys.path.append("..")

import torch_nerf.runners.runner_utils as runner_utils


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
)
def main(cfg: DictConfig) -> None:
    """The entry point of training code."""
    train_session = runner_utils.init_session(cfg, mode="train")
    train_session()


if __name__ == "__main__":
    main()
