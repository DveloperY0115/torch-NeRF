"""A script for scene rendering."""

import sys

import hydra
from omegaconf import DictConfig

sys.path.append(".")
sys.path.append("..")

import torch_nerf.runners.runner_utils as runner_utils


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """The entry point of rendering code."""
    render_session = runner_utils.init_session(cfg, mode="render")
    render_session()
    print("Rendering done.")


if __name__ == "__main__":
    main()
