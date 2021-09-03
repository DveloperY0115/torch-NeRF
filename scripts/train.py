"""
train.py - Script for training
"""

import sys

sys.path.append(".")
sys.path.append("..")

from training.nerf_trainer import NeRFTrainer


def main():
    trainer = NeRFTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
