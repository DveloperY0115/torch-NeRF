"""
train.py - Script for training
"""
import argparse
import sys

sys.path.append(".")
sys.path.append("..")

from training.nerf_trainer import NeRFTrainer

parser = argparse.ArgumentParser()

# Project settings
parser.add_argument("--model", type=str, default="NeRF")
parser.add_argument("--log_wandb", type=bool, default=False)

# CUDA settings
parser.add_argument("--no_cuda", type=bool, default=False)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--use_multi_gpu", type=bool, default=False)

# Dataset
parser.add_argument("--out_dir", type=str, default="out")
parser.add_argument("--dataset_type", type=str, default="Blender")
parser.add_argument("--dataset_dir", type=str, default="data/nerf_synthetic/lego")

# Training
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-4)

args = parser.parse_args()


def main():
    trainer = NeRFTrainer(opts=args)
    trainer.train()


if __name__ == "__main__":
    main()
