"""
base_trainer.py - Base Trainer class for Pytorch model.
"""
import os
from typing import Dict, Iterable, Tuple
from collections import namedtuple

import torch
import torch.optim as optim


class BaseTrainer:
    def __init__(self, *args, **kwargs):
        """
        Base class of trainers used in the project.
        Args:
        - opts (collections.namedtuple)): namedtuple instance containing trainer options
        - logger ()
        """
        self.opts = kwargs["opts"]

        self.initial_epoch = 0
        self.epoch = 0

        _, self.log_dir, self.checkpoint_dir = self.configure_output_directories(
            self.opts.out_dir, self.opts.dataset_type, self.opts.model
        )

        self.device = self.configure_device(
            self.opts.device_id, self.opts.no_cuda, self.opts.use_multi_gpu
        )

    def train(self):
        """
        Train the model.
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        Train the model for one epoch.
        """
        raise NotImplementedError

    def test_one_epoch(self):
        """
        Test the model every epoch.
        """
        raise NotImplementedError

    # Helper functions for configuring the trainer

    def configure_device(
        self, device_id: int, no_cuda: bool = False, use_multi_gpu: bool = False
    ) -> torch.device:
        """
        Configure which device to run training on.
        Inform users various tips based on the status of their machine.
        Args:
        - device_id (int): Index of device to be used.
        - no_cuda (bool): Switch for enabling / disabling CUDA usage.
        - use_multi_gpu (bool): Switch for multi GPU usage.
        Returns:
        - device (torch.device): Context-manager in Pytorch which designates the selected device.
        """

        print("======== Device Configuration ========")
        device = torch.device(
            "cuda:{}".format(device_id) if torch.cuda.is_available() and not no_cuda else "cpu"
        )

        if device.type != "cpu":
            torch.cuda.set_device(device)
            print("[!] Set {} as default device".format(device))
        else:
            print("[!] Using cpu for training..")

        if torch.cuda.is_available() and no_cuda:
            print("[!] Your system is capable of GPU computing but is set not to use it.")
            print("[!] It's highly recommended to use GPUs for training!")

        if (device.type == "cuda") and (torch.cuda.device_count() > 1):
            print(f"[!] Number of GPUs availalbe: {torch.cuda.device_count()}")
            if not use_multi_gpu:
                print("[!] But it's set to start with single GPU")
                print("[!] It's highly recommended to use multiple GPUs if possible!")

        print("======================================")

        return device

    def configure_output_directories(
        self,
        out_root: str,
        dataset_type: str,
        model_type: str,
    ) -> Tuple[str, str]:
        """
        Setup directories where outputs (log, checkpoint, etc) will be saved.
        Configure directory differently for different experiment settings.
        Args:
        - out_root (str or os.path): Root of the output directory.
        - dataset_type (str): Type of dataset used.
        Returns:
        - Tuple of strings each representing directories created inside this function.
        """

        if not os.path.exists(out_root):
            os.mkdir(out_root)

        # dataset specific directory
        datatype_dir = os.path.join(out_root, dataset_type)
        if not os.path.exists(datatype_dir):
            os.mkdir(datatype_dir)

        # model specific directory
        modeltype_dir = os.path.join(datatype_dir, model_type)
        if not os.path.exists(modeltype_dir):
            os.mkdir(modeltype_dir)

        # log output directory
        log_dir = os.path.join(modeltype_dir, "runs")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # checkpoint directory
        checkpoint_dir = os.path.join(log_dir, "checkpoint")

        return out_root, log_dir, checkpoint_dir

    def save_checkpoint(self):
        """
        Save checkpoint.
        """
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        save_path = os.path.join(
            self.checkpoint_dir, "checkpoint-epoch-{}.tar".format(self.epoch + 1)
        )

        checkpoint = {}

        checkpoint["epoch"] = self.epoch
        checkpoint["model_state_dict"] = (
            self.model.module.state_dict() if self.opts.use_multi_gpu else self.model.state_dict()
        )
        checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, save_path)
        print("[!] Saved model at: {}".format(save_path))

    def load_checkpoint(self, checkpoint_file: str):
        """
        Load existing checkpoint to resume training.
        Args:
        - checkpoint (str or os.path): Name of checkpoint file.
        Returns:
        - True if loading was successful, False otherwise.
        """

        if not os.path.exists(checkpoint_file):
            print("[!] No checkpoint loaded")
            return False

        print("[!] Loading the latest checkpoint {}".format(checkpoint_file))

        checkpoint = torch.load(checkpoint_file)

        self.initial_epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.train()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return True

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configure optimizer used for training.
        Expected to return:
        - optimizer (torch.optim.Optimizer): Pytorch optimizer instance.
        """
        raise NotImplementedError

    def configure_lr_scheduler(self) -> torch.optim.lr_scheduler:
        """
        Configure learning rate scheduler.
        Expected to return:
        - lr_scheduler (torch.optim.lr_scheduler): Pytorch learning rate scheduler.
        """
        raise NotImplementedError

    def configure_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Configure dataset.
        Expected to return:
        - train_dataset (torch.utils.data.Dataset): Pytorch Dataset instance representing training dataset.
        - test_dataset (torch.utils.data.Dataset): Pytorch Dataset instance representing test dataset.
        """
        raise NotImplementedError

    def configure_dataloader(
        self,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Configure data loader.
        Expected to return:
        - train_dataset (torch.utils.data.DataLoader): Pytorch DataLoader instance representing training data loader.
        - test_dataset (torch.utils.data.DataLoader): Pytorch DataLoader instance representing test data loader.
        """
        raise NotImplementedError
