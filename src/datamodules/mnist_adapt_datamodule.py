import datetime
import os
from typing import Optional, Tuple

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split


from pytorch_adapt.datasets import DataloaderCreator, get_mnist_mnistm
from pytorch_adapt.validators import MultipleValidators, AccuracyValidator, IMValidator
from pytorch_adapt.frameworks.utils import filter_datasets


import src.datamodules.mnist_generate.mnist as mnist
import src.datamodules.mnist_generate.generate_data as generate_data

class MnistAdaptDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/mnistm/",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 11,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.dataloaders = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        if not os.path.exists(self.hparams.data_dir):
            print("downloading dataset")
            get_mnist_mnistm(["mnist"], ["mnistm"], folder=self.hparams.data_dir, download=True)
        return


    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            datasets = get_mnist_mnistm(["mnist"], ["mnistm"], folder=self.hparams.data_dir, download=False, return_target_with_labels=True)
            datasets["target_train"] = datasets["target_train_with_labels"]
            datasets["target_val"] = datasets["target_val_with_labels"]
            dc = DataloaderCreator(batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
            validator = AccuracyValidator(key_map={"target_train": "src_val"})
            self.dataloaders = dc(**filter_datasets(datasets, validator))
            self.data_train = self.dataloaders.pop("train")
            self.data_val = list(self.dataloaders.values())
            # test_dc = DataloaderCreator(batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, val_names=["target_test"])
            # self.data_test = test_dc(target_test=datasets['target_val'])['target_test']
            return            

    def train_dataloader(self):
        return self.data_train

    def val_dataloader(self):
        return self.data_val

    # This can't be added until pytorch-adapt extends the Lightning class.
    # def test_dataloader(self):
    #     return self.data_test