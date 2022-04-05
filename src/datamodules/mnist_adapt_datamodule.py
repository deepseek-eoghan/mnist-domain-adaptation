import datetime
import os
from typing import Optional, Tuple

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from tqdm import tqdm
from urllib import request
import pathlib
import pickle
import gzip
import numpy as np

from pytorch_adapt.datasets import DataloaderCreator, get_mnist_mnistm
from pytorch_adapt.validators import IMValidator
from pytorch_adapt.frameworks.utils import filter_datasets

from src.datamodules.datasets.mnist_detection_dataset import MnistDetectionDataset

import src.datamodules.mnist_generate.mnist as mnist
import src.datamodules.mnist_generate.generate_data as generate_data

class Collater:
    # https://shoarora.github.io/2020/02/01/collate_fn.html
    def __call__(self, batch):
        return tuple(zip(*batch))


class MnistAdaptDataModule(LightningDataModule):
    """
    MnistDetectionDataModule for Mnist object detection.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

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

        self.transforms = A.Compose(
            [
                A.Normalize(),
                A.HorizontalFlip(p=0.5),
                A.Blur(blur_limit=3, p=0.15),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        )

        self.notransforms = A.Compose(
            [
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        )

        self.collater = Collater()

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 300, 300)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.dataloaders = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        if not os.path.exists(self.hparams.data_dir):
            print("downloading dataset")
            get_mnist_mnistm(["mnist"], ["mnistm"], folder=self.hparams.data_dir, download=True)
        return


    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            datasets = get_mnist_mnistm(["mnist"], ["mnistm"], folder=self.hparams.data_dir, download=False)
            dc = DataloaderCreator(batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
            validator = IMValidator()
            self.dataloaders = dc(**filter_datasets(datasets, validator))
            self.data_train = self.dataloaders.pop("train")
            self.data_val = list(self.dataloaders.values())
            return            

    def train_dataloader(self):
        return self.data_train

    def val_dataloader(self):
        return self.data_val