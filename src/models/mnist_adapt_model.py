from math import nan
from typing import Any, List

import numpy as np
import torch
from pl_bolts.losses.object_detection import iou_loss
from pl_bolts.metrics.object_detection import iou
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.detection.map import MAP
from torchvision import models
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from pytorch_adapt.frameworks.lightning import Lightning
from pytorch_adapt.adapters import DANN
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.models import Discriminator, mnistC, mnistG
from pytorch_adapt.validators import AccuracyValidator


class MnistAdaptLitModel(LightningModule):
    """
    LightningModule for mnist object detection.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        num_classes: int = 11,
        lr: float = 0.0001,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        trainable_backbone_layers: int = 0,
        batch_size: int = 4,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.G = mnistG(pretrained=True)
        self.C = mnistC(pretrained=True)
        self.D = Discriminator(in_size=1200, h=256)
        self.models = Models({"G": self.G, "C": self.C, "D": self.D})
        self.validator = AccuracyValidator(key_map={"target_train": "src_val"})
        self.optimizers = Optimizers((torch.optim.Adam, {"lr": self.hparams.lr}))

        self.adapter = DANN(models=self.models, optimizers=self.optimizers)
