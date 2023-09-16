"""
File: src/models/candlestick.modules.py
Creation Date: 2023-08-22

This contains all the LightningModule definitions for the candlestick plot stock predictor. Of course,
it will also use classes from the nets subfolder, with adjusted classes.
"""
from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy, F1Score
from torchmetrics import Dice

import pytorch_lightning as pl


class CandlestickClassifierModule(pl.LightningModule):
    def __init__(self, neural_net: nn.Module, optimizer: optim.Optimizer):
        super().__init__()
        self.save_hyperparameters(ignore=['neural_net'])
        self.neural_net = neural_net
        self.optimizer = optimizer
        # This is a binary classifier, so we use BCEWithLogitsLoss
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([9]))
        # For metrics, we can use accuracy, and F1 score as well to get a measure of precision/recall.
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        self.train_f1 = F1Score('binary')
        self.val_f1 = F1Score('binary')

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x):
        return self.neural_net(x)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(params=self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        # Calculate training accuracy, training loss, and training F1 score and update everything.
        # Only log the accuracy and F1
        self.train_acc(output.squeeze(), labels)
        loss = self.loss(output.squeeze(), labels)
        self.train_loss(loss)
        self.train_f1(output.squeeze(), labels)
        # Log. The update and clear will happen when I use the object directly.
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train/f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': self.train_acc}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        # Calculate training accuracy, training loss, and training F1 score and update everything.
        # Only log the accuracy and F1
        self.val_acc(output.squeeze(), labels)
        loss = self.loss(output.squeeze(), labels)
        self.val_loss(loss)
        self.val_f1(output.squeeze(), labels)
        # Log. The update and clear will happen when I use the object directly.
        self.log('val/acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/loss', self.val_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('val/f1', self.val_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': self.val_acc}

