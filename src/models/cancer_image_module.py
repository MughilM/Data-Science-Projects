"""
File: cancer_image_module.py
Creation Date: 2023-01-07

Contains LightningModule for detecting cancer images. Should be used
with the DataModule from the histopathologic-cancer-detection Kaggle competition
"""
from typing import Any

import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification import BinaryAccuracy

import wandb


class CancerImageClassifier(pl.LightningModule):
    def __init__(self, net: nn.Module, optimizer: optim.Optimizer):
        super().__init__()
        # Save all the hyperparameters, they'll become available as self.hparams
        self.save_hyperparameters(logger=True, ignore=['net'])

        self.net = net
        self.optimizer = optimizer

        # The loss function
        self.loss = nn.BCEWithLogitsLoss()

        # All tasks will use binary accuracy
        self.binary_accuracy = BinaryAccuracy()

        # We also need to average losses across batches, so set MeanMetrics up...
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        # TODO: Add schedulers if you want
        return optimizer

    # def on_train_start(self) -> None:
    #     # Reset the best validation accuracy due to sanity checks that pl does
    #     self.best_val_acc.reset()

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        train_accuracy = self.binary_accuracy(outputs.squeeze(), targets)
        loss = self.loss(outputs, targets.unsqueeze(dim=-1).float())
        self.train_loss(loss)  # Update our current loss, will hold average loss so far...
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': train_accuracy}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_accuracy = self.binary_accuracy(outputs.squeeze(), targets)
        loss = self.loss(outputs, targets.unsqueeze(dim=-1).float())
        self.val_loss(loss)  # Update our current validation loss
        self.log('val/loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': val_accuracy}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs, _ = batch
        outputs = self(inputs)
        return nn.Sigmoid()(outputs)


