"""
File: models.py
Creation Date: 2025-06-19

Contains all the LightningModule definitions for all the models we would want to train.
All have similar inputs, namely the neural net and its optimizer. Nets are defined
in the nets.py file.
"""
from typing import Any

import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MaxMetric, ConfusionMatrix
from torchmetrics.classification import BinaryAccuracy


class CancerImageClassifier(pl.LightningModule):
    def __init__(self, net: nn.Module, optimizer: optim.Optimizer):
        super().__init__()
        # Save all the hyperparameters, they'll become available as self.hparams
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        self.optimizer = optimizer

        # The loss function
        self.loss = nn.BCEWithLogitsLoss()

        # All tasks will use binary accuracy
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        # We also need to average losses across batches, so set MeanMetrics up...
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # Finally, have a confusion matrix...
        self.matrix = ConfusionMatrix(task='binary')

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
        loss = self.loss(outputs, targets.unsqueeze(dim=-1).float())
        self.train_loss(loss)  # Update our current loss, will hold average loss so far...
        self.train_acc(outputs.squeeze(), targets)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': self.train_acc}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets.unsqueeze(dim=-1).float())
        self.val_loss(loss)  # Update our current validation loss
        self.val_acc(outputs.squeeze(), targets)
        self.matrix.update(outputs.squeeze(), targets)
        self.log('val/loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': self.val_acc}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs, _ = batch
        outputs = self(inputs)
        return nn.Sigmoid()(outputs)