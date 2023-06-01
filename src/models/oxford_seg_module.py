import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

import pytorch_lightning as pl

from src.models.nets.unet import UNet


class SegmentationClassifierModule(pl.LightningModule):
    def __init__(self, output_classes: int, optimizer: optim.Optimizer):
        super().__init__()
        self.output_classes = output_classes
        self.model = UNet(output_classes)
        self.optimizer = optimizer

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(output_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()
        self.test_acc = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        input_images, masks = batch
        output_masks = self(input_images)
        # Calculate training accuracy and training loss and log it
        accuracy = self.accuracy(output_masks, masks)
        loss = self.loss(output_masks, masks.to(torch.long))
        # Update our metrics
        self.train_acc(accuracy)
        self.train_loss(loss)
        # Log
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        input_images, masks = batch
        output_masks = self(input_images)
        # Calculate training accuracy and training loss and log it.
        accuracy = self.accuracy(output_masks, masks)
        loss = self.loss(output_masks, masks.to(torch.long))
        # Update our metrics
        self.val_acc(accuracy)
        self.val_loss(loss)
        # Log
        self.log('val/loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def test_step(self, batch, batch_idx):
        input_images, masks = batch
        output_masks = self(input_images)
        # Calculate training accuracy and training loss and log it.
        accuracy = self.accuracy(output_masks, masks)
        loss = self.loss(output_masks, masks.to(torch.long))
        # Update our metrics
        self.test_acc(accuracy)
        self.test_loss(loss)
        # Log
        self.log('val/loss', self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def predict_step(self, batch, batch_idx, dataloader_idx):
        input_images, _ = batch
        output_masks = self(input_images)
        softmaxxed = nn.Softmax(dim=1)(output_masks)
        return torch.argmax(softmaxxed, dim=1)
