from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy
from torchmetrics import Dice

import pytorch_lightning as pl

from src.models.nets.unet import UMobileNet


class OxfordSegmentationClassifierModule(pl.LightningModule):
    def __init__(self, image_size: int, in_image_channels: int,
                 output_classes: int, optimizer: optim.Optimizer):
        super().__init__()
        self.output_classes = output_classes
        self.model = UMobileNet(image_size, in_image_channels, output_classes)
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
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
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
        self.log('val/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def predict_step(self, batch, batch_idx, dataloader_idx):
        input_images, _ = batch
        output_masks = self(input_images)
        softmaxxed = nn.Softmax(dim=1)(output_masks)
        return torch.argmax(softmaxxed, dim=1)


class TGSSaltClassifierModule(pl.LightningModule):
    def __init__(self, image_size: int, in_image_channels: int,
                 optimizer: optim.Optimizer):
        super().__init__()
        self.model = UMobileNet(image_size, in_image_channels, 1)
        self.optimizer = optimizer
        # For TGS Salt, we need binary cross entropy
        self.loss = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()

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
        loss = self.loss(output_masks, masks.to(torch.float))
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
        loss = self.loss(output_masks, masks.to(torch.float))
        # Update our metrics
        self.val_acc(accuracy)
        self.val_loss(loss)
        # Log
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def predict_step(self, batch, batch_idx, dataloader_idx):
        input_images, _ = batch
        output_masks = self(input_images)
        sigmoided = nn.Sigmoid()(output_masks)
        return sigmoided


class GRContrailsClassifierModule(pl.LightningModule):
    def __init__(self, image_size: int, in_image_channels: int, optimizer: optim.Optimizer):
        super().__init__()
        self.model = UMobileNet(image_size, in_image_channels, 1)
        self.optimizer = optimizer
        # This is a binary classifier, so we need to use BCELogits Loss
        # Weight the positive class, as there are 99x contrail pixels than non ones.
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([99]))
        # For metrics, we'll have accuracy, but we'll also use the Dice coefficient,
        # which is what the competition uses.
        self.accuracy = BinaryAccuracy()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()

        # Create additional variables for the dice coefficients.
        self.train_dice = Dice()
        self.val_dice = Dice()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        input_images, masks = batch
        output_masks = self(input_images)
        # Calculate training accuracy, training loss, and training dice. Only log the dice since that's important.
        accuracy = self.accuracy(output_masks, masks)
        loss = self.loss(output_masks, masks)
        dice = self.train_dice(output_masks, masks.to(int))
        # Update accuracy and loss.
        self.train_acc(accuracy)
        self.train_loss(loss)
        # Log
        self.log('train/dice', self.train_dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        input_images, masks = batch
        output_masks = self(input_images)
        # Calculate training accuracy, training loss, and training dice. Only log the dice since that's important.
        accuracy = self.accuracy(output_masks, masks)
        loss = self.loss(output_masks, masks)
        dice = self.val_dice(output_masks, masks.to(int))
        # Update accuracy and loss.
        self.val_acc(accuracy)
        self.val_loss(loss)
        # Log
        self.log('val/dice', self.val_dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': accuracy}

    def test_step(self, batch, batch_idx):
        # For testing, only the images will be there. In this case we can just return the dice coefficient and accuracy.
        images = batch
        output_masks = self(images)
        # Apply sigmoid
        return {'predictions': torch.sigmoid(output_masks)}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # For prediction, we just need to return the images
        images = batch
        output_masks = self(images)
        return torch.sigmoid(output_masks)

