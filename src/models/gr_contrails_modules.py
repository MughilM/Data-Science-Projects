"""
File: src/models/gr_contrails_modules.py
Creation Date: 2023-07-27

Contains all LightningModule definitions for the Google Research Contrail Detection challenge.
Some of these require their own LightningDataModules, which will be checked.
"""
from typing import Any, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy
from torchmetrics import Dice

import torchvision.transforms.functional as TF

import pytorch_lightning as pl

from src.models.nets.unet import UMobileNet


class GRContrailsClassifierModule(pl.LightningModule):
    def __init__(self, image_size: int, in_image_channels: int, optimizer: optim.Optimizer):
        super().__init__()
        self.model = UMobileNet(image_size, in_image_channels, 1)
        self.optimizer = optimizer
        # This is a binary classifier (contrail, or not contrail), so we need to use BCELogits Loss
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

    # def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    #     print(outputs)
    #     print(outputs.shape)
    #     return outputs


class GRContrailBinaryClassifier(pl.LightningModule):
    def __init__(self, binary_model: nn.Module, seg_model: nn.Module, binary_image_size: int, lr: float = 0.005,
                 beta1: float = 0.5, beta2: float = 0.999):
        super().__init__()
        self.save_hyperparameters(ignore=['binary_model', 'seg_model'])
        self.binary_model = binary_model
        self.seg_model = seg_model
        self.binary_image_size = binary_image_size
        # Both the image binary classifier and the image segmentation tasks will use
        # the BCEWithLogitsLoss. The segmentation will be weighted, while the whole image binary
        # will be unweighted.
        self.seg_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([90]))
        self.binary_loss = nn.BCEWithLogitsLoss()
        # For metrics, we'll have accuracy, but we'll also use the Dice coefficient,
        # which is what the competition uses.
        self.accuracy = BinaryAccuracy()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()


        self.train_b_acc = MeanMetric()
        self.train_s_acc = MeanMetric()
        self.val_b_acc = MeanMetric()
        self.val_s_acc = MeanMetric()

        # Create additional variables for the dice coefficients.
        self.train_dice = Dice()
        self.val_dice = Dice()

    def forward(self, x):
        """
        The forward method will receive a set of images, which may or may not be preprocessed.
        The assumption is that they will be resized to the `binary_image_size` before being passed through
        the binary image model, and will also be passed just like that through the segmentation model.
        :param x:
        :return:
        """
        transformed = TF.resize(x, [self.binary_image_size, self.binary_image_size])
        # Get binary outputs
        binary_outputs = self.binary_model(transformed)
        # Get the locations where the binary model predicted presence of a contrail, and pass the original images
        # through to the segmentation model to get masks.
        seg_inputs = x[torch.where(binary_outputs >= 0.5)]
        res = self.seg_model(seg_inputs).numpy()
        # Next, for the images where we don't have predicted contrails, we insert empty masks.
        insert_locs = torch.where(binary_outputs < 0.5)[0].numpy()
        zero_masks = np.zeros((len(insert_locs),) + res.shape[1:])
        res = np.insert(res, insert_locs, zero_masks, axis=0)
        return res

    def configure_optimizers(self) -> Any:
        """
        We have two optimizers, one for the binary model and one for the segmentation model.
        :return:
        """
        lr = self.hparams.lr
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2
        optimizerB = optim.Adam(params=self.binary_model.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerS = optim.Adam(params=self.seg_model.parameters(), lr=lr, betas=(beta1, beta2))
        return [optimizerB, optimizerS], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        For the training step, we get as input a CombinedLoader, so we extract both the images
        for the binary model and the ones for the segmentation model, and train both depending
        on the selected optimizer.
        :param batch:
        :param batch_idx:
        :param optimizer_idx:
        :return:
        """
        # Train binary model
        if optimizer_idx == 0:
            bin_batch = batch['binary']
            images, labels = bin_batch
            output = self.binary_model(images)
            accuracy = self.accuracy(output, labels)
            self.train_b_acc(accuracy)
            # Calculate loss
            b_loss = self.binary_loss(output, labels)
            self.log('train/b_loss', b_loss, prog_bar=False)
            self.log('train/b_acc', self.train_b_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return b_loss
        # Train segmentation model (not necessarily the same batch)
        if optimizer_idx == 1:
            seg_batch = batch['cr_only']
            images, masks = seg_batch
            output_masks = self.seg_model(images)
            accuracy = self.accuracy(output_masks, masks)
            self.train_s_acc(output_masks, masks)
            # Calculate loss (it's still binary cross entropy, but the weightings are different)
            s_loss = self.seg_loss(output_masks, masks)
            self.log('train/s_loss', s_loss, prog_bar=False)
            self.log('train/s_acc', self.train_s_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return s_loss

    def validation_step(self, batch, batch_idx):
        """
        Just like training, we'll also report binary accuracy and segmentation accuracy.
        But we'll also report the dice value on just the contrails.
        :param batch:
        :param batch_idx:
        :return:
        """
        bin_batch = batch['binary']
        images, labels = bin_batch
        output = self.binary_model(images)
        accuracy = self.accuracy(output, labels)
        self.val_b_acc(accuracy)
        self.log('val/b_acc', self.val_b_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        seg_batch = batch['cr_only']
        images, masks = seg_batch
        output_masks = self.seg_model(images)
        accuracy = self.accuracy(output_masks, masks)
        self.val_s_acc(accuracy)
        self.val_dice(output_masks, masks)
        self.log('val/s_acc', self.val_s_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/s_dice', self.val_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)










