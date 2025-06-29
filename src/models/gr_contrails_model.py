"""
File: src/models/gr_contrails_model.py
Creation Date: 2023-07-27

Contains all LightningModule definitions for the Google Research Contrail Detection challenge.
Some of these require their own LightningDataModules, which will be checked.
"""
from typing import Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics import Dice

import torchvision.transforms.functional as TF
from torch.hub import load_state_dict_from_url

import pytorch_lightning as pl

from src.models.nets.unet import UMobileNet
from src.models.nets.deeplab import DeepLabV3, DeepLabHeadV3Plus, ResNet
from src.models.nets.special_modules import Bottleneck, IntermediateLayerGetter


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
        self.log('val/dice', self.val_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
                 beta1: float = 0.5, beta2: float = 0.999, seg_threshold: float = 0.5):
        super().__init__()
        self.save_hyperparameters(ignore=['binary_model', 'seg_model'])
        self.automatic_optimization = False

        self.binary_model = binary_model
        self.seg_model = seg_model
        self.binary_image_size = binary_image_size
        # Both the image binary classifier and the image segmentation tasks will use
        # the BCEWithLogitsLoss. The segmentation will be weighted, while the whole image binary
        # will be unweighted.
        self.seg_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]))
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
        self.train_dice = Dice(ignore_index=0)
        self.val_dice = Dice(ignore_index=0)

    def forward(self, x):
        """
        The forward method will receive a set of images, which may or may not be preprocessed.
        The assumption is that they will be resized to the `binary_image_size` before being passed through
        the binary image model, and will also be passed just like that through the segmentation model.
        :param x:
        :return:
        """
        transformed = TF.resize(x, [self.binary_image_size, self.binary_image_size])
        # Get binary outputs, and apply sigmoid on them, and squeeze them so they are flattened.
        binary_outputs = self.binary_model(transformed).sigmoid().squeeze()
        # Get the locations where the binary model predicted presence of a contrail, and pass the original images
        # through to the segmentation model to get masks.
        locs = torch.where(binary_outputs >= self.hparams.seg_threshold)
        seg_inputs = x[locs]
        # Because this is binary, the segmodel will return (samples, 1, image size), so squeeze it out.
        # Then, apply sigmoid to it, so we can extract the probabilities, and finally convert to 0, 1.
        res = self.seg_model(seg_inputs).squeeze().sigmoid().cpu().numpy()
        # For the result, we need to collapse into a mask by calling argmax on each pixel.
        class_outputs = res.round()
        # Next, for the images where we don't have predicted contrails, we substitute empty masks.
        # To do so, first create a full zero array the same shape as the input.
        # Using locs, find where we must replace the zero masks with the actual masks.
        output = np.zeros((len(x),) + x.shape[2:])
        output[locs[0].cpu().numpy()] = class_outputs
        return output

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
        return optimizerB, optimizerS

    def training_step(self, batch, batch_idx):
        """
        For the training step, we get as input a CombinedLoader, so we extract both the images
        for the binary model and the ones for the segmentation model, and train both depending
        on the selected optimizer.
        :param batch:
        :param batch_idx:
        :param dataloader_idx:
        :return:
        """
        b_opt, s_opt = self.optimizers()
        # Get the full list of binary labels and its masks.
        # We will only forward the images with contrails to the segmentation model.
        # While the binary model will get all the images.
        images, binary_labels, masks = batch
        contrail_image_locs = torch.where(binary_labels == 1)
        seg_inputs = images[contrail_image_locs]
        seg_masks = masks[contrail_image_locs]

        # Train binary model
        output = self.binary_model(images)
        accuracy = self.accuracy(output.squeeze(), binary_labels)
        self.train_b_acc(accuracy)
        # Calculate loss
        b_loss = self.binary_loss(output.squeeze(), binary_labels.to(float))
        self.log('train/b_loss', b_loss, prog_bar=False)
        self.log('train/b_acc', self.train_b_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        b_opt.zero_grad()
        self.manual_backward(b_loss)
        b_opt.step()

        # Train segmentation model with just the contrail images
        # There is a small chance that the batch does not contain ANY
        # contrail images. In this case, just skip this batch.
        output_masks = self.seg_model(seg_inputs)
        if len(output_masks) > 0:
            accuracy = self.accuracy(output_masks, seg_masks)
            self.train_s_acc(accuracy)
            # Calculate loss (it's still binary cross entropy, but the weightings are different)
            s_loss = self.seg_loss(output_masks, seg_masks)
            self.log('train/s_loss', s_loss, prog_bar=False)
            self.log('train/s_acc', self.train_s_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            s_opt.zero_grad()
            self.manual_backward(s_loss)
            s_opt.step()

    def validation_step(self, batch, batch_idx):
        """
        Just like training, we'll also report binary accuracy and segmentation accuracy.
        But we'll also report the dice value on just the contrails.
        :param batch:
        :param batch_idx:
        :return:
        """
        images, binary_labels, masks = batch
        contrail_image_locs = torch.where(binary_labels == 1)
        seg_inputs = images[contrail_image_locs]
        seg_masks = masks[contrail_image_locs]

        output = self.binary_model(images)
        accuracy = self.accuracy(output.squeeze(), binary_labels)
        self.val_b_acc(accuracy)
        self.log('val/b_acc', self.val_b_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        output_masks = self.seg_model(seg_inputs)
        accuracy = self.accuracy(output_masks, seg_masks)
        self.val_s_acc(accuracy)
        self.val_dice(output_masks, seg_masks.to(torch.int))
        self.log('val/s_acc', self.val_s_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/s_dice', self.val_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)


# Source: https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master
class GRContrailsDeepLabResNet50(pl.LightningModule):
    def __init__(self, optimizer: torch.optim, output_stride=8, aspp_dilate=[12, 24, 36], inplanes=2048, low_level_planes=256,
                 return_layers={'layer4': 'out', 'layer1': 'low_level'}):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        # We will build a DeepLabV3+ model with a ResNet50 backbone.
        # TODO: Allow other backbones (may not be necessary)
        # Get the ResNet50 model from the URL
        RESNET50_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        model = ResNet(Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=[False, True, True])
        state_dict = load_state_dict_from_url(RESNET50_URL, progress=True)
        model.load_state_dict(state_dict)
        # Create the classifier, and attach the backbone and the classifier to create the full model.
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, 2, aspp_dilate)
        backbone = IntermediateLayerGetter(model, return_layers=return_layers)
        self.final_model = DeepLabV3(backbone, classifier)

        # Create loss and metric classes
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 99], dtype=torch.float32))
        self.accuracy = MulticlassAccuracy(num_classes=2)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()

        # Create additional variables for the dice coefficients.
        self.train_dice = Dice()
        self.val_dice = Dice()

    def forward(self, x):
        return self.final_model(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer(params=self.parameters())
        # Set the learning rate scheduler
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):
        input_images, masks = batch
        output_masks = self(input_images)
        masks = masks.squeeze()
        # Calculate training accuracy, training loss, and training dice. Only log the dice since that's important.
        # With DeepLabV3, the output is 2 x 256 x 256, so we have to call argmax before passing through BinaryAccuracy
        accuracy = self.accuracy(output_masks, masks)
        loss = self.loss(output_masks, masks.to(int))
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
        masks = masks.squeeze()
        # Calculate training accuracy, training loss, and training dice. Only log the dice since that's important.
        accuracy = self.accuracy(output_masks, masks)
        loss = self.loss(output_masks, masks.to(int))
        dice = self.val_dice(output_masks, masks.to(int))
        # Update accuracy and loss.
        self.val_acc(accuracy)
        self.val_loss(loss)
        # Log
        self.log('val/dice', self.val_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': accuracy}














