"""
File: src/custom_callbacks.py

This file contains any custom callbacks in the form of classes. They will be used in
the callbacks/default.yaml configuration file. Some are general-use, while others are
specific to the dataset due to other considerations.
"""

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from src.datamodules.gr_contrails import GRContrailsFalseColorDataset

class SegmentationImageCallback(Callback):
    def __init__(self, num_samples=10, num_classes=3, class_labels: DictConfig = None, wandb_enabled: bool = True):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.class_labels: dict = OmegaConf.to_object(class_labels)
        self.wandb_enabled = wandb_enabled

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Grab a batch of validation and masks.
        self.val_images, self.val_masks = next(iter(trainer.val_dataloaders))
        # Only grab the first num_samples
        self.val_images = self.val_images[:self.num_samples]
        self.val_masks = self.val_masks[:self.num_samples]
        # Get the prediction
        self.val_images = self.val_images.to(pl_module.device)
        preds = pl_module(self.val_images)
        if self.num_classes == 2:
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            # Get rid of the singleton dimension...
            preds = preds.squeeze().cpu().numpy()
        else:
            preds = torch.argmax(preds, dim=1).cpu().numpy()

        # class_labels = {0: 'object', 1: 'blank', 2: 'edge'}

        images = self.val_images.cpu().numpy()
        masks = self.val_masks.squeeze().cpu().numpy()

        if self.wandb_enabled:
            trainer.logger.experiment.log({
                'examples': [wandb.Image(image.transpose((1, 2, 0)), masks={
                    'predictions': {'mask_data': pred, 'class_labels': self.class_labels},
                    'ground_truth': {'mask_data': mask, 'class_labels': self.class_labels}
                }) for image, pred, mask in zip(images, preds, masks)]
            })

        else:
            # If wandb is disabled, then create a grid of however many images we have,
            # and color them. For matplotlib, the channels need to be last.
            images = images.transpose((0, 2, 3, 1))
            # Each image will be "6 x 6", and we'll have 3 columns, with num_samples rows.
            plt.figure(figsize=(18, 6 * self.num_samples))
            # Our indices go 1, 4, 7, ..., until 3n - 2, where n = num_samples
            for image, pred, mask, i in zip(images, preds, masks, range(1, 3 * self.num_samples - 1, 3)):
                # Plot the original image, the prediction, and the ground truth
                ax = plt.subplot(self.num_samples, 3, i)
                ax.imshow(image)
                ax = plt.subplot(self.num_samples, 3, i + 1)
                ax.imshow(pred)
                ax = plt.subplot(self.num_samples, 3, i + 2)
                ax.imshow(mask)
            plt.savefig('./seg_images.png')


class ContrailCallback(Callback):
    """
    This callback is specifically for the contrail image segmentation task.
    Due to the wide range of inputs, from empty to images to ones full of contrails,
    taking a random subset of the whole dataset will not yield satisfying test cases.
    Therefore, this callback is designed to accept a specific list of samples to act as test cases.
    The sample_list is expected to come from the validation dataset.
    """
    def __init__(self, image_dir: str, sample_list: list, wandb_enabled: bool = True):
        super().__init__()
        self.sample_list = sample_list
        self.num_samples = len(self.sample_list)
        self.wandb_enabled = wandb_enabled
        # Create a FalseColorImageDataset from the given sample_list
        dataset = GRContrailsFalseColorDataset(image_dir, sample_list, test=False)
        # Gather up the images and its mask labels
        self.val_images = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
        self.val_masks = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)

    # def __plot_contrails(self, trainer: pl.Trainer, pl_module: pl.LightningModule, ):

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Every time the validation ends, test the model
        self.val_images = self.val_images.to(pl_module.device)
        preds = pl_module(self.val_images)
        # Anything larger than 0.5 is a contrail
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        # Get rid of the singleton dimension...
        preds = preds.squeeze().cpu().numpy()

        images = self.val_images.cpu().numpy()
        masks = self.val_masks.squeeze().cpu().numpy()

        class_labels = {0: 'sky', 1: 'contrail'}
        # Either log to W & B or save a local file.
        if self.wandb_enabled:
            trainer.logger.experiment.log({
                'examples': [wandb.Image(image.transpose((1, 2, 0)), masks={
                    'predictions': {'mask_data': pred, 'class_labels': class_labels},
                    'ground_truth': {'mask_data': mask, 'class_labels': class_labels}
                }) for image, pred, mask in zip(images, preds, masks)]
            })

        else:
            # If wandb is disabled, then create a grid of however many images we have,
            # and color them. For matplotlib, the channels need to be last.
            images = images.transpose((0, 2, 3, 1))
            # Each image will be "6 x 6", and we'll have 3 columns, with num_samples rows.
            plt.figure(figsize=(18, 6 * self.num_samples))
            # Our indices go 1, 4, 7, ..., until 3n - 2, where n = num_samples
            for image, pred, mask, i in zip(images, preds, masks, range(1, 3 * self.num_samples - 1, 3)):
                # Plot the original image, the prediction, and the ground truth
                ax = plt.subplot(self.num_samples, 3, i)
                ax.imshow(image)
                ax = plt.subplot(self.num_samples, 3, i + 1)
                ax.imshow(pred)
                ax = plt.subplot(self.num_samples, 3, i + 2)
                ax.imshow(mask)
            plt.savefig('./seg_images.png')

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        This will be called when we run trainer.test, and it should be on the test dataset.
        :param trainer:
        :param pl_module:
        :return:
        """
        pass


