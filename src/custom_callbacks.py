"""
File: src/custom_callbacks.py

This file contains any custom callbacks in the form of classes. They will be used in
the callbacks/default.yaml configuration file. Some are general-use, while others are
specific to the dataset due to other considerations.
"""
import sys
from typing import List, Optional
import io
import logging

import torch
from torchmetrics import ConfusionMatrix, Accuracy
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
import matplotlib.pyplot as plt
import plotly.express as px

from omegaconf import DictConfig, OmegaConf
from src.datamodules.gr_contrails_data import GRContrailsFalseColorDataset

log = logging.getLogger('train.callback')

class SegmentationImageCallback(Callback):
    def __init__(self, num_samples: int = 10, num_classes=3,
                 class_labels: DictConfig = None, wandb_enabled: bool = True):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.wandb_enabled = wandb_enabled

        self.val_images: Optional[torch.Tensor] = None
        self.val_masks: Optional[torch.Tensor] = None

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # At the start, get the set of validation images and masks to plot...
        # Grab a batch of validation and masks.
        self.val_images, self.val_masks = next(iter(trainer.val_dataloaders))
        # Only grab the first num_samples
        self.val_images = self.val_images[:self.num_samples]
        self.val_masks = self.val_masks[:self.num_samples]

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Get the prediction
        self.val_images = self.val_images.to(pl_module.device)
        preds = pl_module(self.val_images)
        if self.num_classes == 1:
            preds = preds.sigmoid()
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            # Get rid of the singleton dimension...
            preds = preds.squeeze().cpu().numpy()
        else:
            preds = preds.softmax(dim=1).argmax(dim=1).cpu().numpy()

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
            plt.close()


class ContrailCallback(SegmentationImageCallback):
    """
    This callback is specifically for the contrail image segmentation task.
    Due to the wide range of inputs, from empty to images to ones full of contrails,
    taking a random subset of the whole dataset will not yield satisfying test cases.
    Therefore, this callback is designed to accept a specific list of samples to act as test cases.
    The sample_list is expected to come from the validation dataset.

    It subclasses the SegmentationImageCallback, because the plotting method is exactly the same.
    The only difference is how it reads the images and masks, which is adjusted in __init__
    """
    def __init__(self, image_dir: str, sample_list: list, wandb_enabled: bool = True, num_classes=2):
        super().__init__(num_samples=len(sample_list), num_classes=num_classes, class_labels={0: 'sky', 1: 'contrail'},
                         wandb_enabled=wandb_enabled)
        self.sample_list = sample_list
        # Create a FalseColorImageDataset from the given sample_list
        dataset = GRContrailsFalseColorDataset(image_dir, sample_list, test=False)
        # Gather the images and its mask labels
        self.val_images = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
        self.val_masks = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # An override, because by default, the SegmentationCallback extracts samples from a single
        # batch, but we've already generated the list of images and masks to use.
        # So we do nothing here.
        pass

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        This will be called when we run trainer.test, and it should be on the test dataset.
        :param trainer:
        :param pl_module:
        :return:
        """
        pass


class PlotMulticlassConfusionMatrix(Callback):
    """
    This callback plots a simple confusion matrix, and logs it to Weights and Biases as well.
    This is designed to be used for multiclass classification, where each label is mutually exclusive.
    To plot a multilabel matrix, where each label is NOT exclusive, please use PlotMultilabelConfusionMatrix.
    The plot only happens at the end of validation.
    """
    def __init__(self, labels: List, matrix_attr: str = 'matrix', val_acc_attr: str = 'val_acc'):
        self.labels = list(labels)
        self.matrix_attr = matrix_attr
        self.val_acc_attr = val_acc_attr

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Get the objects for the confusion matrix and validation accuracy, and exit the
        # program if the specific ones aren't available.
        matrix: ConfusionMatrix = getattr(pl_module, self.matrix_attr)
        acc_metric: Accuracy = getattr(pl_module, self.val_acc_attr)

        if matrix is None:
            log.error(f'Matrix of name "{self.matrix_attr}" not available! Exiting...')
            sys.exit(1)
        if acc_metric is None:
            log.error(f'Accuracy metrix of name "{self.val_acc_attr}" not available! Exiting...')
            sys.exit(1)

        result: np.ndarray = pl_module.matrix.compute().cpu().numpy().T
        fig = px.imshow(result, text_auto=True, x=self.labels, y=self.labels,
                        title=f'Accuracy: {pl_module.val_acc.compute() * 100:2.3f}%')
        fig.update_xaxes(side='top', type='category', title='Actual')
        fig.update_yaxes(type='category', title='Predicted')
        img_bytes = fig.to_image(engine='kaleido')

        # Reset the matrix
        pl_module.matrix.reset()
        # Log it in W and B
        pl_module.logger.log_image('conf_matrix', [Image.open(io.BytesIO(img_bytes))])
