import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb

from omegaconf import DictConfig, OmegaConf

class SegmentationImageCallback(Callback):
    def __init__(self, num_samples=10, num_classes=3, class_labels: DictConfig = None):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.class_labels: dict = OmegaConf.to_object(class_labels)

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

        trainer.logger.experiment.log({
            'examples': [wandb.Image(image.transpose((1, 2, 0)), masks={
                'predictions': {'mask_data': pred, 'class_labels': self.class_labels},
                'ground_truth': {'mask_data': mask, 'class_labels': self.class_labels}
            }) for image, pred, mask in zip(images, preds, masks)]
        })


