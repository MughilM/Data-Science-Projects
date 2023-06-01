import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb

class SegmentationImageCallback(Callback):
    def __init__(self, dm: pl.LightningDataModule, num_samples=10):
        super().__init__()
        self.val_images, self.val_masks = next(iter(dm.val_dataloader()))
        # Only grab the first num_samples
        self.val_images = self.val_images[:num_samples]
        self.val_masks = self.val_masks[:num_samples]
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Get the prediction
        self.val_images = self.val_images.to(pl_module.device)
        preds = pl_module(self.val_images)
        preds = torch.argmax(preds, dim=1).cpu().numpy()

        class_labels = {0: 'object', 1: 'blank', 2: 'edge'}

        images = self.val_images.cpu().numpy()

        trainer.logger.experiment.log({
            'examples': [wandb.Image(image.transpose((1, 2, 0)), masks={
                'predictions': {'mask_data': pred, 'class_labels': class_labels},
                'ground_truth': {'mask_data': mask, 'class_labels': class_labels}
            }) for image, pred, mask in zip(images, preds, self.val_masks.cpu().numpy())]
        })


