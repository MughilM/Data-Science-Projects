from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T

import pytorch_lightning as pl


def decrement_mask_squeeze(x):
    return (x - 1).squeeze()


class OxfordDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, comp_name: str = 'oxford', image_size: int = 128, batch_size: int = 1024,
                 downsample_n: int = 1000, validation_split: float = 0.2,
                 num_workers: int = 2, pin_memory: bool = True):
        super().__init__()
        self.save_hyperparameters()

        # Define the image and mask transforms
        self.image_transform = T.Compose([
            T.Resize((self.hparams.image_size, self.hparams.image_size)),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize((self.hparams.image_size, self.hparams.image_size)),
            T.PILToTensor(),
            T.Lambda(decrement_mask_squeeze)
        ])

        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None

    def prepare_data(self) -> None:
        # Just download the OxFord III Pet Dataset
        OxfordIIITPet(root=self.hparams.data_dir, target_types='segmentation', download=True,
                      transform=self.image_transform, target_transform=self.mask_transform)
        OxfordIIITPet(root=self.hparams.data_dir, target_types='segmentation', download=True, split='test',
                      transform=self.image_transform, target_transform=self.mask_transform)

    def setup(self, stage: str) -> None:
        # Assign training and validation based on stage.
        if stage == 'fit' or stage is None:
            oxford_full = OxfordIIITPet(root=self.hparams.data_dir, target_types='segmentation', download=True,
                                        transform=self.image_transform, target_transform=self.mask_transform)
            # Cut down some more if downsample is positive
            if self.hparams.downsample_n != -1:
                oxford_full, _ = random_split(oxford_full, lengths=[self.hparams.downsample_n,
                                                                    len(oxford_full) - self.hparams.downsample_n])
            self.train_dataset, self.val_dataset = random_split(oxford_full, lengths=[0.8, 0.2])
        if stage == 'test' or stage is None:
            self.test_dataset = OxfordIIITPet(root=self.hparams.data_dir, target_types='segmentation',
                                              download=True, split='test',
                                              transform=self.image_transform, target_transform=self.mask_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)
