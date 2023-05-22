"""
File: dataset.py
Model/Competition: GI Tract Image

This file holds the dataset definitions for the data that we use.
For the GI Tract images, we have all the images from the folder
specified. We also resize them so they're all the same size.
The masks undergo the same resizing, so the masks do not get messed up.
We also use Pytorch Lightning's implementation of a dataset.
"""
from typing import Optional
from pathlib import Path
import os.path

import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule


class TGSSaltDataset(Dataset):
    def __init__(self, image_folder, rle_masks, image_width, image_height, num_workers=1):
        super().__init__()
        self.image_folder = image_folder
        # Get the list of all the images present in the folder
        self.image_paths = [path.resolve() for path in Path(self.image_folder).rglob('*.png')]
        self.rle_masks = rle_masks  # A Series keyed by filename with the rle mask strings
        self.image_width = image_width
        self.image_height = image_height
        self.num_workers = num_workers
        # Transform, reszie the images and convert to tensors...
        self.transforms = transforms.Compose([
            transforms.Resize((self.image_width, self.image_height)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        return self.transforms(read_image(self.image_paths[index], mode=ImageReadMode.GRAY))


class TGSSaltDataModule(LightningDataModule):
    def __init__(self, image_folder, rle_masks, image_width, image_height,
                 batch_size, num_workers):
        self.dataset = TGSSaltDataset(image_folder, rle_masks, image_width, image_height)
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)
    
    def train_dataloader(self):
        return super().train_dataloader()
    
    def val_dataloader(self):
        return super().val_dataloader()
    
    def test_dataloader(self):
        return super().test_dataloader()
    
    def predict_dataloader(self):
        return super().predict_dataloader()
