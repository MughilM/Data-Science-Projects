"""
File: tgs_salt_module.py
Creation Date: 2023-06-05

This file contains DataModule definitions for the TGS Salt Identification
Kaggle challenge. The segmentation salt images are 101 x 101 square images.
"""
import os
import zipfile
import kaggle
import glob
from typing import Optional
import logging

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
import torchvision.transforms.functional as TF

logger = logging.getLogger('train.datamodule.tgs_salt')

class TGSSaltDataset(Dataset):
    def __init__(self, data_dir, hflip=True, vflip=True, train=True, file_ids=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.mask_dir = os.path.join(self.data_dir, 'masks')
        self.hflip = hflip
        self.vflip = vflip
        self.train = train

        # Get the list of file IDs
        self.file_ids = file_ids


    def transform(self, image, mask):
        # We need to transform the image and mask in the same way
        # e.g. give it the same horizontal flip at random probability...
        if self.hflip and torch.rand(1)[0].item() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if self.vflip and torch.rand(1)[0].item() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Convert to tensors, and convert mask to integer
        image = TF.to_tensor(image)
        mask = TF.convert_image_dtype(TF.to_tensor(mask), dtype=torch.uint8)
        return image, mask

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        # Read the image and its mask at the corresponding index.
        file_id = self.file_ids[idx]
        image_path = os.path.join(self.image_dir, file_id)
        mask_path = os.path.join(self.mask_dir, file_id)
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        # To ensure the same random transformation is applied to both the image
        # and the mask, we create a dictionary and call the transform on it as a whole.
        # Only transform if we are dealing with the training dataset
        if self.train:
            image, mask = self.transform(image, mask)
        # Otherwise, only convert to tensor and the target dtype
        else:
            image = TF.to_tensor(image)
            mask = TF.convert_image_dtype(TF.to_tensor(mask), dtype=torch.uint8)
            # We have 0 and 255 here, clip values to 0 and 1
            mask.clip_(0, 1)
        return image, mask


class TGSSaltDataModule(pl.LightningDataModule):
    def __init__(self, comp_name: str = 'tgs-salt-identification-challenge', data_dir: str = 'data/',
                 downsample_n: int = 1000, validation_split: float = 0.2,
                 batch_size: int = 1024, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.COMP_DATA_PATH = os.path.join(self.hparams.data_dir, self.hparams.comp_name)

        self.train_dataset: Optional[Dataset] = None
        self.vali_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 1

    def prepare_data(self) -> None:
        """
        Download TGS Salt from Kaggle. No assignments here.
        :return:
        """
        if not os.path.exists(self.COMP_DATA_PATH):
            logger.info(f'Downloading {self.hparams.comp_name} competition files...')
            kaggle.api.competition_download_files(self.hparams.comp_name, path=self.hparams.data_dir, quiet=False)
            logger.info(f'Extracting contents into {self.COMP_DATA_PATH}...')
            with zipfile.ZipFile(os.path.join(self.hparams.data_dir, f'{self.hparams.comp_name}.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.COMP_DATA_PATH)
            os.remove(os.path.join(self.hparams.data_dir, f'{self.hparams.comp_name}.zip'))
            logger.info('Extracting sub zip files...')
            files = ['competition_data', 'flamingo', 'test', 'train']
            for file in files:
                with zipfile.ZipFile(os.path.join(self.COMP_DATA_PATH, f'{file}.zip'), 'r') as zip_ref:
                    zip_ref.extractall(self.COMP_DATA_PATH)
                    # Delete zip file
                os.remove(os.path.join(self.COMP_DATA_PATH, f'{file}.zip'))

    def setup(self, stage: str) -> None:
        """
        Load the training, validation, and testing Datasets. We will be using
        the competition-data subfolder...
        :param stage:
        :return:
        """
        if not self.train_dataset and not self.vali_dataset and not self.test_dataset:
            # Read the list of training files available
            train_datapath = os.path.join(self.COMP_DATA_PATH, 'competition_data', 'train')
            test_datapath = os.path.join(self.COMP_DATA_PATH, 'competition_data', 'test')
            all_files = glob.glob(os.path.join(train_datapath, 'images', '*.png'))
            test_files = glob.glob(os.path.join(test_datapath, 'images', '*.png'))
            # Convert all to basenames...
            all_files = [os.path.basename(f) for f in all_files]
            test_files = [os.path.basename(f) for f in test_files]
            # Downsample, if applicable
            if self.hparams.downsample_n != -1:
                all_files = np.random.choice(all_files, size=self.hparams.downsample_n, replace=False)
            # Split training and validation
            train_files, vali_files = train_test_split(all_files, test_size=self.hparams.validation_split)
            logger.info(f'Number of training images: {len(train_files)}')
            logger.info(f'Number of validation images: {len(vali_files)}')
            # Load the dataset objects, only transforming the training images...
            self.train_dataset = TGSSaltDataset(data_dir=train_datapath, file_ids=train_files, train=True)
            self.vali_dataset = TGSSaltDataset(data_dir=train_datapath, file_ids=vali_files, train=False)
            self.test_dataset = TGSSaltDataset(data_dir=test_datapath, file_ids=test_files, train=False)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.vali_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)



