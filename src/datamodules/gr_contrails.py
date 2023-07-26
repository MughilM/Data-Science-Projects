"""
File: gr_contrails.py
Creation Date: 2023-07-10

This file contains DataModule definitions for the Google Research Contrails challenge.
This is an image segmentation challenge where the goal is to predict the location of plane contrails
in a satellite image given input time series images from a series of IR image bands.

WARNING: The training data is 450 GB in size, while the validation data is 33 GB.
Please ensure that you have space in your disk to contain all the data if you need.
"""
import os
import sys
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

import pytorch_lightning as pl
import torchvision.transforms.functional as TF

from src.utils.general_funcs import *

logger = logging.getLogger('train.datamodule.gr_contrails')

class GRContrailsFalseColorDataset(Dataset):
    """
    A Dataset object for the GR Contrails competition. This dataset represents the false color
    image version. Of the 9 bands that are included for each sample, a false color image will be
    generated using the "Ash" color scheme as detailed https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf.
    This color scheme utilizes bands 11, 14, 15 (IR wavelengths 8.4 μm, 11.2 μm, and 12.3 μm respectively).
    Since it's a color image, it will have 3 channels, and will be standardized by subtracting mean and dividing
    by standard deviation.
    # TODO: Add time steps and transforms later
    """
    def __init__(self, image_dir, directory_ids=None, test=False):
        """
        If directory_ids is None, then all the samples in image_dir will be read.
        :param image_dir:
        :param directory_ids:
        """
        super().__init__()
        self.image_dir = image_dir
        if directory_ids is None:
            # Get a list of all the subdirectories in image_dir.
            # The first element is the directory itself, so index it out.
            self.directory_ids = [os.path.basename(subdir) for subdir, _, _ in os.walk(self.image_dir)][1:]
        else:
            self.directory_ids = directory_ids
        self.test = test

    def __len__(self):
        return len(self.directory_ids)

    def __getitem__(self, idx):
        dir_id = self.directory_ids[idx]
        # Read in bands 11, 14, and 15 at the exact time step.
        band_11 = np.load(os.path.join(self.image_dir, dir_id, 'band_11.npy'))[:, :, 5]
        band_14 = np.load(os.path.join(self.image_dir, dir_id, 'band_14.npy'))[:, :, 5]
        band_15 = np.load(os.path.join(self.image_dir, dir_id, 'band_15.npy'))[:, :, 5]
        # Calculate R, G, and B channels
        red = ((band_15 - band_14 + 4) / (2 + 4)).clip(0, 1)
        green = ((band_14 - band_11 + 4) / (5 + 4)).clip(0, 1)
        blue = ((band_11 - 243) / (303 - 243)).clip(0, 1)
        # Concatenate them to create a false color image.
        # Do CHANNELS FIRST ordering (axis=0), the default for PyTorch.
        image = np.stack((red, green, blue), axis=0)
        # Read in the mask, unless this is a test directory
        if not self.test:
            mask = np.load(os.path.join(self.image_dir, dir_id, 'human_pixel_masks.npy'))
            # Mask is 256 x 256 x 1, do a transpose so both input image and mask are the same shape.
            # Also convert to float, since it will be compared with binary cross entropy.
            return torch.from_numpy(image), torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(float)
        else:
            return torch.from_numpy(image)


# TODO: Allow for different kinds of backend Dataset objects (false color, using all bands, etc.)
class GRContrailDataModule(pl.LightningDataModule):
    def __init__(self, comp_name: str = 'google-research-identify-contrails-reduce-global-warming',
                 data_dir: str = 'data/', frac: float = 1.0, batch_size: int = 128, num_workers: int = 4,
                 pin_memory: bool = True, use_val_as_train: bool = True, train_url: str = None,
                 val_url: str = None, test_url: str = None):
        super().__init__()
        if (frac > 1) or (frac <= 0):
            logger.error(f'Invalid value for fraction ({frac})!')
            sys.exit(1)
        self.save_hyperparameters()
        self.COMP_DATA_PATH = os.path.join(self.hparams.data_dir, self.hparams.comp_name)

        self.train_dataset: Optional[Dataset] = None
        self.vali_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        Download the Contrail dataset from Kaggle. If use_validation_as_training is True,
        then DOES NOT CHECK for the training set, as that is huge. It will only check for
        existence of the validation and test directories, and will download them if they are absent.
        :return:
        """
        # Check for existence of each of validation, test, and training folders.
        # Testing
        if not os.path.exists(os.path.join(self.COMP_DATA_PATH, 'test')):
            logger.info(f'Downloading {self.hparams.comp_name} test competition files...')
            test_zip_path = download_file_url(self.hparams.test_url, self.COMP_DATA_PATH, 'test.zip')
            with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.COMP_DATA_PATH, 'test'))
            os.remove(test_zip_path)
        # Validation
        if not os.path.exists(os.path.join(self.COMP_DATA_PATH, 'validation')):
            logger.info(f'Downloading {self.hparams.comp_name} validation competition files...')
            # The Kaggle API does not allow downloading of individual folders, so use the requests package directly.
            val_zip_path = download_file_url(self.hparams.val_url, self.COMP_DATA_PATH, 'validation.zip')
            with zipfile.ZipFile(val_zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.COMP_DATA_PATH, 'validation'))
            os.remove(val_zip_path)
        # If we need to download the training data and it's not there.
        if not os.path.exists(os.path.join(self.COMP_DATA_PATH, 'train')) and not self.hparams.use_val_as_train:
            logger.info(f'Downloading TRAINING {self.hparams.comp_name} competition files...')
            train_zip_path = download_file_url(self.hparams.train_url, self.COMP_DATA_PATH, 'train.zip')
            with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.COMP_DATA_PATH, 'train'))
            os.remove(train_zip_path)

    def setup(self, stage: str) -> None:
        """
        Set up the dataset objects. We'll be using the subfolders in the competition folder.
        If use_val_as_train is true, then the training and validation loader will be the same,
        with the test loader being unchanged.
        :param stage:
        :return:
        """
        if not self.train_dataset and not self.vali_dataset and not self.test_dataset:
            if self.hparams.use_val_as_train:
                train_datapath = os.path.join(self.COMP_DATA_PATH, 'validation')
            else:
                train_datapath = os.path.join(self.COMP_DATA_PATH, 'train')
            val_datapath = os.path.join(self.COMP_DATA_PATH, 'validation')
            test_datapath = os.path.join(self.COMP_DATA_PATH, 'test')
            # Create a sublist if frac is not 1.
            if self.hparams.frac != 1:
                train_files = [os.path.basename(subdir) for subdir, _, _ in os.walk(train_datapath)][1:]
                train_files = np.random.choice(train_files, size=int(self.hparams.frac * len(train_files)), replace=False)
                self.train_dataset = GRContrailsFalseColorDataset(train_datapath, train_files)
            else:
                self.train_dataset = GRContrailsFalseColorDataset(train_datapath)
            # Simple setup of validation and testing files
            self.vali_dataset = GRContrailsFalseColorDataset(val_datapath)
            # self.test_dataset = GRContrailsFalseColorDataset(test_datapath, test=True)
            self.test_dataset = GRContrailsFalseColorDataset(val_datapath, test=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.vali_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)


