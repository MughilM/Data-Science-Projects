"""
File: datasets.py
Creation Date: 2025-06-18

Contains Dataset definitions which are to be used along with the Lightning DataModules
defined in the datamodules.py file. Inheritance will be done if it is deemed necessary,
as Datasets have much greater variability.
"""
import os
from typing import Optional
import kaggle
import glob
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

logger = logging.getLogger('train.datasets')

class CancerDataset(Dataset):
    def __init__(self, data_folder, file_id_df=None, transform=T.Compose([T.CenterCrop(32), T.ToTensor()]),
                 dict_labels={}):
        self.data_folder = data_folder
        # Create the file list from the IDs
        if file_id_df is None:
            self.file_ids = [file[:-4] for file in glob.glob(os.path.join(data_folder, '*.tif'))]
            self.labels = [-1] * len(self.file_ids)  # This should not be used in test_step for the model
        else:
            self.file_ids = [os.path.join(self.data_folder, file_id) for file_id in file_id_df['id']]
            self.labels = file_id_df['label'].values
        self.transform = transform
        self.dict_labels = dict_labels

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        # Read the image using the filepath and apply the transform
        image = Image.open(f'{self.file_ids[idx]}.tif')
        image = self.transform(image)
        # Return the label as well...
        return image, self.labels[idx]

class GRContrailsFalseColorDataset(Dataset):
    """
    A Dataset object for the GR Contrails competition. This dataset represents the false color
    image version. Of the 9 bands that are included for each sample, a false color image will be
    generated using the "Ash" color scheme as detailed https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf.
    This color scheme utilizes bands 11, 14, 15 (IR wavelengths 8.4 μm, 11.2 μm, and 12.3 μm respectively).
    Since it's a color image, it will have 3 channels, and will be standardized by subtracting mean and dividing
    by standard deviation.

    There is also an option to yield binary labels instead of the raw mask values.
    # TODO: Add time steps and transforms later
    """

    def __init__(self, image_dir, directory_ids=None, test=False, binary=False):
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
        self.binary = binary

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
        image = torch.from_numpy(np.stack((red, green, blue), axis=0))
        # Read in the mask, unless this is a test directory
        if not self.test:
            mask = np.load(os.path.join(self.image_dir, dir_id, 'human_pixel_masks.npy'))
            # Mask is 256 x 256 x 1, do a transpose so both input image and mask are the same shape.
            # Also convert to float, since it will be compared with binary cross entropy.
            mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(float)
            # Include binary labels if we need to.
            if self.binary:
                return image, int(torch.any(mask)), mask
            else:
                return image, mask
        # Test samples don't include ground truth, so just return the image.
        else:
            return image