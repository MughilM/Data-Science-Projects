"""
File: cancer_module.py
Creation Date: 2023-01-07

Contains the DataModule definition for the histopathologic-cancer-detection
Kaggle competition. Also contains the accompanying pure Dataset definition
"""
import os
from typing import Optional
import zipfile
import kaggle
import glob

from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


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


class CancerDataModule(pl.LightningDataModule):
    def __init__(self, comp_name: str = 'histopathologic-cancer-detection', data_dir: str = 'data/',
                 downsample_n: int = 10000, validation_split: float = 0.2,
                 batch_size: int = 2048, num_workers: int = 1, pin_memory: bool = True,
                 image_size: int = 32):
        super().__init__()
        # Saving the hyperparameters allows all the parameters to be accessible with self.hparams
        self.save_hyperparameters()
        self.COMP_DATA_PATH = os.path.join(self.hparams.data_dir, self.hparams.comp_name)
        # Transforms
        self.train_transform = T.Compose([
            T.CenterCrop(self.hparams.image_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor()
        ])
        self.vali_test_transform = T.Compose([T.CenterCrop(self.hparams.image_size), T.ToTensor()])

        self.train_dataset: Optional[Dataset] = None
        self.vali_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 1

    def prepare_data(self) -> None:
        """
        Download data from Kaggle if we need. No assignments here.
        :return:
        """
        # Download the competition files if the directory doesn't exist
        if not os.path.exists(self.COMP_DATA_PATH):
            print(f'Downloading {self.hparams.comp_name} competition files...')
            kaggle.api.competition_download_files(self.hparams.comp_name, path=self.hparams.data_dir, quiet='False')
            print(f'Extracting contents into {self.COMP_DATA_PATH}...')
            with zipfile.ZipFile(os.path.join(self.hparams.data_dir, f'{self.hparams.comp_name}.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.COMP_DATA_PATH)
            print('Deleting zip file...')
            os.remove(os.path.join(self.hparams.data_dir, f'{self.hparams.comp_name}.zip'))

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load the data and actually assign the Dataset objects (train, vali, and test).
        This is where the splitting happens as well.
        :param stage:
        :return:
        """
        if not self.train_dataset and not self.vali_dataset and not self.test_dataset:
            # Read the training labels from the standalone csv
            labels = pd.read_csv(os.path.join(self.COMP_DATA_PATH, 'train_labels.csv'))
            # Downsample however many we need
            # If downsample_n = -1, then we take the entire dataset.
            if self.hparams.downsample_n != -1:
                labels = labels.sample(n=self.hparams.downsample_n)
            # Split training and validation
            train_files, validation_files = train_test_split(labels, test_size=self.hparams.validation_split)
            print(f'Number of images in training: {len(train_files)}')
            print(f'Number of images in validation: {len(validation_files)}')
            # Now load the Dataset objects
            self.train_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'train'),
                                               file_id_df=train_files,
                                               transform=self.train_transform)
            self.vali_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'train'),
                                              file_id_df=validation_files,
                                              transform=self.vali_test_transform)
            self.test_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'test'),
                                              transform=self.vali_test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.vali_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)