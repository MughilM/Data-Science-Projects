"""
File: datamodules.py
Creation Date: 2025-06-18

Contains all datamodule definitions. Each datamodule is a subclass of pl.LightningDataModule.
Their corresponding Dataset classes (which are just regular torch Datasets) are in the sister
file datasets.py.

To streamline things and reduce the repetitive code, I have defined a BaseDataModule which
defines some consistent methods and parameters e.g. the {train|val|test}_dataloader functions
pretty much never change across classes.
"""
import sys
import os
from typing import Optional
import zipfile
import kaggle
import glob
import logging

from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torchvision.transforms as T

from src.datasets import *
from src.utils.general_funcs import *

logger = logging.getLogger('train.lit_datamodule')

# Base class to remove some redundancies
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, comp_name: str, data_dir: str, downsample_n: int, validation_split: float,
                 batch_size: int, num_workers: int, pin_memory: bool, image_size: int):
        super().__init__()
        # Saving the hyperparameters allows all the parameters to be accessible with self.hparams
        self.save_hyperparameters()
        self.train_dataset: Optional[Dataset] = None
        self.vali_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.vali_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, persistent_workers=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, persistent_workers=True)

class CancerDataModule(BaseDataModule):
    def __init__(self, comp_name: str = 'histopathologic-cancer-detection', data_dir: str = 'data/',
                 downsample_n: int = 10000, validation_split: float = 0.2,
                 batch_size: int = 2048, num_workers: int = 1, pin_memory: bool = True,
                 image_size: int = 32):
        super().__init__(comp_name, data_dir, downsample_n, validation_split, batch_size, num_workers,
                         pin_memory, image_size)
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
            logger.info(f'Downloading {self.hparams.comp_name} competition files...')
            kaggle.api.competition_download_files(self.hparams.comp_name, path=self.hparams.data_dir, quiet='False')
            logger.info(f'Extracting contents into {self.COMP_DATA_PATH}...')
            with zipfile.ZipFile(os.path.join(self.hparams.data_dir, f'{self.hparams.comp_name}.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.COMP_DATA_PATH)
            logger.info('Deleting zip file...')
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
            logger.info(f'Number of images in training: {len(train_files)}')
            logger.info(f'Number of images in validation: {len(validation_files)}')
            # Now load the Dataset objects
            self.train_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'train'),
                                               file_id_df=train_files,
                                               transform=self.train_transform)
            self.vali_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'train'),
                                              file_id_df=validation_files,
                                              transform=self.vali_test_transform)
            self.test_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'test'),
                                              transform=self.vali_test_transform)


# TODO: Allow for different kinds of backend Dataset objects (false color, using all bands, etc.)
class GRContrailDataModule(BaseDataModule):
    def __init__(self, comp_name: str = 'google-research-identify-contrails-reduce-global-warming',
                 data_dir: str = 'data/', frac: float = 1.0, batch_size: int = 128, num_workers: int = 4,
                 pin_memory: bool = True, use_val_as_train: bool = True, validation_split: float = 0.2,
                 train_url: str = None, val_url: str = None, test_url: str = None, binary: bool = False):
        super().__init__(comp_name, data_dir)
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
            # Use validation as train ==> do a train validation split on the full validation set.
            # We also want to include the frac value as well, in case that is less than one.
            # Subset based on frac ==> do train test split on validation if use_val_as_train is true,
            # on train if false.
            val_datapath = os.path.join(self.COMP_DATA_PATH, 'validation')
            if self.hparams.use_val_as_train:
                # Read all the subdirectories from the validation datapath
                all_files = [os.path.basename(subdir) for subdir, _, _ in os.walk(val_datapath)][1:]
                # If the fraction is less than 1, then subset it again.
                if self.hparams.frac < 1:
                    all_files = np.random.choice(all_files, size=int(self.hparams.frac * len(all_files)),
                                                 replace=False)
                # Next, use train_test_split to get a distinct set of files to use for training and validation.
                # This prevents data leakage and having to validate on the same set as training.
                train_files, val_files = train_test_split(all_files, test_size=self.hparams.validation_split)
                # Finally, create the Dataset objects
                self.train_dataset = GRContrailsFalseColorDataset(val_datapath, train_files, binary=self.hparams.binary)
                self.vali_dataset = GRContrailsFalseColorDataset(val_datapath, val_files, binary=self.hparams.binary)

            # If we ARE using the actual training set, then there is only a small change from the above,
            # in that we use the train datapath and do the cutdown on the training files rather than validation files.
            # The validation Dataset will read from the validation datapath completely.
            else:
                train_datapath = os.path.join(self.COMP_DATA_PATH, 'train')
                if self.hparams.frac < 1:
                    train_files = [os.path.basename(subdir) for subdir, _, _ in os.walk(val_datapath)][1:]
                    train_files = np.random.choice(train_files, size=int(self.hparams.frac * len(train_files)),
                                                   replace=False)
                    self.train_dataset = GRContrailsFalseColorDataset(train_datapath, train_files, binary=self.hparams.binary)
                else:
                    self.train_dataset = GRContrailsFalseColorDataset(train_datapath, binary=self.hparams.binary)
                self.vali_dataset = GRContrailsFalseColorDataset(val_datapath, binary=self.hparams.binary)

            # Create the test Dataset, doesn't matter about training and validation splits or whatever
            test_datapath = os.path.join(self.COMP_DATA_PATH, 'test')
            self.test_dataset = GRContrailsFalseColorDataset(test_datapath, binary=self.hparams.binary)

