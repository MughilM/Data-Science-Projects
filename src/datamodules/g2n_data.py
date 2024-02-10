"""
File: src/datamodules/g2n_data.py
Creation Date: 2023-12-02

Contains definitions for the G2Net Continuous Gravitational Wave Detection challenge, located at
https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/overview.
Max score on private leaderboard is around 0.85 AUC.
Total data size is around 227 GB (most of this is the test data), so please ensure enough space is available
on host machine for downloading and unzipping.
"""
import os
from typing import Optional
import zipfile
import kaggle
import glob
import logging

from sklearn.model_selection import train_test_split
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class G2NetCGWDataset(Dataset):
    def __init__(self, data_dir: str, file_label_df: pd.DataFrame):
        self.data_dir = data_dir
        self.file_label_df = file_label_df

    def __len__(self):
        return self.file_label_df.shape[0]

    def __getitem__(self, idx):
        pass




