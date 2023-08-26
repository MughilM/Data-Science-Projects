"""
File: src/datamodules/candlestick.py
Creation Date: 2023-08-08

This file contains the definitions to generate candlestick plots, and their corresponding
labels given a large list of price history. The default data is minute-by-minute,
so there is a one-time generation of converting to daily. Then, depending on the look-back
time (in days), and the maximum date cutoff, it will generate candlestick plots
(with or without volume) and save them as pngs.

During actual loading, it will simply read these images, because it is difficult to know how many
images will be generated from the beginning.
"""
import os
import glob
import datetime as dt
import sys
from typing import Optional
import logging

import numpy as np
import pandas as pd
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from tqdm import tqdm
import mplfinance as mpf
from PIL import Image
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

logger = logging.getLogger('train.datamodule.candlestick')

class CandleDataset(Dataset):
    """
    The CandleDataset will produce an image and percent increase/decrease for the price x days in the future.
    The given folder will have subfolders for each stock, and each image will be labeled by the END DATE.
    """
    def __init__(self, image_folder: str, price_df: pd.DataFrame, target_day: int, transform=T.ToTensor()):
        super().__init__()
        self.image_folder = image_folder
        self.price_df = price_df
        self.target_day = target_day
        self.transform = transform

    def __len__(self):
        return len(self.price_df)

    def __getitem__(self, idx):
        """
        Given an index, read the file using the given filepath, and return the target price
        of the given day.
        :param idx:
        :return:
        """
        image_path = self.price_df.iloc[idx, 2]  # The third column is the image path
        image = Image.open(image_path).convert('RGB')
        # Apply the transforms
        image = self.transform(image)
        # Grab the correct price difference (column index 2 = target_day of 1, so add 1 to target_day)
        # Return 1 if it went up, 0 if it went down (or somehow stayed exactly the same)
        price_perc_diff = self.price_df.iloc[idx, self.target_day + 1]
        if price_perc_diff <= 0:
            return image, 0.0
        else:
            return image, 1.0
        # return image, price_perc_diff


class CandleDataModule(pl.LightningDataModule):
    """
    The CandleDataModule will produce a CandleDataset depending on the look-back and look-forward period
    in days. For each value of look-back, it has to generate a new set of images, which could take a while.
    For each value of look-forward, it has to generate another csv file with the target prices. Each set of images
    will go in its own subfolder under the images folder in NASDAQ 100. The name of the subfolder will be the
    value of look-back, in days. Each stock's images will also be in its own folder.
    """
    def __init__(self, comp_name: str, data_dir: str, look_back_days: int, max_look_forward_days: int,
                 target_look_forward_day: int, validation_start_date: dt.datetime = dt.datetime(2012, 1, 1),
                 batch_size: int = 128, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        self.save_hyperparameters()
        stock_folder = os.path.join(data_dir, comp_name)
        self.MINUTE_STOCK_PATH = os.path.join(stock_folder, 'minute_txt_files')
        self.DAILY_STOCK_PATH = os.path.join(stock_folder, 'dailies')
        self.IMAGE_DATA_PATH = os.path.join(stock_folder, 'images', f'{look_back_days}')
        os.makedirs(self.MINUTE_STOCK_PATH, exist_ok=True)
        os.makedirs(self.DAILY_STOCK_PATH, exist_ok=True)
        # os.makedirs(self.IMAGE_DATA_PATH, exist_ok=True)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        Only preparation, such as image generation takes place here. No assignment of the
        Dataset objects take place. If the image directory is empty, then we need to read all
        the stock files and produce the images, which might take a few hours.
        Additionally, if daily data is unavailable, we need a one-time generation of that.
        :return:
        """
        # Check if the daily directory is empty, if it is then we have to generate
        # the daily csvs for each stock and populate it.
        if len(os.listdir(self.DAILY_STOCK_PATH)) == 0:
            logger.info('Daily stocks not available. Generating...')
            # Use glob to get all the txt files
            minute_files = glob.glob(os.path.join(self.MINUTE_STOCK_PATH, '*.txt'))
            stock_names = [os.path.basename(file)[:-4] for file in minute_files]
            # For each file, read it, and generate the open, high, low, close,
            # and volume. Write it back under the stock name.
            for file, name in tqdm(zip(minute_files, stock_names)):
                stock = pd.read_csv(file, parse_dates=[0])
                grouped = stock.groupby('Date')
                open_indices = [min(index_list) for _, index_list in grouped.groups.items()]
                open_ = stock.loc[open_indices, ['Date', 'Open']].set_index('Date')
                high = grouped.max()['High']
                low = grouped.min()['Low']
                close_indices = [max(index_list) for _, index_list in grouped.groups.items()]
                close = stock.loc[close_indices, ['Date', 'Close']].set_index('Date')
                volume = grouped.sum()['Volume']

                stock = pd.concat((open_, high, low, close, volume), axis=1)
                stock.to_csv(os.path.join(self.DAILY_STOCK_PATH, f'{name}.csv'))
        # Check if the image directory for this specific look back period is there.
        # If it's not, then we have to take all the stocks and generate all the images
        # for each of them. This might take a while.
        if not os.path.exists(self.IMAGE_DATA_PATH):
            os.makedirs(self.IMAGE_DATA_PATH)
            labels = []
            # Get all the daily csv files
            daily_csvs = glob.glob(os.path.join(self.DAILY_STOCK_PATH, '*.csv'))
            # For each stock file, produce an image set for that stock.
            for csv in tqdm(daily_csvs):
                stock_name = os.path.basename(csv)[:-4]
                os.makedirs(os.path.join(self.IMAGE_DATA_PATH, stock_name), exist_ok=True)
                stock = pd.read_csv(csv, parse_dates=[0], index_col=0)
                # Give a little bit of buffer, we would be starting at look_back TRADING days,
                # not necessarily actual days. Slightly reduces the number of images we have,
                # but it's ok.
                offset = 3
                for i in trange(self.hparams.look_back_days + offset, len(stock) - self.hparams.max_look_forward_days):
                # for end_date in tqdm(stock.index[self.hparams.look_back_days + 3:-self.hparams.look_forward_days]):
                    end_date = stock.index[i]
                    # Find the start date for the period we are looking at using searchsorted
                    # start_date = stock.index[np.searchsorted(stock.index, end_date - dt.timedelta(days=self.hparams.look_back_days))]
                    # Next, actually plot it.
                    # It will go under a directory of the stock, and the filename will be the end_date
                    filename = os.path.join(self.IMAGE_DATA_PATH, stock_name, f'{end_date.strftime("%Y%m%d")}.png')
                    mpf.plot(stock.iloc[i - self.hparams.look_back_days : i + 1, :], type='candle', style='yahoo',
                             savefig=dict(fname=filename, dpi=50), volume=True, xrotation=20, tight_layout=True)
                    # Find the target stock prediction
                    # target_date = stock.index[np.searchsorted(stock.index, end_date + dt.timedelta(days=self.hparams.look_forward_days))]
                    # target_date = stock.index[i + self.hparams.max_look_forward_days]
                    target_price_diffs = (stock.iloc[i + 1: i + self.hparams.max_look_forward_days + 1]['Open'] -
                                         stock.loc[end_date, 'Close']) / stock.loc[end_date, 'Close']
                    labels.append([stock_name, end_date, filename] + target_price_diffs.tolist())
            # After all the images have been generated and saved, next we save
            # the target price dataframe into a csv.
            target_df = pd.DataFrame(data=labels, columns=['Stock', 'Date', 'Filename'] +
                                                          [f'Target Price ({d} Days)' for d in
                                                           range(1, self.hparams.max_look_forward_days + 1)])
            target_df.to_csv(os.path.join(self.IMAGE_DATA_PATH, 'target_prices.csv'), index=False)

    def setup(self, stage: str) -> None:
        """
        Set up the dataset objects. Each stock is in its own folder, and this is also where we do a cutdown
        of the full image dataset if we need to.
        The validation data is taken from the start of the validation start date. Everything before is training.
        :param stage:
        :return:
        """
        if not self.train_dataset and not self.val_dataset:
            logger.info(f'Validation start date: {self.hparams.validation_start_date}')
            cutoff_date = dt.datetime.fromisoformat(self.hparams.validation_start_date)
            prices = pd.read_csv(os.path.join(self.IMAGE_DATA_PATH, 'target_prices.csv'), parse_dates=[1])
            # Training is before the cutoff date, validation is after the cutoff date
            train_prices = prices[prices['Date'] <= cutoff_date]
            val_prices = prices[prices['Date'] > cutoff_date]
            logger.info(f'Training size: {train_prices.shape[0]}')
            logger.info(f'Validation size: {val_prices.shape[0]}')
            self.train_dataset = CandleDataset(self.IMAGE_DATA_PATH, train_prices,
                                               target_day=self.hparams.target_look_forward_day,
                                               transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
            self.val_dataset = CandleDataset(self.IMAGE_DATA_PATH, val_prices,
                                             target_day=self.hparams.target_look_forward_day,
                                             transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)





