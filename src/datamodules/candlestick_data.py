"""
File: src/datamodules/candlestick_data.py
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
import multiprocessing

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

    def __init__(self, image_folder: str, price_df: pd.DataFrame, target_day: int, transform: T.Compose,
                 min_price_increase: float = 0.0):
        super().__init__()
        self.image_folder = image_folder
        self.price_df = price_df
        self.target_day = target_day
        self.transform = transform
        self.min_price_increase = min_price_increase

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
        if price_perc_diff <= self.min_price_increase:
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
                 min_price_increase: float = 0.0, batch_size: int = 128, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        self.save_hyperparameters()
        stock_folder = os.path.join(data_dir, comp_name)
        self.MINUTE_STOCK_PATH = os.path.join(stock_folder, 'minute_txt_files')
        self.DAILY_STOCK_PATH = os.path.join(stock_folder, 'dailies')
        self.IMAGE_DATA_PATH = os.path.join(stock_folder, 'images', f'{look_back_days}')
        os.makedirs(self.MINUTE_STOCK_PATH, exist_ok=True)
        os.makedirs(self.DAILY_STOCK_PATH, exist_ok=True)
        # os.makedirs(self.IMAGE_DATA_PATH, exist_ok=True)

        self.train_transform = T.Compose([
            T.Resize((401, 401)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = T.Compose([
            T.Resize((401, 401)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def plot_all_stock_timeframes(self, stock_name: str, stock: pd.DataFrame, lock, offset: int, index: int):
        """
        Given a single stock dataframe, plots all the frames using mpf. This is designed
        to be wrapped around a multiprocessing pool.
        :param stock_name: Name of the stock
        :param stock: The DataFrame object for the stock's data
        :param lock: A lock for the multiprocessor
        :param offset: The number of days to offset the dataframe. The plots will start from look_back_days + offset
        :param index: The index of the called stock (controls the location of this stock's progress bar)
        :return:
        """
        total_plots = len(stock) - self.hparams.max_look_forward_days - self.hparams.look_back_days - offset + 1
        # Set the progress bar, we will have to update the bar manually.
        with lock:
            bar = tqdm(desc=stock_name, total=total_plots, position=index, leave=False)

        # Now actually go through the stock's frames and produce the plots. Additionally, return the percent changes
        # in prices as well.
        # labels = []
        for i in range(self.hparams.look_back_days + offset, len(stock) - self.hparams.max_look_forward_days):
            end_date = stock.index[i]
            # Find the start date for the period we are looking at using searchsorted
            # Next, actually plot it.
            # It will go under a directory of the stock, and the filename will be the end_date
            filename = os.path.join(self.IMAGE_DATA_PATH, stock_name, f'{end_date.strftime("%Y%m%d")}.png')
            mpf.plot(stock.iloc[i - self.hparams.look_back_days: i + 1, :], type='candle', style='yahoo',
                     savefig=dict(fname=filename, dpi=50), volume=True, xrotation=20, tight_layout=True)
            with lock:
                bar.update(1)
        # Find the target stock prediction
        #     target_price_diffs = (stock.iloc[i + 1: i + self.hparams.max_look_forward_days + 1]['Open'] -
        #                           stock.loc[end_date, 'Close']) / stock.loc[end_date, 'Close']
        #     labels.append([stock_name, end_date, filename] + target_price_diffs.tolist())
        # return pd.DataFrame(data=labels, columns=['Stock', 'Date', 'Filename'] +
        #                                          [f'Target Price ({d} Days)' for d in
        #                                           range(1, self.hparams.max_look_forward_days + 1)])

    def prepare_data(self) -> None:
        """
        Only preparation, such as image generation takes place here. No assignment of the
        Dataset objects take place. If the image directory is empty, then we need to read all
        the stock files and produce the images, which might take a few hours.
        Additionally, if daily data is unavailable, we need a one-time generation of that.
        :return:
        """
        # Set up the lock for the multiprocessor
        lock = multiprocessing.Manager().Lock()
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
        COLUMNS = ['Stock', 'Date', 'Filename'] + [f'Target Price ({d} Days)'
                                                   for d in range(1, self.hparams.max_look_forward_days + 1)]
        target_prices = pd.DataFrame(data=None, columns=COLUMNS)  # Will hold all our target prices.
        if not os.path.exists(self.IMAGE_DATA_PATH):
            os.makedirs(self.IMAGE_DATA_PATH)
            labels = []
            # Get all the daily csv files
            daily_csvs = glob.glob(os.path.join(self.DAILY_STOCK_PATH, '*.csv'))
            # Create batches of csv of 4 at a time. To handle the last batch, we adjust the
            # stop limit on the range.
            BATCH_SIZE = 4
            OFFSET = 3
            batched_csv = [daily_csvs[i: i + BATCH_SIZE]
                           for i in range(0, len(daily_csvs) + len(daily_csvs) % BATCH_SIZE, BATCH_SIZE)]
            for batch in tqdm(batched_csv[:-1]):
                # Read in each stock and append to ongoing dataframe
                for csv in batch:
                    stock_name = os.path.basename(csv)[:-4]
                    os.makedirs(os.path.join(self.IMAGE_DATA_PATH, stock_name), exist_ok=True)
                    stock = pd.read_csv(csv, parse_dates=[0], index_col=0)
                    # Give a little bit of buffer, we would be starting at look_back TRADING days,
                    # not necessarily actual days. Slightly reduces the number of images we have,
                    # but it's ok.
                    OFFSET = 3
                    # First, we will generate a table of the target prices for this.
                    # We need to calculate the percentage difference for each day's Open to the end_date's Close.
                    # Using consecutive price percentage changes, we can easily calculate this.
                    # np.diff() and np.cumprod() will be really useful, by concatenating all columns we need.
                    end_date_prices = (stock.iloc[self.hparams.look_back_days + OFFSET:
                                                  len(stock) - self.hparams.max_look_forward_days]['Close'].
                                       values.reshape(-1, 1))
                    # Concatenate price columns all at once, making sure to push each column forward by one day.
                    future_prices = [stock.iloc[self.hparams.look_back_days + OFFSET + day:
                                                len(stock) - self.hparams.max_look_forward_days + day]
                                     ['Open'].values.reshape(-1, 1)
                                     for day in range(1, self.hparams.max_look_forward_days + 1)]
                    all_prices = np.concatenate([end_date_prices] + future_prices, axis=1)
                    # To calculate percentage change from each day's Open to the first day's Close, we use
                    # np.diff to calculate changes in consecutive days, add 1 to each, and then np.cumprod to calculate
                    # cumulative changes starting from the first day. Subtract one from each and we're done.
                    price_changes = np.cumprod(np.diff(all_prices) / all_prices[:, :-1] + 1, axis=1) - 1
                    # Next, create the columns for the dates, filenames, and the stock names
                    dates = stock.iloc[self.hparams.look_back_days + OFFSET:
                                       len(stock) - self.hparams.max_look_forward_days].index.values.astype(
                        str)
                    filenames = np.array(self.IMAGE_DATA_PATH + os.sep + stock_name + os.sep +
                                         pd.to_datetime(dates).strftime('%Y%m%d') + '.png').reshape(-1, 1)
                    stock_column = np.full((len(dates), 1), stock_name)
                    # Concatenate them all, and package them into a dataframe just for this stock
                    price_df = pd.DataFrame(data=np.concatenate([stock_column, dates.reshape(-1, 1), filenames,
                                                                 price_changes], axis=1), columns=COLUMNS)
                    # Append to our master dataframe
                    target_prices = pd.concat((target_prices, price_df), axis=0, ignore_index=True)

                # Second step is we need to produce our images.
                # We will take our multiprocessor, spawn 4 processes on separate cores, and run through each
                # csv in the batch at the same time. Be careful to match the parameters correctly.
                with multiprocessing.Pool(len(batch)) as pool:
                    for i, stock_filepath in enumerate(batch, 1):
                        stock_name = os.path.basename(stock_filepath)[:-4]
                        stock = pd.read_csv(stock_filepath, parse_dates=[0], index_col=0)
                        pool.apply_async(self.plot_all_stock_timeframes, args=(stock_name, stock, lock, OFFSET, i))
                    pool.close()
                    pool.join()

                # for i in trange(self.hparams.look_back_days + offset, len(stock) - self.hparams.max_look_forward_days):
                #     end_date = stock.index[i]
                #     # Find the start date for the period we are looking at using searchsorted
                #     # Next, actually plot it.
                #     # It will go under a directory of the stock, and the filename will be the end_date
                #     filename = os.path.join(self.IMAGE_DATA_PATH, stock_name, f'{end_date.strftime("%Y%m%d")}.png')
                #     mpf.plot(stock.iloc[i - self.hparams.look_back_days: i + 1, :], type='candle', style='yahoo',
                #              savefig=dict(fname=filename, dpi=50), volume=True, xrotation=20, tight_layout=True)

            # After all the images have been generated and saved, next we save
            # the target price dataframe into a csv.
            target_prices.to_csv(os.path.join(self.IMAGE_DATA_PATH, 'target_prices.csv'), index=False)

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
                                               transform=self.train_transform,
                                               min_price_increase=self.hparams.min_price_increase)
            self.val_dataset = CandleDataset(self.IMAGE_DATA_PATH, val_prices,
                                             target_day=self.hparams.target_look_forward_day,
                                             transform=self.val_transform,
                                             min_price_increase=self.hparams.min_price_increase)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
