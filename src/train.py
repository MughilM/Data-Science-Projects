import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['.idea', '.git'],
    pythonpath=True,
)

from typing import Tuple, Optional, List
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import Callback
from torchinfo import summary

from custom_callbacks import SegmentationImageCallback
from utils.instantiators import instantiate_callbacks

import wandb

torch.set_float32_matmul_precision('medium')
os.environ['HYDRA_FULL_ERROR'] = '1'

def train(cfg: DictConfig) -> Tuple[dict, dict]:
    log.info(f'Full hydra configuration:\n{OmegaConf.to_yaml(cfg)}')
    # Set seed for everything (numpy, random number generator, etc.)
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)
    # Create the data directory in case it's completely missing
    os.makedirs(cfg.paths.data_dir, exist_ok=True)

    wandb_logger = WandbLogger(project=cfg.datamodule.comp_name)

    log.info(f'Instantiating datamodule <{cfg.datamodule._target_}>...')
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup('fit')

    log.info(f'Instantiating model <{cfg.model._target_}>...')
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # seg_image_callback = SegmentationImageCallback(datamodule, num_classes=1)
    log.info('Instantiating callbacks')
    callbacks: List[Callback] = instantiate_callbacks(cfg.get('callbacks'))


    log.info(f'Instantiating Trainer...')
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger, callbacks=callbacks)
    log.debug(f'Trainer logger:{trainer.logger}')

    if cfg.get('task_name') == 'train':
        log.info('Starting training...')
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path'))


@hydra.main(version_base='1.3', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)
    return 1.0

if __name__ == '__main__':
    log = logging.getLogger('train')
    main()
