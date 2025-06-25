import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['.idea', '.git'],
    pythonpath=True,
)

from typing import Optional, List
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import Callback, RichProgressBar

from utils.instantiators import instantiate_callbacks


torch.set_float32_matmul_precision('medium')
os.environ['HYDRA_FULL_ERROR'] = '1'


def train(cfg: DictConfig):
    log.info(f'Full hydra configuration:\n{OmegaConf.to_yaml(cfg)}')
    # Set seed for everything (numpy, random number generator, etc.)
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)
    # Create the data directory in case it's completely missing
    os.makedirs(cfg.paths.data_dir, exist_ok=True)

    if cfg.get('wandb_enabled'):
        wandb_logger = WandbLogger(project=cfg.datamodule.comp_name, save_dir=cfg.paths.log_dir)
    else:
        wandb_logger = WandbLogger(project=cfg.datamodule.comp_name, save_dir=cfg.paths.log_dir, mode='disabled')

    log.info(f'Instantiating datamodule <{cfg.datamodule._target_}>...')
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f'Instantiating model <{cfg.model._target_}>...')
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info('Instantiating callbacks...')
    if cfg.get('callbacks'):
        callbacks = instantiate_callbacks(cfg.get('callbacks'))

    log.info(f'Instantiating Trainer...')
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger, callbacks=callbacks)

    log.info('Starting training...')
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path'))
    log.info('Training complete.')


@hydra.main(version_base='1.3', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)
    return 1.0


if __name__ == '__main__':
    log = logging.getLogger('train')
    main()
