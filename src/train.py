import os

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['.idea', '.git'],
    pythonpath=True,
)

from typing import Tuple, Optional
import logging
import datetime as dt

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb

os.environ['WANDB_SILENT'] = 'true'
torch.set_float32_matmul_precision('medium')

def train(cfg: DictConfig) -> Tuple[dict, dict]:
    log.info(f'Full hydra configuration:\n{OmegaConf.to_yaml(cfg)}')
    # Set seed for everything (numpy, random number generator, etc.)
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)
    # Create the data directory in case it's completely missing
    os.makedirs(cfg.paths.data_dir, exist_ok=True)

    # Next, instantiate all the needed parts.
    wandb.init(project=cfg.datamodule.comp_name,
               name=f'{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}|{cfg.model.net.name}')
    log.info(f'Instantiating datamodule <{cfg.datamodule._target_}>...')
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f'Instantiating model <{cfg.model._target_}>...')
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f'Instatiating logger <{cfg.logger._target_}>...')
    wandb_logger: WandbLogger = hydra.utils.instantiate(cfg.logger)

    log.info(f'Instantiating Trainer...')
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger)
    log.debug(f'Trainer logger:{trainer.logger}')

    if cfg.get('task_name') == 'train':
        log.info('Starting training...')
        dm_ignored = ['_target_', 'comp_name', 'data_dir']
        # Log training metrics to Weights and Biases
        wandb_logger.experiment.config.update({k: v for k, v in cfg['datamodule'].items() if k not in dm_ignored})
        wandb_logger.experiment.config.update(cfg['model']['net'])
        wandb_logger.experiment.config.update({
            'lr': cfg['model']['optimizer']['lr'],
            'weight_decay': cfg['model']['optimizer']['weight_decay'],
            'epochs': cfg['trainer']['max_epochs']
        })
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path'))
    wandb.finish()


@hydra.main(version_base='1.3', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)
    return 1.0

if __name__ == '__main__':
    log = logging.getLogger('train')
    main()
