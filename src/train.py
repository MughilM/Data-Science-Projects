import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['.idea', '.git'],
    pythonpath=True,
)

from typing import Tuple, Optional
import logging
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer

import mlflow


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    logger.debug(f'Full hydra configuration:\n{OmegaConf.to_yaml(cfg)}')
    # Set seed for everything (numpy, random number generator, etc.)
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)

    # Next, instantiate all the needed parts.
    logger.info(f'Instantiating datamodule <{cfg.datamodule._target_}>...')
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    logger.info(f'Instantiating model <{cfg.model._target_}>...')
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info(f'Instantiating Trainer...')
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    # We are dealing with PyTorch, so make MLFlow autolog it
    mlflow.set_experiment(cfg.datamodule.comp_name)
    mlflow.pytorch.autolog()

    if cfg.get('task_name') == 'train':
        with mlflow.start_run(run_name=datetime.datetime.today().strftime('%Y%M%D-%H:%M:%S')) as run:
            logger.info('Logging parameters to MLFlow...')

            logger.info('Starting training...')
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path'))


@hydra.main(version_base='1.3', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)
    return 1.0

if __name__ == '__main__':
    logger = logging.getLogger('train')
    main()
