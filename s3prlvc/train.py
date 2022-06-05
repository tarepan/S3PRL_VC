"""Train PyTorch Lightning model"""


from typing import Optional
import os
from datetime import timedelta
from dataclasses import dataclass

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cloud_io import get_filesystem
from omegaconf import MISSING


@dataclass
class ConfCkptLog:
    """Configuration of checkpointing and logging, directly unpackable with `**asdict(this)`.
    """
    dir_root: str = MISSING
    name_exp: str  = "default"
    name_version: str  = "version_-1"

class CheckpointAndLogging:
    """Generate path of checkpoint & logging.
    {dir_root}/
        {name_exp}/
            {name_version}/
                checkpoints/
                    {name_ckpt} # PyTorch-Lightning Checkpoint. Resume from here.
                hparams.yaml
                events.out.tfevents.{xxxxyyyyzzzz} # TensorBoard log file.
    """

    def __init__(self, conf: ConfCkptLog) -> None:
        # Checkpointing
        ## Storing: [Trainer's `default_root_dir`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer.default_root_dir)
        self.default_root_dir: Optional[str] = conf.dir_root
        ## ModelCheckpoint(dir_path=None) infer this adress
        path_ckpt = os.path.join(conf.dir_root, conf.name_exp, conf.name_version, "checkpoints", "last.ckpt")
        ## Resuming: [Trainer.fit's `ckpt_path`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer.fit)
        exists = get_filesystem(path_ckpt).exists(path_ckpt) # type: ignore ; because of fsspec
        self.ckpt_path: Optional[str] = path_ckpt if exists else None

        # Logging
        ## [PL's TensorBoardLogger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html)
        self.logger: TensorBoardLogger = TensorBoardLogger(conf.dir_root, conf.name_exp, conf.name_version)


@dataclass
class ConfTrain:
    """Configuration of train.
    Args:
        gradient_clipping - Maximum gradient L2 norm, clipped when bigger than this (None==∞)
        max_epochs - Number of maximum training epoch
        val_interval_epoch - Interval epoch between validation
        profiler - Profiler setting
    """
    gradient_clipping: Optional[float] = MISSING
    max_epochs: int = MISSING
    val_interval_epoch: int = MISSING
    profiler: Optional[str] = MISSING
    ckpt_log: ConfCkptLog = ConfCkptLog()


def train(model: pl.LightningModule, conf: ConfTrain, datamodule: LightningDataModule) -> None:
    """Train the PyTorch-Lightning model.
    """

    ckpt_log = CheckpointAndLogging(conf.ckpt_log)

    # Save checkpoint as `last.ckpt` every 15 minutes (Path is inferred by Trainer with `ckpt_log`)
    ckpt_cb = ModelCheckpoint(
        train_time_interval=timedelta(minutes=15),
        save_last=True,
        save_top_k=0,
    )

    # Mixed precision training on fast accelerator
    trainer = pl.Trainer(
        accelerator="auto",
        precision=16,
        gradient_clip_val=conf.gradient_clipping,
        max_epochs=conf.max_epochs,
        check_val_every_n_epoch=conf.val_interval_epoch,
        profiler=conf.profiler,
        # checkpoint/logging
        default_root_dir=ckpt_log.default_root_dir,
        logger=ckpt_log.logger,
        callbacks=[ckpt_cb],
    )

    # training
    trainer.fit(model, ckpt_path=ckpt_log.ckpt_path, datamodule=datamodule) # pyright: ignore E[reportUnknownMemberType] ; because of PL
