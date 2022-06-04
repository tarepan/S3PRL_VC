"""Train RNNMS"""


from typing import Optional
from enum import Enum
import os
from datetime import timedelta
from dataclasses import dataclass

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cloud_io import get_filesystem
from omegaconf import MISSING

from .model import Taco2ARVC


class Profiler(Enum):
    """PyTorch-Lightning's Profiler types"""
    SIMPLE = "simple"
    ADVANCED = "advanced"


@dataclass
class ConfCkptLog:
    """Configuration of checkpointing and logging.
    """
    dir_root: str = MISSING
    name_exp: str  = MISSING
    name_version: str  = MISSING

@dataclass
class ConfTrain:
    """Configuration of train.
    Args:
        max_epochs: Number of maximum training epoch
        val_interval_epoch: Interval epoch between validation
        profiler: Profiler setting
    """
    gradient_clipping: Optional[float] = MISSING
    max_epochs: int = MISSING
    val_interval_epoch: int = MISSING
    profiler: Optional[Profiler] = MISSING
    ckpt_log: ConfCkptLog = ConfCkptLog()


def train(model: Taco2ARVC, conf: ConfTrain, datamodule: LightningDataModule) -> None:
    """Train Taco2ARVC on PyTorch-Lightning.
    """

    ckpt_and_logging = CheckpointAndLogging(
        conf.ckpt_log.dir_root,
        conf.ckpt_log.name_exp,
        conf.ckpt_log.name_version
    )

    # Save checkpoint as `last.ckpt` every 15 minutes.
    ckpt_cb = ModelCheckpoint(
        train_time_interval=timedelta(minutes=15),
        save_last=True,
        save_top_k=0,
    )

    trainer = pl.Trainer(
        gradient_clip_val=conf.gradient_clipping,
        accelerator="auto",
        precision=16,
        max_epochs=conf.max_epochs,
        check_val_every_n_epoch=conf.val_interval_epoch,
        # logging/checkpointing
        default_root_dir=ckpt_and_logging.default_root_dir,
        logger=pl_loggers.TensorBoardLogger(
            ckpt_and_logging.save_dir, ckpt_and_logging.name, ckpt_and_logging.version
        ),
        callbacks=[ckpt_cb],
        # reload_dataloaders_every_epoch=True,
        profiler=conf.profiler,
    )

    # training
    trainer.fit(model, ckpt_path=ckpt_and_logging.resume_from_checkpoint, datamodule=datamodule)

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

    # [PL's Trainer]
    # (https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-class-api)
    default_root_dir: Optional[str]
    resume_from_checkpoint: Optional[str]
    # [PL's TensorBoardLogger]
    # (https://pytorch-lightning.readthedocs.io/en/stable/logging.html#tensorboard)
    save_dir: str
    name: str
    version: str
    # [PL's ModelCheckpoint]
    # (https://pytorch-lightning.readthedocs.io/en/stable/generated/
    # pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)
    # dirpath: Inferred from `default_root_dir`, `name` and `version` by PyTorch-Lightning

    def __init__(
        self,
        dir_root: str,
        name_exp: str = "default",
        name_version: str = "version_-1",
        name_ckpt: str = "last.ckpt",
    ) -> None:

        path_ckpt = os.path.join(dir_root, name_exp, name_version, "checkpoints", name_ckpt)

        # PL's Trainer
        self.default_root_dir = dir_root
        exists = get_filesystem(path_ckpt).exists(path_ckpt) # type: ignore ; because of fsspec
        self.resume_from_checkpoint = path_ckpt if exists else None

        # TB's TensorBoardLogger
        self.save_dir = dir_root
        self.name = name_exp
        self.version = name_version
