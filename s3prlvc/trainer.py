"""PyTorch Lightning trainer"""


from typing import Optional
from dataclasses import dataclass

import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from omegaconf import MISSING

from .lightlightning.train import CheckpointAndLogging, ConfCkptLog


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

    # Harmonized setups of checkpointing/logging
    ckpt_log = CheckpointAndLogging(conf.ckpt_log)

    # Trainer for mixed precision training on fast accelerator
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
        callbacks=[ckpt_log.ckpt_cb],
    )

    # training
    trainer.fit(model, ckpt_path=ckpt_log.ckpt_path, datamodule=datamodule) # pyright: ignore E[reportUnknownMemberType] ; because of PL
