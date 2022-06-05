"""PyTorch Lightning checking and logging"""


from typing import Optional
import os
from dataclasses import dataclass
from datetime import timedelta

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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
        self.ckpt_cb = ModelCheckpoint(
            dirpath=None, # Path is inferred by Trainer with `ckpt_log`
            train_time_interval=timedelta(minutes=15), # every 15 minutes
            save_last=True,
            save_top_k=0,
        )
        path_ckpt = os.path.join(conf.dir_root, conf.name_exp, conf.name_version, "checkpoints", "last.ckpt")
        ## Resuming: [Trainer.fit's `ckpt_path`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer.fit)
        exists = get_filesystem(path_ckpt).exists(path_ckpt) # type: ignore ; because of fsspec
        self.ckpt_path: Optional[str] = path_ckpt if exists else None

        # Logging
        ## [PL's TensorBoardLogger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html)
        self.logger: TensorBoardLogger = TensorBoardLogger(conf.dir_root, conf.name_exp, conf.name_version)
