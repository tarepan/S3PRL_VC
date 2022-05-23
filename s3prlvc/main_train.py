"""Run S3PRL-VC training"""


import pytorch_lightning as pl
import torchaudio

from .model import Taco2ARVC
from .data.datamodule import generate_datamodule
from .train import train
from .config import load_conf


def main_train():
    """Train s3prlvc with cli arguments and the default dataset.
    """

    # Load default/extend/CLI configs.
    conf = load_conf()

    # Setup
    pl.seed_everything(conf.seed)
    torchaudio.set_audio_backend("sox_io")
    model = Taco2ARVC(conf.model)
    datamodule = generate_datamodule(conf.data)

    # Train
    train(model, conf.train, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()
