"""Run S3PRL-VC training"""


import pytorch_lightning as pl
import torchaudio # pyright: ignore [reportMissingTypeStubs]; bacause of torchaudio

from .config import load_conf
from .model import Taco2ARVC
from .data.datamodule import WavMelEmbVcData
from .trainer import train


def main_train():
    """Run s3prlvc Taco2AR training with cli arguments.
    """

    # Load default/extend/CLI configs.
    conf = load_conf()

    # Setup
    pl.seed_everything(conf.seed)
    torchaudio.set_audio_backend("sox_io")
    datamodule = WavMelEmbVcData(conf.data)
    ## For stats
    datamodule.prepare_data()
    datamodule.setup()
    model = Taco2ARVC(conf.model, datamodule.dataset_train.acquire_spec_stat())

    # Train
    train(model, conf.train, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()
