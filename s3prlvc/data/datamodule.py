"""Data wrapper by PL-datamodule"""


from typing import Optional
from dataclasses import dataclass

from pytorch_lightning import LightningDataModule
from omegaconf import MISSING
from speechdatasety.helper.loader import LoaderGenerator, ConfLoader

from .corpora import load_corpora, ConfCorpora
from .dataset import WavMelEmbVcDataset, ConfWavMelEmbVcDataset


@dataclass
class ConfWavMelEmbVcData:
    """Configuration of WavMelEmbVcData.
    """
    adress_data_root: Optional[str] = MISSING
    corpus: ConfCorpora = ConfCorpora(
        root="${..adress_data_root}")
    dataset: ConfWavMelEmbVcDataset = ConfWavMelEmbVcDataset(
        adress_data_root="${..adress_data_root}")
    loader: ConfLoader = ConfLoader()

class WavMelEmbVcData(LightningDataModule):
    """PL-DataModule of wave/melspec/embedding/VcTuple.
    """
    def __init__(self, conf: ConfWavMelEmbVcData):
        super().__init__()
        self._conf = conf

        # Init
        self._corpora = load_corpora(conf.corpus)
        self._loader_generator = LoaderGenerator(conf.loader)

    def prepare_data(self) -> None:
        """(PL-API) Prepare data in dataset.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """(PL-API) Setup train/val/test datasets.
        """

        self.dataset_train = WavMelEmbVcDataset("train", self._conf.dataset, self._corpora["train"])
        self.dataset_val =   WavMelEmbVcDataset("dev",   self._conf.dataset, self._corpora["val"])
        self.dataset_test =  WavMelEmbVcDataset("test",  self._conf.dataset, self._corpora["test"])

        if stage == "fit" or stage is None:
            pass
        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        """(PL-API) Generate training dataloader."""
        return self._loader_generator.train(self.dataset_train)

    def val_dataloader(self):
        """(PL-API) Generate validation dataloader."""
        return self._loader_generator.val(self.dataset_val)

    def test_dataloader(self):
        """(PL-API) Generate test dataloader."""
        return self._loader_generator.test(self.dataset_test)
