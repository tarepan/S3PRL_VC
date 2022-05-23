"""Corpuses"""


from typing import Optional
from dataclasses import dataclass

from omegaconf import MISSING
# from speechcorpusy.interface import ConfCorpus
from speechcorpusy import load_preset


# Could be pushed to upstream library
@dataclass
class ConfCorpus:
    name: str = MISSING
    root: Optional[str] = MISSING
    download: bool = MISSING

@dataclass
class ConfCorpora:
    root: Optional[str] = MISSING
    download: bool = MISSING
    train: ConfCorpus = ConfCorpus(
        root="${..root}",
        download="${..download}")
    val: ConfCorpus = ConfCorpus(
        root="${..root}",
        download="${..download}")
    test: ConfCorpus = ConfCorpus(
        root="${..root}",
        download="${..download}")

def load_corpora(conf: ConfCorpora):
    """Load corpuses."""
    # todo: data split
    # todo: `name` property in `load_preset`
    return {
        "train": load_preset(conf.train.name, conf=conf.train),
        "val":   load_preset(conf.val.name,   conf=conf.val),
        "test":  load_preset(conf.test.name,  conf=conf.test),
    }
