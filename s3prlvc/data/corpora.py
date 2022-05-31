"""Corpuses"""


from typing import List, Optional
from dataclasses import dataclass

from omegaconf import MISSING, SI
from speechcorpusy import load_preset # pyright: ignore [reportMissingTypeStubs]; bacause of library
from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId # pyright: ignore [reportMissingTypeStubs]; bacause of library

from .split import split_spk_uttr


@dataclass
class CorpusData:
    corpus: AbstractCorpus
    utterances: List[ItemId]


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
        root=SI("${..root}"),
        download=SI("${..download}"))
    val: ConfCorpus = ConfCorpus(
        root=SI("${..root}"),
        download=SI("${..download}"))
    test: ConfCorpus = ConfCorpus(
        root=SI("${..root}"),
        download=SI("${..download}"))

def load_corpora(conf: ConfCorpora):
    """Load corpuses."""
    corpus_train = load_preset(conf=conf.train)
    corpus_val = load_preset(conf=conf.val)
    corpus_test = load_preset(conf=conf.test)

    # Corpus split for train/val
    assert conf.train.name == conf.val.name, "Currently support only data.train.name==data.val.name"
    corpus = corpus_train
    uttrs = corpus.get_identities()
    if conf.train.name == "JVS":
        spk_unseen = ["jvs_095", "jvs_096", "jvs_098"]
        spk_seen_val = ["jvs_094", "jvs_099"]
        num_val_uttr = 10
    # if corpus_name == "VCC20":
    #     # Missing utterances in original code: E10001-E10050 (c.f. tarepan/s3prl#2)
    #     self._sources = list(filter(lambda item_id: item_id.subtype == "eval_source", all_utterances))
    #     self._targets = list(filter(lambda i: i.subtype == "train_target_task1", all_utterances))
    # elif corpus_name == "AdHoc":
    #     self._sources = list(filter(lambda item_id: item_id.subtype == "s", all_utterances))
    #     self._targets = list(filter(lambda item_id: item_id.subtype == "t", all_utterances))
    else:
        raise Exception(f"Specified corpus do not supported in corpus split.")
    splits = split_spk_uttr(uttrs, spk_unseen, spk_seen_val, num_val_uttr)
    corpus_train_both = CorpusData(corpus, splits.seen_spk_seen_uttr)
    corpus_val_seen = CorpusData(corpus, splits.seen_spk_unseen_uttr)
    corpus_val_unseen = CorpusData(corpus, splits.unseen_spk_unseen_uttr)

    # Test
    # todo: Implement (currently, just place-holder)
    corpus_test_seen = CorpusData(corpus_test, corpus_test.get_identities()[:10])
    corpus_test_unseen = CorpusData(corpus_test, corpus_test.get_identities()[:10])

    return {
        "train": (corpus_train_both, corpus_train_both),
        "val": (corpus_val_seen, corpus_val_unseen),
        "test": (corpus_test_seen, corpus_test_unseen),
    }
