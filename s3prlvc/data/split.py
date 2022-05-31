"""Split corpus items."""


from dataclasses import dataclass
from typing import List
from speechdatasety.interface.speechcorpusy import ItemId # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety

from .pairs import speakers, utterances_of


@dataclass
class CorpusSplits:
    """
    unseen_spk_unseen_uttr - Used for Any-to & to-Any validation
    seen_spk_unseen_uttr - Used for to-Many validation
    seen_spk_seen_uttr - Used for training ('training' define 'seen')
    """
    unseen_spk_unseen_uttr: List[ItemId]
    seen_spk_unseen_uttr: List[ItemId]
    seen_spk_seen_uttr: List[ItemId]

def split_spk_uttr(
    uttrs: List[ItemId],
    spk_unseen: List[str],
    spk_seen_with_unseen_uttr: List[str],
    num_unseen_uttr: int,
    ) -> CorpusSplits:
    """
    Args:
        uttrs - All utterances
        spk_unseen - Unseen speaker list
        spk_seen_with_unseen_uttr - List of 'seen speaker with unseen utterances'
        num_unseen_uttr - The number of unseen utterances per speaker
    Returns - splitted utterances
    """
    unseen_spk_unseen_uttr: List[ItemId] = []
    seen_spk_unseen_uttr: List[ItemId] = []
    seen_spk_seen_uttr: List[ItemId] = []

    idx_seen_unseen_uttr = -1*num_unseen_uttr
    for spk in speakers(uttrs):
        uttr_spk = utterances_of(spk, uttrs)
        if spk in spk_unseen:
            unseen_spk_unseen_uttr += uttr_spk[idx_seen_unseen_uttr:]
        else:
            seen_spk_seen_uttr += uttr_spk[:idx_seen_unseen_uttr]
            # N_seen_spk is usually big, so seen_spk_unseen_uttr become huge.
            # It could be too much for validation, so speakers are selected here.
            if spk in spk_seen_with_unseen_uttr:
                seen_spk_unseen_uttr += uttr_spk[idx_seen_unseen_uttr:]

    return CorpusSplits(unseen_spk_unseen_uttr, seen_spk_unseen_uttr, seen_spk_seen_uttr)
