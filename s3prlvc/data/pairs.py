"""VC tuples"""


from dataclasses import dataclass
from typing import List, Tuple

from speechdatasety.interface.speechcorpusy import ItemId # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety


@dataclass(frozen=True)
class VCPair:
    """Voive conversion source-target pair with VC setup."""
    source: ItemId
    targets: List[ItemId]
    setup: Tuple[str, str] # ["O","O"] (O2O) | ["A","M"] (A2M) | ["A","A"] (A2A)


def speakers(item_ids: List[ItemId]) -> List[str]:
    """Speakers within the utterances."""
    # `set` do not preserve the order.
    spks: List[str] = []
    for item_id in item_ids:
        if item_id.speaker not in spks:
            spks.append(item_id.speaker)
    return spks


def utterances_of(speaker_id: str, utterances: List[ItemId]) -> List[ItemId]:
    """Filter utterances of a speaker."""
    return list(filter(lambda item_id: item_id.speaker == speaker_id, utterances))


def source_to_self_target(source_uttrs: List[ItemId]) -> List[VCPair]:
    """
    utterance contents -> self-target utterance embedding
    Generate tuple [spk_S_uttr_#N, spk_S_uttr_#N] for all utterances.
    """
    return list(map(lambda uttr: VCPair(uttr, [uttr], ("O","O")), source_uttrs))


def all_source_no1_to_all_targets(
    source_uttrs: List[ItemId],
    target_uttrs: List[ItemId],
    setup: Tuple[str, str],
    ) -> List[VCPair]:
    """
    1st utterance contents of each source speaker -> all utterances averaged embedding of each target speaker
    Generate tuple [spk_S_uttr_#0, spk_T_uttr_#0, #1, ..., #N] for all S in source & T in target.
    """
    vc_tuples: List[VCPair] = []
    for source_spk_id in speakers(source_uttrs):
        source_uttr_1st = utterances_of(source_spk_id, source_uttrs)[0]
        for target_spk_id in speakers(target_uttrs):
            target_uttr_all = utterances_of(target_spk_id, target_uttrs)
            vc_tuples.append(VCPair(source_uttr_1st, target_uttr_all, setup))
    return vc_tuples
