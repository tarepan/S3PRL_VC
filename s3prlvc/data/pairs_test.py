"""Test pairs.py"""


from typing import List
from .pairs import VCPair, speakers, utterances_of, source_to_self_target, all_source_no1_to_all_targets

from speechdatasety.interface.speechcorpusy import ItemId # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety


uttr_spk_1 = list(map(lambda idx: ItemId("subtype", "spk1", f"uttr{idx}"), range(0, 3)))
uttr_spk_2 = list(map(lambda idx: ItemId("subtype", "spk2", f"uttr{idx}"), range(0, 3)))
uttr_spk_3 = list(map(lambda idx: ItemId("subtype", "spk3", f"uttr{idx}"), range(0, 3)))
utterances = uttr_spk_1 + uttr_spk_2 + uttr_spk_3


def test_speakers():
    """Check speaker extraction."""
    spks = speakers(utterances)
    spks.sort()
    assert spks == ["spk1", "spk2", "spk3"]


def test_utterances_of():
    """Check utterance extraction."""
    uttrs = utterances_of("spk3", utterances)
    assert uttrs != uttr_spk_2
    assert uttrs == uttr_spk_3


def test_source_to_self_target():
    "Check self-targeted VC tuple generation."
    gen_pairs = source_to_self_target(utterances)
    ref_pairs: List[VCPair] = []
    for spk in range(1, 4):
        for uttr in range(0, 3):
            ref_pairs.append(VCPair(
                ItemId("subtype", f"spk{spk}", f"uttr{uttr}"),
                [ItemId("subtype", f"spk{spk}", f"uttr{uttr}")],
                ("O","O")
            ))

    assert gen_pairs == ref_pairs


def test_all_source_no1_to_all_targets():
    """Check tuple generation."""
    gen_pairs = all_source_no1_to_all_targets(utterances, utterances, ("A","A"))
    ref_pairs: List[VCPair] = []
    for spk in range(1, 4):
        ref_pair_spkx_1 = VCPair(ItemId("subtype", f"spk{spk}", "uttr0"), uttr_spk_1, ("A","A"))
        ref_pair_spkx_2 = VCPair(ItemId("subtype", f"spk{spk}", "uttr0"), uttr_spk_2, ("A","A"))
        ref_pair_spkx_3 = VCPair(ItemId("subtype", f"spk{spk}", "uttr0"), uttr_spk_3, ("A","A"))
        ref_pairs += [ref_pair_spkx_1, ref_pair_spkx_2, ref_pair_spkx_3]

    assert len(gen_pairs) == len(ref_pairs)
    assert gen_pairs == ref_pairs
