"""Test split.py"""

from speechdatasety.interface.speechcorpusy import ItemId # pyright: ignore [reportMissingTypeStubs]; bacause of speechdatasety

from .split import split_spk_uttr


def test_split_spk_uttr():
    """Test split_spk_uttr."""

    # X: seen-spk-seen-uttr(train), Y: seen-spk-unseen-uttr (val, Many), Z: unseen-spk-unseen-uttr (val, Any)
    # speakers of Y could be subset of X for small val dataset
    #
    #                     seen      |    unseen
    #                 uttr0   uttr1 | uttr2   uttr3
    #   seen   spk0     X       X   |   Y       Y
    #          spk1     X       X   |
    #  ---------------------------------------------
    #  unseen  spk2                 |   Z       Z
    #          spk3                 |   Z       Z

    utterances = [ItemId("", f"spk{j_spk}", f"uttr{i_uttr}") for i_uttr in range(4) for j_spk in range(4)]
    spk_unseen = ["spk2", "spk3"]
    spk_seen_val = ["spk0"] # spk1 not included
    num_unseen_uttr = 2
    splits = split_spk_uttr(utterances, spk_unseen, spk_seen_val, num_unseen_uttr)

    assert splits.unseen_spk_unseen_uttr == [ItemId("", f"spk{j_spk}", f"uttr{i_uttr}") for j_spk in range(2, 4) for i_uttr in range(2, 4)] # Z
    assert splits.seen_spk_unseen_uttr == [ItemId("", f"spk{j_spk}", f"uttr{i_uttr}") for j_spk in range(1) for i_uttr in range(2, 4)] # Y
    assert splits.seen_spk_seen_uttr == [ItemId("", f"spk{j_spk}", f"uttr{i_uttr}") for j_spk in range(2) for i_uttr in range(2)] # X
