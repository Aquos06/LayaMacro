"""The multi-task question bank keeps its held-out promises."""

from __future__ import annotations

import random

import pytest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def bank():  # type: ignore[no-untyped-def]
    from layamacro.data import questions as Q
    from layamacro.data.item_labels import item_names
    from layamacro.train.common import load_data

    samples, _ = load_data()
    train = [s for s in samples if s.split == "train" and s.dataset in Q.DATASETS]
    test = [s for s in samples if s.split == "test" and s.dataset in Q.DATASETS]
    tr = [a for s in train for a in Q.ask(s, random.Random(s.id), train=True)]
    te = [a for s in test for a in Q.ask(s, random.Random(s.id), train=False)]
    return Q, item_names(), tr, te


def _names_in(a):  # type: ignore[no-untyped-def]
    if a.family == "which":
        return set(a.question.options.values())
    return {a.about} if a.family == "ingredient" else set()


def test_heldout_names_never_trained(bank) -> None:  # type: ignore[no-untyped-def]
    Q, _, tr, _ = bank
    held = set().union(*(h for _, h in Q.vocab().values()))
    assert held, "no held-out names"
    leaked = {n for a in tr for n in _names_in(a)} & held
    assert not leaked, f"held-out names in training questions: {sorted(leaked)[:5]}"


def test_heldout_phrasings_and_datasets_never_trained(bank) -> None:  # type: ignore[no-untyped-def]
    Q, _, tr, te = bank
    held_texts = {Q.WHICH_PHRASES[-1]} | {p[-1] for _, _, p in Q.THRESHOLDS}
    for a in tr:
        text = a.question.instructions
        assert text not in held_texts, a
        assert text != Q.ING_PHRASES[-1].format(x=a.about), a
        assert a.held_out == ""
        if a.family == "threshold":
            assert a.dataset in Q.T_TRAIN_DATASETS
    tags = {a.held_out for a in te}
    assert {"name", "dataset"} <= tags


def test_answers_match_the_recorded_items(bank) -> None:  # type: ignore[no-untyped-def]
    _, names, tr, te = bank
    for a in tr + te:
        opts = a.question.options
        assert a.answer in opts
        if a.family == "ingredient":
            assert (opts[a.answer] == "yes") == (a.about in names[a.photo])
        elif a.family == "which":
            assert opts[a.answer] in names[a.photo]
            others = [v for k, v in opts.items() if k != a.answer]
            assert not set(others) & names[a.photo], "a decoy is actually on the plate"


def test_yes_is_not_always_the_same_key(bank) -> None:  # type: ignore[no-untyped-def]
    _, _, tr, _ = bank
    yes = [a.yes_key for a in tr if a.yes_key]
    share_a = sum(k == "A" for k in yes) / len(yes)
    assert 0.45 < share_a < 0.55
