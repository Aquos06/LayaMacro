"""Guard the Laya facts docs/LAYA.md and the Option B graft depend on.

These are *contract* tests against an upstream dependency, not tests of our own code:
if one fails, a `laya` or `transformers` upgrade has moved something the graft is built
on, and docs/LAYA.md is now stale. Fix the docs and the graft together — do not just
relax the assertion.

No weights are downloaded here (that is scripts/verify_laya.py, and the `slow` tests).
"""

from __future__ import annotations

import inspect

import pytest

laya_common = pytest.importorskip("laya.common", reason="laya not installed")


def test_decision_model_does_not_accept_inputs_embeds():
    """The core surgery: we must plumb inputs_embeds through ourselves.

    If this ever fails, upstream added the kwarg and our subclass may be unnecessary.
    """
    params = inspect.signature(laya_common.DecisionModel.forward).parameters
    assert "inputs_embeds" not in params, (
        "upstream now accepts inputs_embeds — revisit the graft in docs/ARCHITECTURE.md"
    )
    # The signature the graft wraps.
    assert {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"} <= set(params)


def test_modernbert_backbone_still_accepts_inputs_embeds():
    """Injecting soft tokens requires this; it is the other half of the graft."""
    m = pytest.importorskip("transformers.models.modernbert.modeling_modernbert")
    assert "inputs_embeds" in inspect.signature(m.ModernBertModel.forward).parameters
    # And it normalizes what we inject, so we must NOT pre-normalize.
    src = inspect.getsource(m.ModernBertEmbeddings.forward)
    assert "self.norm(inputs_embeds)" in src, (
        "ModernBertEmbeddings no longer normalizes inputs_embeds — "
        "docs/LAYA.md tells us not to pre-normalize, which may now be wrong"
    )


def test_qtypes_are_the_three_primitives():
    assert laya_common.QTYPES == {"choice": 0, "score": 1, "noul": 2}


def test_temperature_clamp_range():
    """We rely on knowing calibration is clamped; see docs/LAYA.md §5."""
    assert (laya_common.TEMP_MIN, laya_common.TEMP_MAX) == (0.5, 5.0)


@pytest.mark.parametrize(
    ("question", "n_options"),
    [
        (
            {
                "t": "choice",
                "ins": "Which category?",
                "crit": {"grain": "rice, noodles", "meat": "pork", "vegetable": "greens"},
            },
            3,
        ),
        (
            {
                "t": "score",
                "ins": "How oily?",
                "crit": ["not oily", "slightly", "moderately", "very"],
            },
            4,
        ),
        ({"t": "noul", "ins": "Is it deep-fried?"}, 2),
    ],
)
def test_one_mask_marker_per_option(question, n_options):
    """marker_pos is absolute and lands on [MASK]; the whole decision gathers there.

    This is the invariant the graft protects: inject tokens before these markers
    and they shift silently.
    """
    tok = _tokenizer()
    ids, markers = laya_common.build_sequence(tok, "a plate of white rice", question, 512, 192)

    assert len(markers) == n_options
    assert all(ids[m] == tok.mask_token_id for m in markers), "a marker does not point at [MASK]"
    assert markers == sorted(markers)
    assert len(ids) <= 512


def test_noul_option_order_is_false_then_true():
    """p[1] is always P(true) regardless of `labels`; relied on when reading answers."""
    opts = laya_common.render_options({"t": "noul", "ins": "x"})
    assert opts[0].startswith("false"), opts
    assert opts[1].startswith("true"), opts


def test_internal_question_keys_are_t_ins_crit():
    """The public predict() format is type/instructions/criteria; internals differ.

    Our training-item builders and graft call the internal form, so this guards against
    reintroducing the public key names there by habit.
    """
    with pytest.raises((KeyError, TypeError, AttributeError)):
        laya_common.render_options({"type": "choice", "criteria": {"a": "b"}})


# --- helpers ---------------------------------------------------------------------

_TOK = None


def _tokenizer():
    """ModernBERT's tokenizer, fetched from the hub (small, ~3.5 MB) and cached."""
    global _TOK
    if _TOK is None:
        transformers = pytest.importorskip("transformers")
        try:
            _TOK = transformers.AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        except Exception as exc:  # offline, rate-limited, etc.
            pytest.skip(f"tokenizer unavailable: {type(exc).__name__}: {exc}")
    return _TOK
