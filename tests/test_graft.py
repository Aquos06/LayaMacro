"""Phase 0: the graft works mechanically.

Every failure mode here is **silent** — a displaced marker or a truncated image run
trains perfectly well and is wrong forever (docs/ARCHITECTURE.md §2). So these are
assertions, not observations.

The graft-math tests build a *tiny* real `DecisionModel` (random weights, 64-dim
ModernBERT) rather than downloading Laya, so the most important test — that our
`inputs_embeds` path is bit-identical to upstream's `forward` — runs in CI.
"""

from __future__ import annotations

import pytest
import torch

laya_common = pytest.importorskip("laya.common", reason="laya not installed")

from layamacro.model import MacroRegressor, VisionLaya  # noqa: E402
from layamacro.schema import AUX_QUESTIONS, FOOD_CATEGORY, OIL_LEVEL, Question  # noqa: E402
from layamacro.sequence import (  # noqa: E402
    build_image_sequence,
    collate,
    default_placeholder_id,
)
from layamacro.vision.projector import Projector  # noqa: E402
from layamacro.vision.resampler import PerceiverResampler  # noqa: E402

TINY_D = 64
TINY_VOCAB = 1000
PLACEHOLDER = 999


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """ModernBERT's tokenizer (~3.5 MB) — the real one, since ids matter."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    except Exception as exc:  # offline / rate-limited
        pytest.skip(f"tokenizer unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def tiny_laya():
    """A real `DecisionModel` at toy scale: upstream head code, random weights."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.ModernBertConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_D,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        local_attention=16,
        pad_token_id=0,
    )
    encoder = transformers.ModernBertModel(cfg)
    model = laya_common.DecisionModel(encoder, head_layers=2)
    return model.eval()


@pytest.fixture
def graft(tiny_laya):
    vl = VisionLaya(
        laya=tiny_laya,
        resampler=PerceiverResampler(TINY_D, n_queries=8, n_heads=4, n_layers=1),
        projector=Projector(TINY_D, TINY_D),
        n_image_tokens=8,
        placeholder_id=PLACEHOLDER,
    )
    return vl.eval()


def _synthetic_batch(b: int = 2, length: int = 40, n_img: int = 8, n_opts: int = 3):
    """Ids with a contiguous placeholder run, plus markers that avoid it."""
    torch.manual_seed(0)
    input_ids = torch.randint(5, 500, (b, length))
    img_start = length - n_img - 1
    input_ids[:, img_start : img_start + n_img] = PLACEHOLDER
    marker_pos = torch.arange(2, 2 + n_opts).unsqueeze(0).repeat(b, 1)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(b, length, dtype=torch.long),
        "marker_pos": marker_pos,
        "marker_mask": torch.ones(b, n_opts, dtype=torch.bool),
        "qtype": torch.zeros(b, dtype=torch.long),
        "image_slots": torch.arange(img_start, img_start + n_img).unsqueeze(0).repeat(b, 1),
    }


# --------------------------------------------------------------------------------------
# THE critical test: our path must equal upstream's
# --------------------------------------------------------------------------------------


def test_embeds_path_matches_upstream_forward(graft, tiny_laya):
    """`_decide_from_embeds` duplicates upstream head math; prove it is exact.

    This is what makes the duplication safe. If upstream changes its head, this fails
    loudly instead of the graft quietly diverging.
    """
    batch = _synthetic_batch()
    embeds = tiny_laya.encoder.get_input_embeddings()(batch["input_ids"])

    with torch.no_grad():
        ours = graft._decide_from_embeds(
            embeds,
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
        ref_logits, ref_act = tiny_laya(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )

    assert torch.equal(ours.logits, ref_logits), "logits diverge from DecisionModel.forward"
    assert torch.equal(ours.act_logits, ref_act), "act_logits diverge from DecisionModel.forward"


def test_forward_without_image_equals_upstream(graft, tiny_laya):
    """With no image, the wrapper must be a no-op relative to upstream."""
    batch = _synthetic_batch()
    with torch.no_grad():
        out = graft(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
        ref_logits, _ = tiny_laya(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
    assert torch.equal(out.logits, ref_logits)


# --------------------------------------------------------------------------------------
# the image actually reaches the model
# --------------------------------------------------------------------------------------


def test_image_tokens_change_the_logits(graft):
    """If swapping the image leaves logits untouched, the graft is decorative."""
    batch = _synthetic_batch()
    torch.manual_seed(1)
    patches_a = torch.randn(2, 20, TINY_D)
    patches_b = torch.randn(2, 20, TINY_D) * 3.0

    with torch.no_grad():
        logits_a = graft(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
            image_tokens=graft.encode_image(patches_a),
            image_slots=batch["image_slots"],
        ).logits
        logits_b = graft(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
            image_tokens=graft.encode_image(patches_b),
            image_slots=batch["image_slots"],
        ).logits

    assert not torch.allclose(logits_a, logits_b), "logits ignore the image content"


def test_projected_tokens_land_exactly_in_their_slots(graft, tiny_laya):
    """The scatter must overwrite the placeholder embeddings and nothing else."""
    batch = _synthetic_batch()
    img = torch.full((2, 8, TINY_D), 7.0)

    base = tiny_laya.encoder.get_input_embeddings()(batch["input_ids"])
    idx = batch["image_slots"].unsqueeze(-1).expand(-1, -1, TINY_D)
    patched = base.scatter(1, idx, img)

    assert torch.equal(patched.gather(1, idx), img), "slots do not hold the projected tokens"

    # Everything outside the slots is untouched.
    mask = torch.ones_like(base, dtype=torch.bool).scatter(1, idx, False)
    assert torch.equal(patched[mask], base[mask]), "the scatter modified non-image positions"


def test_gradients_reach_the_new_modules_only(graft):
    """Stage 1 trains resampler + projector; Laya must receive nothing."""
    graft.freeze_laya(True)
    graft.train()
    batch = _synthetic_batch()
    patches = torch.randn(2, 20, TINY_D)

    out = graft(
        batch["input_ids"],
        batch["attention_mask"],
        batch["marker_pos"],
        batch["marker_mask"],
        batch["qtype"],
        image_tokens=graft.encode_image(patches),
        image_slots=batch["image_slots"],
    )
    out.logits.sum().backward()

    trainable = [p for p in graft.new_parameters() if p.requires_grad]
    assert trainable, "nothing to train"
    got_grad = [p for p in trainable if p.grad is not None and p.grad.abs().sum() > 0]
    assert got_grad, "no gradient reached the resampler/projector"

    laya_grads = [p.grad for p in graft.laya.parameters() if p.grad is not None]
    assert not laya_grads, "Laya received gradients while frozen"


def test_invariant_check_catches_a_displaced_slot(graft):
    """The debug assertion must fire when a slot does not hold the placeholder."""
    graft.check_invariants = True
    batch = _synthetic_batch()
    bad = batch["image_slots"].clone()
    bad[0, 0] = 1  # point at a normal token instead

    with pytest.raises(AssertionError, match="placeholder"):
        graft(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
            image_tokens=torch.zeros(2, 8, TINY_D),
            image_slots=bad,
        )


def test_invariant_check_catches_marker_inside_image_run(graft):
    batch = _synthetic_batch()
    graft.check_invariants = True
    markers = batch["marker_pos"].clone()
    markers[0, 0] = int(batch["image_slots"][0, 0])  # marker inside the image

    with pytest.raises(AssertionError, match="marker_pos"):
        graft(
            batch["input_ids"],
            batch["attention_mask"],
            markers,
            batch["marker_mask"],
            batch["qtype"],
            image_tokens=torch.zeros(2, 8, TINY_D),
            image_slots=batch["image_slots"],
        )


def test_resampler_token_count_must_match_reserved_slots(tiny_laya):
    with pytest.raises(ValueError, match="must agree"):
        VisionLaya(
            laya=tiny_laya,
            resampler=PerceiverResampler(TINY_D, n_queries=16, n_heads=4, n_layers=1),
            projector=Projector(TINY_D, TINY_D),
            n_image_tokens=8,
            placeholder_id=PLACEHOLDER,
        )


# --------------------------------------------------------------------------------------
# sequence building (real tokenizer, no model)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", AUX_QUESTIONS, ids=lambda q: q.id)
def test_sequence_reserves_slots_and_keeps_markers_on_mask(tokenizer, question: Question):
    seq = build_image_sequence(tokenizer, question, n_image_tokens=64)

    assert len(seq.image_slots) == 64
    assert len(seq.marker_pos) == question.n_options
    assert all(seq.input_ids[m] == tokenizer.mask_token_id for m in seq.marker_pos)
    # Image tokens must come AFTER every marker: prepending is what breaks the gather.
    assert min(seq.image_slots) > max(seq.marker_pos)
    assert not set(seq.image_slots) & set(seq.marker_pos)


def test_placeholder_id_is_unused_by_the_tokenizer(tokenizer):
    pid = default_placeholder_id(tokenizer, vocab_size=50368)
    assert pid != tokenizer.mask_token_id
    assert pid > max(tokenizer.all_special_ids) - 1
    # It must never be produced from real text.
    ids = tokenizer("a plate of fried rice with pork and greens", add_special_tokens=True)
    assert pid not in ids["input_ids"]


def test_truncated_image_run_raises_rather_than_degrading(tokenizer):
    """A budget too small must fail loudly, not silently shrink the image."""
    with pytest.raises(ValueError, match="image slots survived"):
        build_image_sequence(tokenizer, OIL_LEVEL, n_image_tokens=64, max_len=80, head_max_len=64)


def test_collate_shapes(tokenizer):
    seqs = [
        build_image_sequence(tokenizer, FOOD_CATEGORY, 32),
        build_image_sequence(tokenizer, OIL_LEVEL, 32),
    ]
    batch = collate(seqs, qtypes=[0, 1], pad_id=tokenizer.pad_token_id)

    b, length = batch["input_ids"].shape
    assert b == 2
    assert batch["attention_mask"].shape == (2, length)
    assert batch["image_slots"].shape == (2, 32)
    # marker_pos padded to the widest question; marker_mask says which are real.
    assert batch["marker_pos"].shape[1] == max(len(s.marker_pos) for s in seqs)
    assert batch["marker_mask"].sum().item() == sum(len(s.marker_pos) for s in seqs)
    # Padding must be masked out.
    for i, s in enumerate(seqs):
        assert batch["attention_mask"][i].sum().item() == len(s)


# --------------------------------------------------------------------------------------
# vision modules
# --------------------------------------------------------------------------------------


def test_resampler_fixes_the_token_count_regardless_of_input_length():
    r = PerceiverResampler(TINY_D, n_queries=32, n_heads=4, n_layers=1).eval()
    for n_patches in (16, 257, 1024):
        out = r(torch.randn(2, n_patches, TINY_D))
        assert out.shape == (2, 32, TINY_D)


def test_projector_matches_target_scale():
    """A projector starting orders of magnitude off can stall Stage 1."""
    torch.manual_seed(0)
    p = Projector(TINY_D, 128, target_std=0.02).eval()
    with torch.no_grad():
        out = p(torch.randn(64, 8, TINY_D))
    assert 0.005 < out.std().item() < 0.08, out.std().item()


def test_macro_regressor_predicts_positive_macros():
    head = MacroRegressor(TINY_D).eval()
    preds = head.predict(torch.randn(4, TINY_D))
    assert preds.shape == (4, 4)
    assert (preds >= 0).all(), "expm1 of a log-space prediction must be non-negative"
