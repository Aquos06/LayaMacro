"""Building Laya sequences that reserve slots for image tokens.

The graft's whole correctness argument lives here. Laya reads each option's logit
from the hidden state at that option's ``[MASK]``, gathered by an **absolute** index
(``marker_pos``). Shift those and you get plausible wrong numbers with no error. That
is why image tokens go *after* the options, in Laya's state region, never before them.

So we never build sequences by hand. We call Laya's own ``build_sequence`` with
``state_ids`` set to a run of placeholder tokens — it stays the source of truth for
``marker_pos``, truncation and the template — then locate the placeholders and
overwrite *their embeddings* later. Every invariant is asserted, because all of them
fail silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from laya import common as laya_common

from layamacro.schema import Question


def default_placeholder_id(tok, vocab_size: int) -> int:
    """An id the tokenizer can never emit, but the embedding table still has a row for.

    ModernBERT's embedding matrix (50368) is larger than the tokenizer's id space
    (special tokens top out at 50284), so the slots above are unused vocabulary. Using
    one guarantees a placeholder can never collide with real text or with ``[MASK]``,
    which keeps the debug assertions meaningful.
    """
    pid = max(tok.all_special_ids) + 1
    if pid >= vocab_size:
        raise ValueError(
            f"no unused vocab slot: special ids reach {pid - 1}, vocab_size is {vocab_size}"
        )
    return pid


@dataclass(frozen=True)
class ImageSequence:
    """One question's tokenized sequence, with the image slots located."""

    input_ids: list[int]
    marker_pos: list[int]
    image_slots: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)


def build_image_sequence(
    tok,
    question: Question,
    n_image_tokens: int,
    max_len: int = 512,
    head_max_len: int = 192,
    placeholder_id: int | None = None,
    vocab_size: int = 50368,
) -> ImageSequence:
    """Build ``[CLS] q [SEP] [MASK] opt0 … [SEP] <image slots> [SEP]``.

    Raises rather than silently degrading: a truncated image run or a displaced marker
    would train perfectly well and be wrong forever.
    """
    if placeholder_id is None:
        placeholder_id = default_placeholder_id(tok, vocab_size)
    if placeholder_id == tok.mask_token_id:
        raise ValueError("placeholder id must differ from [MASK] or assertions become useless")

    ids, markers = laya_common.build_sequence(
        tok,
        state="",  # ignored when state_ids is given
        q=question.to_laya_internal(),
        max_len=max_len,
        head_max_len=head_max_len,
        state_ids=[placeholder_id] * n_image_tokens,
    )

    slots = [i for i, t in enumerate(ids) if t == placeholder_id]

    # --- invariants, all of which fail silently if unchecked ---------------------
    if len(slots) != n_image_tokens:
        raise ValueError(
            f"{question.id}: {len(slots)} of {n_image_tokens} image slots survived. "
            f"The options and instructions consumed the budget (max_len={max_len}, "
            f"head_max_len={head_max_len}). Shorten the options or raise max_len — "
            "do not proceed, the model would silently see a truncated image."
        )
    if len(markers) != question.n_options:
        raise ValueError(
            f"{question.id}: {len(markers)} markers for {question.n_options} options — "
            "an option was truncated away; its logit would read someone else's token."
        )
    if any(ids[m] != tok.mask_token_id for m in markers):
        raise ValueError(f"{question.id}: a marker does not point at [MASK]; the gather is wrong")
    if slots != list(range(slots[0], slots[0] + n_image_tokens)):
        raise ValueError(f"{question.id}: image slots are not contiguous: {slots}")
    if set(slots) & set(markers):
        raise ValueError(f"{question.id}: an image slot collides with a marker")

    return ImageSequence(input_ids=ids, marker_pos=markers, image_slots=slots)


def collate(
    sequences: list[ImageSequence],
    qtypes: list[int],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a list of sequences into the batch `VisionLaya.forward` expects.

    Mirrors upstream's `collate_items` contract: `marker_pos` is padded to the batch's
    widest option count and `marker_mask` says which entries are real.
    """
    if not sequences:
        raise ValueError("nothing to collate")
    max_l = max(len(s) for s in sequences)
    k_max = max(len(s.marker_pos) for s in sequences)
    n_img = len(sequences[0].image_slots)
    if any(len(s.image_slots) != n_img for s in sequences):
        raise ValueError("all sequences must reserve the same number of image slots")

    b = len(sequences)
    input_ids = torch.full((b, max_l), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((b, max_l), dtype=torch.long)
    marker_pos = torch.zeros((b, k_max), dtype=torch.long)
    marker_mask = torch.zeros((b, k_max), dtype=torch.bool)
    image_slots = torch.zeros((b, n_img), dtype=torch.long)

    for i, s in enumerate(sequences):
        n = len(s)
        input_ids[i, :n] = torch.tensor(s.input_ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        k = len(s.marker_pos)
        marker_pos[i, :k] = torch.tensor(s.marker_pos, dtype=torch.long)
        marker_mask[i, :k] = True
        image_slots[i] = torch.tensor(s.image_slots, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": torch.tensor(qtypes, dtype=torch.long),
        "image_slots": image_slots,
    }
