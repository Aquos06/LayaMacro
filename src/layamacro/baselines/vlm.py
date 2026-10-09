"""Decoder VLMs as nutrition baselines: zero-shot, fine-tuned and multi-task.

Ways to get an 18-bin distribution per target out of a chat VLM without letting it generate
text (every model is scored on a full distribution, not a single number):

- **zero-shot**: one prompt per target listing the 18 bin ranges as options A-R; the
  next-token logits of the 18 letters are the bin logits. Letters are verified to be single
  tokens for every model.
- **fine-tuned**: vision tower frozen by default; LoRA on the language model's linear
  layers; one fixed prompt; the final hidden state at the last prompt token → LayerNorm →
  Linear → 5 x 18 bin logits. Same HL-Gauss soft labels as every other model here.
- **multi-task**: LoRA, and every question (the 5 nutrition prompts of the zero-shot
  readout plus the bank's ingredient / which-food / threshold questions) answered through
  the model's own next-token letter logits, so new questions need no new head.

Model facts verified 2026-09-29 (transformers 5.17): Qwen3.5's chat template already emits
an empty `<think></think>` block, so it answers directly; Gemma 4 E2B has no thinking
block and a fixed 280 soft tokens per image.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from layamacro.schema import TARGETS, MacroBins

MODELS = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
}
#: Qwen resizes to the image; cap the long side so a photo costs ~150-300 tokens.
QWEN_LONG_SIDE = 448

QUESTIONS = {
    "kcal": "How many kilocalories does all the food in this photo contain?",
    "protein_g": "How many grams of protein does all the food in this photo contain?",
    "carbs_g": "How many grams of carbohydrate does all the food in this photo contain?",
    "fat_g": "How many grams of fat does all the food in this photo contain?",
    "mass_g": "How much does all the food in this photo weigh, in grams?",
}
FINETUNE_PROMPT = (
    "Estimate the calories, protein, carbohydrate, fat and total weight of all the food in "
    "this photo."
)


@dataclass
class VLM:
    key: str
    model: Any  # a transformers PreTrainedModel (peft-wrapped when fine-tuning)
    processor: Any

    @property
    def is_qwen(self) -> bool:
        return self.key.startswith("qwen")

    def prompt(self, text: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        kw = {"enable_thinking": False} if self.is_qwen else {}
        return self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, **kw
        )

    def prepare_image(self, img: Image.Image) -> Image.Image:
        if self.is_qwen and max(img.size) > QWEN_LONG_SIDE:
            s = QWEN_LONG_SIDE / max(img.size)
            img = img.resize((max(32, round(img.width * s)), max(32, round(img.height * s))))
        return img

    def batch(self, texts: list[str], images: list[Image.Image]) -> Any:
        """A transformers BatchFeature (dict-like, with `.to(device)`)."""
        prepared = [self.prepare_image(i) for i in images]
        # Gemma 4's processor wants one image list per prompt; Qwen takes a flat list.
        imgs: list[Any] = list(prepared) if self.is_qwen else [[i] for i in prepared]
        return self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)

    @property
    def hidden_size(self) -> int:
        cfg = self.model.config
        text = getattr(cfg, "text_config", cfg)
        return int(text.hidden_size)


def load_vlm(key: str, device: str, dtype: torch.dtype = torch.bfloat16) -> VLM:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    mid = MODELS[key]
    proc = AutoProcessor.from_pretrained(mid)
    proc.tokenizer.padding_side = "left"  # last position = last prompt token for every row
    model: Any = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype)
    model = model.to(device)
    model.eval()
    return VLM(key, model, proc)


def letter_ids(vlm: VLM, n: int) -> list[int]:
    tok = vlm.processor.tokenizer
    ids = []
    for L in string.ascii_uppercase[:n]:
        enc = tok.encode(L, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"{vlm.key}: letter {L!r} is {len(enc)} tokens, readout invalid")
        ids.append(enc[0])
    return ids


def range_text(b: MacroBins, i: int) -> str:
    lo, hi = b.edges[i], b.edges[i + 1]
    return f"more than {lo:.3g} {b.unit}" if i == b.n_bins - 1 else f"{lo:.3g} to {hi:.3g} {b.unit}"


def zero_shot_prompt(target: str, b: MacroBins) -> str:
    opts = "\n".join(f"{string.ascii_uppercase[i]}) {range_text(b, i)}" for i in range(b.n_bins))
    return f"{QUESTIONS[target]}\nChoose the best range:\n{opts}\nAnswer with the letter only."


def choice_prompt(instructions: str, options: dict[str, str]) -> str:
    """A bank `choice` question (data/questions.py) in the same letter format as the
    nutrition prompts; its keys are already A, B, ... in order."""
    opts = "\n".join(f"{k}) {v}" for k, v in options.items())
    return f"{instructions}\n{opts}\nAnswer with the letter only."


def language_linear_names(model: nn.Module) -> list[str]:
    """Every nn.Linear inside the language model — LoRA targets. Never the vision tower,
    never the LM head (unused: the fine-tuned model reads hidden states)."""
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        low = name.lower()
        if "language_model" not in low or any(v in low for v in ("vision", "visual", "lm_head")):
            continue
        names.append(name)
    if not names:
        raise ValueError("no language-model Linear layers found: module naming changed")
    return names


class NutritionHead(nn.Module):
    def __init__(self, d: int, bins: dict[str, MacroBins]) -> None:
        super().__init__()
        self.sizes = [bins[t].n_bins for t in TARGETS]
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, sum(self.sizes))

    def forward(self, h: Tensor) -> list[Tensor]:
        return list(self.out(self.norm(h.float())).split(self.sizes, dim=-1))
