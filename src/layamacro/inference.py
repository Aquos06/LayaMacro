"""Run a trained LayaMacro model on photos.

    from layamacro.inference import load_predictor
    model = load_predictor("weights/layamacro-laya.pt")   # or hf://<user>/<repo>/<file>
    model.predict(["meal.jpg"])                          # kcal, protein, carbs, fat, grams
    model.ask(["meal.jpg"], "Does this meal contain rice?", ["yes", "no"])   # Laya only

Every number comes with its central 80 % range, from the model's probability distribution
over the bins. A checkpoint carries the temperatures fitted on the calibration split
(`temperatures`); without them the raw (uncalibrated) distribution is used, with a warning.

The image goes through SigLIP 2 exactly as in training: EXIF-rotated, the model's own
processor, bf16, and for models with trained SigLIP 2 blocks the frozen first blocks followed
by the trained ones.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from layamacro.data.images import load_image
from layamacro.posterior import MacroPosterior, distributions
from layamacro.schema import TARGETS, MacroBins, Question, load_macro_bins
from layamacro.vision.encoder import SIGLIP2_LARGE_384
from layamacro.vision.tail import SiglipTail

COVERAGE = 0.8


def _bins(ckpt: dict[str, Any]) -> dict[str, MacroBins]:
    """The bin edges the model was trained with: shipped inside release checkpoints; for a
    training run's checkpoint, read from the repository's configs/target_bins.yaml."""
    if "bins" in ckpt:
        return {t: MacroBins(**b) for t, b in ckpt["bins"].items()}
    return load_macro_bins("configs/target_bins.yaml", required=TARGETS)


class _Encoder:
    """SigLIP 2, frozen, returning either the final patch tokens or a hidden state."""

    def __init__(self, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.proc = AutoImageProcessor.from_pretrained(SIGLIP2_LARGE_384)
        vision = AutoModel.from_pretrained(SIGLIP2_LARGE_384).vision_model
        self.model = vision.to(device, torch.bfloat16).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, paths: list[str | Path], hidden_layer: int | None) -> torch.Tensor:
        imgs = [load_image(Path(p)) for p in paths]
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"]
        px = px.to(self.device, torch.bfloat16)
        if hidden_layer is None:
            out = self.model(pixel_values=px).last_hidden_state
        else:
            out = self.model(pixel_values=px, output_hidden_states=True).hidden_states[hidden_layer]
        # Training read fp16 caches: same rounding here.
        return out.to(torch.float16).float()


def _summary(dists: list[Any]) -> list[dict[str, dict[str, float]]]:
    out: list[dict[str, dict[str, float]]] = [{} for _ in range(len(dists[0].probs))]
    for j, t in enumerate(TARGETS):
        mean = dists[j].expected()
        lo, hi = dists[j].interval(COVERAGE)
        for i in range(len(out)):
            out[i][t] = {"value": float(mean[i]), "low": float(lo[i]), "high": float(hi[i])}
    return out


class _Tuned(nn.Module):
    def __init__(self, tail: SiglipTail, head: MacroPosterior) -> None:
        super().__init__()
        self.tail, self.head = tail, head


class MacroPosteriorPredictor:
    """SigLIP 2 (optionally with trained last blocks) -> MacroPosterior head."""

    def __init__(self, ckpt: dict[str, Any], device: torch.device) -> None:
        self.bins = _bins(ckpt)
        head = MacroPosterior(self.bins, d_in=1024, targets=TARGETS)
        k = int(ckpt["args"].get("tune_last", 0))
        self.tail: SiglipTail | None = None
        if k:
            self.tail = SiglipTail(k)
            module: nn.Module = _Tuned(self.tail, head)
        else:
            module = head
        module.load_state_dict(ckpt["model"])
        self.head = head.to(device).eval()
        if self.tail is not None:
            self.tail.to(device).eval()
        self.hidden_layer = self.tail.first if self.tail is not None else None
        self.temps = ckpt.get("temperatures")
        self.encoder = _Encoder(device)

    @torch.no_grad()
    def predict(self, paths: list[str | Path]) -> list[dict[str, dict[str, float]]]:
        feats = self.encoder(paths, self.hidden_layer)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feats.is_cuda):
            if self.tail is not None:
                feats = self.tail(feats)
            logits, _ = self.head(feats)
        return _summary(
            distributions(
                [lg.float().cpu() for lg in logits], [self.bins[t] for t in TARGETS], self.temps
            )
        )


class LayaPredictor:
    """SigLIP 2 (optionally with trained last blocks) -> projector -> Laya's decision head."""

    def __init__(self, ckpt: dict[str, Any], device: torch.device) -> None:
        import laya

        from layamacro.graft import build_graft, nutrition_questions

        a = ckpt["args"]
        self.agent = laya.load(
            a.get("checkpoint", "convaiinnovations/laya-typed-decisions"), device="cpu"
        )
        self.model = build_graft(
            self.agent,
            1024,
            576,
            no_resampler=a.get("no_resampler", True),
            n_image_tokens=a.get("n_image_tokens", 64),
        ).float()
        if ckpt.get("slim"):
            # Release checkpoints carry only what was trained; the rest is pristine Laya.
            missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
            if unexpected or any(not k.startswith("laya.") for k in missing):
                raise ValueError(
                    f"checkpoint does not fit the graft: {unexpected[:3]} {missing[:3]}"
                )
        else:
            self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self.tail: SiglipTail | None = None
        if ckpt.get("tail") is not None:
            self.tail = SiglipTail(int(a["tune_last"]))
            self.tail.load_state_dict(ckpt["tail"])
            self.tail.to(device).eval()
        self.hidden_layer = self.tail.first if self.tail is not None else None
        self.bins = _bins(ckpt)
        self.questions = nutrition_questions(self.bins)
        self.temps = ckpt.get("temperatures")
        #: Food names a multi-task checkpoint was trained to recognise (empty otherwise).
        self.foods: list[str] = list(ckpt.get("foods", []))
        self.device = device
        self.encoder = _Encoder(device)

    @torch.no_grad()
    def _option_probs(
        self, feats: torch.Tensor, questions: list[Question], temps: list[float] | None = None
    ) -> list[torch.Tensor]:
        """Per question, [n_photos, n_options] probabilities."""
        from layamacro.graft import QTYPE_CHOICE, QTYPE_SCORE
        from layamacro.sequence import build_image_sequence, collate

        seqs = [
            build_image_sequence(
                self.agent.tok,
                q,
                n_image_tokens=self.model.n_image_tokens,
                max_len=self.agent.cfg["max_len"],
                head_max_len=self.agent.cfg["head_max_len"],
                placeholder_id=self.model.placeholder_id,
                vocab_size=self.model.vocab_size,
            )
            for q in questions
        ]
        qt = [QTYPE_SCORE if q.kind == "score" else QTYPE_CHOICE for q in questions]
        n, nq = len(feats), len(questions)
        b = collate(seqs * n, qt * n, pad_id=self.agent.tok.pad_token_id)
        b = {k: v.to(self.device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feats.is_cuda):
            px = self.tail(feats) if self.tail is not None else feats
            img = self.model.encode_image(px).repeat_interleave(nq, dim=0)
            lg = self.model(
                b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"],
                b["qtype"], image_tokens=img, image_slots=b["image_slots"],
            ).logits.float().view(n, nq, -1)  # fmt: skip
        out = []
        for j, q in enumerate(questions):
            t = temps[j] if temps else 1.0
            out.append(torch.softmax(lg[:, j, : q.n_options] / t, -1).cpu())
        return out

    def _distributions(self, paths: list[str | Path]) -> list[Any]:
        feats = self.encoder(paths, self.hidden_layer)
        probs = self._option_probs(feats, self.questions)
        logits = [torch.log(p.clamp_min(1e-12)) for p in probs]
        return distributions(logits, [self.bins[t] for t in TARGETS], self.temps)

    def predict(self, paths: list[str | Path]) -> list[dict[str, dict[str, float]]]:
        return _summary(self._distributions(paths))

    def prob_over(self, paths: list[str | Path], target: str, value: float) -> list[float]:
        """P(target > value), e.g. ("kcal", 500): read from the calibrated nutrition
        distribution, so more reliable than asking the same thing as a text question."""
        d = self._distributions(paths)[TARGETS.index(target)]
        return [float(x) for x in d.prob_above(value)]

    def ask(
        self, paths: list[str | Path], question: str, options: list[str]
    ) -> list[dict[str, float]]:
        """Any question with up to 8 options, answered without retraining (raw probabilities)."""
        if not 2 <= len(options) <= 8:
            raise ValueError("ask() takes 2-8 options")
        keys = "ABCDEFGH"[: len(options)]
        q = Question(
            id="ask",
            kind="choice",
            instructions=question,
            options=dict(zip(keys, options, strict=True)),
        )
        feats = self.encoder(paths, self.hidden_layer)
        p = self._option_probs(feats, [q])[0]
        return [{o: float(p[i, k]) for k, o in enumerate(options)} for i in range(len(paths))]


def load_predictor(
    path: str | Path, device: str | None = None
) -> MacroPosteriorPredictor | LayaPredictor:
    """Load a LayaMacro checkpoint (MacroPosterior or Laya graft) for inference."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if str(path).startswith("hf://"):
        # hf://<user>/<repo>/<file>: download (and cache) from the Hugging Face Hub
        from huggingface_hub import hf_hub_download

        user, repo, filename = str(path)[5:].split("/", 2)
        path = hf_hub_download(f"{user}/{repo}", filename)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    if not isinstance(args, dict):
        args = vars(args)
        ckpt["args"] = args
    if ckpt.get("temperatures") is None:
        warnings.warn(
            "checkpoint has no fitted temperatures: ranges are uncalibrated", stacklevel=2
        )
    if ckpt.get("kind") == "laya" or "no_resampler" in args or "scope" in args:
        return LayaPredictor(ckpt, dev)
    return MacroPosteriorPredictor(ckpt, dev)
