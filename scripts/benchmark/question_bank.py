"""Ask trained Laya models (and untrained baselines) food questions; score them against the
recorded item lists.

Two question sets, both on **test** photos of Nutrition5k, DFood-TJ and NutritionVerse-Real:

- default: a fixed probe (2 present + 2 absent ingredients, one "which of 5", two thresholds
  per photo) for models never trained on such questions.
- `--bank`: the multi-task question bank's test set (layamacro.data.questions), with held-out
  food names, a held-out phrasing and held-out datasets reported separately.

Question kinds: **ingredient** "Does this meal contain X?" (yes/no as a 2-option `choice` with
neutral keys A/B in random order: Laya's `noul` type can follow its option text instead of
the input); **which** "Which of these foods is on this plate?" (1 recorded item + 4 absent);
**threshold** "Is this meal over 500 kcal?" etc. from the weighed totals.

Answer key: recorded item lists (`data/item_labels.py`) and weighed totals, never an estimate.
Checks before any number counts:
1. **replay**: the 5 trained nutrition questions through this script's code path must
   reproduce the run's saved predictions (same batching: bf16 results depend on it);
2. the graft invariants (asserted by `build_image_sequence` and the model's slot check);
3. **wrong photo**: the same questions about a photo of another dish / patient;
4. a printed spot check (Nutrition5k / NV-Real only; DFood-TJ photos are never shown).

Also runs untrained baselines: `--run runs/vlm_<model>_zeroshot` (a VLM's letter readout,
scripts/benchmark/vlm.py) and `--run pure_laya_text_only` (Laya with no image).
Probabilities are raw (temperature 1): nothing was fitted for these questions.

    uv run python scripts/benchmark/question_bank.py --bank --run runs/<laya run> [--run ...]
"""

from __future__ import annotations

import argparse
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from layamacro.data.item_labels import INVISIBLE, item_names
from layamacro.eval.metrics import ece
from layamacro.graft import INSTRUCTIONS, build_graft, level_labels
from layamacro.model import VisionLaya
from layamacro.schema import TARGETS, Question
from layamacro.sequence import build_image_sequence, collate
from layamacro.train.common import load_data

DATASETS = ("nutrition5k", "dfood_tj", "nutritionverse_real")
QTYPE = {"choice": 0, "score": 1}
OUT = Path("docs/results/question_probe.md")


@dataclass
class Item:
    photo: str
    dataset: str
    family: str  # ingredient | which | threshold
    q: Question
    answer: str  # correct option key
    yes_key: str | None = None  # 2-option yes/no questions: the key meaning "yes"
    invisible: bool = False
    about: str = ""
    held: str = ""  # bank mode: "" | name | phrasing | dataset (DECISIONS 033)


def yes_no(qid: str, text: str, truth: bool, rng: random.Random) -> tuple[Question, str, str]:
    yes_first = rng.random() < 0.5
    yk, nk = ("A", "B") if yes_first else ("B", "A")
    opts = {yk: "yes", nk: "no"}
    q = Question(id=qid, kind="choice", instructions=text, options=dict(sorted(opts.items())))
    return q, (yk if truth else nk), yk


def bank_items(photos: list[Any], seed: int) -> list[Item]:
    """The question bank's fixed test set (data/questions.py), held-out kinds tagged."""
    from layamacro.data.questions import ask

    return [
        Item(
            a.photo,
            a.dataset,
            a.family,
            a.question,
            a.answer,
            a.yes_key,
            a.invisible,
            a.about,
            a.held_out,
        )
        for p in photos
        for a in ask(p, random.Random(f"{seed}:{p.id}"), train=False)
    ]


def build_items(photos: list[Any], seed: int) -> list[Item]:
    names = item_names()
    vocab = {
        d: [
            n
            for n, c in Counter(n for p in photos if p.dataset == d for n in names[p.id]).items()
            if c >= 3
        ]
        for d in DATASETS
    }
    items: list[Item] = []
    for p in photos:
        rng = random.Random(f"{seed}:{p.id}")
        have = sorted(names[p.id])
        absent = sorted(set(vocab[p.dataset]) - names[p.id])
        for n in rng.sample(have, min(2, len(have))):
            q, a, yk = yes_no("ing", f"Does this meal contain {n}?", True, rng)
            items.append(Item(p.id, p.dataset, "ingredient", q, a, yk, n in INVISIBLE, n))
        for n in rng.sample(absent, min(2, len(absent))):
            q, a, yk = yes_no("ing", f"Does this meal contain {n}?", False, rng)
            items.append(Item(p.id, p.dataset, "ingredient", q, a, yk, n in INVISIBLE, n))
        if have and len(absent) >= 4:
            opts = [rng.choice(have), *rng.sample(absent, 4)]
            rng.shuffle(opts)
            keys = "ABCDE"
            q = Question(
                id="which",
                kind="choice",
                instructions="Which of these foods is on this plate?",
                options=dict(zip(keys, opts, strict=True)),
            )
            true_key = keys[[o in names[p.id] for o in opts].index(True)]
            items.append(
                Item(p.id, p.dataset, "which", q, true_key, about=opts[keys.index(true_key)])
            )
        for text, truth in (
            ("Is this meal over 500 kcal?", p.targets["kcal"] > 500),
            ("Does this meal have more than 30 g of protein?", p.targets["protein_g"] > 30),
        ):
            q, a, yk = yes_no("thr", text, truth, rng)
            items.append(Item(p.id, p.dataset, "threshold", q, a, yk, about=text))
    return items


class Prober:
    """The saved graft, rebuilt exactly as trained, answering arbitrary questions."""

    def __init__(self, run: Path, dev: torch.device) -> None:
        import laya

        saved = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        a = argparse.Namespace(**saved["args"])
        self.agent = laya.load(a.checkpoint, device="cpu")
        cache = torch.load(a.features, weights_only=False)
        self.row = {i: k for k, i in enumerate(cache["ids"])}
        self.feats = cache["features"]
        self.model = build_graft(
            self.agent,
            int(self.feats.shape[-1]),
            int(self.feats.shape[1]),
            no_resampler=a.no_resampler,
            n_image_tokens=a.n_image_tokens,
        )
        self.model = self.model.float().to(dev)
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.tail = None
        if saved.get("tail") is not None:
            from layamacro.vision.tail import SiglipTail

            self.tail = SiglipTail(a.tune_last).to(dev)
            self.tail.load_state_dict(saved["tail"])
            self.tail.eval()
        self.args, self.dev, self.run = a, dev, run
        self._seq: dict[tuple[str, str], Any] = {}

    def seq(self, q: Question) -> Any:
        key = (q.instructions, repr(q.options))
        if key not in self._seq:
            self._seq[key] = build_image_sequence(
                self.agent.tok,
                q,
                n_image_tokens=self.args.n_image_tokens,
                max_len=self.agent.cfg["max_len"],
                head_max_len=self.agent.cfg["head_max_len"],
                placeholder_id=self.model.placeholder_id,
                vocab_size=self.model.vocab_size,
            )
        return self._seq[key]

    @torch.no_grad()
    def probs(self, qs: list[Question], photo_ids: list[str], batch: int = 16) -> list[np.ndarray]:
        """Option probabilities (raw softmax over each question's options), row by row."""
        out: list[np.ndarray] = []
        for k in range(0, len(qs), batch):
            bq, bp = qs[k : k + batch], photo_ids[k : k + batch]
            b = collate(
                [self.seq(q) for q in bq], [QTYPE[q.kind] for q in bq], self.agent.tok.pad_token_id
            )
            b = {n: v.to(self.dev) for n, v in b.items()}
            # Encode each photo once and share it across its questions, as training does
            # (bf16 results depend on batch size, so this also keeps replay exact).
            uniq = list(dict.fromkeys(bp))
            px = self.feats[[self.row[p] for p in uniq]].to(self.dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.tail is not None:
                    px = self.tail(px)
                img = self.model.encode_image(px)[[uniq.index(p) for p in bp]]
                lg = self.model(
                    b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"],
                    b["qtype"], image_tokens=img, image_slots=b["image_slots"],
                ).logits.float()  # fmt: skip
            for i, q in enumerate(bq):
                out.append(torch.softmax(lg[i, : q.n_options], -1).cpu().numpy())
        return out

    def replay(self, photos: list[Any]) -> float:
        """Max |logit diff| between this path and the run's saved predictions (score questions)."""
        bins = load_data()[1]
        qs = [
            Question(
                id=t, kind="score", instructions=INSTRUCTIONS[t], options=level_labels(bins[t])
            )
            for t in TARGETS
        ]
        pred = np.load(self.run / "pred.npz")
        prow = {str(i): k for k, i in enumerate(pred["ids"])}
        # Same batching as the training script's evaluation (16 photos x 5 questions per
        # forward): bf16 kernels differ by batch size, so other batchings shift probabilities
        # by up to ~0.03 (measured 2026-10-07); identical batching must match exactly.
        sel = photos[:48]
        got = self.probs(
            [q for _ in sel for q in qs], [p.id for p in sel for _ in qs], batch=16 * len(qs)
        )
        worst = 0.0
        for i, p in enumerate(sel):
            for j, t in enumerate(TARGETS):
                ref = torch.softmax(torch.from_numpy(pred[f"logits_{t}"][prow[p.id]]).float(), -1)
                worst = max(worst, float(np.abs(got[i * len(qs) + j] - ref.numpy()).max()))
        return worst


class VLMProber:
    """A LoRA-tuned decoder VLM (LoRA, letter readout) behind the same
    interface: every answer is the softmax over the option letters' next-token logits."""

    def __init__(self, run: Path, dev: torch.device) -> None:
        from peft import LoraConfig, get_peft_model

        from layamacro.baselines.vlm import language_linear_names, letter_ids, load_vlm

        saved = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        a = argparse.Namespace(**saved["args"])
        self.vlm = load_vlm(a.model, str(dev))
        lora = LoraConfig(
            r=a.lora_r,
            lora_alpha=2 * a.lora_r,
            lora_dropout=0.05,
            target_modules=language_linear_names(self.vlm.model),
        )
        self.vlm.model = get_peft_model(self.vlm.model, lora)
        missing, unexpected = self.vlm.model.load_state_dict(saved["lora"], strict=False)
        assert not unexpected and not [k for k in missing if "lora_" in k], "LoRA keys mismatch"
        self.vlm.model.eval()
        self.letters = torch.tensor(letter_ids(self.vlm, 18), device=dev)
        samples, self.bins = load_data()
        self.images = {s.id: s.image for s in samples}
        self.run, self.dev, self.batch = run, dev, a.batch * 2
        self.autocast = True  # a LoRA-tuned VLM is evaluated under bf16 autocast
        self.replay_photos: list[Any] | None = None

    def text(self, q: Question) -> str:
        from layamacro.baselines.vlm import choice_prompt, zero_shot_prompt

        if q.kind == "score":
            return self.vlm.prompt(zero_shot_prompt(q.id, self.bins[q.id]))
        assert isinstance(q.options, dict)
        return self.vlm.prompt(choice_prompt(q.instructions, q.options))

    @torch.no_grad()
    def probs(self, qs: list[Question], photo_ids: list[str], batch: int = 0) -> list[np.ndarray]:
        from layamacro.data.images import load_image

        out: list[np.ndarray] = []
        batch = batch or self.batch
        for k in range(0, len(qs), batch):
            bq, bp = qs[k : k + batch], photo_ids[k : k + batch]
            b = self.vlm.batch(
                [self.text(q) for q in bq], [load_image(self.images[p]) for p in bp]
            ).to(self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                lg = self.vlm.model(**b, logits_to_keep=1).logits[:, -1][:, self.letters].float()
            for i, q in enumerate(bq):
                out.append(torch.softmax(lg[i, : q.n_options], -1).cpu().numpy())
        return out

    def replay(self, photos: list[Any]) -> float:
        """Nutrition prompts with the training script's batching (one target at a time)."""
        pred = np.load(self.run / "pred.npz")
        prow = {str(i): k for k, i in enumerate(pred["ids"])}
        sel = (self.replay_photos or photos)[: self.batch]
        worst = 0.0
        for t in TARGETS:
            q = Question(id=t, kind="score", instructions=t, options=["x"] * 18)
            got = self.probs([q] * len(sel), [p.id for p in sel])
            for i, p in enumerate(sel):
                ref = torch.softmax(torch.from_numpy(pred[f"logits_{t}"][prow[p.id]]).float(), -1)
                worst = max(worst, float(np.abs(got[i] - ref.numpy()).max()))
        return worst


class ZeroShotVLMProber(VLMProber):
    """A downloaded VLM with no training at all (runs/vlm_<model>_zeroshot): same letter
    readout; replay against that zero-shot run's saved nutrition predictions."""

    def __init__(self, run: Path, dev: torch.device) -> None:
        from layamacro.baselines.vlm import MODELS, letter_ids, load_vlm

        key = next(k for k in MODELS if f"_{k}_zeroshot" in run.name)
        self.vlm = load_vlm(key, str(dev))
        self.letters = torch.tensor(letter_ids(self.vlm, 18), device=dev)
        samples, self.bins = load_data()
        self.images = {s.id: s.image for s in samples}
        self.run, self.dev, self.batch = run, dev, 8  # scripts/benchmark/vlm.py default batch
        # The zero-shot run evaluated without autocast, in eval-sample order (its first batch
        # is the first 8 eval samples); replay must match both to be exact.
        self.autocast = False
        from layamacro.eval.evaluate import eval_samples

        self.replay_photos = eval_samples(samples)


class TextLayaProber:
    """Pure Laya, as downloaded, with NO image: the question about "a meal" in text only.

    Laya cannot see images without our trained graft, so this is its honest untrained
    baseline: what the words alone give. Raw probabilities (temperature 1), like every probe.
    """

    STATE = "A photo of a meal."

    def __init__(self, dev: torch.device) -> None:
        import laya
        from laya import common as laya_common

        self.agent = laya.load("convaiinnovations/laya-typed-decisions", device="cpu")
        self.model = VisionLaya.from_agent(self.agent, encoder_dim=1024).float().to(dev).eval()
        self.lc, self.dev, self.run = laya_common, dev, Path("pure_laya_text_only")

    @torch.no_grad()
    def probs(self, qs: list[Question], photo_ids: list[str], batch: int = 32) -> list[np.ndarray]:
        from layamacro.sequence import ImageSequence

        out: list[np.ndarray] = []
        for k in range(0, len(qs), batch):
            seqs = []
            for q in qs[k : k + batch]:
                ids, markers = self.lc.build_sequence(
                    self.agent.tok,
                    state=self.STATE,
                    q=q.to_laya_internal(),
                    max_len=self.agent.cfg["max_len"],
                    head_max_len=self.agent.cfg["head_max_len"],
                )
                assert len(markers) == q.n_options
                seqs.append(ImageSequence(input_ids=ids, marker_pos=markers, image_slots=[]))
            bq = qs[k : k + batch]
            b = collate(seqs, [QTYPE[q.kind] for q in bq], self.agent.tok.pad_token_id)
            b = {n: v.to(self.dev) for n, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = self.model(
                    b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"],
                    b["qtype"],
                ).logits.float()  # fmt: skip
            for i, q in enumerate(bq):
                out.append(torch.softmax(lg[i, : q.n_options], -1).cpu().numpy())
        return out

    def replay(self, photos: list[Any]) -> float:
        return 0.0  # nothing trained, nothing to reproduce


def make_prober(run: Path, dev: torch.device) -> Any:
    if run.name == "pure_laya_text_only":
        return TextLayaProber(dev)
    if run.name.endswith("_zeroshot"):
        return ZeroShotVLMProber(run, dev)
    return VLMProber(run, dev) if "_letters_" in run.name else Prober(run, dev)


def score(items: list[Item], probs: list[np.ndarray]) -> dict[str, float]:
    pick = np.array([int(np.argmax(p)) for p in probs])
    keys = [list(it.q.options) for it in items]  # type: ignore[arg-type]
    correct = np.array([keys[i][pick[i]] == it.answer for i, it in enumerate(items)])
    conf = np.array([float(np.max(p)) for p in probs])
    out = {"n": float(len(items)), "acc": float(correct.mean()), "ece": ece(conf, correct)}
    if items[0].yes_key is not None:
        from sklearn.metrics import roc_auc_score

        p_yes = np.array(
            [
                p[keys[i].index(str(it.yes_key))]
                for i, (it, p) in enumerate(zip(items, probs, strict=True))
            ]
        )
        truth = np.array([it.answer == it.yes_key for it in items])
        out["auroc"] = (
            float(roc_auc_score(truth, p_yes)) if 0 < truth.sum() < len(truth) else float("nan")
        )
        out["majority"] = float(max(truth.mean(), 1 - truth.mean()))
        out["says_yes"] = float((p_yes > 0.5).mean())
    else:
        out["majority"] = 1.0 / len(keys[0])
    out["picks_A"] = float(np.mean([keys[i][pick[i]] == "A" for i in range(len(items))]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--bank",
        action="store_true",
        help="ask the multi-task question bank's test set (held-out kinds reported apart)",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    dev = torch.device(args.device)

    samples, _ = load_data()
    photos = [s for s in samples if s.dataset in DATASETS and s.split == "test"]
    items = bank_items(photos, args.seed) if args.bank else build_items(photos, args.seed)
    # wrong photo: every question asked about a photo of another group (dish / patient) of the
    # same dataset. Neighbouring photos can be the same dish (NV-Real has ~4 per dish), which
    # would make the control useless, so the replacement comes from a different group.
    nxt: dict[str, str] = {}
    for d in DATASETS:
        ps = [p for p in photos if p.dataset == d]
        for i, p in enumerate(ps):
            for step in range(1, len(ps)):
                q = ps[(i + step * 7) % len(ps)]
                if q.group != p.group:
                    nxt[p.id] = q.id
                    break
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    title = (
        "# Question-bank test: trained, held-out names, held-out phrasings, held-out datasets"
        if args.bank
        else "# Zero-shot probe: Laya asked questions it was never trained on"
    )
    L = [
        title,
        "",
        "`scripts/benchmark/question_bank.py`, "
        f"commit `{commit}`, seed {args.seed}. Test photos only: "
        + ", ".join(f"{d} {sum(p.dataset == d for p in photos)}" for d in DATASETS)
        + ". Answer key: recorded item lists and weighed totals. Probabilities are raw "
        "(no temperature fitted for these questions). **Wrong photo** = the same question "
        "about another meal of the same dataset: if accuracy does not drop, the answer did "
        "not come from the image.",
    ]
    for run in args.run:
        pr = make_prober(run, dev)
        rep = pr.replay(photos)
        print(f"{run.name}: replay max prob diff {rep:.2e}", flush=True)
        if rep > 1e-3:
            raise AssertionError(f"probe path does not reproduce {run.name}'s predictions: {rep}")
        P = pr.probs([it.q for it in items], [it.photo for it in items])
        W = pr.probs([it.q for it in items], [nxt[it.photo] for it in items])
        L += [
            "",
            f"## {run.name}",
            "",
            f"Replay check (5 trained questions through this script vs the run's saved "
            f"predictions, 48 photos, same batching): max probability difference **{rep:.1e}**: "
            "passed. (With a different batching, 16-bit GPU kernels shift probabilities by up "
            "to ~0.03, so all probe questions use one fixed batching.)",
            "",
            "| question | dataset | n | accuracy | chance / majority | wrong photo | AUROC | ECE | says yes | picks A |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        groups: list[tuple[str, str, Any]] = []
        for fam in ("ingredient", "which", "threshold"):
            for d in DATASETS:
                if args.bank:
                    for held in ("", "name", "phrasing", "dataset"):
                        tag = f" [held-out {held}]" if held else " [trained]"
                        groups.append(
                            (
                                fam + tag,
                                d,
                                lambda it, f=fam, d=d, h=held: (
                                    it.family == f and it.dataset == d and it.held == h
                                ),
                            )
                        )
                    continue
                for vis, tag in (
                    (None, ""),
                    (False, " (visible items)"),
                    (True, " (seasoning / oil)"),
                ):
                    if vis is not None and not (fam == "ingredient" and d == "nutrition5k"):
                        continue
                    groups.append(
                        (
                            fam + tag,
                            d,
                            lambda it, f=fam, d=d, v=vis: (
                                it.family == f
                                and it.dataset == d
                                and (v is None or it.invisible == v)
                            ),
                        )
                    )
        for label, d, sel in groups:
            idx = [i for i, it in enumerate(items) if sel(it)]
            if not idx:
                continue
            s = score([items[i] for i in idx], [P[i] for i in idx])
            w = score([items[i] for i in idx], [W[i] for i in idx])
            L.append(
                f"| {label} | {d} | {int(s['n'])} | **{s['acc']:.3f}** | {s['majority']:.3f} "
                f"| {w['acc']:.3f} | {s.get('auroc', float('nan')):.3f} | {s['ece']:.3f} "
                f"| {s.get('says_yes', float('nan')):.2f} | {s['picks_A']:.2f} |"
            )
        print(f"{run.name}: scored {len(items)} questions", flush=True)
        # spot check, public datasets only
        spot = [i for i, it in enumerate(items) if it.dataset != "dfood_tj"][
            :: max(1, len(items) // 40)
        ][:10]
        L += ["", "Spot check (public datasets only):", ""]
        L += [
            "| photo | question | options | Laya's probabilities | correct |",
            "|---|---|---|---|---|",
        ]
        for i in spot:
            it = items[i]
            opts = " · ".join(f"{k}: {v}" for k, v in it.q.options.items())  # type: ignore[union-attr]
            pr_s = " · ".join(f"{k} {p:.2f}" for k, p in zip(it.q.options, P[i], strict=True))  # type: ignore[arg-type]
            L.append(f"| {it.photo} | {it.q.instructions} | {opts} | {pr_s} | {it.answer} |")
        del pr
        torch.cuda.empty_cache()
    args.out.write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
