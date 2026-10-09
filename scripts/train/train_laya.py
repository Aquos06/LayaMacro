"""Train the Laya graft: SigLIP 2 patch tokens -> projector -> Laya's typed decision head.

SigLIP 2 patch tokens (all 576, `--no-resampler`) are projected to Laya's width and written
into image slots in Laya's state region, after the question and its options. Each nutrition
target is one `score` question whose 18 options are calorie / gram ranges; the answer is read
from Laya's own `[MASK]` markers (its decision head is not changed) and trained with soft
cross-entropy against HL-Gauss targets. `--multitask` adds ingredient, which-food and
threshold `choice` questions from the question bank (layamacro.data.questions).

Recipe: Laya's backbone frozen (`--scope head`), 10 epochs, LRs: new modules 3e-4, Laya's
decision head 1e-4, SigLIP 2 blocks 1e-5 (`--tune-last`). Every graft invariant (markers on
`[MASK]`, image slots holding exactly the projected tokens) is asserted on the first batch.

Released configurations:

    # nutrition-only, last 4 SigLIP 2 blocks trained (needs cache_features --hidden-layer 20)
    uv run python scripts/train/train_laya.py --scope head --no-resampler --images-per-batch 2 \
        --epochs 10 --tune-last 4 --bench --seed 0
    # multi-task on 2 GPUs, 1 photo per GPU (= 2 per update)
    uv run torchrun --nproc_per_node 2 scripts/train/train_laya.py --scope head --no-resampler \
        --images-per-batch 1 --epochs 10 --multitask --bench --seed 0
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from layamacro.eval.evaluate import eval_samples, save_predictions
from layamacro.graft import (
    INSTRUCTIONS,
    QTYPE_CHOICE,
    QTYPE_SCORE,
    build_graft,
    level_labels,
)
from layamacro.model import VisionLaya
from layamacro.schema import TARGETS, Question
from layamacro.sequence import build_image_sequence, collate
from layamacro.train.common import (
    load_data,
    save_weights,
    seed_everything,
    soft_ce,
    soft_targets,
    target_matrix,
)


class Graft(torch.nn.Module):
    """Trainable SigLIP2 blocks (optional) + the graft, behind one forward for DDP.

    Each photo is encoded once and its image tokens are shared by all of its question rows
    (`img_idx` maps row -> photo), as in single-task training.
    """

    def __init__(self, model: VisionLaya, tail: Any) -> None:
        super().__init__()
        self.model, self.tail = model, tail

    def forward(
        self, px: torch.Tensor, img_idx: torch.Tensor, b: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.tail is not None:
                px = self.tail(px)
            img = self.model.encode_image(px)[img_idx]
            out = self.model(
                b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"],
                b["qtype"], image_tokens=img, image_slots=b["image_slots"],
            )  # fmt: skip
        return out.logits.float()


def build_model(agent: Any, d_feat: int, n_patches: int, args: argparse.Namespace) -> VisionLaya:
    """The graft exactly as released (layamacro.graft); records the image-token count."""
    if args.no_resampler:
        args.n_image_tokens = n_patches
    return build_graft(
        agent, d_feat, n_patches, no_resampler=args.no_resampler, n_image_tokens=args.n_image_tokens
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/features/siglip2.pt"))
    ap.add_argument("--checkpoint", default="convaiinnovations/laya-typed-decisions")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--images-per-batch", type=int, default=4)  # x5 questions per step
    ap.add_argument("--lr-new", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--n-image-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop early")
    ap.add_argument(
        "--scope",
        choices=("all", "head", "new"),
        default="all",
        help="all = full fine-tune; head = Laya backbone frozen (resampler, projector and "
        "Laya's decision head train); new = only resampler + projector (VisionLaya.set_trainable)",
    )
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. 'long'")
    ap.add_argument(
        "--no-resampler",
        action="store_true",
        help="skip the Perceiver: all patch tokens go to Laya (the released configuration)",
    )
    ap.add_argument(
        "--tune-last",
        type=int,
        default=0,
        help="also train SigLIP2's last k blocks, on the L(24-k) cache",
    )
    ap.add_argument("--lr-vision", type=float, default=1e-5, help="LR for --tune-last blocks")
    ap.add_argument(
        "--vision-accum",
        type=int,
        default=1,
        help="--tune-last blocks: accumulate gradients over N steps (N x images-per-batch "
        "photos) before an update; the rest of Laya still updates every step",
    )
    ap.add_argument(
        "--bench",
        action="store_true",
        help="also predict the unseen-dataset benchmark",
    )
    ap.add_argument("--bench-features", type=Path, default=Path("data/features/siglip2-bench.pt"))
    ap.add_argument(
        "--multitask",
        action="store_true",
        help="also train ingredient / which-food / threshold questions",
    )
    ap.add_argument(
        "--aux-weight", type=float, default=0.5, help="multitask: non-nutrition loss weight"
    )
    ap.add_argument("--n-other", type=int, default=4, help="multitask: extra questions per photo")
    args = ap.parse_args()
    # Launched by torchrun: one process per GPU, data parallel.
    ddp = "LOCAL_RANK" in os.environ
    rank, world = 0, 1
    if ddp:
        import torch.distributed as dist

        if not args.multitask:
            ap.error("multi-GPU training is implemented for --multitask runs only")
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(args.device)
    seed_everything(args.seed)
    dev = torch.device(args.device)

    import laya

    samples, bins = load_data()
    tail = None
    if args.tune_last:
        from layamacro.vision.tail import SiglipTail

        tail = SiglipTail(args.tune_last).to(dev)
        final = torch.load(args.features, weights_only=False)
        args.features = args.features.with_name(
            args.features.stem + f"-L{tail.first}" + args.features.suffix
        )
    cache = torch.load(args.features, weights_only=False)
    row = {i: k for k, i in enumerate(cache["ids"])}
    feats = cache["features"]  # stays on CPU; Laya + Adam need the GPU memory
    if tail is not None:
        assert cache["hidden_layer"] == tail.first and final["ids"] == cache["ids"]
        err = tail.check_against(feats[:16], final["features"][:16])
        print(f"tail reproduces the frozen encoder: rel err {err:.4f}", flush=True)
        del final

    agent = laya.load(args.checkpoint, device="cpu")
    model = build_model(agent, int(feats.shape[-1]), int(feats.shape[1]), args).float().to(dev)
    model.enable_gradient_checkpointing()
    tok = agent.tok

    # One sequence per target question, all invariants asserted by build_image_sequence.
    questions = [
        Question(id=t, kind="score", instructions=INSTRUCTIONS[t], options=level_labels(bins[t]))
        for t in TARGETS
    ]
    seqs = [
        build_image_sequence(
            tok,
            q,
            n_image_tokens=args.n_image_tokens,
            max_len=agent.cfg["max_len"],
            head_max_len=agent.cfg["head_max_len"],
            placeholder_id=model.placeholder_id,
            vocab_size=model.vocab_size,
        )
        for q in questions
    ]
    print("sequence lengths", [len(s) for s in seqs], "markers", [len(s.marker_pos) for s in seqs])
    template = collate(seqs, [QTYPE_SCORE] * len(seqs), pad_id=tok.pad_token_id)
    template = {k: v.to(dev) for k, v in template.items()}
    nq = len(TARGETS)

    def batch_inputs(n_img: int) -> dict[str, torch.Tensor]:
        """The 5 question rows tiled for n_img images: row = image * nq + question."""
        return {k: v.repeat(n_img, *([1] * (v.dim() - 1))) for k, v in template.items()}

    def logits_of(m: VisionLaya, t: Any, px: torch.Tensor) -> torch.Tensor:
        """[n_img, P, D] encoder features → option logits [n_img, nq, 18]."""
        px = px.to(dev, non_blocking=True).float()
        n = len(px)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if t is not None:
                px = t(px)  # trainable SigLIP2 blocks, then the usual graft
            img = m.encode_image(px)  # encode once per image …
            img = img.repeat_interleave(nq, dim=0)  # … reuse for all 5 questions
            b = batch_inputs(n)
            out = m(
                b["input_ids"],
                b["attention_mask"],
                b["marker_pos"],
                b["marker_mask"],
                b["qtype"],
                image_tokens=img,
                image_slots=b["image_slots"],
            )
        return out.logits.float().view(n, nq, -1)

    def forward(idx: list[int]) -> torch.Tensor:
        return logits_of(model, tail, feats[idx])

    # --- optimiser: three LRs (fresh modules, Laya head, Laya encoder) -----------------
    model.set_trainable(args.scope)
    new_ids = {id(p) for p in model.new_parameters()}
    enc_ids = {id(p) for p in model.laya.encoder.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if id(p) in new_ids], "lr": args.lr_new},
        {
            "params": [p for p in model.parameters() if id(p) in enc_ids and p.requires_grad],
            "lr": args.lr_encoder,
        },
        {
            "params": [
                p
                for p in model.parameters()
                if p.requires_grad and id(p) not in new_ids and id(p) not in enc_ids
            ],
            "lr": args.lr_head,
        },
    ]
    accum = args.vision_accum if tail is not None else 1
    if tail is not None and accum == 1:
        groups.append({"params": list(tail.parameters()), "lr": args.lr_vision})
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    trainable = [p for g in groups for p in g["params"]]
    # With accumulation the SigLIP2 blocks get their own optimiser, stepped every `accum`
    # steps on the averaged gradient; everything else is unchanged.
    tail_params = list(tail.parameters()) if tail is not None and accum > 1 else []
    opt_v = (
        torch.optim.AdamW(tail_params, lr=args.lr_vision, weight_decay=0.01)
        if tail_params
        else None
    )
    train = [s for s in samples if s.split == "train"]
    tr_rows = [row[s.id] for s in train]
    soft = soft_targets(target_matrix(train), bins)  # list over targets of [N, 18]
    soft_all = torch.stack(soft, 1)  # [N, nq, 18]
    per_rank = len(train) // world  # equal shards, or the ranks fall out of step
    steps_per_epoch = math.ceil(per_rank / args.images_per_batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, [g["lr"] for g in groups], total_steps=total, pct_start=0.05
    )
    sched_v = (
        torch.optim.lr_scheduler.OneCycleLR(
            opt_v, args.lr_vision, total_steps=math.ceil(total / accum) + 1, pct_start=0.05
        )
        if opt_v is not None
        else None
    )
    n_params = sum(p.numel() for p in trainable) + sum(p.numel() for p in tail_params)
    if rank == 0:
        print(
            f"train {len(train)} images x {nq} questions"
            + (f" + {args.n_other} other (multitask)" if args.multitask else "")
            + f", {world} GPU(s), {total} steps, {n_params / 1e6:.1f}M trainable",
            flush=True,
        )

    # --- multitask: questions built per photo per step from the bank -----------------
    seq_cache: dict[tuple[str, str], Any] = {}

    def seq_of(q: Question) -> Any:
        key = (q.instructions, repr(q.options))
        if key not in seq_cache:
            seq_cache[key] = build_image_sequence(
                tok,
                q,
                n_image_tokens=args.n_image_tokens,
                max_len=agent.cfg["max_len"],
                head_max_len=agent.cfg["head_max_len"],
                placeholder_id=model.placeholder_id,
                vocab_size=model.vocab_size,
            )
        return seq_cache[key]

    graft = Graft(model, tail).to(dev)
    net: torch.nn.Module = graft
    if ddp:
        from torch.nn.parallel import DistributedDataParallel

        net = DistributedDataParallel(graft, device_ids=[dev.index], find_unused_parameters=True)

    def mt_step(b_idx: list[int], rng: random.Random) -> torch.Tensor:
        from layamacro.data.questions import ask

        rows: list[Any] = []
        qtypes: list[int] = []
        img_idx: list[int] = []
        n_rows: list[int] = []
        aux: list[tuple[int, int]] = []
        for j, i in enumerate(b_idx):
            for sq in seqs:  # the 5 nutrition questions, unchanged
                n_rows.append(len(rows))
                rows.append(sq)
                qtypes.append(QTYPE_SCORE)
                img_idx.append(j)
            for a in ask(train[i], rng, train=True, n_other=args.n_other):
                aux.append((len(rows), list(a.question.options).index(a.answer)))  # type: ignore[arg-type]
                rows.append(seq_of(a.question))
                qtypes.append(QTYPE_CHOICE)
                img_idx.append(j)
        bt = {k: v.to(dev) for k, v in collate(rows, qtypes, pad_id=tok.pad_token_id).items()}
        px = feats[[tr_rows[i] for i in b_idx]].to(dev, non_blocking=True).float()
        lg = net(px, torch.tensor(img_idx, device=dev), bt)
        n_lg = lg[n_rows].view(len(b_idx), nq, -1)
        tgt = soft_all[b_idx].to(dev)
        loss = soft_ce(n_lg.reshape(-1, n_lg.shape[-1]), tgt.reshape(-1, tgt.shape[-1]))
        if aux:
            pos = torch.tensor([r for r, _ in aux], device=dev)
            ans = torch.tensor([k for _, k in aux], device=dev)
            loss = loss + args.aux_weight * torch.nn.functional.cross_entropy(lg[pos], ans)
        return loss

    model.check_invariants = True  # assert slots/markers on the first batch, then off
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        if tail is not None:
            tail.train()
        if args.multitask:
            # Same permutation on every rank, each takes its own equal shard.
            g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
            order = torch.randperm(len(train), generator=g)[: per_rank * world][rank::world]
        else:
            order = torch.randperm(len(train))
        run_loss = 0.0
        for k in range(0, len(order), args.images_per_batch):
            b = order[k : k + args.images_per_batch]
            if args.multitask:
                loss = mt_step(b.tolist(), random.Random(f"{args.seed}:{epoch}:{k}:{rank}"))
            else:
                logits = forward([tr_rows[i] for i in b])
                target = soft_all[b].to(dev)
                loss = soft_ce(
                    logits.reshape(-1, logits.shape[-1]), target.reshape(-1, target.shape[-1])
                )
            opt.zero_grad(set_to_none=True)  # only this optimiser's parameters
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            model.check_invariants = False
            step += 1
            if opt_v is not None and sched_v is not None and step % accum == 0:
                for q in tail_params:
                    if q.grad is not None:
                        q.grad.div_(accum)
                torch.nn.utils.clip_grad_norm_(tail_params, 1.0)
                opt_v.step()
                sched_v.step()
                opt_v.zero_grad(set_to_none=True)
            run_loss += loss.item()
            if step % 100 == 0 and rank == 0:
                el = time.time() - t0
                print(
                    f"epoch {epoch + 1} step {step}/{total} loss {run_loss / 100:.4f} "
                    f"({el / step:.2f}s/step, ETA {(total - step) * el / step / 60:.0f} min)",
                    flush=True,
                )
                run_loss = 0.0
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    train_s = time.time() - t0
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
        if rank != 0:
            return  # rank 0 evaluates, saves and benchmarks alone

    ev = eval_samples(samples)
    ev_rows = [row[s.id] for s in ev]
    model.eval()
    if tail is not None:
        tail.eval()
    outs = []
    t1 = time.time()
    with torch.no_grad():
        for k in range(0, len(ev_rows), 16):
            outs.append(forward(ev_rows[k : k + 16]).cpu().numpy())
    ms = (time.time() - t1) / len(ev_rows) * 1e3
    lg = np.concatenate(outs)  # [N, nq, 18]
    run = save_predictions(
        f"laya{'_' + args.tag if args.tag else ''}_s{args.seed}",
        [s.id for s in ev],
        {t: lg[:, j] for j, t in enumerate(TARGETS)},
        {
            "method": "Laya graft: "
            + (
                f"SigLIP 2 patches, last {args.tune_last} blocks trained (LR {args.lr_vision}) → "
                if args.tune_last
                else "frozen SigLIP2 patches → "
            )
            + (
                f"NO resampler (all {args.n_image_tokens} patches)"
                if args.no_resampler
                else f"Perceiver resampler ({args.n_image_tokens} tok)"
            )
            + " → projector → Laya (laya-typed-decisions) image slots; each target a `score` "
            "question over 18 bin ranges, read from Laya's [MASK] option markers",
            "trained": f"{args.epochs} epochs, scope {args.scope!r}, LR new {args.lr_new} / head "
            f"{args.lr_head} / encoder {args.lr_encoder}"
            + (f", SigLIP2 blocks updated every {accum} steps (accumulated)" if accum > 1 else "")
            + f", {args.images_per_batch} images x 5 "
            f"questions per step, soft CE (HL-Gauss), seed {args.seed}"
            + (f", STOPPED at step {step}" if args.max_steps else ""),
            "params_trainable": n_params,
            "train_seconds": round(train_s),
            "ms_per_image_5_questions": round(ms, 1),
            "checkpoint": args.checkpoint,
        },
    )
    print(f"wrote {run}")

    # --- weights (always), checked by rebuilding the graft from scratch and reloading them
    state = {
        "model": model.state_dict(),
        "tail": tail.state_dict() if tail is not None else None,
        "args": vars(args),
    }
    ckpt = save_weights(run, state)
    agent2 = laya.load(args.checkpoint, device="cpu")
    m2 = build_model(agent2, int(feats.shape[-1]), int(feats.shape[1]), args).float().to(dev)
    saved = torch.load(ckpt, map_location=dev, weights_only=False)
    m2.load_state_dict(saved["model"])
    m2.eval()
    t2 = None
    if tail is not None:
        from layamacro.vision.tail import SiglipTail

        t2 = SiglipTail(args.tune_last).to(dev)
        t2.load_state_dict(saved["tail"])
        t2.eval()
    with torch.no_grad():
        probe = feats[ev_rows[:8]]
        diff = float((logits_of(model, tail, probe) - logits_of(m2, t2, probe)).abs().max())
    if diff > 1e-3:
        raise AssertionError(f"reloaded weights give different logits (max diff {diff})")
    print(f"weights {ckpt}; reload check max logit diff {diff:.2e}", flush=True)
    del m2, t2, agent2, saved
    torch.cuda.empty_cache()

    if args.bench:
        from layamacro.eval.bench import bench_samples

        bpath = args.bench_features
        if tail is not None:
            bpath = bpath.with_name(bpath.stem + f"-L{tail.first}" + bpath.suffix)
        del feats, cache
        bc = torch.load(bpath, weights_only=False)
        assert bc.get("hidden_layer") == (tail.first if tail is not None else None), bpath
        brow = {i: k for k, i in enumerate(bc["ids"])}
        bs = bench_samples()
        bfeats = bc["features"]
        bidx = [brow[s.id] for s in bs]
        bouts = []
        with torch.no_grad():
            for k in range(0, len(bidx), 16):
                bouts.append(logits_of(model, tail, bfeats[bidx[k : k + 16]]).cpu().numpy())
        blg = np.concatenate(bouts)
        arrays = {"ids": np.array([s.id for s in bs])} | {
            f"logits_{t}": blg[:, j] for j, t in enumerate(TARGETS)
        }
        np.savez(Path(run) / "bench.npz", **arrays)  # type: ignore[arg-type]
        print(f"wrote {run}/bench.npz ({len(bs)} benchmark photos)", flush=True)


if __name__ == "__main__":
    main()
