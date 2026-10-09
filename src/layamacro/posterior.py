"""MacroPosterior: encoder patch tokens → one distribution per target.

No language model: one learned posterior per target over the frozen log-spaced bins, and
**questions answered in code** from that posterior (`MacroDistribution`). Every output is a
full distribution, ranges come from it rather than from a fixed ±%, and temperatures are
fitted on the calibration split, never on test.

A log1p regression head shares the pooled features as a point-estimate baseline; the
released trainer (scripts/train/train_macroposterior.py) does not train it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layamacro.schema import MACROS, MacroBins


class AttentionPool(nn.Module):
    """Learned queries cross-attend over patch tokens (SigLIP's MAP head, re-learned)."""

    def __init__(self, d_in: int, d: int, n_queries: int, n_heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.inp = nn.Linear(d_in, d)
        self.queries = nn.Parameter(torch.randn(n_queries, d) * d**-0.5)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, patches: Tensor) -> Tensor:
        """[B, P, d_in] → [B, n_queries, d]."""
        kv = self.inp(self.norm(patches))
        q = self.queries.expand(patches.shape[0], -1, -1)
        pooled, _ = self.attn(q, kv, kv, need_weights=False)
        return self.out_norm(pooled)


class MacroPosterior(nn.Module):
    """One query per macro → per-macro bin logits; plus a shared regression head."""

    def __init__(
        self,
        bins: dict[str, MacroBins],
        d_in: int = 1024,
        d: int = 512,
        dropout: float = 0.1,
        targets: tuple[str, ...] = MACROS,
    ) -> None:
        super().__init__()
        #: Defaults to the 4 macros; the trainers pass schema.TARGETS (macros + mass).
        self.macros = tuple(targets)
        self.n_bins = [bins[m].n_bins for m in self.macros]
        self.pool = AttentionPool(d_in, d, n_queries=len(self.macros))
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
        )
        self.bin_heads = nn.ModuleList(nn.Linear(d, k) for k in self.n_bins)
        self.reg_head = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, patches: Tensor) -> tuple[list[Tensor], Tensor]:
        """→ (per-macro bin logits [B, K_m], log1p regression [B, n_macros])."""
        h = self.pool(patches.float())
        h = h + self.mlp(h)
        h = self.drop(h)
        logits = [head(h[:, i]) for i, head in enumerate(self.bin_heads)]
        reg = self.reg_head(h).squeeze(-1)
        return logits, reg


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------


def hl_gauss_target(values: Tensor, edges: Tensor, offset: float, sigma: float) -> Tensor:
    """Soft ordinal target: a Gaussian in log(v + offset) space, integrated over each bin.

    "Histogram loss" (Imani & White 2018; Farebrother et al. 2024). The mass that falls
    outside the edges is folded into the end bins, so the target always sums to 1.
    """
    t = torch.log(values.clamp_min(0) + offset).unsqueeze(-1)  # [B, 1]
    e = torch.log(edges + offset).unsqueeze(0)  # [1, K+1]
    cdf = 0.5 * (1 + torch.erf((e - t) / (sigma * math.sqrt(2))))
    cdf[:, 0] = 0.0
    cdf[:, -1] = 1.0
    return cdf[:, 1:] - cdf[:, :-1]


# --------------------------------------------------------------------------------------
# Answering questions from the posterior — the part that replaces Laya's option scoring
# --------------------------------------------------------------------------------------


@dataclass
class MacroDistribution:
    """A batch of posteriors for one macro. All question answering is arithmetic on this.

    Within a bin, probability mass is spread uniformly in log(v + offset) space — consistent
    with how the bins are spaced and how the targets were built.
    """

    probs: Tensor  # [B, K]
    edges: Tensor  # [K+1], natural units
    offset: float

    @property
    def _log_edges(self) -> Tensor:
        return torch.log(self.edges + self.offset)

    def _unlog(self, t: Tensor) -> Tensor:
        return torch.exp(t) - self.offset

    def expected(self) -> Tensor:
        """Mean of the piecewise-log-uniform density, in natural units."""
        a, b = self._log_edges[:-1], self._log_edges[1:]
        # E[exp(U) - offset] for U ~ Uniform(a, b)
        bin_mean = (torch.exp(b) - torch.exp(a)) / (b - a) - self.offset
        return self.probs @ bin_mean

    def quantile(self, q: float) -> Tensor:
        """Inverse CDF, interpolated linearly in log space within the bin."""
        cdf = torch.cumsum(self.probs, dim=-1)
        k = torch.searchsorted(cdf, torch.full_like(cdf[:, :1], q)).clamp(max=cdf.shape[1] - 1)
        c_hi = cdf.gather(1, k)
        p_k = self.probs.gather(1, k).clamp_min(1e-12)
        frac = ((q - (c_hi - p_k)) / p_k).clamp(0, 1)
        le = self._log_edges
        t = le[k] + frac * (le[k + 1] - le[k])
        return self._unlog(t).squeeze(-1)

    def median(self) -> Tensor:
        return self.quantile(0.5)

    def interval(self, coverage: float) -> tuple[Tensor, Tensor]:
        """Central interval: the honest range, read straight off the distribution."""
        a = (1 - coverage) / 2
        return self.quantile(a), self.quantile(1 - a)

    def prob_above(self, x: float) -> Tensor:
        """noul: P(value > x). A question never trained on, answered exactly."""
        lx = math.log(max(x, 0.0) + self.offset)
        a, b = self._log_edges[:-1], self._log_edges[1:]
        frac_above = ((b - lx) / (b - a)).clamp(0, 1)
        return self.probs @ frac_above

    def prob_between(self, lo: float, hi: float) -> Tensor:
        return self.prob_above(lo) - self.prob_above(hi)

    def choice(self, thresholds: list[float]) -> Tensor:
        """choice over the partition (-inf, t0], (t0, t1], ..., (tn, inf) → [B, n+1]."""
        above = [torch.ones_like(self.probs[:, 0])] + [self.prob_above(t) for t in thresholds]
        above.append(torch.zeros_like(self.probs[:, 0]))
        return torch.stack([above[i] - above[i + 1] for i in range(len(above) - 1)], dim=-1)


def distributions(
    logits: list[Tensor], bins: list[MacroBins], temperatures: list[float] | None = None
) -> list[MacroDistribution]:
    temps = temperatures or [1.0] * len(logits)
    return [
        MacroDistribution(
            F.softmax(lg.float() / t, dim=-1), torch.tensor(b.edges, dtype=torch.float32), b.offset
        )
        for lg, b, t in zip(logits, bins, temps, strict=True)
    ]
