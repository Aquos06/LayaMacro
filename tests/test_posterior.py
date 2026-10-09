"""MacroPosterior: the question-answering arithmetic, and the Nutrition5k parser.

Every answer this model gives is arithmetic on a bin distribution, so that arithmetic is
where a silent error would live. These tests pin it against closed-form cases.
"""

from __future__ import annotations

import math

import pytest
import torch

from layamacro.data.nutrition5k import parse_dish_row
from layamacro.posterior import MacroDistribution, MacroPosterior, hl_gauss_target
from layamacro.schema import MACROS, MacroBins

OFFSET = 10.0


def _edges(n: int = 8, top: float = 1000.0) -> torch.Tensor:
    t = torch.linspace(math.log(OFFSET), math.log(top + OFFSET), n + 1)
    e = torch.exp(t) - OFFSET
    e[0] = 0.0
    return e


def _point_mass(k: int, n: int = 8) -> MacroDistribution:
    p = torch.zeros(1, n)
    p[0, k] = 1.0
    return MacroDistribution(p, _edges(n), OFFSET)


def test_quantiles_stay_inside_a_point_mass_bin() -> None:
    e = _edges()
    d = _point_mass(3)
    lo, hi = d.interval(0.8)
    assert e[3] <= lo < hi <= e[4]
    assert e[3] <= d.median() <= e[4]
    assert e[3] <= d.expected() <= e[4]


def test_median_is_log_midpoint_of_a_single_bin() -> None:
    e = _edges()
    d = _point_mass(5)
    mid = math.exp((math.log(e[5] + OFFSET) + math.log(e[6] + OFFSET)) / 2) - OFFSET
    assert d.median().item() == pytest.approx(mid, rel=1e-5)


def test_prob_above_matches_cdf_and_edges() -> None:
    e = _edges()
    d = _point_mass(4)
    assert d.prob_above(float(e[4])).item() == pytest.approx(1.0, abs=1e-6)
    assert d.prob_above(float(e[5])).item() == pytest.approx(0.0, abs=1e-6)
    # consistency with the quantile function: P(X > q_0.3) = 0.7
    p = torch.softmax(torch.randn(3, 8), -1)
    dd = MacroDistribution(p, e, OFFSET)
    q = dd.quantile(0.3)
    for b in range(3):
        assert dd.prob_above(float(q[b])).tolist()[b] == pytest.approx(0.7, abs=1e-4)


def test_choice_partition_sums_to_one() -> None:
    p = torch.softmax(torch.randn(4, 8), -1)
    d = MacroDistribution(p, _edges(), OFFSET)
    c = d.choice([100.0, 300.0, 600.0])
    assert c.shape == (4, 4)
    assert torch.allclose(c.sum(-1), torch.ones(4), atol=1e-5)
    assert (c >= -1e-6).all()


def test_hl_gauss_target_is_a_distribution_peaked_at_the_true_bin() -> None:
    e = _edges()
    v = torch.tensor([0.0, 50.0, 400.0, 5000.0])  # includes out-of-range on both sides
    t = hl_gauss_target(v, e, OFFSET, sigma=0.2)
    assert torch.allclose(t.sum(-1), torch.ones(4), atol=1e-5)
    for i, x in enumerate(v.tolist()):
        k = min(int(torch.searchsorted(e, torch.tensor(x), right=True)) - 1, 7)
        assert int(t[i].argmax()) == k


def test_model_shapes() -> None:
    bins = {
        m: MacroBins(macro=m, edges=_edges().tolist(), unit="x", offset=OFFSET, fitted_on="test")
        for m in MACROS
    }
    model = MacroPosterior(bins, d_in=32, d=16)
    logits, reg = model(torch.randn(2, 10, 32))
    assert [lg.shape for lg in logits] == [(2, 8)] * len(MACROS)
    assert reg.shape == (2, len(MACROS))


def test_parser_real_layout_macro_order_fat_carb_protein() -> None:
    line = (
        "dish_1,300.5,193.0,12.3,28.2,18.6,"
        "ingr_0000000508,soy sauce,3.4,1.8,0.02,0.17,0.28,"
        "ingr_0000000122,,2.1,3.2,0.01,0.70,0.14"  # blank name happens: join on id
    )
    d = parse_dish_row(line.split(","))
    assert (d.kcal, d.mass_g, d.fat_g, d.carbs_g, d.protein_g) == (300.5, 193.0, 12.3, 28.2, 18.6)
    assert [i.id for i in d.ingredients] == ["ingr_0000000508", "ingr_0000000122"]
    assert d.ingredients[1].name == ""


def test_parser_rejects_the_readme_layout() -> None:
    # README claims a num_ingrs field and 8 fields per ingredient; 0 of 5,006 rows match.
    with pytest.raises(ValueError):
        parse_dish_row(
            [
                "dish_1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "ingr_0000000001",
                "x",
                "1",
                "1",
                "1",
                "1",
                "1",
            ]
        )
