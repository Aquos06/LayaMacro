"""Draw the README's result charts as SVG, a light and a dark file for each.

The numbers are copied from docs/results.md: rerun this after they change.

    uv run python scripts/docs/readme_figures.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures"

# LayaMacro in the accent blue, the other models in a recessive gray. Checked with a
# colour-blindness simulation against GitHub's page backgrounds (#ffffff, #0d1117). Each
# file paints that background itself, so a viewer that shows the wrong theme's file still
# gets a legible chart.
THEMES = {
    "light": {
        "bg": "#ffffff",
        "focus": "#2a78d6",
        "other": "#b4b3ab",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "rule": "#c3c2b7",
    },
    "dark": {
        "bg": "#0d1117",
        "focus": "#3987e5",
        "other": "#686761",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "rule": "#45443f",
    },
}
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'


@dataclass
class Column:
    title: str
    subtitle: str
    values: tuple[float, ...]  # one per model, in Chart.models order
    scale: float  # value at the full bar length
    ticks: tuple[float, ...]
    fmt: str = "{:.1f}"
    chance: float | None = None


@dataclass
class Chart:
    name: str
    title: str
    description: str
    models: tuple[str, ...]  # the first one is LayaMacro, drawn in the accent colour
    columns: list[Column]
    groups: list[tuple[int, int, str]] = field(default_factory=list)  # (first, end, label)


NUTRITION = Chart(
    name="nutrition",
    title="Calorie error per meal (kcal, lower is better)",
    description=(
        "Mean absolute calorie error in kcal. Nutrition5k test dishes: LayaMacro 48.5, "
        "OmniFood8K 36.8, FoodLMM 48.6. Never-seen dishes: 62.7, 50.5, 66.2. DFood-TJ: "
        "63.1, 168.9, 209.3. NutritionVerse-Real: 169.9, 639.3, 684.7."
    ),
    models=("LayaMacro", "OmniFood8K", "FoodLMM (7B)"),
    groups=[
        (0, 2, "Nutrition5k: all three trained on it"),
        (2, 4, "Only LayaMacro trained on these"),
    ],
    columns=[
        Column("All test dishes", "US cafeteria", (48.5, 36.8, 48.6), 50, (0, 25, 50)),
        Column("Never-seen dishes", "new ingredient mix", (62.7, 50.5, 66.2), 75, (0, 25, 50, 75)),
        Column("DFood-TJ", "hospital trays", (63.1, 168.9, 209.3), 250, (0, 100, 200)),
        Column("NutritionVerse-Real", "phone photos", (169.9, 639.3, 684.7), 750, (0, 250, 500)),
    ],
)

QUESTIONS = Chart(
    name="questions",
    title="Food question accuracy (higher is better)",
    description=(
        "Accuracy on test photos; the line marks chance. Contains X, trained foods: LayaMacro "
        "0.89, Qwen3.5-2B 0.80, Gemma 4 E2B 0.74 (chance 0.56). Contains X, new wording: 0.88, "
        "0.79, 0.68 (chance 0.55). Which of 5 foods: 0.86, 0.62, 0.52 (chance 0.20). Contains "
        "X, food names never trained on: 0.65, 0.83, 0.75 (chance 0.63)."
    ),
    models=("LayaMacro", "Qwen3.5-2B", "Gemma 4 E2B"),
    groups=[(0, 3, "The 243 foods LayaMacro was trained on"), (3, 4, "Other foods")],
    columns=[
        Column("Contains X?", "trained wording", (0.89, 0.80, 0.74), 1, (0, 1), "{:.2f}", 0.56),
        Column("Contains X?", "new wording", (0.88, 0.79, 0.68), 1, (0, 1), "{:.2f}", 0.55),
        Column("Which of 5?", "pick the food", (0.86, 0.62, 0.52), 1, (0, 1), "{:.2f}", 0.20),
        Column("Contains X?", "new food names", (0.65, 0.83, 0.75), 1, (0, 1), "{:.2f}", 0.63),
    ],
)

WIDTH = 860
LABEL_W = 112  # row labels, right-aligned at LABEL_W - 8
COL_GAP = 20
COL_W = (WIDTH - LABEL_W - 3 * COL_GAP) // 4
BAR_MAX = COL_W - 46  # room for the value label at the bar's tip
ROW_H, BAR_H = 28, 16


def text(x: float, y: float, s: str, fill: str, size: float, **attrs: str) -> str:
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}"{extra}>{escape(s)}</text>'
    )


def bar(x: float, y: float, w: float, fill: str) -> str:
    """Square at the baseline, 4px rounded at the data end."""
    r = min(4.0, w, BAR_H / 2)
    return (
        f'<path fill="{fill}" d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} '
        f'v{BAR_H - 2 * r:.1f} a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )


def render(chart: Chart, theme: dict[str, str]) -> str:
    t = theme
    out = [text(0, 18, chart.title, t["ink"], 15, font_weight="600")]
    y = 34
    if chart.groups:
        y += 14
        for first, end, label in chart.groups:
            x0 = LABEL_W + first * (COL_W + COL_GAP)
            x1 = LABEL_W + end * (COL_W + COL_GAP) - COL_GAP
            out.append(text(x0, y, label, t["ink2"], 12))
            out.append(line(x0, y + 6, x1, y + 6, t["rule"]))
        y += 12
    head = y + 18  # column titles
    top = head + 26  # first row
    if any(col.chance is not None for col in chart.columns):
        top += 14  # room for the "chance" labels above the bars
    axis = top + ROW_H * len(chart.models) + 4

    for i, model in enumerate(chart.models):
        cy = top + i * ROW_H + ROW_H / 2
        weight = "600" if i == 0 else "400"
        colour = t["ink"] if i == 0 else t["ink2"]
        out.append(
            text(LABEL_W - 8, cy + 4.5, model, colour, 13, text_anchor="end", font_weight=weight)
        )

    for c, col in enumerate(chart.columns):
        x0 = LABEL_W + c * (COL_W + COL_GAP)
        out.append(text(x0, head, col.title, t["ink"], 13, font_weight="600"))
        out.append(text(x0, head + 15, col.subtitle, t["ink2"], 12))
        for i, v in enumerate(col.values):
            w = BAR_MAX * v / col.scale
            by = top + i * ROW_H + (ROW_H - BAR_H) / 2
            out.append(bar(x0, by, w, t["focus"] if i == 0 else t["other"]))
            weight = "600" if i == 0 else "400"
            colour = t["ink"] if i == 0 else t["ink2"]
            out.append(
                text(
                    x0 + w + 5,
                    by + BAR_H / 2 + 4.5,
                    col.fmt.format(v),
                    colour,
                    12.5,
                    font_weight=weight,
                )
            )
        out.append(line(x0, top - 2, x0, axis, t["rule"]))
        out.append(line(x0, axis, x0 + BAR_MAX, axis, t["rule"]))
        for tick in col.ticks:
            tx = x0 + BAR_MAX * tick / col.scale
            anchor = "start" if tick == 0 else "middle"
            out.append(text(tx, axis + 14, f"{tick:g}", t["muted"], 11.5, text_anchor=anchor))
        if col.chance is not None:
            cx = x0 + BAR_MAX * col.chance / col.scale
            out.append(line(cx, top - 4, cx, axis, t["ink2"]))
            out.append(text(cx, top - 8, "chance", t["muted"], 11.5, text_anchor="middle"))

    height = axis + 24
    body = "\n  ".join(out)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family=\'{FONT}\' role="img">\n'
        f"  <title>{escape(chart.title)}</title>\n  <desc>{escape(chart.description)}</desc>\n"
        f'  <rect width="100%" height="100%" fill="{t["bg"]}"/>\n'
        f"  {body}\n</svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for chart in (NUTRITION, QUESTIONS):
        for mode, theme in THEMES.items():
            path = OUT / f"{chart.name}-{mode}.svg"
            path.write_text(render(chart, theme))
            print(path.relative_to(OUT.parents[1]))


if __name__ == "__main__":
    main()
