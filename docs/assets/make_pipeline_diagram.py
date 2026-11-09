"""Draw ``docs/assets/pipeline.svg``, the process diagram in the README.

Run from the repository root:  ``python docs/assets/make_pipeline_diagram.py``
Text is converted to outlines, so the SVG looks the same on every machine; it is set in Inter when installed.
Every card's text is measured and wrapped to fit, so editing the wording below cannot make text overflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.textpath import TextToPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from foodvision.tokens import INK, MODEL_COLORS, MUTED, SAFFRON  # noqa: E402

OUT = ROOT / "docs" / "assets" / "pipeline.svg"
FONT = "Inter" if "inter" in Path(findfont(FontProperties(family="Inter"))).name.lower() else "DejaVu Sans"
PANEL_FILL, PANEL_EDGE, CARD_EDGE, ARROW = "#F5F7FA", "#E1E6ED", "#D5DBE3", "#A3AEBC"

# (title, subtitle, accent colour, panel width, [(card title, card text), ...])
STAGES = [
    ("Data", "downloaded and audited", MODEL_COLORS["effnetb2"], 236, [
        ("Food-101 subset", "pizza, steak and sushi, split into 450 training and 150 test images"),
        ("Data audit", "integrity, duplicates and train-test leakage"),
        ("Non-food photos", "Flowers-102 and Oxford-IIIT Pet test the not-food check"),
    ]),
    ("Train", "one run on a Colab GPU", MODEL_COLORS["effnetb3"], 292, [
        ("A · Assignment protocol", "9 runs: B2, B3, B4 × 15, 20, 30 epochs, logged to TensorBoard"),
        ("B · Research protocol", "3 seeds on a validation split pick the model; one test run"),
        ("B · Reliability", "temperature scaling and a not-food check, both set on validation data"),
        ("C · Analysis", "robustness, t-SNE feature map, data efficiency and CPU speed"),
    ]),
    ("Results", "artifacts/, committed to Git", MODEL_COLORS["effnetb4"], 262, [
        ("Metrics and predictions", "JSON and CSV files with a hashed manifest"),
        ("Reports", "experiments, model card and data audit, all generated"),
        ("Delta checkpoint", "under 1 MB: the new head and BatchNorm statistics"),
        ("Figures", "summary figure and Grad-CAM gallery"),
    ]),
    ("Ship", "deployed on every push", SAFFRON, 242, [
        ("GitHub Actions", "ruff, mypy, pytest and an app smoke test"),
        ("Streamlit app", "seven pages, running on a free CPU instance"),
        ("Your photo", "label, calibrated confidence, Grad-CAM and a not-food warning"),
    ]),
]
FOOTER = "The app never trains: it only reads the small files in artifacts/, so a free CPU instance can serve it."

WIDTH, MARGIN, GAP = 1200, 24, 40
TITLE_SIZE, TEXT_SIZE, LINE = 16.5, 14.0, 19.0  # sizes in px (the figure uses 72 dpi, so 1 pt = 1 px)
CARD_PAD, CARD_GAP, ACCENT = 12, 12, 5
HEADER, PANEL_PAD = 74, 14

_measure = TextToPath()


def _width(text: str, size: float, weight: str = "normal") -> float:
    prop = FontProperties(family=FONT, size=size, weight=weight)
    return float(_measure.get_text_width_height_descent(text, prop, ismath=False)[0])


def _wrap(text: str, size: float, limit: float) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and _width(trial, size) > limit:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def _card_height(n_lines: int) -> float:
    return CARD_PAD + TITLE_SIZE + 7 + n_lines * LINE + CARD_PAD - 3


def draw() -> Figure:
    inner = {i: w - 2 * PANEL_PAD - ACCENT - 2 * CARD_PAD for i, (_, _, _, w, _) in enumerate(STAGES)}
    wrapped = {i: [_wrap(text, TEXT_SIZE, inner[i]) for _, text in cards] for i, (*_, cards) in enumerate(STAGES)}
    for i, (title, subtitle, _, panel_w, cards) in enumerate(STAGES):  # titles and headers must fit on one line
        for card_title, _ in cards:
            assert _width(card_title, TITLE_SIZE, "semibold") <= inner[i], f"card title too long: {card_title}"
        room = panel_w - 2 * PANEL_PAD - 38
        assert _width(title, 19, "bold") <= room and _width(subtitle, 13) <= room, f"stage header too long: {title}"
    stacks = [sum(_card_height(len(lines)) for lines in wrapped[i]) + CARD_GAP * (len(wrapped[i]) - 1)
              for i in range(len(STAGES))]
    panel_h = HEADER + max(stacks) + PANEL_PAD
    height = MARGIN + panel_h + 54
    assert MARGIN * 2 + sum(s[3] for s in STAGES) + GAP * (len(STAGES) - 1) == WIDTH, "panel widths must fill the page"

    fig = Figure(figsize=(WIDTH / 72, height / 72), dpi=72, facecolor="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)  # y grows downwards, like a page
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), WIDTH - 2, height - 2, boxstyle="round,pad=0,rounding_size=18",
                                fc="white", ec=PANEL_EDGE, lw=1.2))

    x = MARGIN
    for i, (title, subtitle, accent, panel_w, cards) in enumerate(STAGES):
        y = MARGIN
        ax.add_patch(FancyBboxPatch((x, y), panel_w, panel_h, boxstyle="round,pad=0,rounding_size=14",
                                    fc=PANEL_FILL, ec=PANEL_EDGE, lw=1))
        ax.add_patch(Circle((x + PANEL_PAD + 14, y + 30), 14, fc=INK, ec="none"))
        ax.text(x + PANEL_PAD + 14, y + 30.5, str(i + 1), color="white", fontsize=14, fontweight="bold",
                family=FONT, ha="center", va="center")
        ax.text(x + PANEL_PAD + 38, y + 16, title, color=INK, fontsize=19, fontweight="bold", family=FONT, va="top")
        ax.text(x + PANEL_PAD + 38, y + 41, subtitle, color=MUTED, fontsize=13, family=FONT, va="top")
        cy = y + HEADER
        for (card_title, _), lines in zip(cards, wrapped[i]):
            h = _card_height(len(lines))
            cx, cw = x + PANEL_PAD, panel_w - 2 * PANEL_PAD
            box = FancyBboxPatch((cx, cy), cw, h, boxstyle="round,pad=0,rounding_size=9", fc="white",
                                 ec=CARD_EDGE, lw=1)
            ax.add_patch(box)
            bar = ax.add_patch(FancyBboxPatch((cx, cy), ACCENT, h, boxstyle="square,pad=0", fc=accent, ec="none"))
            bar.set_clip_path(box)  # a coloured left edge that follows the card's rounded corners
            tx = cx + ACCENT + CARD_PAD
            ax.text(tx, cy + CARD_PAD - 1, card_title, color=INK, fontsize=TITLE_SIZE, fontweight="semibold",
                    family=FONT, va="top")
            for k, line in enumerate(lines):
                ax.text(tx, cy + CARD_PAD + TITLE_SIZE + 7 + k * LINE, line, color=MUTED, fontsize=TEXT_SIZE,
                        family=FONT, va="top")
            cy += h + CARD_GAP
        if i < len(STAGES) - 1:
            mid = y + HEADER + 0.42 * (panel_h - HEADER)
            ax.add_patch(FancyArrowPatch((x + panel_w + 7, mid), (x + panel_w + GAP - 7, mid), arrowstyle="-|>",
                                         mutation_scale=20, lw=2.2, color=ARROW, shrinkA=0, shrinkB=0))
        x += panel_w + GAP
    ax.text(MARGIN + 2, MARGIN + panel_h + 30, FOOTER, color=MUTED, fontsize=13.5, family=FONT, va="center")
    return fig


def main() -> None:
    matplotlib.rcParams["svg.fonttype"] = "path"  # text as outlines: identical rendering everywhere
    matplotlib.rcParams["svg.hashsalt"] = "foodvision-pipeline"  # stable ids, so re-running gives the same file
    fig = draw()
    fig.savefig(OUT, format="svg", metadata={"Date": None})
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB, font: {FONT})")


if __name__ == "__main__":
    main()
