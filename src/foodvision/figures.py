"""A static, shareable summary figure (PNG, 16:9) generated from the artifacts, for the README and social posts.

Uses matplotlib's object-oriented API only, so nothing is shown or changed in a notebook's plotting state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import CORRUPTIONS, efficiency_summary
from .artifacts import ArtifactStore, Files
from .tokens import CORRUPTION_COLORS, GRID, INK, MODEL_COLORS, MUTED, SAFFRON

PANEL_TITLE = "panel-title"


def _fit_to_page(fig: Any, grid: Any, margin: float = 0.01) -> None:
    """Keep every label on the page whatever fonts the machine has (Colab draws text wider than other setups).

    Measures the drawn text: moves the panels right if the first panel's tick labels would run off the left edge,
    then gives all panel titles one size small enough that none runs into the next panel or off the page.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    renderer = FigureCanvasAgg(fig).get_renderer()
    width = fig.bbox.width
    pad = margin * width
    panels = fig.axes
    overflow = pad - panels[0].get_tightbbox(renderer).x0
    if overflow > 0:
        grid.update(left=grid.left + overflow / width)
        for ax in panels:  # update() only moves the subplots of pyplot-managed figures
            ax.set_position(ax.get_subplotspec().get_position(fig))
    titles = [ax.findobj(lambda a: a.get_gid() == PANEL_TITLE)[0] for ax in panels]
    limits = [ax.get_position().x0 * width - pad for ax in panels[1:]] + [width - pad]
    size = min(t.get_fontsize() for t in titles)
    while size > 8 and any(t.get_window_extent(renderer).x1 > limit for t, limit in zip(titles, limits)):
        size -= 0.5
        for t in titles:
            t.set_fontsize(size)


def _read(store: ArtifactStore, name: str) -> Any:
    return store.read_json(name) if store.exists(name) else None


def _style(ax: Any, title: str, grid_axis: str = "both") -> None:
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=10, gid=PANEL_TITLE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if grid_axis != "none":
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _missing(ax: Any, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=10, gid=PANEL_TITLE)
    ax.text(0.5, 0.5, "not generated in this run", ha="center", va="center", color=MUTED, fontsize=10,
            transform=ax.transAxes)


def _validation_panel(ax: Any, sel: dict[str, Any]) -> None:
    table = pd.DataFrame(sel["table"]).sort_values(["model", "epochs"]).reset_index(drop=True)
    y = np.arange(len(table))[::-1]
    selected = ((table["model"] == sel["model"]) & (table["epochs"] == sel["epochs"])).to_numpy()
    mean = 100 * table["val_acc_mean"].to_numpy()
    spread = 100 * pd.to_numeric(table["val_acc_sd"], errors="coerce").fillna(0.0).to_numpy()
    ax.errorbar(mean, y, xerr=spread, fmt="none", ecolor="#C9CED6", elinewidth=2, zorder=1)
    ax.scatter(mean, y, s=np.where(selected, 95, 55), zorder=2,
               c=[SAFFRON if s else MODEL_COLORS.get(m, MUTED) for m, s in zip(table["model"], selected)],
               edgecolors=[INK if s else "none" for s in selected], linewidths=1.2)
    ax.set_yticks(y, [f"B{m[-1]} · {e} epochs" for m, e in zip(table["model"], table["epochs"])])
    ax.set_xlabel("validation accuracy, mean ± SD over seeds (%)")
    _style(ax, f"{len(table)} configurations compared", grid_axis="x")


def _confusion_panel(ax: Any, report: dict[str, Any]) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cm = np.asarray(report["confusion_matrix"])
    shares = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(shares, cmap=LinearSegmentedColormap.from_list("fv", ["#FFFFFF", MODEL_COLORS["effnetb4"]]),
              vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if shares[i, j] > 0.5 else INK, fontweight="bold" if i == j else "normal")
    classes = report["classes"]
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    _style(ax, f"Test set: {report['n_correct']} of {report['n_test']} correct", grid_axis="none")
    ax.set_anchor("N")  # the square matrix is shorter than its column: keep its title level with the others
    for spine in ax.spines.values():
        spine.set_visible(False)


def _robustness_panel(ax: Any, store: ArtifactStore) -> None:
    title = "Accuracy as photos get worse"
    if not store.exists(Files.ROBUSTNESS):
        return _missing(ax, title)
    frame = store.read_csv(Files.ROBUSTNESS)
    clean = float(frame.loc[frame["corruption"] == "none", "accuracy"].iloc[0])
    for kind, spec in CORRUPTIONS.items():
        part = frame[frame["corruption"] == kind].sort_values("severity")
        if part.empty:
            continue
        ax.plot([0, *part["severity"]], 100 * np.array([clean, *part["accuracy"]]), marker="o", markersize=4,
                linewidth=2, color=CORRUPTION_COLORS[kind], label=spec["label"])
    ax.set_xticks([0, 1, 2, 3], ["clean", "mild", "medium", "strong"])
    ax.set_ylabel("test accuracy (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="best", handlelength=1.5)
    _style(ax, title, grid_axis="y")


def _efficiency_panel(ax: Any, store: ArtifactStore) -> None:
    title = "Accuracy vs. training photos"
    if not store.exists(Files.DATA_EFFICIENCY):
        return _missing(ax, title)
    frame = store.read_csv(Files.DATA_EFFICIENCY)
    summary = efficiency_summary(frame)
    ax.scatter(frame["n_train"], 100 * frame["val_acc"], s=20, color=MODEL_COLORS["effnetb2"], label="one seed",
               zorder=2)
    ax.plot(summary["n_train"], 100 * summary["val_acc_mean"], marker="o", color=INK, linewidth=2, label="mean",
            zorder=3)
    ax.set_xscale("log")
    ticks = [int(round(v)) for v in summary["n_train"]]
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("training photos (log scale)")
    ax.set_ylabel("validation accuracy (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax, title, grid_axis="y")


def summary_figure(store: ArtifactStore) -> Path | None:
    """Write ``figures/summary.png``; returns None while the research results are missing."""
    if store.missing(Files.TEST_REPORT, Files.SELECTION, Files.MODEL_META):
        return None
    from matplotlib.figure import Figure

    report = store.read_json(Files.TEST_REPORT)
    sel = store.read_json(Files.SELECTION)
    meta = store.read_json(Files.MODEL_META)
    ood = _read(store, Files.OOD)
    bench = _read(store, Files.BENCHMARK) or []

    fig = Figure(figsize=(40 / 3, 7.5), dpi=150, facecolor="white")  # exactly 2000 x 1125 px (16:9)
    fig.text(0.04, 0.935, "FoodVision Lab", fontsize=27, fontweight="bold", color=INK, family="serif", va="top")
    fig.text(0.04, 0.855, f"{meta['label']} transfer learning for pizza, steak and sushi photos", fontsize=12.5,
             color=MUTED, va="top")
    low, high = report["accuracy_wilson"]
    tiles = [(f"{100 * report['accuracy']:.1f}%", f"test accuracy\n95% CI {100 * low:.1f} to {100 * high:.1f}%")]
    if ood and ood.get("status") == "ok":
        tiles.append((f"{100 * ood['test']['rejected_non_food']:.0f}%", "of non-food photos\nflagged"))
    tiles.append((f"{meta['file']['bytes'] / 1024:.0f} KB", "model file\n(delta checkpoint)"))
    cpu = [b for b in bench if b["model"] == meta["arch"] and b["device"] == "cpu"]
    if cpu:
        tiles.append((f"{cpu[0]['median_ms']:.0f} ms", "per photo\non a CPU"))
    for i, (value, caption) in enumerate(tiles):
        x = 0.985 - (len(tiles) - i) * 0.125
        fig.text(x, 0.94, value, fontsize=21, fontweight="bold", color=INK, va="top")
        fig.text(x, 0.865, caption, fontsize=9.5, color=MUTED, va="top", linespacing=1.35)

    grid = fig.add_gridspec(1, 4, left=0.075, right=0.985, bottom=0.14, top=0.70, wspace=0.45)
    _validation_panel(fig.add_subplot(grid[0, 0]), sel)
    _confusion_panel(fig.add_subplot(grid[0, 1]), report)
    _robustness_panel(fig.add_subplot(grid[0, 2]), store)
    _efficiency_panel(fig.add_subplot(grid[0, 3]), store)
    fig.text(0.04, 0.035, "Model chosen on validation data over three seeds; the test set was used once. "
             "Every number is generated by the pipeline.", fontsize=9, color=MUTED)

    _fit_to_page(fig, grid)
    path = store.path(Files.SUMMARY_FIGURE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    return path
