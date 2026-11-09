"""Plotly figure factories shared by the Streamlit app.

Design tokens: model size is an ordinal variable, so B2 -> B4 use a light-to-dark blue ramp (darker = bigger
network). Classes use food colours (tomato, char, nori). Saffron marks the selected model everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .analysis import efficiency_summary
from .tokens import (
    CLASS_COLORS,
    CORRUPTION_COLORS,
    GRID,
    INK,
    METRIC_COLORS,
    MODEL_COLORS,
    MODEL_LABELS,
    MUTED,
    NON_FOOD,
    OOD_COLORS,
    SAFFRON,
)

FONT = "IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif"


def _layout(fig: go.Figure, height: int = 380, **overrides: Any) -> go.Figure:
    layout: dict[str, Any] = {
        "template": "simple_white",
        "height": height,
        "font": {"family": FONT, "size": 13, "color": INK},
        "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": {"text": ""}},
        "hoverlabel": {"font": {"family": FONT}},
    }
    fig.update_layout(**{**layout, **overrides})
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="")
    fig.update_yaxes(gridcolor=GRID, showgrid=True, linecolor=GRID, ticks="")
    return fig


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def scenario_heatmap(summary: pd.DataFrame, value: str, title: str, fmt: str, reverse: bool = False) -> go.Figure:
    """Model x epoch grid for one metric; the selected cell is outlined in saffron."""
    models = [m for m in MODEL_LABELS if m in set(summary["model"])]
    epochs = sorted(summary["epochs"].unique())
    grid = summary.pivot(index="model", columns="epochs", values=value).reindex(index=models, columns=epochs)
    text = [[fmt.format(v) for v in row] for row in grid.to_numpy()]
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[f"{e} epochs" for e in epochs],
            y=[label(m) for m in models],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 15},
            colorscale="Blues_r" if reverse else "Blues",
            showscale=False,
            hovertemplate="%{y}, %{x}: %{text}<extra></extra>",
        )
    )
    if "selected" in summary:
        best = summary[summary["selected"].astype(bool)]
        for _, row in best.iterrows():
            i, j = models.index(row["model"]), epochs.index(row["epochs"])
            fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line={"color": SAFFRON, "width": 4})
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, height=260, title={"text": title, "x": 0, "font": {"size": 15}})


def interval_plot(frame: pd.DataFrame, value: str, low: str, high: str, highlight: str | None = None) -> go.Figure:
    """Point estimates with asymmetric confidence intervals, one row per scenario, coloured by model."""
    data = frame.reset_index(drop=True)
    names = [f"{label(m)}, {e} ep" for m, e in zip(data["model"], data["epochs"])]
    fig = go.Figure()
    if highlight is not None:
        best = data[[n == highlight for n in names]]
        if not best.empty:
            fig.add_vrect(x0=float(best[low].iloc[0]) * 100, x1=float(best[high].iloc[0]) * 100,
                          fillcolor=SAFFRON, opacity=0.12, line_width=0)
    for model, group in data.groupby("model", sort=False):
        idx = group.index
        fig.add_trace(
            go.Scatter(
                x=group[value] * 100,
                y=[names[i] for i in idx],
                mode="markers",
                name=label(str(model)),
                marker={"color": MODEL_COLORS.get(str(model), MUTED), "size": 11,
                        "line": {"color": [SAFFRON if names[i] == highlight else "white" for i in idx], "width": 2.5}},
                error_x={"type": "data", "symmetric": False, "array": (group[high] - group[value]) * 100,
                         "arrayminus": (group[value] - group[low]) * 100, "color": MODEL_COLORS.get(str(model), MUTED),
                         "thickness": 2, "width": 4},
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="accuracy (%)", showgrid=True, gridcolor=GRID)
    return _layout(fig, height=60 + 34 * len(data))


def pareto_plot(frame: pd.DataFrame, x: str, y: str, x_title: str, highlight: str | None = None) -> go.Figure:
    """Accuracy against a cost; dashed line = configurations no other configuration beats on both axes."""
    data = frame.sort_values(x).reset_index(drop=True)
    frontier, best = [], -np.inf
    for _, row in data.iterrows():
        if row[y] > best:
            frontier.append(row)
            best = row[y]
    fig = go.Figure()
    if len(frontier) > 1:
        fig.add_trace(go.Scatter(x=[r[x] for r in frontier], y=[r[y] * 100 for r in frontier], mode="lines",
                                 line={"color": MUTED, "dash": "dash", "width": 1.5}, name="Pareto frontier",
                                 hoverinfo="skip"))
    for model, group in data.groupby("model", sort=False):
        names = [f"{label(str(model))}, {e} ep" for e in group["epochs"]]
        fig.add_trace(
            go.Scatter(
                x=group[x], y=group[y] * 100, mode="markers+text", name=label(str(model)),
                text=[f"{e}" for e in group["epochs"]], textposition="top center", textfont={"size": 11, "color": MUTED},
                marker={"size": 13, "color": MODEL_COLORS.get(str(model), MUTED),
                        "line": {"color": [SAFFRON if n == highlight else "white" for n in names], "width": 3}},
                customdata=names, hovertemplate="%{customdata}<br>%{y:.1f}% at %{x:.1f}<extra></extra>",
            )
        )
    fig.update_xaxes(title=x_title, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="test accuracy (%)")
    return _layout(fig, height=400)


def learning_curves(histories: dict[str, Any], keys: Sequence[str], metric: str, splits: Sequence[str]) -> go.Figure:
    """Per-epoch curves; solid = evaluation split, dotted = training split."""
    fig = go.Figure()
    for key in keys:
        model = key.split("_")[0]
        history = histories[key]
        for split in splits:
            values = np.asarray(history[split][metric], dtype=float)
            values = values * 100 if metric == "acc" else values
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(values) + 1)), y=values, mode="lines",
                    name=f"{key.replace('_', ' ')} {split}",
                    line={"color": MODEL_COLORS.get(model, MUTED), "width": 2.4 if split != "train" else 1.6,
                          "dash": "dot" if split == "train" else "solid"},
                )
            )
    fig.update_xaxes(title="epoch", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="accuracy (%)" if metric == "acc" else "cross-entropy loss")
    return _layout(fig, height=420, legend={"orientation": "v", "x": 1.01, "y": 1})


def confusion_heatmap(cm: Sequence[Sequence[int]], classes: Sequence[str], normalized: bool) -> go.Figure:
    counts = np.asarray(cm, dtype=float)
    rates = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
    z = rates if normalized else counts
    text = [[f"{rates[i, j]:.0%}" if normalized else f"{int(counts[i, j])}" for j in range(len(classes))]
            for i in range(len(classes))]
    fig = go.Figure(
        go.Heatmap(z=z, x=list(classes), y=list(classes), text=text, texttemplate="%{text}", textfont={"size": 16},
                   colorscale="Blues", showscale=False,
                   hovertemplate="true %{y}, predicted %{x}: %{text}<extra></extra>")
    )
    fig.update_xaxes(title="predicted class", side="bottom")
    fig.update_yaxes(title="true class", autorange="reversed", showgrid=False)
    return _layout(fig, height=360)


def per_class_bars(per_class: Sequence[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(per_class)
    fig = go.Figure()
    for metric in ("precision", "recall", "f1"):
        fig.add_trace(go.Bar(x=frame["class"], y=frame[metric] * 100, name=metric.upper() if metric == "f1" else metric,
                             marker_color=METRIC_COLORS[metric], text=[f"{v:.1f}" for v in frame[metric] * 100],
                             textposition="outside"))
    fig.update_yaxes(title="%", range=[0, 108])
    return _layout(fig, height=360, barmode="group")


def reliability_diagram(before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]) -> go.Figure:
    """Accuracy against confidence per bin; the diagonal is perfect calibration. Marker area = bin size."""
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": GRID, "width": 2},
                               name="perfect calibration", hoverinfo="skip"))
    for bins, name, color in ((before, "softmax", "#9DB7DA"), (after, "temperature-scaled", "#1F3A68")):
        used = [b for b in bins if b["count"]]
        fig.add_trace(
            go.Scatter(
                x=[b["confidence"] for b in used], y=[b["accuracy"] for b in used], mode="lines+markers", name=name,
                line={"color": color, "width": 2},
                marker={"size": [6 + 2.5 * np.sqrt(b["count"]) for b in used], "color": color},
                customdata=[b["count"] for b in used],
                hovertemplate="confidence %{x:.2f}<br>accuracy %{y:.2f}<br>%{customdata} images<extra></extra>",
            )
        )
    fig.update_xaxes(title="mean confidence in bin", range=[0, 1.02], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="accuracy in bin", range=[0, 1.05])
    return _layout(fig, height=420)


def confidence_histogram(predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for correct, name, color in ((True, "correct", "#4F75B3"), (False, "incorrect", "#C8402F")):
        subset = predictions[predictions["correct"] == correct]
        fig.add_trace(go.Histogram(x=subset["confidence"], name=f"{name} ({len(subset)})", marker_color=color,
                                   xbins={"start": 0.3, "end": 1.0, "size": 0.035}, opacity=0.85))
    fig.update_xaxes(title="confidence (calibrated top-class probability)")
    fig.update_yaxes(title="test images")
    return _layout(fig, height=340, barmode="overlay")


def abstention_plot(curve: Sequence[dict[str, Any]], threshold: float) -> go.Figure:
    frame = pd.DataFrame(curve)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["threshold"], y=frame["coverage"] * 100, name="coverage (answered)",
                             line={"color": "#9DB7DA", "width": 2.5}))
    fig.add_trace(go.Scatter(x=frame["threshold"], y=frame["accuracy"] * 100, name="accuracy on answered",
                             line={"color": "#1F3A68", "width": 2.5}))
    fig.add_vline(x=threshold, line={"color": SAFFRON, "width": 2})
    fig.update_xaxes(title="minimum confidence to answer", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="%", range=[0, 102])
    return _layout(fig, height=360)


def ood_histogram(frame: pd.DataFrame, score: str, threshold: float, score_label: str) -> go.Figure:
    """Score distributions of food and non-food test photos; photos left of the threshold are rejected."""
    values = frame[score].to_numpy(dtype=float)
    edges = np.histogram_bin_edges(values, bins=30)
    size = float(edges[1] - edges[0]) or 1.0
    fig = go.Figure()
    for group, name in (("non-food", "non-food photos"), ("food", "pizza, steak and sushi")):
        fig.add_trace(
            go.Histogram(
                x=frame.loc[frame["group"] == group, score], name=name, opacity=0.72,
                marker={"color": OOD_COLORS[group]}, xbins={"start": float(edges[0]), "end": float(edges[-1]), "size": size},
            )
        )
    fig.add_vline(x=threshold, line={"color": SAFFRON, "width": 2.5},
                  annotation_text="threshold", annotation_position="top")
    fig.update_xaxes(title=f"{score_label.lower()} (higher = more like the training photos)")
    fig.update_yaxes(title="test photos")
    return _layout(fig, height=340, barmode="overlay")


def probability_bars(classes: Sequence[str], probabilities: Sequence[float]) -> go.Figure:
    order = np.argsort(probabilities)
    fig = go.Figure(
        go.Bar(
            x=[probabilities[i] * 100 for i in order], y=[classes[i] for i in order], orientation="h",
            marker_color=[CLASS_COLORS.get(classes[i], MUTED) for i in order],
            text=[f"{probabilities[i]:.1%}" for i in order], textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 115], title="probability (%)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _layout(fig, height=60 + 48 * len(classes))


def latency_bars(benchmark: Sequence[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(benchmark)
    fig = go.Figure()
    for device, group in frame.groupby("device"):
        fig.add_trace(
            go.Bar(x=[label(m) for m in group["model"]], y=group["median_ms"], name=str(device).upper(),
                   marker_color="#1F3A68" if device == "cuda" else "#9DB7DA",
                   text=[f"{v:.0f} ms" for v in group["median_ms"]], textposition="outside")
        )
    fig.update_yaxes(title="median latency per image (ms)")
    return _layout(fig, height=360, barmode="group")


def class_counts(counts: dict[str, dict[str, int]]) -> go.Figure:
    frame = pd.DataFrame(counts).fillna(0)
    fig = go.Figure()
    for split, opacity in (("train", 1.0), ("test", 0.55)):
        if split in frame:
            fig.add_trace(go.Bar(x=frame.index, y=frame[split], name=split, opacity=opacity,
                                 marker_color=[CLASS_COLORS.get(c, MUTED) for c in frame.index],
                                 text=frame[split].astype(int), textposition="outside"))
    fig.update_yaxes(title="images")
    return _layout(fig, height=320, barmode="group")


def size_scatter(images: pd.DataFrame) -> go.Figure:
    data = images.dropna(subset=["width", "height"])
    fig = go.Figure()
    for cls, group in data.groupby("label"):
        fig.add_trace(go.Scatter(x=group["width"], y=group["height"], mode="markers", name=str(cls),
                                 marker={"color": CLASS_COLORS.get(str(cls), MUTED), "size": 7, "opacity": 0.6}))
    fig.update_xaxes(title="width (px)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="height (px)")
    return _layout(fig, height=340)


def robustness_plot(frame: pd.DataFrame, value: str = "accuracy") -> go.Figure:
    """A metric per corruption as its strength rises; the clean score is the shared starting point."""
    clean = frame.loc[frame["corruption"] == "none"].iloc[0]
    fig = go.Figure()
    for kind, color in CORRUPTION_COLORS.items():
        part = frame[frame["corruption"] == kind].sort_values("severity")
        if part.empty:
            continue
        y = 100 * np.array([clean[value], *part[value]], dtype=float)
        if value == "accuracy":
            lows = 100 * np.array([clean["ci_low"], *part["ci_low"]], dtype=float)
            highs = 100 * np.array([clean["ci_high"], *part["ci_high"]], dtype=float)
            hover = "%{y:.1f}% (95% CI %{customdata[0]:.1f} to %{customdata[1]:.1f})<extra>%{fullData.name}</extra>"
            custom = np.c_[lows, highs]
        else:
            hover, custom = "%{y:.1f}%<extra>%{fullData.name}</extra>", None
        fig.add_trace(go.Scatter(x=[0, *part["severity"]], y=y, name=str(part["label"].iloc[0]), mode="lines+markers",
                                 line={"color": color, "width": 2.5}, marker={"size": 7}, customdata=custom,
                                 hovertemplate=hover))
    fig.update_xaxes(tickvals=[0, 1, 2, 3], ticktext=["clean", "mild", "medium", "strong"], title="corruption strength")
    fig.update_yaxes(title="test accuracy (%)" if value == "accuracy" else "food photos accepted (%)")
    return _layout(fig, height=360)


def feature_map_plot(frame: pd.DataFrame) -> go.Figure:
    """t-SNE map of test photos: food coloured by true class, non-food in grey, misclassified photos circled."""
    fig = go.Figure()
    non_food = frame[frame["group"] == "non-food"]
    for label_, symbol in zip(sorted(non_food["label"].unique()), ("diamond", "square", "triangle-up", "cross")):
        part = non_food[non_food["label"] == label_]
        fig.add_trace(go.Scatter(x=part["x"], y=part["y"], mode="markers", name=f"non-food: {label_}",
                                 marker={"color": NON_FOOD, "size": 6, "symbol": symbol, "opacity": 0.75},
                                 text=part["prediction"], hovertemplate="looks most like %{text}<extra>%{fullData.name}</extra>"))
    food = frame[frame["group"] == "food"]
    for cls in sorted(food["label"].unique()):
        part = food[food["label"] == cls]
        fig.add_trace(go.Scatter(x=part["x"], y=part["y"], mode="markers", name=str(cls),
                                 marker={"color": CLASS_COLORS.get(str(cls), MUTED), "size": 8},
                                 text=part["prediction"], hovertemplate="predicted %{text}<extra>%{fullData.name}</extra>"))
    wrong = frame[frame["outcome"] == "misclassified"]
    if not wrong.empty:
        fig.add_trace(go.Scatter(x=wrong["x"], y=wrong["y"], mode="markers", name="misclassified",
                                 marker={"symbol": "circle-open", "size": 16, "color": INK, "line": {"width": 2}},
                                 text=wrong["label"] + " as " + wrong["prediction"],
                                 hovertemplate="%{text}<extra>misclassified</extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, height=480)


def data_efficiency_plot(frame: pd.DataFrame) -> go.Figure:
    """Validation accuracy against the number of training photos: each seed as a dot, the mean as a line."""
    summary = efficiency_summary(frame)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["n_train"], y=100 * frame["val_acc"], mode="markers", name="one seed",
                             marker={"color": MODEL_COLORS["effnetb2"], "size": 9}, text=frame["seed"],
                             hovertemplate="%{x} photos, seed %{text}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=summary["n_train"], y=100 * summary["val_acc_mean"], mode="lines+markers",
                             name="mean over seeds", line={"color": INK, "width": 2.5}, marker={"size": 7}))
    ticks = [float(v) for v in summary["n_train"]]
    fig.update_xaxes(type="log", title="training photos (log scale)", tickvals=ticks,
                     ticktext=[str(int(round(v))) for v in ticks])
    fig.update_yaxes(title="validation accuracy (%)")
    return _layout(fig, height=360)
