"""Training curves: the per-epoch scalars logged to TensorBoard, plus the BatchNorm statistics ablation."""

import streamlit as st

from foodvision import ui
from foodvision.artifacts import Files
from foodvision.viz import MODEL_LABELS, data_efficiency_plot, learning_curves

ui.page("Training curves", "Every epoch of every run, as logged to TensorBoard. Dotted lines are the training "
        "split; solid lines are the evaluation split (test for the assignment runs, validation for the research runs).")

protocol = st.radio("Protocol", ["Assignment (train and test)", "Research (train and validation)"], horizontal=True)
research = protocol.startswith("Research")
name = Files.RESEARCH_HISTORY if research else Files.ASSIGNMENT_HISTORY
ui.require(name)
histories = ui.read_json(name)
keys = sorted(histories)
models = st.multiselect("Backbones", list(MODEL_LABELS), default=list(MODEL_LABELS), format_func=MODEL_LABELS.get)
if research:
    seeds = sorted({k.split("seed")[-1] for k in keys})
    seed = st.selectbox("Seed", seeds)
    chosen = [k for k in keys if k.split("_")[0] in models and k.endswith(f"seed{seed}")]
    splits = ["train", "val"]
else:
    budgets = sorted({int(k.split("_")[1]) for k in keys})
    budget = st.selectbox("Epoch budget", budgets, index=len(budgets) - 1)
    chosen = [k for k in keys if k.split("_")[0] in models and k.endswith(f"_{budget}")]
    splits = ["train", "test"]
metric = st.radio("Metric", ["Accuracy", "Loss"], horizontal=True)
if chosen:
    st.plotly_chart(learning_curves(histories, chosen, "acc" if metric == "Accuracy" else "loss", splits))
else:
    st.warning("Select at least one backbone.")
ui.note("Every run is seeded, so runs of one backbone with different epoch budgets follow the same trajectory over "
        "their shared epochs: the budgets are stopping points on one curve. Training accuracy can sit below "
        "evaluation accuracy because dropout and stochastic depth are active while training metrics are collected.")

st.subheader("Do frozen layers really stay frozen?")
st.markdown(
    "Setting `requires_grad=False` stops weight updates, but BatchNorm layers still update their running mean and "
    "variance whenever the network is in training mode. The ablation below trains each backbone a second time with "
    "the backbone kept in evaluation mode, so its BatchNorm statistics stay at their ImageNet values."
)
if ui.available(Files.BN_ABLATION):
    bn = ui.read_csv(Files.BN_ABLATION)
    pivot = bn.pivot_table(index=["model", "epochs"], columns="bn_statistics", values="val_acc").reset_index()
    pivot["model"] = pivot["model"].map(MODEL_LABELS)
    for column in [c for c in pivot.columns if c not in ("model", "epochs")]:
        pivot[column] = pivot[column].map(ui.pct)
    st.dataframe(pivot.rename(columns={"model": "Backbone", "epochs": "Epochs", "adapting": "Statistics adapt",
                                       "frozen": "Statistics frozen"}), hide_index=True)
    ui.note("Validation accuracy, one seed. Differences of one or two images are within noise; this is an exploratory check.")
else:
    st.info("The ablation was not run (research.bn_ablation is false or the research stage is missing).")

st.subheader("How much training data does it need?")
if not ui.missing_notice(Files.DATA_EFFICIENCY):
    st.plotly_chart(data_efficiency_plot(ui.read_csv(Files.DATA_EFFICIENCY)))
    ui.note("The selected configuration retrained on nested, class-balanced subsets of the training photos with the same "
            "epochs and learning rate, then scored on the validation photos. The full-data points are the research runs. "
            "Smaller subsets also mean fewer training steps, which is part of what this curve shows.")
