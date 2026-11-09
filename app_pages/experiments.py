"""Experiment lab: the nine assignment runs, validation-based selection, cost, and paired tests."""

import pandas as pd
import streamlit as st

from foodvision import ui
from foodvision.artifacts import Files
from foodvision.viz import interval_plot, label, pareto_plot, scenario_heatmap

ui.page("Experiment lab", "Nine configurations (three backbones, three epoch budgets) under two protocols. "
        "The assignment protocol selects on the test set; the research protocol selects on validation data "
        "and keeps the test set for one final measurement.")

tab_a, tab_b, tab_cost, tab_tests = st.tabs(
    ["Assignment protocol", "Research protocol", "Accuracy and cost", "Significance tests"]
)

with tab_a:
    if ui.available(Files.ASSIGNMENT_SUMMARY):
        summary = ui.read_csv(Files.ASSIGNMENT_SUMMARY)
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(scenario_heatmap(summary.assign(v=summary["test_acc"] * 100), "v", "Test accuracy (%)",
                                         "{:.1f}"))
        c2.plotly_chart(scenario_heatmap(summary, "test_loss", "Test loss", "{:.3f}", reverse=True))
        c3.plotly_chart(scenario_heatmap(summary, "train_seconds", "Training time (s)", "{:.0f}", reverse=True))
        best = summary[summary["selected"].astype(bool)].iloc[0]
        highlight = f"{label(best['model'])}, {best['epochs']} ep"
        st.markdown("**Test accuracy with 95% Wilson intervals**")
        st.plotly_chart(interval_plot(summary, "test_acc", "test_ci_low", "test_ci_high", highlight))
        ui.note("Each run logged train and test loss and accuracy to TensorBoard every epoch (see Training curves). "
                "The saffron outline marks the run with the highest final test accuracy, the assignment's selection "
                "rule. Because the test set chose it, its score overstates how a new sample would fare.")
    else:
        st.info("Assignment results are missing; run the `assignment` stage of the pipeline.")

with tab_b:
    if not ui.missing_notice(Files.SELECTION, Files.VALIDATION):
        selection = ui.read_json(Files.SELECTION)
        table = pd.DataFrame(selection["table"])
        split = selection["split"]
        st.markdown(
            f"The {split['train'] + split['validation']} training images were split, stratified by class, into "
            f"{split['train']} for training and {split['validation']} for validation; the {split['test']} test images "
            f"were not used. Each configuration was trained with seeds {selection['seeds']}."
        )
        view = table.assign(
            Model=table["model"].map(label),
            **{"Mean validation accuracy": table["val_acc_mean"].map(ui.pct),
               "SD across seeds": table["val_acc_sd"].map(lambda v: "n/a" if pd.isna(v) else f"{100 * v:.1f} pts"),
               "Mean validation loss": table["val_loss_mean"].map("{:.3f}".format)},
        )[["Model", "epochs", "Mean validation accuracy", "SD across seeds", "Mean validation loss"]]
        st.dataframe(view.rename(columns={"epochs": "Epochs"}), hide_index=True)
        st.success(f"Selected: {selection['label']} at {selection['epochs']} epochs. Rule: {selection['rule']}.")
        if ui.available(Files.TEST_REPORT):
            report = ui.read_json(Files.TEST_REPORT)
            repeat = ", ".join(f"seed {r['seed']}: {ui.pct(r['test_acc'])}" for r in report["repeatability"])
            ui.note(f"Repeatability of the selected configuration on the test set: {repeat}.")

with tab_cost:
    if not ui.missing_notice(Files.ASSIGNMENT_SUMMARY):
        summary = ui.read_csv(Files.ASSIGNMENT_SUMMARY)
        options = {"Training time (s)": "train_seconds", "Parameters (millions)": "params_m"}
        if ui.available(Files.BENCHMARK):
            cpu = {b["model"]: b["median_ms"] for b in ui.read_json(Files.BENCHMARK) if b["device"] == "cpu"}
            summary["cpu_ms"] = summary["model"].map(cpu)
            options["CPU latency per image (ms)"] = "cpu_ms"
        summary["params_m"] = summary["total_params"] / 1e6
        choice = st.radio("Cost axis", list(options), horizontal=True)
        best = summary[summary["selected"].astype(bool)].iloc[0]
        highlight = f"{label(best['model'])}, {best['epochs']} ep"
        st.plotly_chart(pareto_plot(summary, options[choice], "test_acc", choice, highlight))
        ui.note("Assignment-protocol accuracies. Labels give the epoch budget. The dashed line joins configurations that "
                "no other configuration beats on both accuracy and the chosen cost.")

with tab_tests:
    if not ui.missing_notice(Files.TEST_REPORT):
        report = ui.read_json(Files.TEST_REPORT)
        rows = [
            {
                "Comparison": f"{label(c['model_a'])} vs {label(c['model_b'])} ({c['epochs']} epochs)",
                "Accuracy": f"{ui.pct(c['acc_a'])} vs {ui.pct(c['acc_b'])}",
                "Difference, 95% paired bootstrap CI": f"{ui.pct(c['difference']['estimate'])} "
                f"({ui.pct(c['difference']['low'])} to {ui.pct(c['difference']['high'])})",
                "Images only A got right": c["a_only_correct"],
                "Images only B got right": c["b_only_correct"],
                "Exact McNemar p": f"{c['p_value']:.3f}",
                "Holm-adjusted p": f"{c['p_holm']:.3f}",
            }
            for c in report["comparisons"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        ui.note("These tests were run after model selection, so they are exploratory. McNemar's test uses only the images "
                "on which the two models disagree. A p-value above 0.05 means the data cannot distinguish the models; it "
                "does not show that they are equivalent.")
