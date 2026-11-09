"""Overview: what was built, what the selected model looks at, and the headline findings."""

import streamlit as st

from foodvision import ui
from foodvision.artifacts import Files
from foodvision.viz import label

st.markdown(ui.CSS, unsafe_allow_html=True)
left, right = st.columns([1.25, 1], gap="large")
with left:
    st.markdown("<div class='fv-hero'><h1>Which dish is it, and what did the model look at?</h1></div>",
                unsafe_allow_html=True)
    st.markdown(
        "<p class='fv-lede'>FoodVision Lab compares EfficientNet-B2, B3 and B4 as frozen ImageNet feature extractors "
        "on pizza, steak and sushi photos from Food-101. It reproduces the course assignment, then selects a model on "
        "held-out validation data, tests it once, and checks its confidence intervals, calibration and visual "
        "evidence.</p>",
        unsafe_allow_html=True,
    )

if not ui.available(Files.TEST_REPORT, Files.SELECTION):
    st.info(
        "No results yet. Run `notebooks/run_pipeline_colab.ipynb` on a Colab GPU (about 15 minutes), download "
        "`artifacts.zip`, unzip it into the repository's `artifacts/` folder and commit it. Every page then fills in."
    )
    st.stop()

report = ui.read_json(Files.TEST_REPORT)
selection = ui.read_json(Files.SELECTION)
low, high = report["accuracy_wilson"]

with right:
    if ui.available(Files.GALLERY):
        entries = ui.read_json(Files.GALLERY)["entries"]
        example = next((e for e in entries if e["correct"] and e["kind"] == "most confident"), entries[0])
        view, cam = st.columns(2)
        view.image(str(ui.store().path(f"gallery/{example['view']}")), caption="What the model sees")
        cam.image(str(ui.store().path(f"gallery/{example['cam']}")), caption="Regions that raised the score")
        ui.note(f"{report['label']} labels this {example['label']} photo as {example['prediction']} with "
                f"{example['confidence']:.0%} confidence. Grad-CAM shows correlation with the decision, not its cause.")

st.subheader("Findings")
findings = [
    f"<b>{report['label']} trained for {report['epochs']} epochs</b> was selected on validation data and classified "
    f"{report['n_correct']} of {report['n_test']} untouched test images correctly ({ui.pct(report['accuracy'])}; "
    f"95% Wilson interval {ui.pct(low)} to {ui.pct(high)}).",
]
table = selection["table"]
if len(table) > 1:
    runner = table[1]
    findings.append(
        f"On validation data it averaged {ui.pct(table[0]['val_acc_mean'])} across seeds {selection['seeds']}, "
        f"against {ui.pct(runner['val_acc_mean'])} for the runner-up ({label(runner['model'])}, {runner['epochs']} epochs)."
    )
comparisons = report.get("comparisons", [])
if comparisons:
    significant = [c for c in comparisons if c["p_holm"] < 0.05]
    findings.append(
        f"In exploratory exact McNemar tests on the same test images, {len(significant)} of {len(comparisons)} "
        "comparisons with the other backbones stayed significant after Holm correction. A non-significant result "
        "does not show the models are equivalent; 150 images resolve only large differences."
    )
if ui.available(Files.CALIBRATION):
    cal = ui.read_json(Files.CALIBRATION)
    findings.append(
        f"Temperature scaling (T = {cal['temperature']:.2f}, fitted on validation data) changed the test expected "
        f"calibration error from {cal['uncalibrated']['ece']:.3f} to {cal['temperature_scaled']['ece']:.3f}."
    )
if ui.available(Files.ASSIGNMENT_SUMMARY):
    summary = ui.read_csv(Files.ASSIGNMENT_SUMMARY)
    best = summary[summary["selected"].astype(bool)].iloc[0]
    findings.append(
        f"The assignment protocol, which picks the winner on the test set, favoured {label(best['model'])} at "
        f"{best['epochs']} epochs ({ui.pct(best['test_acc'])}). Choosing on the test set makes that score optimistic."
    )
if ui.available(Files.BENCHMARK):
    bench = [b for b in ui.read_json(Files.BENCHMARK) if b["device"] == "cpu"]
    by_model = {b["model"]: b["median_ms"] for b in bench}
    if report["model"] in by_model and "effnetb4" in by_model and report["model"] != "effnetb4":
        findings.append(
            f"On a CPU, {report['label']} answers in about {by_model[report['model']]:.0f} ms per image, against "
            f"{by_model['effnetb4']:.0f} ms for EfficientNet-B4."
        )
if ui.available(Files.OOD):
    ood = ui.read_json(Files.OOD)
    if ood.get("status") == "ok":
        findings.append(
            f"A not-food check, tuned on validation photos, flagged {ui.pct(ood['test']['rejected_non_food'])} of "
            f"{ood['test']['n_non_food']} unseen flower and pet photos while accepting "
            f"{ui.pct(ood['test']['accepted_food'])} of the real food test photos."
        )
if ui.available(Files.ROBUSTNESS):
    robustness = ui.read_csv(Files.ROBUSTNESS)
    worst = robustness[robustness["severity"] == 3].sort_values("accuracy").iloc[0]
    findings.append(
        f"Under the strongest {str(worst['label']).lower()} tested, accuracy fell to {ui.pct(worst['accuracy'])}, "
        "the largest drop among five everyday photo problems."
    )
if ui.available(Files.DATA_EFFICIENCY):
    efficiency = ui.read_csv(Files.DATA_EFFICIENCY)
    smallest = efficiency[efficiency["fraction"] == efficiency["fraction"].min()]
    largest = efficiency[efficiency["fraction"] == efficiency["fraction"].max()]
    if efficiency["fraction"].nunique() > 1:
        findings.append(
            f"Trained on only {int(smallest['n_train'].mean())} photos, the same configuration averaged "
            f"{ui.pct(smallest['val_acc'].mean())} on validation data, against {ui.pct(largest['val_acc'].mean())} with "
            f"all {int(largest['n_train'].mean())}."
        )
for text in findings:
    st.markdown(f"<div class='fv-finding'>{text}</div>", unsafe_allow_html=True)
ui.note("Use the sidebar to inspect the experiments, the errors, the reliability of the probabilities, "
        "or to try the model on your own photo.")

if ui.available(Files.SUMMARY_FIGURE):
    st.subheader("Share these results")
    figure = ui.store().path(Files.SUMMARY_FIGURE)
    st.image(str(figure))
    st.download_button("Download the summary figure (PNG)", figure.read_bytes(), file_name="foodvision-lab-summary.png",
                       mime="image/png")
