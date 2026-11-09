"""Errors and diagnostics of the selected model on the test set."""

import streamlit as st

from foodvision import ui
from foodvision.artifacts import Files
from foodvision.viz import confusion_heatmap, feature_map_plot, per_class_bars

ui.page("Errors and diagnostics", "How the selected model performs on each class of the untouched test set, "
        "which images it gets wrong, and what Grad-CAM highlights in them.")
ui.require(Files.TEST_REPORT, Files.PREDICTIONS)
report = ui.read_json(Files.TEST_REPORT)
predictions = ui.read_csv(Files.PREDICTIONS)
f1_boot = report["macro_f1_bootstrap"]
low, high = report["accuracy_wilson"]

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", ui.pct(report["accuracy"]), help=f"95% Wilson CI {ui.pct(low)} to {ui.pct(high)}")
c2.metric("Macro-F1", ui.pct(report["macro_f1"]),
          help=f"95% bootstrap CI {ui.pct(f1_boot['low'])} to {ui.pct(f1_boot['high'])}")
c3.metric("Misclassified", f"{report['n_test'] - report['n_correct']} of {report['n_test']}")
ui.note(f"{report['label']}, {report['epochs']} epochs. Hover the values for 95% intervals "
        "(accuracy: Wilson; macro-F1: percentile bootstrap over the test images).")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Confusion matrix**")
    normalized = st.toggle("Show row percentages (recall)", value=False)
    st.plotly_chart(confusion_heatmap(report["confusion_matrix"], report["classes"], normalized))
with right:
    st.markdown("**Per-class precision, recall and F1**")
    st.plotly_chart(per_class_bars(report["per_class"]))
    support = ", ".join(f"{c['class']} {c['support']}" for c in report["per_class"])
    ui.note(f"Test images per class: {support}. With about 50 images per class, one image moves a class's recall "
            "by about two points.")

st.subheader("Grad-CAM gallery")
if ui.available(Files.GALLERY):
    entries = ui.read_json(Files.GALLERY)["entries"]
    kinds = sorted({e["kind"] for e in entries})
    shown = st.multiselect("Show", kinds, default=kinds)
    selected = [e for e in entries if e["kind"] in shown]
    for start in range(0, len(selected), 3):
        columns = st.columns(3)
        for column, entry in zip(columns, selected[start : start + 3]):
            with column:
                pair = st.columns(2)
                pair[0].image(str(ui.store().path(f"gallery/{entry['view']}")))
                pair[1].image(str(ui.store().path(f"gallery/{entry['cam']}")))
                verdict = "correct" if entry["correct"] else f"wrong (true: {entry['label']})"
                st.caption(f"Predicted {entry['prediction']} at {entry['confidence']:.0%}, {verdict}. {entry['kind'].capitalize()}.")
    ui.note("Warm colours mark regions whose features raised the predicted class score. Use the maps to check "
            "whether the model attends to the food rather than plates, tables or text; they are not causal explanations.")
else:
    st.info("The Grad-CAM gallery was not exported (evaluation.export_gallery is false).")

st.subheader("All misclassified test images")
wrong = predictions[~predictions["correct"].astype(bool)][["path", "label", "prediction", "confidence"]]
if wrong.empty:
    st.success("The selected model classified every test image correctly.")
else:
    st.dataframe(wrong.assign(confidence=wrong["confidence"].map(ui.pct)).rename(columns=str.capitalize),
                 hide_index=True)

st.subheader("What the network sees")
if not ui.missing_notice(Files.FEATURE_MAP):
    st.plotly_chart(feature_map_plot(ui.read_csv(Files.FEATURE_MAP)))
    ui.note("Each point is one test photo, placed with t-SNE so that photos the network describes similarly sit close "
            "together. Clusters and neighbours are meaningful; distances between clusters and the axes are not. Circled "
            "points were misclassified; grey points are the non-food photos used to test the not-food check.")
