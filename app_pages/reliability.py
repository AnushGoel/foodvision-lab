"""Reliability: are the probabilities trustworthy, what does abstaining buy, and can it spot non-food photos?"""

import numpy as np
import pandas as pd
import streamlit as st

from foodvision import ui
from foodvision.artifacts import Files
from foodvision.calibration import abstention_curve
from foodvision.viz import abstention_plot, confidence_histogram, ood_histogram, reliability_diagram, robustness_plot

ui.page("Reliability", "A probability of 90% should be right about 90% of the time. Temperature scaling, fitted "
        "on validation data, rescales the model's confidence without changing any prediction.")
ui.require(Files.CALIBRATION, Files.PREDICTIONS)
cal = ui.read_json(Files.CALIBRATION)
predictions = ui.read_csv(Files.PREDICTIONS)

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown("**Reliability diagram (test set)**")
    st.plotly_chart(reliability_diagram(cal["uncalibrated"]["bins"], cal["temperature_scaled"]["bins"]))
with right:
    st.markdown(f"**Calibration metrics** (T = {cal['temperature']:.2f})")
    metrics = pd.DataFrame(
        {
            "Metric": ["Expected calibration error", "Negative log-likelihood", "Brier score"],
            "Softmax": [cal["uncalibrated"][k] for k in ("ece", "nll", "brier")],
            "Temperature-scaled": [cal["temperature_scaled"][k] for k in ("ece", "nll", "brier")],
        }
    )
    st.dataframe(metrics.round(4), hide_index=True)
    ui.note(f"Lower is better for all three. ECE uses {cal['n_bins']} equal-width bins; with about 150 test images "
            "several bins hold only a few images, so small ECE differences are not meaningful. T above 1 means the raw "
            "softmax was over-confident; below 1, under-confident.")

st.subheader("Confidence of right and wrong answers")
st.plotly_chart(confidence_histogram(predictions))

st.subheader("Answer only when confident")
classes = [c[2:] for c in predictions.columns if c.startswith("p_")]
probs = predictions[[f"p_{c}" for c in classes]].to_numpy()
labels = np.array([classes.index(label) for label in predictions["label"]])
threshold = st.slider("Minimum confidence to answer", 0.34, 0.99, 0.70, 0.01)
curve = abstention_curve(probs, labels)
kept = probs.max(axis=1) >= threshold
answered = int(kept.sum())
accuracy = (probs.argmax(axis=1) == labels)[kept].mean() if answered else float("nan")
c1, c2 = st.columns(2)
c1.metric("Images answered", f"{answered} of {len(labels)}")
c2.metric("Accuracy on answered images", ui.pct(accuracy))
st.plotly_chart(abstention_plot(curve, threshold))
ui.note("Raising the threshold trades coverage for accuracy. Choose a threshold on validation data before deployment; "
        "picking it here would tune it to the test set.")

st.subheader("Photos that aren't pizza, steak or sushi")
if not ui.missing_notice(Files.OOD):
    ood = ui.read_json(Files.OOD)
    if ood.get("status") == "disabled":
        st.info("The not-food check was switched off for this run (`ood.enabled: false` in the configuration).")
    elif ood.get("status") != "ok":
        st.info(f"The not-food check was skipped: {ood.get('reason', 'unknown reason')}. Re-run the research stage to "
                "retry: `python -m foodvision.pipeline --stages research --force`.")
    else:
        test = ood["test"]
        auroc = test["auroc"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Non-food photos caught", ui.pct(test["rejected_non_food"]),
                  help=f"Of {test['n_non_food']} non-food test photos, the share rejected at the chosen threshold.")
        c2.metric("Food photos let through", ui.pct(test["accepted_food"]),
                  help=f"Of {test['n_food']} pizza, steak and sushi test photos. Validation target: "
                  f"{ui.pct(ood['target_tpr'], 0)}.")
        c3.metric("AUROC", f"{auroc['estimate']:.3f}",
                  help=f"95% bootstrap CI {auroc['low']:.3f} to {auroc['high']:.3f}. 0.5 is chance, 1.0 is perfect "
                  "separation.")
        if ui.available(Files.OOD_SCORES):
            scores = ui.read_csv(Files.OOD_SCORES)
            st.plotly_chart(ood_histogram(scores[scores["split"] == "test"], ood["score"], ood["threshold"],
                                          ood["score_label"]))
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Caught, by photo collection (test)**")
            st.dataframe(pd.DataFrame({"Collection": [r["label"] for r in ood["per_source"]],
                                       "Photos": [r["n"] for r in ood["per_source"]],
                                       "Rejected": [ui.pct(r["rejected"]) for r in ood["per_source"]]}),
                         hide_index=True)
        with right:
            st.markdown("**Which score separates best?**")
            st.dataframe(pd.DataFrame({"Score": [r["label"] for r in ood["scores"]],
                                       "Validation AUROC": [round(r["val_auroc"], 3) for r in ood["scores"]],
                                       "Test AUROC": [round(r["test_auroc"], 3) for r in ood["scores"]]}),
                         hide_index=True)
        ui.note(f"{ood['score_label']} was chosen because it separated food from non-food best on validation photos; "
                f"its threshold keeps {ui.pct(ood['target_tpr'], 0)} of validation food photos. The test photos were "
                "used once, afterwards. Only obvious non-food (flowers, cats and dogs) was tested, so these numbers say "
                "nothing about other foods such as burgers or pasta.")

st.subheader("When photos get worse")
if not ui.missing_notice(Files.ROBUSTNESS):
    robustness = ui.read_csv(Files.ROBUSTNESS)
    st.plotly_chart(robustness_plot(robustness))
    clean = robustness[robustness["corruption"] == "none"].iloc[0]
    worst = robustness[robustness["severity"] == 3].sort_values("accuracy").iloc[0]
    ui.note(f"Clean test accuracy is {ui.pct(clean['accuracy'])}; the hardest strong corruption, "
            f"{str(worst['label']).lower()}, brings it to {ui.pct(worst['accuracy'])}. Corruptions are applied to the "
            "photo the model actually sees (after resizing and cropping). Exploratory: the test photos were reused "
            "here, and no decision was made from these results.")
    if "accepted_as_food" in robustness.columns:
        st.markdown("**Real food that the not-food check still accepts**")
        st.plotly_chart(robustness_plot(robustness, "accepted_as_food"))
        ui.note("A badly degraded photo can look unlike the training photos. Where a line drops, the check starts "
                "turning away real pizza, steak and sushi.")
