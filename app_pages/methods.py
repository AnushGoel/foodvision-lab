"""Data and methods: the dataset audit, the architectures, and the two experimental protocols."""

import pandas as pd
import streamlit as st

from foodvision import ui
from foodvision.artifacts import Files
from foodvision.viz import class_counts, size_scatter

ui.page("Data and methods", "What the data contains, how the three networks differ, and how each number in this "
        "app was produced.")
tab_data, tab_models, tab_protocol, tab_card = st.tabs(
    ["Dataset audit", "Architectures", "Protocol and statistics", "Model card"]
)

with tab_data:
    if not ui.missing_notice(Files.AUDIT, Files.IMAGES):
        audit = ui.read_json(Files.AUDIT)
        images = ui.read_csv(Files.IMAGES)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Images", audit["n_images"])
        c2.metric("Corrupted files", audit["n_corrupted"])
        c3.metric("Near-duplicate pairs", audit["exact_duplicate_pairs"] + audit["near_duplicate_pairs"])
        c4.metric("Pairs across train and test", audit["cross_split_pairs"])
        left, right = st.columns(2, gap="large")
        left.markdown("**Images per class and split**")
        left.plotly_chart(class_counts(audit["counts"]))
        right.markdown("**Image dimensions**")
        right.plotly_chart(size_scatter(images))
        st.markdown(
            f"Formats: {audit['formats']}. Colour modes: {audit['modes']}. Duplicates were screened with a 64-bit "
            f"difference hash (at most {audit['near_duplicate_threshold_bits']} differing bits); every flagged pair "
            "needs a visual check before it is treated as a real duplicate."
        )
        if audit["duplicate_pairs"]:
            st.dataframe(pd.DataFrame(audit["duplicate_pairs"]), hide_index=True)
        ui.note("Source: the 20% pizza, steak and sushi subset of Food-101 (Bossard et al., 2014) distributed with the "
                "Zero to Mastery PyTorch course. Food-101 images come from Foodspotting and remain the property of their "
                "owners; see docs/DATASET_CARD.md.")

with tab_models:
    if not ui.missing_notice(Files.ARCHITECTURE):
        arch = ui.read_json(Files.ARCHITECTURE)
        table = pd.DataFrame(
            {
                "Backbone": [a["label"] for a in arch],
                "Input (px)": [f"{a['crop']} (resize {a['resize']})" for a in arch],
                "MBConv blocks": [a["mbconv_blocks"] for a in arch],
                "Feature dim.": [a["feature_dim"] for a in arch],
                "Dropout": [a["dropout"] for a in arch],
                "Parameters": [f"{a['params']['total'] / 1e6:.2f} M" for a in arch],
                "Trainable": [f"{a['params']['trainable']:,}" for a in arch],
                "ImageNet top-1": [f"{a['imagenet_top1']:.1f}%" if a["imagenet_top1"] else "n/a" for a in arch],
                "GFLOPs": [a["gflops"] if a["gflops"] else "n/a" for a in arch],
            }
        )
        st.dataframe(table, hide_index=True)
        st.markdown(
            "EfficientNet scales one baseline network in depth, width and input resolution together (compound scaling; "
            "Tan & Le, 2019). Each stage stacks MBConv blocks: a 1x1 expansion, a depthwise convolution, "
            "squeeze-and-excitation channel gating (Hu et al., 2018) and a 1x1 projection with a residual connection "
            "(Sandler et al., 2018). Here every convolution is frozen and only a new `Dropout -> Linear(features, 3)` "
            "head is trained."
        )
        choice = st.selectbox("Stage-by-stage structure", [a["label"] for a in arch])
        stages = next(a for a in arch if a["label"] == choice)["stages"]
        st.dataframe(pd.DataFrame(stages).assign(output=lambda d: d["output"].map(lambda s: " x ".join(map(str, s)))),
                     hide_index=True)

with tab_protocol:
    st.markdown(
        """
**Assignment protocol.** Nine runs (EfficientNet-B2, B3, B4 x 15, 20, 30 epochs) on all 450 training images,
evaluated on the 150 test images after every epoch and logged to TensorBoard. The best run is chosen on the test set,
as the assignment asks, so its score is optimistic.

**Research protocol.** The training images are split, stratified by class, into training and validation parts. Each
backbone is trained for 30 epochs with three seeds, with snapshots at 15, 20 and 30 epochs. The configuration with the
best mean validation accuracy is selected, its temperature is fitted on validation data, and only then is it evaluated
once on the test set. Comparisons with the other backbones after that point are exploratory.

**Statistics.** Wilson score intervals for accuracy (Wilson, 1927; Brown et al., 2001); percentile bootstrap intervals
for macro-F1 and for paired accuracy differences; exact McNemar tests on paired predictions (Dietterich, 1998) with
Holm's correction (Holm, 1979); temperature scaling and expected calibration error (Guo et al., 2017); Grad-CAM
(Selvaraju et al., 2017).

**Not-food check.** A three-class model must always answer with one of its classes, so non-food photos are screened
first. Three standard scores computed from the model's outputs were compared: maximum softmax probability (Hendrycks &
Gimpel, 2017), maximum logit (Hendrycks et al., 2022) and negative energy (Liu et al., 2020). The score and its
threshold were chosen on the validation photos plus a validation sample of non-food photos from Flowers-102 (Nilsback &
Zisserman, 2008) and Oxford-IIIT Pet (Parkhi et al., 2012), then tested once. Only the scores are stored, never the
non-food images.

**Training details.** Adam, learning rate 0.001, cross-entropy, batch size 32, no augmentation, ImageNet preprocessing
of each set of weights. All images are decoded once and cached as tensors, which gives identical results to decoding
every epoch because the preprocessing is deterministic.
"""
    )
    if ui.available(Files.MANIFEST):
        manifest = ui.read_json(Files.MANIFEST)
        env = manifest.get("environment", {})
        ui.note("Produced with " + ", ".join(f"{k} {v}" for k, v in env.items() if v) + f". Last update {manifest.get('updated_at')}.")

with tab_card:
    if not ui.missing_notice(Files.MODEL_CARD):
        card = ui.read_text(Files.MODEL_CARD)
        st.markdown(card.split("\n", 1)[1] if card.startswith("# ") else card)  # the page already has a title
