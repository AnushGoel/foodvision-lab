# artifacts/

Results produced by `python -m foodvision.pipeline` and read by the Streamlit app. Do not edit by hand:
re-run the pipeline instead. Everything here is small enough for Git (the model is a delta checkpoint of
well under 1 MB: the new head plus BatchNorm statistics over pinned torchvision ImageNet weights).

| Path | Content |
| --- | --- |
| `manifest.json` | schema version, environment, config, per-stage fingerprints, file hashes |
| `data/` | dataset audit summary and per-image table |
| `models/` | measured architecture summaries and CPU/GPU latency benchmark |
| `assignment/` | the nine assignment runs: summary table and per-epoch histories |
| `research/` | validation results, selection, one-time test report, predictions, calibration, not-food check (`ood.json`, per-photo scores in `ood_scores.csv`), BN ablation |
| `analysis/` | robustness to corrupted photos, the t-SNE feature map, the data-efficiency curve, and the checkpoint reload check |
| `figures/` | `summary.png`, a 16:9 overview of the key results for the README and social posts |
| `gallery/` | Grad-CAM thumbnails of selected test images |
| `model/` | the selected model (delta checkpoint) and its metadata |
| `reports/` | Markdown experiments summary, model card and data audit generated from the files above |
