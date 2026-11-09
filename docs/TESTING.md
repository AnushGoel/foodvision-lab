# Testing

## Strategy
103 test cases across 14 files. Most of the logic (statistics, metrics, calibration, auditing, upload validation, reports) has no PyTorch dependency
and is tested directly against known values and reference implementations. Model, engine, Grad-CAM and inference
tests use randomly initialised networks (no weight downloads) and synthetic images. App tests render every page with
Streamlit's `AppTest`, both with a complete synthetic artifact set and with none, so empty states are tested too.
The synthetic artifacts (`tests/synthetic.py`) follow the real schema and are computed with the real statistics
code; they are never shipped to the app.

| File | What it checks |
| --- | --- |
| `test_stats.py` | Wilson bounds (incl. exact edges), exact McNemar against a known value, Holm, bootstrap reproducibility |
| `test_metrics.py` | confusion matrix orientation, per-class scores on the assignment's real confusion matrix, empty classes |
| `test_calibration.py` | temperature recovers a known over-confidence, ECE/NLL drop after scaling, argmax unchanged |
| `test_config.py` | assignment constants, typo rejection, fingerprint scope, seed consistency, not-food settings |
| `test_analysis.py` | every corruption keeps the size and grows stronger with severity, reproducible noise, nested and stratified subsets, t-SNE separates known clusters, the summary figure renders at exactly 2000 x 1125 px |
| `test_ood.py` | the three scores against their definitions (SciPy), AUROC against scikit-learn including ties, the threshold always keeps the target share, validation-only selection, reproducible disjoint sampling |
| `test_data.py` | class mapping, stratified split is disjoint, complete, balanced and reproducible |
| `test_audit.py` | hash robustness, corrupted-file and cross-split leakage detection |
| `test_imageio.py` | empty, non-image, truncated, tiny and unsupported uploads are rejected with a message |
| `test_artifacts_reports.py` | JSON sanitising, atomic writes, schema checks, report generation, every model card section |
| `test_readme.py` | the README's results table matches the committed artifacts; refreshing it rewrites only the marked block |
| `test_models.py` | head and freezing for B2/B3/B4, BatchNorm statistics drift despite frozen weights, exact delta round trip |
| `test_engine_explain_inference.py` | training history and snapshots, Grad-CAM shape and range, end-to-end `Predictor`, the stored not-food threshold, unreadable non-food files skipped |
| `test_viz.py` | every Plotly figure builds and validates, including the robustness, feature-map and data-efficiency charts |
| `test_app.py` | every page renders with and without results; the entry point registers every page |

## Running
```bash
pip install -r requirements-dev.txt && pip install -e . --no-deps
pytest --cov                        # full suite with coverage
pytest tests/test_stats.py -q       # one file
pre-commit run --all-files          # ruff lint + format, whitespace and file-size checks
```

## Continuous integration
`.github/workflows/ci.yml` runs on every push and pull request: ruff lint, ruff format check, mypy on the typed
core modules, the full pytest suite with coverage, and a live smoke test that starts the Streamlit server and polls
its health endpoint. GPU training is deliberately not part of CI; it runs on Colab and its outputs are committed as
artifacts.
