# Model card: FoodVision EfficientNet-B4

_Generated from the pipeline's artifacts, so no number here was typed by hand. Structure after Mitchell et al. (2019)._

## Model details
- **Task:** classify a food photo as one of 3 classes (pizza, steak, sushi), after a check that flags photos unlike the training data.
- **Architecture:** EfficientNet-B4 with torchvision ImageNet weights `IMAGENET1K_V1`. The feature extractor is frozen; a new `Dropout -> Linear` head is trained.
- **Training:** 30 epochs, seed 42, 360 images (stratified training split).
- **Selection:** best of 9 backbone and epoch-budget configurations by mean validation accuracy over seeds 42, 43, 44; the test set played no part.
- **Preprocessing:** resize 384 px (bicubic), centre crop 380 px, ImageNet normalisation.
- **Calibration:** temperature scaling, T = 0.40, fitted on validation data.
- **Not-food check:** maximum logit, threshold 0.563, set to keep 95% of validation food photos.
- **Checkpoint:** 598 KB, storing only what training changed (head and BatchNorm statistics) over the pinned pretrained weights. SHA-256 `b678294208f2814a...`.
- **Version:** FoodVision Lab 1.0.0, created 2026-11-09T01:38:22+00:00. Code under the MIT licence.

## Intended use
- **Primary use:** showing transfer learning, honest evaluation, calibration and explainability in a portfolio app.
- **Primary users:** students, practitioners and recruiters exploring the project.

**Out of scope:**
- Any decision about allergies, diet, nutrition or food safety, or anything else where a wrong label matters.
- Foods other than these three: a burger or a salad will usually receive one of the three labels (the not-food check was tested on flowers and pets, not on other foods).
- Photos of people, or any use that identifies or profiles people.

## Factors
Lighting, camera angle, plating, regional styles of each dish and plates with several foods can all change performance. The test set is too small to measure these factors separately.

## Metrics
Accuracy with a Wilson score interval; macro-F1 with a bootstrap interval; per-class precision, recall and F1; calibration (expected calibration error, negative log-likelihood, Brier score); for the not-food check, AUROC and the shares of food and non-food photos accepted.

## Evaluation data
- 150 Food-101 test photos, held out from training and from every choice above.
- 200 non-food test photos sampled from Flowers-102 (flowers; Nilsback & Zisserman, 2008) and Oxford-IIIT Pet (cats and dogs; Parkhi et al., 2012), separate from the validation sample used to set the threshold. Only scores are stored, never the images.

## Training data
360 photos for training and 90 for validation, split by class from the training part of the 20% pizza, steak and sushi subset of Food-101 (Bossard et al., 2014). See `docs/DATASET_CARD.md`.

## Quantitative analyses
| Metric | Value | 95% interval |
| --- | --- | --- |
| Accuracy | 96.0% (144/150) | 91.5% to 98.2% (Wilson) |
| Macro-F1 | 95.9% | 92.3% to 98.7% (bootstrap) |
| Cross-entropy | 0.225 | |

Per class (test set):

| Class | Precision | Recall | F1 | Photos |
| --- | --- | --- | --- | --- |
| pizza | 93.8% | 97.8% | 95.7% | 46 |
| steak | 98.2% | 96.6% | 97.4% | 58 |
| sushi | 95.6% | 93.5% | 94.5% | 46 |

The same configuration trained with each seed scored 96.0% (seed 42), 97.3% (seed 43), 95.3% (seed 44).

Calibration, before and after temperature scaling: expected calibration error 0.132 and 0.020; negative log-likelihood 0.225 and 0.101; Brier score 0.100 and 0.058.

Not-food check (test photos, used once): 56.0% of 200 non-food photos rejected and 94.7% of 150 food photos accepted; AUROC 0.926 (95% bootstrap CI 0.894 to 0.952).

| Collection | Test photos | Rejected |
| --- | --- | --- |
| Flowers-102 (flowers) | 100 | 37.0% |
| Oxford-IIIT Pet (cats and dogs) | 100 | 75.0% |

Robustness (exploratory, corrupted copies of the test photos): at the strongest level tested, accuracy was sensor noise 30.7%; low resolution 51.3%; blur 69.3%; JPEG compression 78.7%; low light 94.7%.

Speed: median 81 ms per photo on CPU (6 threads).

## Ethical considerations
- No personal data was used for training. The app classifies uploaded photos in memory and does not store them.
- Food-101 photos come from the Foodspotting website, so they reflect its users' cuisines and photo styles; the model may work worse for these dishes as prepared or photographed elsewhere.
- Confident-looking numbers invite over-trust. The app shows calibrated probabilities, lets users require a minimum confidence, flags photos that look unlike the training data and states these limits next to every prediction.

## Caveats and recommendations
- Expect lower accuracy on blurred, dark, noisy, compressed or low-resolution photos; the robustness results above show by how much.
- With 150 test photos the accuracy interval is about ±3 percentage points, so differences of a point or two between models are not meaningful.
- Grad-CAM heat maps summarise correlations in the last feature map; they are not causal explanations.
- Calibration was fitted on a small validation split; treat probabilities as approximate.
- Retrain and re-evaluate before using the model on other data or for another task.

## Environment
python 3.13.16, platform Linux-6.6.159+-x86_64-with-glibc2.39, torch 2.14.1+cu130, torchvision 0.29.1+cu130, cuda 13.0, gpu NVIDIA L4, cpu_count 12, foodvision 1.0.0.
