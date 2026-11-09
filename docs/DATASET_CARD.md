# Dataset card: pizza, steak and sushi (20% Food-101 subset)

## Provenance
* **Parent dataset.** Food-101 (Bossard et al., 2014): 101 food categories with 1,000 images each, split into 750
  training and 250 test images per class. Images were collected from the Foodspotting website and rescaled to at most
  512 pixels on the longer side. The training images were deliberately not cleaned and contain some label noise; the
  test images were checked by hand.
* **Subset.** `pizza_steak_sushi_20_percent.zip`, a random 20% sample of three classes distributed with the
  *Zero to Mastery Learn PyTorch for Deep Learning* course (Bourke, n.d.). The pipeline downloads it; this repository
  does not redistribute any images.

## Composition (measured)
| Class | Train | Test |
| --- | --- | --- |
| pizza | 154 | 46 |
| steak | 146 | 58 |
| sushi | 150 | 46 |
| **Total** | **450** | **150** |

The subset is close to, but not exactly, balanced; per-class metrics are therefore reported alongside accuracy.
The full measured audit (formats, dimensions, corrupted files, duplicate and leakage screening) is generated in
`artifacts/reports/DATA_AUDIT.md`.

## Intended use
Teaching and demonstrating transfer learning, evaluation and explainability on a small, well-known benchmark.
It is not suitable for nutrition estimation, dietary or medical decisions, or any production food-recognition use.

## Known limitations
* Three classes only; the photos reflect one social platform's photography style, cuisines and plating.
* Small: one test image moves accuracy by 0.67 percentage points.
* Label noise in the training split (inherited from Food-101).
* Possible near-duplicates; the audit flags candidate pairs with a perceptual hash, and they need visual review.

## Licensing
Food-101 images come from Foodspotting and are not the property of ETH Zurich, which released the dataset. Use
beyond scientific fair use must be negotiated with the respective picture owners under Foodspotting's terms of use.
The app shows a small number of Grad-CAM thumbnails of test images for research demonstration. Set
`evaluation.export_gallery: false` in `configs/default.yaml` to publish no dataset images at all.

## Non-food photos for the not-food check
To choose and test the threshold that flags photos that aren't pizza, steak or sushi, the pipeline downloads two public
collections through torchvision: Flowers-102 (Nilsback & Zisserman, 2008) and Oxford-IIIT Pet, which contains cats
and dogs (Parkhi et al., 2012). A seeded sample of each (200 photos by default, `ood.per_source`) is halved into
validation and test parts. Only the resulting scores are kept in `artifacts/research/ood_scores.csv`; no non-food
image is stored in the artifacts, the repository or the app. Both collections remain subject to their creators' and
photographers' terms; check them before reusing any image. Because flowers and pets are far from food, these photos
cannot show how the check behaves on other dishes.

## References
Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 – Mining discriminative components with random forests. In D. Fleet, T. Pajdla, B. Schiele, & T. Tuytelaars (Eds.), *Computer vision – ECCV 2014* (pp. 446–461). Springer. https://doi.org/10.1007/978-3-319-10599-4_29

Bourke, D. (n.d.). *Zero to Mastery learn PyTorch for deep learning* [Online book]. Retrieved November 8, 2026, from https://www.learnpytorch.io/

Nilsback, M.-E., & Zisserman, A. (2008). Automated flower classification over a large number of classes. In *2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing* (pp. 722–729). IEEE. https://doi.org/10.1109/ICVGIP.2008.47

Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs. In *2012 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 3498–3505). IEEE. https://doi.org/10.1109/CVPR.2012.6248092
