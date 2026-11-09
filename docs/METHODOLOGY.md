# Methodology

## Research questions

| | Question | Answered by |
| --- | --- | --- |
| RQ1 | How does EfficientNet scaling (B2, B3, B4) affect accuracy when only the classifier is trained? | validation table, test comparisons |
| RQ2 | How does the training budget (15, 20, 30 epochs) affect convergence and accuracy? | learning curves, validation table |
| RQ3 | How do model size, training time and CPU latency trade off against accuracy? | Pareto plot, latency benchmark |
| RQ4 | Which classes and images cause the errors? | confusion matrix, per-class metrics, Grad-CAM gallery |
| RQ5 | How reliable are the predicted probabilities? | reliability diagram, ECE, abstention curve |

Hypothesis for RQ1 (confirmatory): larger backbones transfer better (Kornblith et al., 2019). Everything beyond the
selection of one configuration and its single test evaluation is labelled exploratory.

## Two protocols, kept separate

**A. Assignment protocol (reproduction).** Nine runs (three backbones x three budgets) on all training images, each
logging `Loss/train`, `Loss/test`, `Accuracy/train` and `Accuracy/test` to TensorBoard every epoch. The best run is
chosen on the test set, as the assignment asks. Because the same test set chooses and measures, that score is
optimistically biased (Cawley & Talbot, 2010).

**B. Research protocol (selection without the test set).**
1. Split the training images, stratified by class, into training (80%) and validation (20%).
2. Train each backbone for 30 epochs with seeds 42, 43 and 44, taking snapshots at 15, 20 and 30 epochs. Because each
   run is seeded, the budgets are stopping points on one trajectory, which costs three runs instead of nine.
3. Select the configuration with the highest mean validation accuracy (ties: lower validation loss, fewer
   parameters, fewer epochs).
4. Fit the softmax temperature on the validation logits of the selected model.
5. Evaluate the selected model once on the untouched test set.
6. Afterwards, and only as exploratory analysis, compare it with the other backbones at the same budget.

## Transfer learning

Each backbone loads torchvision's `IMAGENET1K_V1` weights. All `features` parameters are frozen and the 1,000-class
classifier is replaced by `Dropout(p) -> Linear(C, 3)`, keeping each model's dropout rate. Preprocessing is exactly
the transform shipped with each set of weights (bicubic resize, centre crop to 288, 300 or 380 pixels, ImageNet
normalisation). Optimiser: Adam (Kingma & Ba, 2015), learning rate 0.001, cross-entropy, batch size 32.

**BatchNorm.** `requires_grad=False` stops weight updates but not BatchNorm running-statistic updates: in training
mode the frozen backbone's statistics drift toward the food images. The research protocol includes an ablation that
keeps the backbone in evaluation mode, so the statistics stay at their ImageNet values (Li et al., 2016, discuss
this kind of statistic adaptation as a domain-adaptation mechanism).

**Dropped data.** The original notebook used `drop_last=True`, discarding two training images every epoch to avoid
a two-image BatchNorm batch. With the 360-image training split and batch size 32 the last batch has 8 images, so
the pipeline keeps every image (`drop_last: false`).

## Statistics

* **Accuracy intervals.** Wilson score interval for a binomial proportion (Wilson, 1927), preferred to the normal
  approximation near 0 and 1 (Brown et al., 2001):
  $\frac{\hat p + z^2/2n}{1 + z^2/n} \pm \frac{z}{1 + z^2/n}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}$.
* **Other intervals.** Percentile bootstrap (10,000 resamples) for macro-F1; paired bootstrap for the accuracy
  difference between two models evaluated on the same images.
* **Paired tests.** Exact McNemar test on the discordant pairs (McNemar, 1947; Dietterich, 1998), with Holm's
  step-down correction for the multiple comparisons (Holm, 1979). A non-significant result means the data cannot
  separate the models; it is not evidence of equivalence.
* **Repeatability.** Mean and standard deviation of validation accuracy across three seeds; test accuracy of the
  selected configuration for each seed (Picard, 2021, shows why seed variance matters).

## Calibration and abstention

Temperature scaling divides the logits by one scalar T that minimises validation negative log-likelihood (Guo et
al., 2017); it changes confidence, never the predicted class. Reported on the test set before and after scaling:
expected calibration error (10 equal-width bins), negative log-likelihood and Brier score, plus a reliability
diagram. With about 150 test images several bins are sparse, so small ECE differences are not meaningful. The
abstention curve shows coverage against accuracy when low-confidence predictions are withheld.

## Not-food check (out-of-distribution detection)

A three-class model must assign every photo to pizza, steak or sushi, so the pipeline compares three post-hoc scores
that rank how much a photo resembles the training data, all computed from the selected model's logits `z`: the
maximum softmax probability at the deployed temperature, `max_k softmax(z / T)_k` (Hendrycks & Gimpel, 2017); the
maximum logit, `max_k z_k` (Hendrycks et al., 2022); and the negative free energy, `log sum_k exp(z_k)` (Liu et al.,
2020). Non-food photos come from two public collections downloaded through torchvision, Flowers-102 (Nilsback &
Zisserman, 2008) and Oxford-IIIT Pet (Parkhi et al., 2012); a seeded sample of each is halved into validation and test
parts.

The score with the highest validation AUROC is selected, and its threshold is the highest value that still keeps at
least 95% of the validation food photos. Only then are the test photos of both kinds scored, once. The report gives
the AUROC with a stratified bootstrap interval (food and non-food photos resampled separately), the false-positive
rate at 95% true-positive rate, and the shares of food and non-food test photos accepted at the chosen threshold,
overall and per collection. The chosen score and threshold are stored inside the checkpoint, so the app applies
exactly the validated rule. Only scores are kept; no non-food image is stored or redistributed.

Flowers and pets are far from food, so these results do not measure near-distribution cases such as other dishes,
where a closed-set classifier is most likely to be confidently wrong.

## Analysis of the selected model

These analyses run after selection, on the saved checkpoint reloaded exactly as the app loads it; no choice depends
on them. The first check confirms that the reloaded model reproduces the reported test accuracy.

**Robustness.** Five everyday corruptions (Gaussian blur, sensor noise, low light, JPEG compression and low resolution)
are applied at three strengths to the model's view of each test photo, after resizing and cropping, in the spirit of
ImageNet-C (Hendrycks & Dietterich, 2019). For each condition the report gives accuracy with a Wilson interval and,
with the not-food check active, the share of real food photos it still accepts. Reusing the test photos here is
exploratory and is labelled as such.

**Feature map.** The pooled penultimate features of the test photos and of the non-food test sample are
L2-normalised, reduced with PCA to at most 50 dimensions and projected with t-SNE (van der Maaten & Hinton, 2008).
Neighbourhoods in the map are meaningful; distances between clusters and the axes are not.

**Data efficiency.** The selected configuration is retrained on nested, class-stratified subsets of the training split
(10%, 25% and 50%) with each research seed, using the same epochs and learning rate, and scored on the validation
split. The full-data points are the research runs themselves. Smaller subsets also mean fewer optimisation steps,
which is part of what the curve measures.

## Explainability

Grad-CAM (Selvaraju et al., 2017) weights the final feature maps by the gradient of the predicted class score.
Because the backbone is frozen, its output does not track gradients, so the implementation makes that feature map
a gradient-tracking leaf. Grad-CAM shows where the evidence correlates with the decision, not why the decision was
made; it is used to check whether the model attends to the food rather than plates, tables or text.

## Threats to validity

Small test set (one image is 0.67 percentage points); test-set selection in protocol A; three seeds only; no data
augmentation or fine-tuning; Food-101 training images contain some label noise (Bossard et al., 2014); wall-clock
times depend on shared cloud hardware; the not-food check was evaluated on far-from-food photos (flowers, pets)
only, with about 200 test photos.

## References

Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 – Mining discriminative components with random forests. In D. Fleet, T. Pajdla, B. Schiele, & T. Tuytelaars (Eds.), *Computer vision – ECCV 2014* (pp. 446–461). Springer. https://doi.org/10.1007/978-3-319-10599-4_29

Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science, 16*(2), 101–133. https://doi.org/10.1214/ss/1009213286

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://jmlr.org/papers/v11/cawley10a.html

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

Hendrycks, D., Basart, S., Mazeika, M., Zou, A., Kwon, J., Mostajabi, M., Steinhardt, J., & Song, D. (2022). Scaling out-of-distribution detection for real-world settings. In *Proceedings of the 39th International Conference on Machine Learning* (Vol. 162, pp. 8759–8773). PMLR. https://proceedings.mlr.press/v162/hendrycks22a.html

Hendrycks, D., & Dietterich, T. (2019, May 6–9). *Benchmarking neural network robustness to common corruptions and perturbations* [Paper presentation]. 7th International Conference on Learning Representations, New Orleans, LA, United States. https://doi.org/10.48550/arXiv.1903.12261

Hendrycks, D., & Gimpel, K. (2017, April 24–26). *A baseline for detecting misclassified and out-of-distribution examples in neural networks* [Paper presentation]. 5th International Conference on Learning Representations, Toulon, France. https://doi.org/10.48550/arXiv.1610.02136

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.

Kingma, D. P., & Ba, J. (2015, May 7–9). *Adam: A method for stochastic optimization* [Paper presentation]. 3rd International Conference on Learning Representations, San Diego, CA, United States. https://doi.org/10.48550/arXiv.1412.6980

Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do better ImageNet models transfer better? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2661–2671). IEEE. https://doi.org/10.1109/CVPR.2019.00277

Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2016). *Revisiting batch normalization for practical domain adaptation* (arXiv:1603.04779). arXiv. https://doi.org/10.48550/arXiv.1603.04779

Liu, W., Wang, X., Owens, J. D., & Li, Y. (2020). Energy-based out-of-distribution detection. *Advances in Neural Information Processing Systems, 33*. https://doi.org/10.48550/arXiv.2010.03759

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596

Nilsback, M.-E., & Zisserman, A. (2008). Automated flower classification over a large number of classes. In *2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing* (pp. 722–729). IEEE. https://doi.org/10.1109/ICVGIP.2008.47

Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs. In *2012 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 3498–3505). IEEE. https://doi.org/10.1109/CVPR.2012.6248092

Picard, D. (2021). *Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision* (arXiv:2109.08203). arXiv. https://doi.org/10.48550/arXiv.2109.08203

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626). IEEE. https://doi.org/10.1109/ICCV.2017.74

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97, pp. 6105–6114). PMLR. https://proceedings.mlr.press/v97/tan19a.html

van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research, 9*(86), 2579–2605. https://jmlr.org/papers/v9/vandermaaten08a.html

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953
