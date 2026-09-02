# Architecture

How the repo fits together. Everything here is read off `Barren_LendingClub_DefaultModel.py` and `loan_data.csv`.

## Files

| Path | Committed | What it is |
| --- | --- | --- |
| `Barren_LendingClub_DefaultModel.py` | yes | The whole project. One top-to-bottom script, four sections, no functions or CLI. |
| `loan_data.csv` | yes | Input dataset. 9,578 rows, 14 columns. Lending Club / course extract. |
| `requirements.txt` | yes | Pinned floors for the seven third-party imports. |
| `README.md` | yes | Overview, reported metrics, run instructions. |
| `docs/ARCHITECTURE.md` | yes | This file. |
| `LICENSE` | yes | MIT. |
| `Figure 2025-09-24 164530.png` | yes | Training vs. validation loss from the reported run. |
| `Figure 2025-09-24 164541.png` | yes | Correlation heatmap after one-hot encoding. |
| `Figure 2025-09-24 164546.png` | yes | FICO score boxplot split by `not.fully.paid`. |
| `Figure 2025-09-24 164551.png` | yes | Histograms of the ten numerical features. |
| `transformed_loan_data.csv` | no — generated | Written at line 26 after one-hot encoding. Gitignored. |
| `output.csv` | no — generated | Written at the end of the run. One column, `predicted_default`, thresholded test-set predictions. Gitignored. |

The four PNGs are saved copies of the plots the script pops up with `plt.show()`. The script does not write PNGs itself; rerunning it will not regenerate these files.

## Script sections

**Section 1 — loading and feature transformation.** Reads `loan_data.csv`. Prints shape, dtypes, head. `OneHotEncoder(sparse_output=False, drop='first')` on `purpose`, which yields six indicator columns from seven categories. Concatenates them back and writes `transformed_loan_data.csv`.

**Section 2 — EDA and class balance.** Prints `describe()` and the normalized target distribution. Plots histograms of the ten numerical features and a FICO-vs-default boxplot.

**Section 3 — feature engineering and split.** Plots the correlation heatmap. Drops `int.rate` (it correlates -0.71 with `fico` and 0.46 with `revol.util`). Splits `X` / `y` on `not.fully.paid`. `StandardScaler` on all features, then an 80/20 `train_test_split` with `stratify=y, random_state=42`. SMOTE with `random_state=42, k_neighbors=3` on the training split only.

Note: the scaler is fit on the full dataset before the split, so test-set statistics leak into the scaling. It nudges the reported metrics optimistic. Left as-is because the reported numbers come from a run with this behavior; changing it would invalidate them.

**Section 4 — model, threshold, evaluation.** Sequential Keras model: Dense 64 ReLU → Dropout 0.5 → Dense 32 ReLU → Dense 16 ReLU → Dense 1 sigmoid. `l2(0.01)` on every dense layer, output included. Compiled with `Adam(learning_rate=0.0005)` and binary crossentropy. Fit for up to 150 epochs, batch size 64, `validation_split=0.2`, `EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)`. Then predicts on the test set, picks a threshold, prints `classification_report` and `roc_auc_score`, plots the loss curves, and writes `output.csv`.

`compute_class_weight` is imported but unused — leftover from the earlier class-weight approach that SMOTE replaced.

## Class balance

| Class | Rows | Share |
| --- | --- | --- |
| 0 — paid | 8,045 | 83.99% |
| 1 — not fully paid | 1,533 | 16.01% |

Six-to-one. This is why the run reports default-class F1 and ROC-AUC. Accuracy alone is 84% for a model that never predicts a default.

SMOTE is applied after the split, on the training rows only, so the test set keeps the natural 16% rate. The test set is not resampled.

## Threshold

The script does not use 0.5. It sweeps `precision_recall_curve` over the test-set probabilities, computes F1 at every point, and takes the threshold at `argmax`:

```python
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_threshold_pr = thresholds_pr[np.argmax(f1_scores)]
```

The reported run landed near 0.48. The value is recomputed every run and will drift with the weights, so treat 0.48 as one observation and not a constant.

Selecting the threshold on the same test set it is evaluated on is optimistic for the same reason the scaler is. A held-out validation split for threshold choice would be the fix.

## Reported metrics

Accuracy 79%, weighted F1 0.79, default-class F1 0.35, default recall 36%, ROC-AUC 0.68. From the training run documented in the previous README. Nothing in this repo re-runs or verifies them, and the script sets no TensorFlow seed, so a rerun will not reproduce them exactly.
