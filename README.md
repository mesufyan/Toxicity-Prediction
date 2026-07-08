# Toxicity Prediction in Head and Neck Radiotherapy — Xerostomia (Dry Mouth)

This repository predicts **radiation-induced xerostomia (dry mouth)** in head and neck cancer
patients, using a combination of **clinical features** and **radiomic features** extracted from
CT/dose imaging, evaluated with a regularized deep neural network. The project places particular
emphasis on **avoiding data leakage**, **preserving clinical signal despite radiomic feature
dominance**, **interpretability (SHAP)**, and **quantifying model uncertainty (MC Dropout)**.

---

## 1. Data

Two raw data sources are merged, joined on a shared `Patient` identifier:

| Source | File | Content |
|---|---|---|
| Clinical | `ml_data_0.9.xlsx` | Patient demographics, tumor staging, treatment parameters, dose-volume metrics, toxicity outcome labels |
| Radiomic | `radiomic_features.parquet` | Radiomic features extracted from CT/RT structures (first-order, texture, shape features per organ-at-risk) |

- **Number of patients:** `[TOTAL_N]` (train: `[TRAIN_N]`, test: `[TEST_N]`, 80/20 stratified split, `random_state=42`)
- **Clinical features (raw):** approximately 23 columns before preprocessing
- **Radiomic features (raw):** `[RAW_RADIOMIC_N]` columns before preprocessing
- **Target:** `Drymouth_binary` (binary xerostomia outcome). A second outcome, `Mucositis_binary`,
  exists in the same clinical file but is excluded from this pipeline to avoid leaking a related
  toxicity outcome into the xerostomia model.

> **Note:** exact patient counts and raw radiomic feature counts should be confirmed by re-running
> `data_processing.ipynb` and filling in the bracketed placeholders above — these values were not
> captured in the notebook's saved outputs.

---

## 2. Data Processing Pipeline (`data_processing.ipynb`)

### 2.1 Radiomic feature reduction (unsupervised, structural)

Radiomic features are reduced *before* any label information is used, purely on statistical
grounds:

1. **Variance thresholding** (`VarianceThreshold(threshold=0.2)`) — removes near-constant features
   that carry little to no discriminative information regardless of outcome.
2. **Correlation filtering** — features with pairwise Pearson correlation above 0.80 are dropped
   (only one of each highly correlated pair is kept), reducing redundancy in the radiomic feature
   space before any supervised selection is applied.

### 2.2 Clinical feature preprocessing — explicit leakage removal

Several clinical columns are dropped **by design**, not for data quality reasons, because they
would leak information unavailable at prediction time or leak a *different* toxicity outcome into
this model:

- `PFS_event`, `PFS_duration_m` — survival/progression outcomes measured *after* treatment, not
  available at the point a QA/toxicity-risk prediction would actually be made.
- `Dysphagia_binary`, `Dermatitis_binary`, `Oralpain_binary` — other toxicity outcomes. Keeping
  these as input features when predicting `Drymouth_binary` would let the model implicitly "peek"
  at correlated downstream toxicities rather than learn from pre-treatment/planning information.
- `chemo_fit`, `avelumab_given` — treatment-decision variables that may be confounded with outcome
  in ways that don't generalize.
- `(IMRTHPTV) Total Dose Received in Gy`, `Esophagus_Dnear-max`, `Mandible_Dnear-max`,
  `Oral_Cavity_Dnear-max` — near-duplicate or overly aggregated dose descriptors, removed to avoid
  redundancy with more specific dose-volume features retained elsewhere in the pipeline.
- `(MHYN) Medical Oncological Significant History`, `Unnamed: 0`, `REFERENCE` — administrative or
  non-informative columns.

### 2.3 Data type cleaning

Radiomic columns are coerced to numeric (`pd.to_numeric(..., errors='coerce')`), and any column
with more than 5% missing values after coercion is dropped entirely, rather than imputed, to avoid
introducing noisy, mostly-fabricated columns into the feature space.

### 2.4 Train/test split — done before imputation and selection

The 80/20 stratified split (`train_test_split(..., stratify=y_x, random_state=42)`) is performed
**before** imputation and **before** any feature selection. This ordering matters:

- **Imputation:** a `SimpleImputer(strategy="median")` is fit only on the training set, then
  applied (via `.transform`, not `.fit_transform`) to the test set. This prevents the test set's
  own distribution from influencing the imputed values.
- **Feature selection (below):** all supervised feature selection is performed using training
  labels only, and the resulting feature subset is then applied to the test set.

---

## 3. Feature Selection — Radiomics and Clinical Features Selected *Separately*

### 3.1 Why this matters: preventing radiomic features from crowding out clinical features

The clinical feature set is small (~23 columns), while the radiomic feature set is far larger,
even after variance/correlation filtering. If feature selection were run on the **combined**
matrix (radiomics + clinical together), a generic importance-ranking or selection algorithm would
be dominated by the sheer number of radiomic candidates competing for a limited feature budget —
clinical variables, despite being clinically meaningful and often lower-dimensional/lower-variance,
would be systematically outcompeted and likely excluded entirely, simply due to being vastly
outnumbered, not because they lack signal.

**The fix implemented here:** radiomic and clinical features are selected **on separate tracks**,
each with its own, appropriately scaled selection procedure, and only concatenated together
*after* both tracks are finalized. This guarantees clinical features have a reserved place in the
final feature set regardless of how many radiomic candidates exist.

### 3.2 Radiomics track (supervised, training data only)

1. **mRMR (minimum-Redundancy-Maximum-Relevance)** selection (`mrmr_classif`, `K=150`) — selects
   the 150 radiomic features most relevant to the xerostomia label while minimizing redundancy
   between selected features, computed on training data only.
2. **Mann-Whitney U test** on each of the 150 mRMR-selected features, comparing the two outcome
   groups, followed by **Benjamini-Hochberg FDR correction** for multiple comparisons
   (`p_adj < 0.05` retained). This is a second, independent statistical filter beyond mRMR's
   relevance ranking.
3. **Random Forest importance ranking**, evaluated via 5-fold stratified cross-validation across
   increasing feature-count subsets (5, 10, 15, ... up to 100 features) to find the CV-AUC-optimal
   number of features. The final radiomics feature set used downstream is the **top 10 features**
   by Random Forest importance.

### 3.3 Clinical track (light pruning, not aggressive selection)

Rather than subjecting the already-small clinical set to the same aggressive multi-stage
selection, clinical features are only lightly pruned:

- **Spearman correlation** of each clinical feature with the target is computed (training data
  only) for inspection.
- A small number of redundant dose-related clinical columns
  (`Esophagus_Dmean`, `Mandible_Dmean`, `PCM_(constrictors)_Dmean`) and one low-value categorical
  (`(PTUMLOC) Localization of the Primary Tumor`) are dropped based on this inspection.
- Remaining categorical clinical variables (T stage, N stage, smoking status, alcohol consumption)
  are **one-hot encoded**, fit on the training set, with the test set **aligned** to the training
  columns (`X_train.align(X_test, join='left', axis=1, fill_value=0)`) — this ensures any category
  present only in the test set does not silently create a mismatched or leaking column.

### 3.4 Recombination

The top-10 radiomic features and the (lightly pruned, encoded) clinical features are concatenated
into the final feature matrix, separately for train and test, preserving the train/test boundary
established at split time.

---

## 4. Scaling

A `StandardScaler` is fit **only on the training set** and applied (via `.transform`) to the test
set, consistent with the leakage-avoidance principle applied throughout: no statistic used for
preprocessing is ever computed using test-set data.

Final processed files saved for modeling: `X_train_x.xlsx`, `X_test_x.xlsx`, `y_train_x.xlsx`,
`y_test_x.xlsx`.

---

## 5. Model (`Deep_Learning_Toxicity_Prediction_Drymouth.ipynb`)

- **Architecture:** a small, regularized feed-forward neural network (Dense(32) → Dropout(0.35) →
  Dense(16) → Dense(1, sigmoid)), with L1/L2 kernel regularization and `AdamW` optimizer, built in
  TensorFlow/Keras.
- **Class imbalance:** handled via `compute_class_weight('balanced', ...)`, applied during
  training.
- **Validation:** 5-fold stratified cross-validation is run first to obtain an unbiased estimate of
  generalization performance (mean ± std CV AUC), before a final model is refit on the full
  training set with an internal validation split and early stopping (`monitor='val_auc'`,
  `patience=20`, best weights restored).

### 5.1 Hyperparameter tuning (Optuna)

Network width (`units1`, `units2`), dropout rate, L1/L2 regularization strength, learning rate,
weight decay, and batch size are tuned using **Optuna** (`TPESampler`, 30 trials), with the
objective being mean 5-fold CV AUC — the same cross-validation scheme used for the untuned
baseline, to keep the comparison fair. The final model is refit using the best-found
hyperparameters.

### 5.2 Interpretability (SHAP)

`shap.Explainer` is used with a small, representative background sample (`shap.sample`, 50 points)
— the model-agnostic approach recommended in the SHAP documentation for tabular deep learning
models. Global (summary, bar) and per-patient (waterfall) explanations are generated.

**A key finding from this project:** SHAP's top-ranked feature is not always stable across
retraining. In some retrainings, the model correctly predicted elevated xerostomia risk without
SHAP identifying parotid dose, the established clinical driver (xerostomia risk rises sharply above
26 Gy to the parotid gland), as the leading feature. This indicated that a correct prediction is
not the same as a reliable explanation, and motivated the uncertainty quantification below as a
complementary signal alongside SHAP.

### 5.3 Model uncertainty (Monte Carlo Dropout)

Because the network already includes a Dropout layer, dropout is kept **active at inference time**
(`training=True`), and 100 stochastic forward passes are run per patient (Gal & Ghahramani, 2016).
The resulting standard deviation across passes is used as a per-patient uncertainty estimate,
computed alongside the mean prediction. Patients near the 0.5 decision boundary **and** with high
predictive uncertainty are flagged as the cases a clinician should treat with the most caution —
these are not necessarily the same patients SHAP would flag as having an unstable explanation, and
both signals are reported together rather than one replacing the other.

---

## 6. Results

| Split | AUC | Accuracy | Recall |
|---|---|---|---|
| Train (tuned) | 0.879 | - | - |
| Test (tuned) | 0.677 | 67.7 | 68.1 |

Confusion matrices (tuned model):

- **Train:** TN=221, FP=64, FN=37, TP=153
- **Test:** TN=48, FP=24, FN=15, TP=32

> The gap between train AUC (0.879) and test AUC (0.677) indicates some degree of overfitting
> despite regularization, cross-validation-guided hyperparameter tuning, and the small, carefully
> selected feature set (10 radiomic + clinical features). This is discussed further in
> [Section 7](#7-limitations).

---

## 7. Limitations

- The train/test AUC gap suggests the model may not generalize as strongly to unseen patients as
  the training performance implies; results should be interpreted with this in mind, and external
  validation on an independent cohort would be needed before any clinical use.
- SHAP explanations, while informative, are approximations of model behavior and were empirically
  observed to vary in top-feature ranking across retrainings (Section 5.2) — a limitation of
  post-hoc explainability methods in general, not specific to this dataset.
- Radiomic feature reduction (variance/correlation thresholds, mRMR K=150, top-10 by RF importance)
  involves several manually set thresholds; sensitivity of downstream performance to these choices
  was not exhaustively explored here.

---

## 8. Repository Structure

```
├── data_processing.ipynb          # Raw data → leakage-safe, feature-selected train/test sets
├── Deep_Learning_Toxicity_Prediction_Drymouth.ipynb   # Model training, Optuna, SHAP, MC Dropout
├── Datasets/                      # Raw and processed data (not included in repo if patient data)
└── Results/                       # Saved performance plots, SHAP plots, uncertainty outputs
```

---

## 9. Requirements

```
pandas
numpy
scikit-learn
scipy
statsmodels
mrmr-selection
tensorflow
torch
optuna
shap
matplotlib
```
