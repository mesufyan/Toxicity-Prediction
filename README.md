# Toxicity Prediction — Xerostomia (Drymouth)

Predicting radiation-induced xerostomia in head-and-neck patients from CT
radiomics combined with clinical and dose-volume features.

## Data
- **Radiomic features:** extracted per patient, reduced to ~1,495 features after
  variance and correlation filtering (from an initial ~11,800).
- **Clinical features (22):** demographics and risk factors (age, sex, smoking,
  alcohol, T/N stage, BMI, chemo status) plus dose-volume metrics
  (parotid L/R Dmean, submandibular gland L/R Dmean, oral cavity, mandible,
  esophagus, PCM Dmean, dose per fraction).
- **Target:** `Drymouth_binary` (xerostomia present / absent).
- Patients with a missing label were dropped; patient ID used as the join key.

## Pipeline (separate-stream / late fusion)
Radiomic and clinical features are reduced **independently**, then merged. This
prevents the ~1,495 radiomic features from outvoting the much smaller clinical
block during selection — protecting the dose-volume metrics that are the
established physical drivers of xerostomia.

**1. Single patient-level train/test split** (80/20, stratified on the target).
The same patients define train and test for both streams.

**2. Radiomic stream (fit on train only):**
- Median imputation (features >50% missing dropped first)
- mRMR → top 150 (redundancy-aware relevance)
- Mann-Whitney U test with Benjamini-Hochberg FDR correction (non-parametric,
  suited to non-normal radiomic distributions)
- Random Forest importance → top ~10

**3. Clinical stream (fit on train only):**
- Categorical features one-hot encoded; numerics median-imputed
- Only few clinical features are removed (no aggressive selection on a small,
  domain-selected set)

**4. Merge** the selected radiomic features with the clinical features by patient ID.

## Leakage prevention
- Test set held out **before** any imputation, selection, or encoding.
- All fitting (imputers, encoders, selectors) done on training data only.
- **Permutation test:** with shuffled labels the model collapses to chance
  (AUC ≈ 0.51 ± 0.06 over repeated shuffles), confirming no information leakage.

## Modelling
- Primary model: **XGBoost** (regularized, class-imbalance weighting via
  `scale_pos_weight`, early stopping). Tree models outperform a neural network
  on this small tabular dataset.
- Evaluated with ROC-AUC, precision-recall (AP), and confusion matrices on the
  held-out test set.

## Results (xerostomia)
- Held-out test AUC: ~0.677 
- Permutation null: ~0.51 ± 0.06 → measured signal is above chance.

## Notes / limitations
- Single 80/20 split with ~119 test patients → wide AUC confidence interval;
  nested cross-validation recommended for the final reported estimate.
- Performance appears bounded by the available features; dose-volume metrics
  contribute the clearest signal.
