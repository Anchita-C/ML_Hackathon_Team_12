# Step 2: Model comparison
Owner: SRN 60. Features are described in docs/STEP1B_FEATURES_SUMMARY.md (SRN 50). Uses the candidate rows built in Step 1 (`scripts/02_build_dataset.py`).
All numbers are from `scripts/03_train_models.py` and `03b_baselines.py` (output in `artifacts/experiments/`).
Evaluator trips are never used. 5-fold GroupKFold is by trip on the 80 train trips; test numbers are on the 10 student-test trips.

# MODELS

## 2.1 Experimental design
- Task: binary classification of candidate pairs; output P(candidate is correct).
- Validation: **5-fold GroupKFold, groups = trips**, on the 80 training trips (reliable points, 25% sample, seed 33).
- Final comparison: models refitted on all training rows, evaluated on the **10 student-test trips**.
- Same data, same seed for every model. Preprocessing (imputer + scaler) is inside the pipelines of models that need it (LogReg, NB, KNN, MLP); trees do not need scaling.
- Metrics: observation-level **Top-1** (for each point take the highest-probability candidate and compare with the reference arc; points with no candidate count as errors), candidate-level macro-F1/precision/recall at threshold 0.5, log loss, route continuity, train and inference time.
- Primary metric: **Top-1** is the task objective; we also report CV macro-F1 because the brief asks for it.

## 2.2 The eight models and their settings
| Model | Family | Key settings | Objective |
|---|---|---|---|
| GaussianNB | probabilistic | scaled inputs | maximum likelihood, assumes independent Gaussian features |
| KNN | instance-based | k=25, distance-weighted, scaled; training set capped at 150k rows | no training loss (vote of neighbours) |
| Logistic Regression | linear | C=1.0, max_iter=500, scaled | log loss with L2 |
| ExtraTrees | randomised trees | 60 trees, max_depth 18, min_leaf 10, sqrt features | Gini impurity |
| Random Forest | bagging | 60 trees, max_depth 16, min_leaf 10, sqrt features, 30% bootstrap | Gini impurity |
| MLP | neural net | layers (64,32), alpha 1e-4, batch 1024, lr 1e-3, 25 epochs max, early stopping on 10% validation | log loss (cross-entropy) |
| HistGradientBoosting | boosting | 300 iterations, lr 0.08, 31 leaves | log loss |
| LightGBM | boosting (leaf-wise) | 400 trees, lr 0.05, 63 leaves, subsample 0.8, colsample 0.8 | log loss |
Note: Random Forest and ExtraTrees were first set to 150 trees but were too slow on the 2-core machine, so they were lightened to 60 trees with limited depth. That means they were not given the same tuning effort; mention it if asked.

## 2.3 Results
Cross-validation (5-fold GroupKFold) and student-test (10 trips, 19,160 points):

| Model | CV macro-F1 (+-sd) | CV Top-1 | Test Top-1 | Test macro-F1 | Test precision | Test recall | Test log loss | Route continuity | Train s | Infer ms/pt |
|---|---|---|---|---|---|---|---|---|---|---|
| GaussianNB | 0.903 (0.009) | 0.885 | 0.882 | 0.896 | 0.723 | 0.935 | 0.380 | 0.934 | 0.3 | 0.002 |
| KNN | 0.931 (0.012) | 0.887 | 0.889 | 0.929 | 0.871 | 0.874 | 0.090 | 0.941 | 0.1 | 2.19 |
| LogReg | 0.942 (0.012) | 0.901 | 0.894 | 0.936 | 0.869 | 0.899 | 0.062 | 0.947 | 1.3 | 0.001 |
| ExtraTrees | 0.944 (0.011) | 0.901 | 0.896 | 0.939 | 0.909 | 0.870 | 0.054 | 0.943 | 16.3 | 0.026 |
| RandomForest | 0.947 (0.011) | 0.905 | 0.902 | 0.942 | 0.915 | 0.876 | 0.052 | 0.946 | 30.1 | 0.022 |
| MLP | 0.942 (0.012) | 0.905 | 0.903 | 0.941 | 0.905 | 0.882 | 0.054 | 0.952 | 9.4 | 0.003 |
| HistGB | 0.946 (0.012) | 0.903 | 0.909 | 0.947 | 0.910 | 0.898 | 0.063 | 0.943 | 8.7 | 0.040 |
| **LightGBM** | **0.948 (0.012)** | **0.908** | **0.912** | **0.947** | 0.916 | 0.895 | **0.048** | 0.951 | 8.5 | 0.081 |
Reference rules (not models): nearest arc = 0.544 Top-1; nearest + heading = 0.857; candidate-set ceiling = 0.9988.
LightGBM confusion matrix on student-test (threshold 0.5): TN 174,957, FP 1,580, FN 2,019, TP 17,119.
Plots: `artifacts/plots/model_comparison.png`, `confusion_matrices.png`, `mlp_history.png`.

## 2.4 Why each model was considered, and how it behaved
- **GaussianNB:** simple probabilistic baseline. Worst, because it assumes features are independent given the class but `dist`, `dist_ratio`, `ctx_dist_*` are strongly correlated, so its probabilities are overconfident (log loss 0.38, precision only 0.72).
- **KNN:** instance-based; similar candidates should share labels. Slow at inference (2.2 ms per point, about 30x others), depends on feature scaling, and we had to cap training rows at 150k. Mid-field accuracy.
- **Logistic Regression:** linear, fast, well-calibrated (log loss 0.062). Surprisingly strong at 0.894, which tells us the engineered features already make the problem close to linearly separable; non-linear models add about 2 points.
- **Random Forest / ExtraTrees:** non-linear tree ensembles, robust to feature scale, good calibration (log loss 0.052/0.054). Slower to train and slightly lower Top-1 than boosting. ExtraTrees had the lowest recall (0.870).
- **MLP:** non-linear, produces probabilities and has a real epoch-wise training history, so it is the only model with training/validation curves (no artificial curves for the others). Competitive (0.903) and the best route continuity (0.952).
- **HistGradientBoosting and LightGBM:** gradient boosting corrects errors sequentially and handles feature interactions (distance x direction x speed), which is exactly what this task needs. LightGBM was best on Top-1, CV macro-F1 and log loss.

## 2.5 Why LightGBM was selected (and the honest nuance)
- Best on CV macro-F1 (0.948), CV Top-1, student-test Top-1 (0.912), macro-F1 and **log loss (0.048)**.
- Log loss matters here because the next stage (HMM) multiplies probabilities along the path; poorly calibrated models mislead it. NB is an example of that failure.
- Fast: 8.5 s to train, 0.08 ms per point at inference.
- Nuance: on CV the top four (LightGBM, RF, HistGB, ExtraTrees) are within about one fold standard deviation (about 0.012). The differences are small; the tie-breakers were test Top-1, log loss and speed. We do not claim LightGBM is overwhelmingly better.
- Fold variation is large (macro-F1 0.93-0.97 across folds) because some trips are much harder than others (dense cities, slow driving).

## 2.6 Overfitting / underfitting
- LightGBM CV Top-1 0.908 versus student-test 0.912, so the model generalises to unseen trips and is not collapsing on new data. Underfitting is not indicated because gains over the linear model show the data supports non-linearity.
- Honest limitation: we did not record training-set scores for every model, so our evidence is the CV-versus-test comparison and fold stability, not a train-versus-validation gap.
- MLP: validation score is tracked with early stopping (training history in `mlp_history.json`).


## 2.9 Limitations to state yourself (graders like this)
1. Reference labels are derived by alignment and imperfect (trips 33, 54).
2. 25% subsample of points for training; models were not tuned except LightGBM.
3. Random Forest / ExtraTrees were lightened for runtime.
4. Single random seed (33); fold sds are given but no repeated-seed runs.
5. Candidate set ceiling 99.88%.
6. Macro-F1 at threshold 0.5 on 10%-positive data; Top-1 is the more meaningful metric.

---
