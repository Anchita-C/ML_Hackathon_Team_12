# Step 3: Hyperparameter tuning and HMM route reconstruction
Owner: SRN 9. Scripts: `scripts/04_bayes_opt.py`, `scripts/05_hmm_eval.py`; library: `pathsnap/hmm.py`.
Evaluator trips are never used for tuning or selection. Tuning and decoder-parameter choice use the 80 train trips only; the 10 student-test trips are used for reporting.

## 1. Bayesian hyperparameter tuning (LightGBM, the model chosen in Step 2)
- **Method:** Optuna TPE (Bayesian optimisation) with median pruning, seed 33.
- **Objective:** mean Top-1 accuracy over 5-fold GroupKFold (groups = trips) on the train trips. Splitting by trip prevents leakage.
- **Search space:** n_estimators 100-600, learning_rate 0.01-0.2 (log), num_leaves 15-255 (log), min_child_samples 10-300 (log), subsample 0.5-1.0, colsample_bytree 0.4-1.0, reg_lambda and reg_alpha 0.001-10 (log).
- **Trial 0 is the untuned baseline**, so the search can never end below it. 40 trials: 10 completed, 30 pruned. Total 1,135.9 s.
- **Best parameters:** n_estimators 600, learning_rate 0.0371, num_leaves 239, min_child_samples 105, subsample 0.557, colsample_bytree 0.515, reg_lambda 0.002, reg_alpha 6.42.

| | CV Top-1 | Test Top-1 | Test macro-F1 | Test log loss |
|---|---|---|---|---|
| Untuned | 0.9084 | 0.9120 | 0.9474 | 0.0483 |
| Tuned | 0.9090 | 0.9174 | 0.9484 | 0.0470 |

Honest reading: the gain is small (+0.5 points Top-1 on test, +0.06 on CV). The default LightGBM was already near a plateau, so most of the final accuracy comes from the features and the HMM, not from tuning.
Run history kept for transparency: `bayes_opt_run1_cv3/` is an earlier run that optimised a 3-fold objective but was judged with 5 folds; its tuned model was worse on test (0.9072 vs 0.9120 untuned). We fixed the mismatch (5-fold objective, baseline as trial 0, pruning) and re-ran.

## 2. HMM / Viterbi decoder (`pathsnap/hmm.py`)
Why: the classifier scores each point separately, but a vehicle follows a connected route. The HMM chooses the best connected sequence of roads.
- **Hidden states:** the candidate arcs of each GPS point (the arc id and projection position are kept, so the candidate-to-road correspondence is preserved).
- **Emission:** `w_e * log(classifier probability)` from the tuned LightGBM.
- **Transition:** `-|d_route - d_gps| / beta` (Newson and Krumm, 2009). d_route is the network distance between the two projections (bounded Dijkstra on the directed road graph); d_gps is the straight-line distance between the two GPS fixes. Unreachable pairs get a fixed log penalty (-30).
- **Robustness:** gaps over 60 s or no reachable transition restart the chain; stationary points (speed below 4 m/s) have their emissions down-weighted to 0.1; small backward moves on one arc (up to 40 m) are tolerated. Points with no candidate get no prediction and count as wrong.
- **Final settings:** beta 6.0, w_e 0.5, back_tol 40, stop_speed 4.0, stop_w 0.1, gap_s 60, unreach -30 (`artifacts/final/config.json`).

### How the decoder settings were chosen (no test data)
Grid over beta (3, 6, 12), w_e (0.5, 1.0), back_tol (10, 40), stop_speed (0, 4) on the **train trips' out-of-fold probabilities**: `artifacts/hmm/decoder_grid_train_oof.csv`. Best: beta 6, w_e 0.5, back_tol 40, stop_speed 4 = 0.9377 train OOF Top-1.
A variant that uses the signed along-arc movement in the transition was tried and was worse (0.9313 vs 0.9377 OOF), so it was rejected.
`experiments/hmm_grid_student_test.csv` is an earlier exploratory grid on student-test; it was not used to pick the final settings (beta 3 would have scored slightly higher there, 0.9427, but we kept the train-selected beta 6).

### Result on the 10 student-test trips
| Method | Top-1 | Route F1 | Route continuity |
|---|---|---|---|
| Classifier only (tuned LightGBM) | 0.9174 | 0.9579 | 0.9503 |
| Classifier + HMM | 0.9450 | 0.9827 | 0.9796 |
All 10 trips improved (`artifacts/hmm/student_test_clf_vs_hmm_per_trip.csv`, plot `artifacts/plots/hmm_per_trip.png`).

## 3. Limitations to say yourself
1. Labels are derived by us from the supplied route, not independent ground truth.
2. About 44% of the remaining errors are the wrong direction on a two-way road at low speed; some may be label ambiguity.
3. The tuning gain is small; the HMM is where most of the improvement comes from.
4. Decoder settings come from train out-of-fold data; the true evaluator trips may behave differently.
5. HMM decoding alone takes 0.02 to 0.4 s per student-test trip; the whole API call (features + model + HMM) took 0.2 to 3.3 s per test trip.
