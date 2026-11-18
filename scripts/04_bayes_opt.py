"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (LightGBM).

Objective : mean observation-level Top-1 accuracy over 5-fold GroupKFold (groups = trips)
            on the 80 training trips (same reliable-label point subsample as script 03).
Then      : 5-fold CV + student-test comparison of untuned vs tuned with identical data,
            out-of-fold predictions for every training point (used to tune the HMM decoder),
            and export of the final model + inference config to artifacts/final/.
"""
import argparse, os, sys, time, json, pickle
import numpy as np, pandas as pd
import optuna, lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import GroupKFold
from pathsnap.features import FEATURES
from pathsnap.models import SEED, get_models
from pathsnap.metrics import top1, classification_report

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="artifacts"); ap.add_argument("--trials", type=int, default=30)
ap.add_argument("--timeout", type=int, default=1200); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--sub", type=float, default=0.25)
args = ap.parse_args()
out = os.path.join(args.art, "bayes_opt"); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(SEED)
full = pd.read_parquet(os.path.join(args.art, "features_train.parquet"))
te = pd.read_parquet(os.path.join(args.art, "features_student_test.parquet"))
tr = full[full.ref_dist <= 50]
pts = tr[["trip", "pid"]].drop_duplicates(); pts = pts[rng.random(len(pts)) < args.sub]   # identical to script 03
tr = tr.merge(pts, on=["trip", "pid"]).reset_index(drop=True)
X, y, g = tr[FEATURES].values.astype(np.float32), tr.y.values, tr.trip.values
n_test_pts = te.groupby("trip").pid.max().add(1).sum()


def cv(params, k, trial=None):
    t1, f1, ll = [], [], []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=k).split(X, y, g)):
        m = lgb.LGBMClassifier(**params); m.fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        cr = classification_report(y[b], p)
        t1.append(top1(tr.iloc[b].assign(p=p))[0]); f1.append(cr["macro_f1"]); ll.append(cr["log_loss"])
        if trial is not None:          # median pruning on the running fold mean
            trial.report(float(np.mean(t1)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return np.mean(t1), np.mean(f1), np.mean(ll), np.std(f1)


BASE = dict(random_state=SEED, verbose=-1, n_jobs=2, subsample_freq=1)


def objective(trial):
    params = dict(BASE,
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True))
    t0 = time.time()
    t1, f1, ll, _ = cv(params, args.folds, trial)
    trial.set_user_attr("cv_macro_f1", f1); trial.set_user_attr("cv_log_loss", ll); trial.set_user_attr("sec", time.time() - t0)
    return t1


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
                            study_name="pathsnap_lgbm")
# trial 0 = the untuned configuration, so the search is anchored at the baseline
_u = get_models()["LightGBM"][1].get_params()
study.enqueue_trial({k: _u[k] for k in ["n_estimators", "learning_rate", "num_leaves", "min_child_samples",
                                        "subsample", "colsample_bytree"]} | {"reg_lambda": 1e-3, "reg_alpha": 1e-3})
t0 = time.time()
study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)
opt_cost = time.time() - t0
study.trials_dataframe().to_csv(os.path.join(out, "optuna_trials.csv"), index=False)
best = dict(BASE, **study.best_params)
json.dump({"best_params": study.best_params, "best_cv_top1": study.best_value, "cv_folds": args.folds, "n_trials": len(study.trials),
           "optimisation_seconds": opt_cost}, open(os.path.join(out, "best_params.json"), "w"), indent=2)
vals = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
plt.figure(figsize=(7, 4)); plt.plot(vals, "o", alpha=.6, label="trial"); plt.plot(np.maximum.accumulate(vals), "r-", label="best so far")
plt.xlabel("trial"); plt.ylabel(f"{args.folds}-fold group CV Top-1"); plt.title("Optuna TPE optimisation history (LightGBM)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(out, "optuna_history.png"), dpi=120)

# ---- untuned vs tuned, identical protocol
untuned = get_models()["LightGBM"][1].get_params()
rows = []
for name, params in [("LightGBM_untuned", untuned), ("LightGBM_tuned", best)]:
    t1c, f1c, llc, f1s = cv(params, 5)
    m = lgb.LGBMClassifier(**params); t_ = time.time(); m.fit(X, y); ttrain = time.time() - t_
    t_ = time.time(); p = m.predict_proba(te[FEATURES].values.astype(np.float32))[:, 1]; tinf = time.time() - t_
    cr = classification_report(te.y.values, p)
    acc, _ = top1(te.assign(p=p), n_points=n_test_pts)
    rows.append(dict(model=name, cv_macro_f1=f1c, cv_macro_f1_std=f1s, cv_top1=t1c, cv_log_loss=llc,
                     test_top1=acc, test_macro_f1=cr["macro_f1"], test_log_loss=cr["log_loss"],
                     test_precision=cr["precision"], test_recall=cr["recall"],
                     train_time_s=ttrain, infer_ms_per_point=1000 * tinf / n_test_pts))
    pd.DataFrame({"trip": te.trip, "pid": te.pid, "arc": te.arc, "p": p}).to_parquet(os.path.join(out, f"pred_test_{name}.parquet"))
    if name == "LightGBM_tuned":
        final = m
pd.DataFrame(rows).to_csv(os.path.join(out, "tuned_vs_untuned.csv"), index=False); print(pd.DataFrame(rows).T)

# ---- out-of-fold predictions on ALL training points (for decoder tuning)
oof = np.zeros(len(full))
fg = full.trip.values
for a_trips, b_trips in GroupKFold(n_splits=5).split(np.zeros(len(pts)), groups=pts.trip.values):
    tr_trips = set(pts.trip.values[a_trips])
    msk = np.isin(g, list(tr_trips))
    m = lgb.LGBMClassifier(**best); m.fit(X[msk], y[msk])
    hold = ~np.isin(fg, list(tr_trips))
    hold &= np.isin(fg, list(set(pts.trip.values[b_trips])))
    oof[hold] = m.predict_proba(full.loc[hold, FEATURES].values.astype(np.float32))[:, 1]
pd.DataFrame({"trip": full.trip, "pid": full.pid, "arc": full.arc, "p": oof}).to_parquet(os.path.join(out, "oof_train_tuned.parquet"))

os.makedirs(os.path.join(args.art, "final"), exist_ok=True)
pickle.dump(final, open(os.path.join(args.art, "final", "model.pkl"), "wb"))
final.booster_.save_model(os.path.join(args.art, "final", "model_lgbm.txt"))   # portable across versions
print("saved final model")
