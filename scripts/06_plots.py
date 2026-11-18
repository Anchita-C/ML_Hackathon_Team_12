"""Comparison plots from recorded experiment results (no synthetic numbers)."""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ART = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
out = os.path.join(ART, "plots"); os.makedirs(out, exist_ok=True)
frames = [pd.read_csv(f) for f in glob.glob(os.path.join(ART, "experiments", "model_comparison*.csv"))]
df = pd.concat(frames).drop_duplicates("model", keep="last").sort_values("test_top1").reset_index(drop=True)
df.to_csv(os.path.join(ART, "experiments", "all_models_comparison.csv"), index=False)
cols = ["model", "family", "cv_macro_f1", "cv_macro_f1_std", "cv_top1", "cv_log_loss", "test_top1", "test_macro_f1",
        "test_precision", "test_recall", "test_log_loss", "test_route_continuity", "train_time_s", "infer_ms_per_point"]
print(df[cols].round(4).to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].barh(df.model, df.cv_macro_f1, xerr=df.cv_macro_f1_std, color="#4e79a7"); ax[0].set_xlim(0.85, 0.98)
ax[0].set_title("5-fold GroupKFold CV macro-F1 (±1 sd over folds)")
y = np.arange(len(df)); ax[1].barh(y - .2, df.test_top1, .4, label="Top-1 accuracy", color="#e15759")
ax[1].barh(y + .2, df.test_macro_f1, .4, label="Macro-F1", color="#59a14f"); ax[1].set_yticks(y, df.model); ax[1].set_xlim(0.85, 0.96)
ax[1].legend(loc="lower right"); ax[1].set_title("Student-test: Top-1 and macro-F1")
ax[2].barh(df.model, df.test_log_loss, color="#f28e2b"); ax[2].set_xscale("log"); ax[2].set_title("Student-test log loss (lower is better)")
plt.tight_layout(); plt.savefig(os.path.join(out, "model_comparison.png"), dpi=130)

fig, axs = plt.subplots(2, (len(df) + 1) // 2, figsize=(4 * ((len(df) + 1) // 2), 7))
for a, (_, r) in zip(axs.ravel(), df.iterrows()):
    cm = np.array(json.loads(r.test_confusion)); cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i,j]}\n({cmn[i,j]:.2f})", ha="center", va="center", fontsize=9, color="white" if cmn[i, j] > .5 else "black")
    a.set_xticks([0, 1], ["pred 0", "pred 1"]); a.set_yticks([0, 1], ["true 0", "true 1"]); a.set_title(r.model, fontsize=10)
for a in axs.ravel()[len(df):]: a.axis("off")
plt.suptitle("Candidate-level confusion matrices on student-test (threshold 0.5)"); plt.tight_layout()
plt.savefig(os.path.join(out, "confusion_matrices.png"), dpi=120)

h = os.path.join(ART, "experiments", "mlp_history.json")
if os.path.exists(h):
    hh = json.load(open(h)); fig, a = plt.subplots(figsize=(6, 4)); a.plot(hh["train_loss"], "o-", label="training log-loss")
    a2 = a.twinx(); a2.plot(hh["val_score"], "s--", color="C1", label="validation accuracy (10% hold-out)")
    a.set_xlabel("epoch"); a.set_ylabel("train loss"); a2.set_ylabel("val accuracy"); a.set_title("MLP training history (actual)")
    fig.legend(loc="center right"); plt.tight_layout(); plt.savefig(os.path.join(out, "mlp_history.png"), dpi=120)

tv = os.path.join(ART, "bayes_opt", "tuned_vs_untuned.csv")
if os.path.exists(tv):
    t = pd.read_csv(tv); m = ["cv_macro_f1", "test_top1", "test_macro_f1"]
    fig, a = plt.subplots(1, 2, figsize=(11, 4)); x = np.arange(len(m))
    for i, (_, r) in enumerate(t.iterrows()):
        a[0].bar(x + i * .38, r[m].values.astype(float), .38, label=r.model)
    a[0].set_xticks(x + .19, m); a[0].set_ylim(.88, .97); a[0].legend(); a[0].set_title("Untuned vs tuned LightGBM")
    a[1].bar(t.model, t.test_log_loss, color=["#bab0ac", "#4e79a7"]); a[1].set_title("Student-test log loss")
    plt.tight_layout(); plt.savefig(os.path.join(out, "tuned_vs_untuned.png"), dpi=120)

hs = os.path.join(ART, "hmm", "student_test_clf_vs_hmm_per_trip.csv")
if os.path.exists(hs):
    p = pd.read_csv(hs).pivot(index="trip", columns="method", values="top1")
    ax = p[["classifier_only", "classifier_hmm"]].plot.bar(figsize=(10, 4), color=["#f28e2b", "#e15759"])
    ax.set_ylim(.7, 1); ax.set_title("Student-test Top-1 per trip: classifier-only vs classifier + HMM/Viterbi"); plt.tight_layout()
    plt.savefig(os.path.join(out, "hmm_per_trip.png"), dpi=120)
print("plots written to", out)
