"""Non-learned reference baselines on student_test (for context, not counted as models)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.metrics import top1
te = pd.read_parquet("artifacts/features_student_test.parquet")
n = te.groupby("trip").pid.max().add(1).sum()
res = {}
res["nearest_arc"] = top1(te.assign(p=-te.dist - 1e-6 * te.arc), n_points=n)[0]
res["nearest_arc_plus_heading"] = top1(te.assign(p=-te.dist - 10 * (1 - te.head_cos)), n_points=n)[0]
res["oracle_candidate_recall"] = te.groupby(["trip", "pid"]).y.max().sum() / n
pd.Series(res).to_csv("artifacts/experiments/baselines.csv"); print(pd.Series(res).round(4))
