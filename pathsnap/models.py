"""Model zoo: >=5 algorithm families, each wrapped so it outputs P(candidate is correct)."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

SEED = 33  # team seed


def get_models():
    return {
        "LogReg": ("Linear", make_pipeline(SimpleImputer(), StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=500, class_weight=None))),
        "GaussianNB": ("Probabilistic", make_pipeline(SimpleImputer(), StandardScaler(), GaussianNB())),
        "KNN": ("Instance-based", make_pipeline(SimpleImputer(), StandardScaler(),
                 KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
        "RandomForest": ("Bagging ensemble", RandomForestClassifier(n_estimators=60, max_depth=16, min_samples_leaf=10,
                          max_features="sqrt", max_samples=0.3, n_jobs=-1, random_state=SEED)),
        "ExtraTrees": ("Randomised trees", ExtraTreesClassifier(n_estimators=60, max_depth=18, min_samples_leaf=10,
                        max_features="sqrt", n_jobs=-1, random_state=SEED)),
        "HistGB": ("Boosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                    max_leaf_nodes=31, early_stopping=False, random_state=SEED)),
        "LightGBM": ("Boosting (leaf-wise)", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=2)),
        "MLP": ("Neural network", make_pipeline(SimpleImputer(), StandardScaler(),
                 MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                               max_iter=25, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                               random_state=SEED))),
    }
