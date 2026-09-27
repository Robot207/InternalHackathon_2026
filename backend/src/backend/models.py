from __future__ import annotations

import numpy as np

from .config import MODELS


import warnings


def make_model(name: str):
    if name in ("lightgbm", "spatial_gradient_boosting"):
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "random_forest":
        warnings.warn(
            "Random Forest baseline is deprecated in favor of Spatial LightGBM (Gradient Boosting).",
            DeprecationWarning,
            stacklevel=2,
        )
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42
        )
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, random_state=42
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                early_stopping=True,
                max_iter=300,
                random_state=42,
            ),
        )
    raise ValueError(f"unknown model: {name}. available: {list(MODELS)}")


def feature_importance(model, names: list[str]) -> list[dict]:
    est = model
    if hasattr(model, "named_steps"):
        est = list(model.named_steps.values())[-1]
    importances = None
    if hasattr(est, "booster_"):
        try:
            importances = np.asarray(est.booster_.feature_importance(importance_type="gain"), dtype=float)
        except Exception:
            importances = None
    if importances is None and hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coefs_"):
        importances = np.mean(np.abs(np.asarray(est.coefs_[0], dtype=float)), axis=0)
    elif hasattr(est, "coef_"):
        importances = np.abs(np.asarray(est.coef_, dtype=float)).reshape(-1)
    if importances is None or len(importances) != len(names):
        return []
    order = np.argsort(importances)[::-1]
    total = float(importances.sum()) or 1.0
    return [
        {"feature": names[i], "importance": round(float(importances[i] / total), 4)}
        for i in order
    ]


def extract_shap_importance(model, names: list[str], X_sample: np.ndarray | None = None) -> list[dict]:
    """Extract Tree SHAP feature importance weights for the trained LightGBM/GBDT model."""
    est = model
    if hasattr(model, "named_steps"):
        est = list(model.named_steps.values())[-1]

    if X_sample is not None and len(X_sample) > 0:
        try:
            import shap
            sub = X_sample[: min(len(X_sample), 250)]
            explainer = shap.TreeExplainer(est)
            vals = explainer.shap_values(sub)
            if isinstance(vals, list):
                vals = vals[0]
            mean_abs = np.mean(np.abs(vals), axis=0)
            total = float(np.sum(mean_abs)) or 1.0
            order = np.argsort(mean_abs)[::-1]
            return [
                {
                    "feature": names[i],
                    "importance": round(float(mean_abs[i] / total), 4),
                    "method": "tree_shap",
                }
                for i in order
            ]
        except Exception:
            pass

    base = feature_importance(model, names)
    for b in base:
        b["method"] = "gain"
    return base
