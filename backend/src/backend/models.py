from __future__ import annotations

import numpy as np

from .config import MODELS


def make_model(name: str):
    if name == "random_forest":
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
    if name == "ensemble":
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor
        from xgboost import XGBRegressor

        rf = RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=42
        )
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        return VotingRegressor(estimators=[("rf", rf), ("xgb", xgb)], weights=[0.45, 0.55])
    raise ValueError(f"unknown model: {name}. available: {list(MODELS)}")


def feature_importance(model, names: list[str]) -> list[dict]:
    est = model
    if hasattr(model, "named_steps"):
        est = list(model.named_steps.values())[-1]
    importances = None
    if hasattr(est, "estimators_"):
        # For VotingRegressor / Ensemble: average normalized importances across sub-estimators
        sub_imps = []
        for sub_est in est.estimators_:
            sub_res = feature_importance(sub_est, names)
            if sub_res:
                sub_dict = {x["feature"]: x["importance"] for x in sub_res}
                sub_imps.append([sub_dict.get(n, 0.0) for n in names])
        if sub_imps:
            importances = np.mean(sub_imps, axis=0)
    elif hasattr(est, "feature_importances_"):
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

