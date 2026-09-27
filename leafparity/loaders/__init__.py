"""Loaders: turn a model object or file into a :class:`leafparity.ir.Model`."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..ir import UnsupportedModelError


def load_original(obj: Any, user_dtype=np.float64):
    """Load the *original* (training-library) model: XGBoost, LightGBM or scikit-learn.

    ``obj`` may be a live model object or a path (.json/.ubj XGBoost, .txt LightGBM,
    .pkl/.joblib pickled estimator).  Pickles execute code when loaded: only load
    files you trust.
    """
    if isinstance(obj, (str, os.PathLike)):
        path = os.fspath(obj)
        low = path.lower()
        if low.endswith((".pkl", ".pickle", ".joblib")):
            import joblib
            return load_original(joblib.load(path), user_dtype)
        if low.endswith((".json", ".ubj", ".ubjson", ".model", ".bin")):
            from .xgboost_loader import load_xgboost
            return load_xgboost(path, user_dtype)
        if low.endswith(".txt"):
            from .lightgbm_loader import load_lightgbm
            return load_lightgbm(path, user_dtype)
        raise UnsupportedModelError(f"cannot tell the model format of '{path}' from its extension")
    mod = type(obj).__module__
    if mod.startswith("xgboost"):
        from .xgboost_loader import load_xgboost
        return load_xgboost(obj, user_dtype)
    if mod.startswith("lightgbm"):
        from .lightgbm_loader import load_lightgbm
        return load_lightgbm(obj, user_dtype)
    if mod.startswith("sklearn") or hasattr(obj, "steps") or hasattr(obj, "tree_") or hasattr(obj, "estimators_"):
        from .sklearn_loader import load_sklearn
        return load_sklearn(obj, user_dtype)
    raise UnsupportedModelError(f"unsupported original model type {type(obj)!r}")


def load_onnx(obj: Any, user_dtype=np.float64):
    from .onnx_loader import load_onnx as _l
    return _l(obj, user_dtype)
