"""Model zoo for the test-suite: (name, original model, converted ONNX proto)."""
from __future__ import annotations

import functools
import warnings

import numpy as np

warnings.filterwarnings("ignore")

F = 5


@functools.lru_cache(maxsize=None)
def data(seed: int = 0, n: int = 2500, nan_frac: float = 0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, F)) * np.array([1, 10, 100, 1000, 1e5])
    X[:, 2] = np.rint(X[:, 2])            # an integer-valued feature
    X[rng.random(n) < 0.1, 1] = 0.0  # a feature with many exact zeros
    Xn = X.copy()
    Xn[rng.random(X.shape) < nan_frac] = np.nan
    y = X[:, 0] + 0.3 * X[:, 1] / 10 + X[:, 4] / 1e5 + (X[:, 2] > 5)
    yb = (y > 0).astype(int)
    ym = np.digitize(y, [-0.7, 0.7])
    return X, Xn, y, yb, ym


def _fl():
    from onnxmltools.convert.common.data_types import FloatTensorType
    return [("X", FloatTensorType([None, F]))]


def xgb_models():
    import onnxmltools
    import xgboost as xgb
    X, Xn, y, yb, ym = data()
    out = []
    m = xgb.XGBRegressor(n_estimators=25, max_depth=4).fit(Xn, y)
    out.append(("xgb_reg", m, onnxmltools.convert_xgboost(m, initial_types=_fl(), target_opset=15)))
    m = xgb.XGBClassifier(n_estimators=25, max_depth=4).fit(Xn, yb)
    out.append(("xgb_bin", m, onnxmltools.convert_xgboost(m, initial_types=_fl(), target_opset=15)))
    m = xgb.XGBClassifier(n_estimators=10, max_depth=3).fit(Xn, ym)
    out.append(("xgb_multi", m, onnxmltools.convert_xgboost(m, initial_types=_fl(), target_opset=15)))
    return out


def lgb_models():
    import lightgbm as lgb
    import onnxmltools
    X, Xn, y, yb, ym = data()
    conv = lambda m, **kw: onnxmltools.convert_lightgbm(m, initial_types=_fl(), target_opset=15, **kw)
    out = []
    m = lgb.LGBMRegressor(n_estimators=25, num_leaves=15, verbose=-1).fit(Xn, y)
    out.append(("lgb_reg_nan", m, conv(m)))
    m = lgb.LGBMRegressor(n_estimators=25, num_leaves=15, verbose=-1).fit(X, y)
    out.append(("lgb_reg_none", m, conv(m)))
    m = lgb.LGBMRegressor(n_estimators=25, num_leaves=15, verbose=-1, zero_as_missing=True).fit(X, y)
    out.append(("lgb_reg_zero", m, conv(m)))
    m = lgb.LGBMClassifier(n_estimators=20, num_leaves=15, verbose=-1).fit(Xn, yb)
    out.append(("lgb_bin", m, conv(m, zipmap=False)))
    m = lgb.LGBMClassifier(n_estimators=10, num_leaves=7, verbose=-1).fit(Xn, ym)
    out.append(("lgb_multi", m, conv(m, zipmap=False)))
    m = lgb.LGBMRegressor(boosting_type="rf", n_estimators=15, num_leaves=15, bagging_freq=1,
                          bagging_fraction=0.7, verbose=-1).fit(Xn, y)
    out.append(("lgb_rf", m, conv(m)))
    return out


def sk_models():
    from skl2onnx import to_onnx
    from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                  GradientBoostingRegressor, RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    X, Xn, y, yb, ym = data()
    x32 = X[:1].astype(np.float32)
    nz = lambda m: {id(m): {"zipmap": False}}
    out = []
    m = DecisionTreeRegressor(max_depth=7, random_state=0).fit(Xn, y)
    out.append(("sk_dt_reg", m, to_onnx(m, x32)))
    m = DecisionTreeClassifier(max_depth=6, random_state=0).fit(Xn, yb)
    out.append(("sk_dt_bin", m, to_onnx(m, x32, options=nz(m))))
    m = RandomForestRegressor(8, max_depth=6, random_state=0).fit(Xn, y)
    out.append(("sk_rf_reg", m, to_onnx(m, x32)))
    m = RandomForestClassifier(8, max_depth=5, random_state=0).fit(Xn, ym)
    out.append(("sk_rf_multi", m, to_onnx(m, x32, options=nz(m))))
    m = ExtraTreesClassifier(8, max_depth=5, random_state=0).fit(Xn, yb)
    out.append(("sk_et_bin", m, to_onnx(m, x32, options=nz(m))))
    m = GradientBoostingRegressor(n_estimators=15, max_depth=3, random_state=0).fit(X, y)
    out.append(("sk_gb_reg", m, to_onnx(m, x32)))
    m = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, yb)
    out.append(("sk_gb_bin", m, to_onnx(m, x32, options=nz(m))))
    m = GradientBoostingClassifier(n_estimators=6, max_depth=2, random_state=0).fit(X, ym)
    out.append(("sk_gb_multi", m, to_onnx(m, x32, options=nz(m))))
    for sc in (StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler()):
        p = Pipeline([("s", sc), ("dt", DecisionTreeRegressor(max_depth=7, random_state=0))]).fit(X, y)
        out.append((f"sk_pipe_{type(sc).__name__}", p, to_onnx(p, x32)))
    for sc in (StandardScaler(), MinMaxScaler(), RobustScaler()):
        p = Pipeline([("scaler", sc), ("model", RandomForestClassifier(6, max_depth=5, random_state=0))])
        p.fit(Xn, yb)
        out.append((f"sk_pipe_rf_{type(sc).__name__}", p, to_onnx(p, x32, options=nz(p.steps[-1][1]))))
    return out


@functools.lru_cache(maxsize=None)
def all_models():
    return tuple(xgb_models() + lgb_models() + sk_models())


def by_name(name):
    for n, m, o in all_models():
        if n == name:
            return m, o
    raise KeyError(name)
