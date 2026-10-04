"""scikit-learn trees, forests, gradient boosting, and XGBoost / LightGBM estimators (optionally
inside a Pipeline with monotone scalers or a ColumnTransformer of them in front) -> leafparity IR."""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..ir import Model, Tree, UnsupportedModelError
from ..routers import Chain, SklearnRouter
from .columns import Columns


def _step_variants(ops, D):
    """Two ways scikit-learn may evaluate an in-place op on an array of dtype D:
    in D with the constant cast to D ('native'), or in float64 then rounded to D
    ('wide').  Which one applies differs between transformers and versions, so the
    right one is chosen by calibration against the real transformer."""
    native = [(op, c, D) for op, c in ops]
    wide = []
    for op, c in ops:
        wide += [(op, c, np.float64), ("cast", None, D)]
    return [native, wide]


def _calibrate(tr, variants, D, n_features):
    rng = np.random.default_rng(12345)
    n = 4096
    X = rng.normal(size=(n, n_features)) * rng.choice([1e-3, 1.0, 1e2, 1e5, 1e8], size=(n, n_features))
    X = X.astype(D)
    try:
        ref = np.asarray(tr.transform(X))
    except Exception as exc:  # pragma: no cover
        raise UnsupportedModelError(f"cannot calibrate {type(tr).__name__}: {exc}")
    feat = np.broadcast_to(np.arange(n_features), X.shape).reshape(-1)
    for steps in variants:
        got = Chain(steps).apply(X.reshape(-1), feat).reshape(X.shape)
        if got.dtype == ref.dtype and np.array_equal(got, ref, equal_nan=True):
            return steps
    raise UnsupportedModelError(
        f"{type(tr).__name__}: could not reproduce its arithmetic bit-for-bit with this "
        "scikit-learn version, so it cannot be analysed exactly")


_SCALERS = ("StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler")
_ONLY = ("only StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, CastTransformer, "
         "'passthrough' and a ColumnTransformer of those scalers and 'passthrough' can be "
         "analysed exactly")


def _is_identity(tr) -> bool:
    """'passthrough' (a fitted ColumnTransformer stores it as FunctionTransformer(func=None))."""
    if tr is None or (isinstance(tr, str) and tr == "passthrough"):
        return True
    return type(tr).__name__ == "FunctionTransformer" and getattr(tr, "func", 0) is None


def _scaler_ops(tr) -> list:
    """The in-place operations a fitted scaler performs, in order, with its float64 constants."""
    cls = type(tr).__name__
    ops = []
    if cls == "StandardScaler":
        if getattr(tr, "with_mean", True) and tr.mean_ is not None:
            ops.append(("sub", np.asarray(tr.mean_, np.float64)))
        if getattr(tr, "with_std", True) and tr.scale_ is not None:
            ops.append(("div", np.asarray(tr.scale_, np.float64)))
    elif cls == "MinMaxScaler":
        if getattr(tr, "clip", False):
            raise UnsupportedModelError("MinMaxScaler(clip=True) is not supported yet")
        ops = [("mul", np.asarray(tr.scale_, np.float64)), ("add", np.asarray(tr.min_, np.float64))]
    elif cls == "MaxAbsScaler":
        ops = [("div", np.asarray(tr.scale_, np.float64))]
    elif cls == "RobustScaler":
        if getattr(tr, "with_centering", True) and tr.center_ is not None:
            ops.append(("sub", np.asarray(tr.center_, np.float64)))
        if getattr(tr, "with_scaling", True) and tr.scale_ is not None:
            ops.append(("div", np.asarray(tr.scale_, np.float64)))
    return ops


def _calibrated(tr, D, n) -> list:
    ops = _scaler_ops(tr)
    return _calibrate(tr, _step_variants(ops, D), D, n) if ops else []


def _column_selection(name, ct, tname, columns, n) -> list:
    if callable(columns) or isinstance(columns, str) or np.asarray(columns).dtype.kind in "OUS":
        raise UnsupportedModelError(
            f"pipeline step '{name}' (ColumnTransformer): transformer '{tname}' selects columns "
            "by name or by a callable; leafparity runs the Pipeline on plain numeric arrays, so "
            "select the columns by position")
    known = getattr(ct, "_transformer_to_input_indices", None)
    if known is not None and tname in known:
        return [int(i) for i in known[tname]]
    return [int(i) for i in np.atleast_1d(np.arange(n)[columns])]


def _column_transformer(name, ct, cols: Columns, D) -> Columns:
    """A ColumnTransformer of scalers and 'passthrough' on whole columns: each output
    column is one input column, with that transformer's own calibrated arithmetic."""
    if getattr(ct, "transformer_weights", None):
        raise UnsupportedModelError(
            f"pipeline step '{name}' (ColumnTransformer) uses transformer_weights, which is not supported")
    if getattr(ct, "sparse_output_", False):
        raise UnsupportedModelError(f"pipeline step '{name}' (ColumnTransformer) has sparse output")
    out_idx = getattr(ct, "output_indices_", None)
    if out_idx is None:  # pragma: no cover - scikit-learn < 1.0
        raise UnsupportedModelError(
            f"pipeline step '{name}' (ColumnTransformer): this scikit-learn version does not "
            "record its output columns")
    placed = []
    for tname, tr, columns in ct.transformers_:
        sl = out_idx.get(tname)
        if sl is None or sl.stop <= sl.start:  # 'drop', or no columns
            continue
        part = cols.select(_column_selection(name, ct, tname, columns, cols.n))
        if _is_identity(tr):
            pass
        elif type(tr).__name__ in _SCALERS:
            part.apply(_calibrated(tr, D, part.n))
        else:
            raise UnsupportedModelError(
                f"pipeline step '{name}' (ColumnTransformer): transformer '{tname}' "
                f"({type(tr).__name__}) is not supported: inside a ColumnTransformer only "
                "StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler and 'passthrough' "
                "can be analysed exactly")
        if part.n != sl.stop - sl.start:
            raise UnsupportedModelError(
                f"pipeline step '{name}' (ColumnTransformer): transformer '{tname}' does not map "
                "its columns one to one")
        placed.append((sl.start, part))
    placed.sort(key=lambda sp: sp[0])
    width = 0
    for start, part in placed:
        if start != width:
            raise UnsupportedModelError(f"pipeline step '{name}' (ColumnTransformer): output columns overlap")
        width += part.n
    parts = [p for _, p in placed]
    dtype = np.result_type(*[p.dtype for p in parts]) if parts else D
    return Columns.concat(parts, dtype)


def _preprocessing(steps, user_dtype, n_features) -> Tuple[Columns, list]:
    """The columns the final estimator sees: for each, its input column and the exact
    arithmetic of every step on the way, calibrated against the installed scikit-learn."""
    desc: list = []
    D = np.dtype(user_dtype) if np.dtype(user_dtype) in (np.dtype(np.float32), np.dtype(np.float64)) \
        else np.dtype(np.float64)
    cols = Columns(n_features, D)
    for name, tr in steps:
        if _is_identity(tr):
            continue
        cls = type(tr).__name__
        if cls == "ColumnTransformer":
            cols = _column_transformer(name, tr, cols, D)
        elif cls == "CastTransformer":
            D = np.dtype(getattr(tr, "dtype", np.float32))
            cols.apply([("cast", None, D)])
        elif cls in _SCALERS:
            cols.apply(_calibrated(tr, D, cols.n))
        else:
            raise UnsupportedModelError(f"pipeline step '{name}' ({cls}) is not supported: {_ONLY}")
        desc.append(cls)
    return cols, desc


def _verify_preprocessing(steps, cols: Columns, chain: Chain, user_dtype, n_features, with_nan) -> None:
    """The modelled preprocessing must reproduce the real transform steps bit for bit,
    column by column, on values of every magnitude, signed zeros and missing values."""
    if cols.is_identity():
        return
    rng = np.random.default_rng(54321)
    n = 4096
    X = rng.normal(size=(n, n_features)) * rng.choice([1e-30, 1e-3, 1.0, 1e3, 1e8, 1e30], size=(n, n_features))
    special = rng.random(X.shape)
    X[special < 0.04] = 0.0
    X[(special >= 0.04) & (special < 0.08)] = -0.0
    if with_nan:
        X[special > 0.95] = np.nan
    X = X.astype(user_dtype)
    ref = X
    try:
        for _, tr in steps:
            if not _is_identity(tr):
                ref = tr.transform(ref)
    except Exception as exc:
        raise UnsupportedModelError(f"cannot run the Pipeline's transform steps on a numeric array: {exc}")
    ref = np.asarray(ref)
    rows = np.arange(n)
    got = np.stack([chain.apply(X[rows, cols.src[k]], np.full(n, k)) for k in range(cols.n)], axis=1)
    if ref.shape != got.shape or ref.dtype != got.dtype or not np.array_equal(ref, got, equal_nan=True):
        raise UnsupportedModelError(
            "the Pipeline's preprocessing could not be reproduced bit for bit with this "
            "scikit-learn version, so it cannot be analysed exactly")


def _tree_arrays(tree_):
    left = np.asarray(tree_.children_left, dtype=np.int64)
    right = np.asarray(tree_.children_right, dtype=np.int64)
    is_leaf = left == -1
    feature = np.where(is_leaf, -1, np.asarray(tree_.feature, dtype=np.int64))
    children = np.stack([left, right], axis=1)
    children[is_leaf] = -1
    thr = np.where(is_leaf, 0.0, np.asarray(tree_.threshold, dtype=np.float64))
    mgl = getattr(tree_, "missing_go_to_left", None)
    mgl = np.zeros(left.shape[0], bool) if mgl is None else np.asarray(mgl, dtype=bool)
    return children, feature, thr, mgl, is_leaf


class _Pre:
    def __init__(self, columns: Columns, chain: Chain, desc: list):
        self.columns, self.chain, self.desc = columns, chain, desc


def _pipeline_chain(steps, user_dtype, n_features, accepts_nan) -> _Pre:
    cols, desc = _preprocessing(steps, user_dtype, n_features)
    chain = cols.to_chain()
    D0 = np.dtype(user_dtype) if np.dtype(user_dtype) in (np.dtype(np.float32), np.dtype(np.float64)) \
        else np.dtype(np.float64)
    _verify_preprocessing(steps, cols, chain, D0, n_features, accepts_nan)
    return _Pre(cols, chain, desc)


def _load_booster_pipeline(pipeline, est, steps, user_dtype) -> Model:
    """A Pipeline whose last step is an XGBoost or LightGBM estimator (scikit-learn API):
    the booster's own model, with the Pipeline's preprocessing in front of its routing."""
    if type(est).__module__.startswith("xgboost"):
        from .xgboost_loader import load_xgboost as load
    else:
        from .lightgbm_loader import load_lightgbm as load
    try:
        m = load(est, user_dtype)
    except UnsupportedModelError as exc:
        raise UnsupportedModelError(
            f"pipeline step '{pipeline.steps[-1][0]}' ({type(est).__name__}): {exc}") from exc
    n_features = int(getattr(pipeline, "n_features_in_", 0) or m.n_features)
    accepts_nan = _accepts(pipeline, n_features, np.nan)
    pre = _pipeline_chain(steps, user_dtype, n_features, accepts_nan)
    m.router.chain = pre.chain.then(m.router.chain)
    m.pipeline = pipeline  # the real runtime: Pipeline transforms, then the booster
    if pre.columns.src == list(range(pre.columns.n)):
        m.n_features = n_features
    else:
        m.map_columns_to_inputs(pre.columns.src, n_features)
    names = getattr(pipeline, "feature_names_in_", None)
    m.feature_names = [str(x) for x in names] if names is not None else None
    m.accepts_nan = m.accepts_nan and accepts_nan
    m.accepts_inf = m.accepts_inf and _accepts(pipeline, n_features, np.inf)
    pre_desc = pre.desc
    m.description = "scikit-learn Pipeline: " + " -> ".join(pre_desc + [m.description])
    if pre_desc:
        m.notes.append("preprocessing modelled exactly: " + ", ".join(pre_desc))
    return m


def load_sklearn(obj: Any, user_dtype=np.float64) -> Model:
    pipeline = obj
    steps = []
    est = obj
    if hasattr(obj, "steps"):
        steps = obj.steps[:-1]
        est = obj.steps[-1][1]
        if type(est).__module__.startswith(("xgboost", "lightgbm")):
            return _load_booster_pipeline(obj, est, steps, user_dtype)
    cls = type(est).__name__
    n_inputs = int(getattr(obj, 'n_features_in_', 0) or getattr(est, 'n_features_in_', 0))
    accepts_nan = _accepts(pipeline, n_inputs, np.nan)
    pre = _pipeline_chain(steps, user_dtype, n_inputs, accepts_nan)
    pre_desc = pre.desc
    chain = pre.chain.then(Chain([("cast", None, np.float32)]))

    is_clf = hasattr(est, "classes_")
    base = None
    task = "regression"
    raw_meaning = "prediction"

    if cls in ("DecisionTreeRegressor", "ExtraTreeRegressor", "DecisionTreeClassifier",
               "ExtraTreeClassifier"):
        members = [est]
        scale = 1.0
    elif cls in ("RandomForestRegressor", "ExtraTreesRegressor", "RandomForestClassifier",
                 "ExtraTreesClassifier"):
        members = list(est.estimators_)
        scale = 1.0 / len(members)
    elif cls in ("GradientBoostingRegressor", "GradientBoostingClassifier"):
        members = None
    elif hasattr(obj, "steps"):
        raise UnsupportedModelError(
            f"pipeline step '{obj.steps[-1][0]}' ({cls}) is not supported as the final estimator: "
            "use a scikit-learn tree model, XGBoost or LightGBM")
    else:
        raise UnsupportedModelError(f"scikit-learn estimator {cls} is not supported yet")

    if members is not None:
        n_outputs_sk = int(getattr(est, "n_outputs_", 1))
        if is_clf:
            if n_outputs_sk != 1:
                raise UnsupportedModelError("multi-output classifiers are not supported")
            n_classes = len(est.classes_)
            n_out = 1 if n_classes == 2 else n_classes
            task = "binary" if n_classes == 2 else "multiclass"
            raw_meaning = "probability of class 1" if n_classes == 2 else "class probabilities"
        else:
            n_out = n_outputs_sk
        base = np.zeros(n_out)
        trees = []
        feats, thrs, mgls = [], [], []
        for t_idx, m in enumerate(members):
            children, feature, thr, mgl, is_leaf = _tree_arrays(m.tree_)
            val = np.asarray(m.tree_.value, dtype=np.float64)
            n = feature.shape[0]
            value = np.zeros((n, n_out), dtype=np.float64)
            if is_clf:
                v = val[:, 0, :n_classes]
                s = v.sum(axis=1, keepdims=True)
                s[s == 0] = 1.0
                p = v / s
                if n_classes == 2:
                    value[:, 0] = p[:, 1] * scale
                else:
                    value[:, :] = p * scale
            else:
                value[:, :] = val[:, :, 0] * scale
            value[~is_leaf] = 0.0
            trees.append(Tree(children=children, feature=feature, value=value,
                              node_ids=np.arange(n, dtype=np.int64), tree_id=t_idx))
            feats.append(feature)
            thrs.append(thr)
            mgls.append(mgl)
    else:
        stages = est.estimators_  # (n_stages, K)
        K = stages.shape[1]
        lr = float(est.learning_rate)
        n_out = K
        if is_clf:
            task = "binary" if len(est.classes_) == 2 else "multiclass"
            raw_meaning = "decision_function (raw log-odds)"
        trees, feats, thrs, mgls = [], [], [], []
        t_idx = 0
        for i in range(stages.shape[0]):
            for k in range(K):
                m = stages[i, k]
                children, feature, thr, mgl, is_leaf = _tree_arrays(m.tree_)
                n = feature.shape[0]
                value = np.zeros((n, n_out), dtype=np.float64)
                value[is_leaf, k] = lr * np.asarray(m.tree_.value, dtype=np.float64)[is_leaf, 0, 0]
                trees.append(Tree(children=children, feature=feature, value=value,
                                  node_ids=np.arange(n, dtype=np.int64), tree_id=t_idx))
                feats.append(feature)
                thrs.append(thr)
                mgls.append(mgl)
                t_idx += 1
        base = _gb_init(est, pipeline, n_out)

    router = SklearnRouter(np.concatenate(feats), np.concatenate(thrs), np.concatenate(mgls), chain)
    n_features = int(getattr(pipeline, "n_features_in_", getattr(est, "n_features_in_", 0)))
    names = getattr(pipeline, "feature_names_in_", None)
    desc = cls if not pre_desc else " -> ".join(pre_desc + [cls])
    m = Model(library="sklearn", task=task, n_features=n_features, n_outputs=n_out, trees=trees,
              base=np.asarray(base, dtype=np.float64), router=router,
              description=f"scikit-learn {desc}, {len(trees)} trees",
              feature_names=[str(x) for x in names] if names is not None else None,
              accepts_nan=accepts_nan, accepts_inf=False,
              accumulate_dtype=np.float64, raw_meaning=raw_meaning, source=pipeline)
    if pre.columns.src != list(range(pre.columns.n)):
        m.map_columns_to_inputs(pre.columns.src, n_features)
    if pre_desc:
        m.notes.append("preprocessing modelled exactly: " + ", ".join(pre_desc))
    return m


def _gb_init(est, pipeline, n_out):
    x0 = np.zeros((1, est.n_features_in_), dtype=np.float32)
    try:
        raw = est._raw_predict_init(x0)
        return np.asarray(raw, dtype=np.float64).reshape(-1)[:n_out]
    except Exception as exc:  # pragma: no cover - private API changed
        raise UnsupportedModelError(f"cannot read GradientBoosting init prediction: {exc}")


def _accepts(pipeline, n_features, value) -> bool:
    if n_features <= 0:
        return False
    x = np.zeros((1, n_features), dtype=np.float64)
    x[0, 0] = value
    try:
        if hasattr(pipeline, "predict_proba"):
            pipeline.predict_proba(x)
        else:
            pipeline.predict(x)
        return True
    except Exception:
        return False
