"""scikit-learn trees, forests, gradient boosting (optionally inside a Pipeline with
monotone scalers in front) -> leafparity IR."""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..ir import Model, Tree, UnsupportedModelError
from ..routers import Chain, SklearnRouter


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


def _preprocessing_steps(steps, user_dtype, n_features) -> Tuple[list, list]:
    """Translate supported per-feature monotone transformers into Chain steps whose
    arithmetic is calibrated to match the installed scikit-learn bit-for-bit."""
    out: list = []
    desc: list = []
    D = np.dtype(user_dtype) if np.dtype(user_dtype) in (np.dtype(np.float32), np.dtype(np.float64)) \
        else np.dtype(np.float64)
    for name, tr in steps:
        if tr is None or tr == "passthrough":
            continue
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
        elif cls == "CastTransformer":
            D = np.dtype(getattr(tr, "dtype", np.float32))
            out += [("cast", None, D)]
            desc.append(cls)
            continue
        else:
            raise UnsupportedModelError(
                f"pipeline step '{name}' ({cls}) is not supported: only per-feature monotone "
                "scalers (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, CastTransformer) "
                "can be analysed exactly")
        if ops:
            out += _calibrate(tr, _step_variants(ops, D), D, n_features)
        desc.append(cls)
    return out, desc


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


def load_sklearn(obj: Any, user_dtype=np.float64) -> Model:
    pipeline = obj
    steps = []
    est = obj
    if hasattr(obj, "steps"):
        steps = obj.steps[:-1]
        est = obj.steps[-1][1]
    cls = type(est).__name__
    pre, pre_desc = _preprocessing_steps(steps, user_dtype, int(getattr(obj, 'n_features_in_', 0) or getattr(est, 'n_features_in_', 0)))
    chain = Chain(pre + [("cast", None, np.float32)])

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
              accepts_nan=_accepts(pipeline, n_features, np.nan), accepts_inf=False,
              accumulate_dtype=np.float64, raw_meaning=raw_meaning, source=pipeline)
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
