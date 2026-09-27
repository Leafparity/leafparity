"""XGBoost (gbtree, numerical splits) -> leafparity IR."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from ..ir import Model, Tree, UnsupportedModelError
from ..routers import Chain, XGBoostRouter

_LOGIT = {"binary:logistic", "reg:logistic"}
_LOG = {"count:poisson", "reg:gamma", "reg:tweedie", "survival:cox", "survival:aft"}


def _booster(obj):
    import xgboost as xgb
    if isinstance(obj, xgb.Booster):
        return obj
    if hasattr(obj, "get_booster"):
        return obj.get_booster()
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        b = xgb.Booster()
        b.load_model(str(obj))
        return b
    raise UnsupportedModelError(f"not an XGBoost model: {type(obj)!r}")


def _parse_floats(s) -> list:
    if isinstance(s, (int, float)):
        return [float(s)]
    s = str(s).strip()
    if s.startswith("["):
        s = s[1:-1]
    return [float(x) for x in s.split(",") if x.strip()]


def load_xgboost(obj: Any, user_dtype=np.float64) -> Model:
    booster = _booster(obj)
    j = json.loads(booster.save_raw("json"))
    learner = j["learner"]
    gb = learner["gradient_booster"]
    if gb.get("name") != "gbtree":
        raise UnsupportedModelError(
            f"XGBoost booster '{gb.get('name')}' is not supported (only 'gbtree'; 'dart' "
            "applies per-tree weights at prediction time and 'gblinear' has no trees)")
    model = gb["model"]
    params = learner["learner_model_param"]
    objective = learner["objective"]["name"]
    num_class = int(params.get("num_class", "0"))
    num_target = int(params.get("num_target", "1"))
    n_features = int(params["num_feature"])
    n_out = num_class if num_class > 1 else num_target
    tree_info = [int(x) for x in model["tree_info"]]

    feats, thrs, dls = [], [], []
    trees = []
    for t_idx, tr in enumerate(model["trees"]):
        tp = tr.get("tree_param", {})
        if int(tp.get("size_leaf_vector", "1") or 1) > 1:
            raise UnsupportedModelError("XGBoost multi-target (vector-leaf) trees are not supported")
        left = np.asarray(tr["left_children"], dtype=np.int64)
        right = np.asarray(tr["right_children"], dtype=np.int64)
        split_idx = np.asarray(tr["split_indices"], dtype=np.int64)
        cond = np.asarray(tr["split_conditions"], dtype=np.float64).astype(np.float32)
        dleft = np.asarray(tr["default_left"], dtype=bool)
        stype = np.asarray(tr.get("split_type", [0] * len(left)), dtype=np.int64)
        n = left.shape[0]
        is_leaf = left == -1
        if np.any(stype[~is_leaf] != 0):
            raise UnsupportedModelError("XGBoost categorical splits are not supported yet")
        feature = np.where(is_leaf, -1, split_idx)
        children = np.stack([left, right], axis=1)
        children[is_leaf] = -1
        value = np.zeros((n, n_out), dtype=np.float64)
        grp = tree_info[t_idx]
        value[is_leaf, grp] = cond[is_leaf].astype(np.float64)
        trees.append(Tree(children=children, feature=feature, value=value,
                          node_ids=np.arange(n, dtype=np.int64), tree_id=t_idx))
        feats.append(feature)
        thrs.append(np.where(is_leaf, np.float32(0), cond))
        dls.append(dleft)

    chain = Chain([("cast", None, np.float32)])
    router = XGBoostRouter(np.concatenate(feats) if feats else np.zeros(0, np.int64),
                           np.concatenate(thrs) if thrs else np.zeros(0, np.float32),
                           np.concatenate(dls) if dls else np.zeros(0, bool), chain)

    base_scores = _parse_floats(params["base_score"])
    if len(base_scores) == 1 and n_out > 1:
        base_scores = base_scores * n_out
    base = np.zeros(n_out, dtype=np.float64)
    for k in range(n_out):
        b = base_scores[k]
        if objective in _LOGIT:
            b = -math.log(1.0 / b - 1.0)
        elif objective in _LOG:
            b = math.log(b)
        base[k] = float(np.float32(b))

    if num_class > 1:
        task = "multiclass"
    elif objective.startswith("binary:") or objective == "reg:logistic":
        task = "binary"
    else:
        task = "regression"
    names = booster.feature_names
    m = Model(library="xgboost", task=task, n_features=n_features, n_outputs=n_out,
              trees=trees, base=base, router=router,
              description=f"XGBoost {objective}, {len(trees)} trees",
              feature_names=list(names) if names else None,
              accepts_nan=True, accepts_inf=False, accumulate_dtype=np.float32,
              raw_meaning="margin (output_margin=True)", source=booster)
    _calibrate_base(m, booster)
    best = booster.attributes().get("best_iteration")
    if best is not None:
        m.notes.append(f"booster has best_iteration={best}; analysis uses ALL trees - make sure "
                       "your serving code does too (sklearn-API predict() may stop early)")
    return m


def _calibrate_base(m: Model, booster) -> None:
    """Check the base margin against the real library; adopt the observed one if the
    objective's link function is not one we know."""
    import xgboost as xgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, m.n_features)).astype(np.float32)
    try:
        margin = booster.predict(xgb.DMatrix(X), output_margin=True)
        leaves = booster.predict(xgb.DMatrix(X), pred_leaf=True).astype(np.int64)
    except Exception:  # pragma: no cover - best effort
        return
    margin = np.asarray(margin, dtype=np.float64).reshape(X.shape[0], -1)
    leaves = leaves.reshape(X.shape[0], -1)
    s = np.zeros_like(margin)
    for t, tree in enumerate(m.trees):
        s += tree.value[leaves[:, t]]
    est = np.median(margin - s, axis=0)
    scale = np.maximum(1.0, np.abs(est))
    if np.any(np.abs(est - m.base) > 1e-4 * scale * max(1, len(m.trees))):
        m.notes.append("base margin taken from the library's own output (unknown link function)")
        m.base = est.astype(np.float32).astype(np.float64)
