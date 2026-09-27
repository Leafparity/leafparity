"""LightGBM (numerical splits) -> leafparity IR."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..ir import Model, Tree, UnsupportedModelError
from ..routers import Chain, LightGBMRouter, MISSING_NAN, MISSING_NONE, MISSING_ZERO

_MT = {"None": MISSING_NONE, "Zero": MISSING_ZERO, "NaN": MISSING_NAN}


def _booster(obj):
    import lightgbm as lgb
    if isinstance(obj, lgb.Booster):
        return obj
    if hasattr(obj, "booster_"):
        return obj.booster_
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        return lgb.Booster(model_file=str(obj))
    raise UnsupportedModelError(f"not a LightGBM model: {type(obj)!r}")


def load_lightgbm(obj: Any, user_dtype=np.float64) -> Model:
    booster = _booster(obj)
    # dump_model() uses best_iteration when present, exactly like Booster.predict()
    d = booster.dump_model()
    K = int(d.get("num_tree_per_iteration", 1))
    n_features = int(d["max_feature_idx"]) + 1
    objective = str(d.get("objective", ""))
    average = bool(d.get("average_output", False))
    infos = d["tree_info"]
    n_iter = max(1, len(infos) // K)

    feats, thrs, dls, mts = [], [], [], []
    trees = []
    for t_idx, info in enumerate(infos):
        # pre-order flattening
        order, idx_of = [], {}
        todo = [info["tree_structure"]]
        while todo:
            n = todo.pop()
            idx_of[id(n)] = len(order)
            order.append(n)
            if "split_index" in n:
                todo.append(n["right_child"])
                todo.append(n["left_child"])
        nn = len(order)
        ch = np.full((nn, 2), -1, dtype=np.int64)
        fe = np.full(nn, -1, dtype=np.int64)
        va = np.zeros((nn, K), dtype=np.float64)
        th = np.zeros(nn, dtype=np.float64)
        de = np.zeros(nn, dtype=bool)
        mi = np.zeros(nn, dtype=np.int64)
        labels = np.zeros(nn, dtype=np.int64)
        k = t_idx % K
        for i, n in enumerate(order):
            if "split_index" in n:
                if n.get("decision_type", "<=") != "<=":
                    raise UnsupportedModelError("LightGBM categorical splits are not supported yet")
                ch[i, 0] = idx_of[id(n["left_child"])]
                ch[i, 1] = idx_of[id(n["right_child"])]
                fe[i] = int(n["split_feature"])
                th[i] = float(n["threshold"])
                de[i] = bool(n["default_left"])
                mi[i] = _MT[str(n.get("missing_type", "None"))]
                labels[i] = int(n["split_index"])
            else:
                if "leaf_coeff" in n or "leaf_features" in n:
                    raise UnsupportedModelError("LightGBM linear trees are not supported")
                v = float(n.get("leaf_value", 0.0))
                if average:
                    v = v / n_iter
                va[i, k] = v
                labels[i] = -1 - int(n.get("leaf_index", 0))
        trees.append(Tree(children=ch, feature=fe, value=va, node_ids=labels, tree_id=t_idx))
        feats.append(fe)
        thrs.append(th)
        dls.append(de)
        mts.append(mi)

    chain = Chain([("cast", None, np.float64)])
    router = LightGBMRouter(np.concatenate(feats), np.concatenate(thrs), np.concatenate(dls),
                            np.concatenate(mts), chain)
    if objective.startswith("binary"):
        task = "binary"
    elif objective.startswith("multiclass"):
        task = "multiclass"
    else:
        task = "regression"
    m = Model(library="lightgbm", task=task, n_features=n_features, n_outputs=K, trees=trees,
              base=np.zeros(K), router=router,
              description=f"LightGBM {objective.split(' ')[0] or 'model'}, {len(trees)} trees",
              feature_names=list(d.get("feature_names") or []) or None,
              accepts_nan=True, accepts_inf=True, accumulate_dtype=np.float64,
              raw_meaning="raw score (raw_score=True)", source=booster)
    # LightGBM quirk: in random-forest mode predict() averages over iterations but
    # predict(raw_score=True) returns the plain sum; the canonical raw output here is the
    # averaged value (what predict() and every converter compute).
    m.lgb_average_divisor = n_iter if average else None
    if booster.best_iteration and 0 < booster.best_iteration < booster.current_iteration():
        m.notes.append(f"booster.best_iteration={booster.best_iteration}: LightGBM's predict() "
                       "and this analysis use only those iterations - check your converter did too")
    return m
