"""Run the *real* libraries.

leafparity's analysis is done on its own exact model of each runtime.  Everything
it reports is then re-checked by executing the actual library code (XGBoost,
LightGBM, scikit-learn, onnxruntime) on the witness inputs, and the test-suite
uses these functions to prove the exact model matches the libraries leaf for leaf.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .ir import Model, UnsupportedModelError


# =========================================================================== originals
def _booster_input(model: Model, X: np.ndarray) -> np.ndarray:
    """What an XGBoost / LightGBM model inside a scikit-learn Pipeline receives: X after
    every transform step, exactly as Pipeline.predict computes it."""
    pipeline = getattr(model, "pipeline", None)
    if pipeline is None:
        return X
    for _, tr in pipeline.steps[:-1]:
        if tr is None or tr == "passthrough":
            continue
        X = tr.transform(X)
    return X


def leaf_indices(model: Model, X: np.ndarray) -> np.ndarray:
    """Leaf reached in every tree, as local node indices of ``model.trees[t]``."""
    if model.library == "xgboost":
        import xgboost as xgb
        X = _booster_input(model, X)
        out = model.source.predict(xgb.DMatrix(X, missing=np.nan), pred_leaf=True)
        return np.asarray(out, dtype=np.int64).reshape(X.shape[0], -1)
    if model.library == "lightgbm":
        X = _booster_input(model, X)
        out = np.asarray(model.source.predict(X, pred_leaf=True), dtype=np.int64).reshape(X.shape[0], -1)
        res = np.empty_like(out)
        for t, tree in enumerate(model.trees):
            lut = {-1 - int(lab): i for i, lab in enumerate(tree.node_ids) if tree.feature[i] < 0}
            res[:, t] = [lut[int(v)] for v in out[:, t]]
        return res
    if model.library == "sklearn":
        est, Xt = _sk_split(model, X)
        members = _sk_members(est)
        return np.stack([m.apply(Xt) for m in members], axis=1).astype(np.int64)
    if model.library == "onnx":
        return _onnx_leaf_indices(model, X)
    raise UnsupportedModelError(model.library)


def raw_outputs(model: Model, X: np.ndarray) -> np.ndarray:
    """Canonical raw outputs (n, n_outputs) computed by the real library."""
    n = X.shape[0]
    if model.library == "xgboost":
        import xgboost as xgb
        X = _booster_input(model, X)
        out = model.source.predict(xgb.DMatrix(X, missing=np.nan), output_margin=True)
        return np.asarray(out, dtype=np.float64).reshape(n, -1)
    if model.library == "lightgbm":
        X = _booster_input(model, X)
        out = np.asarray(model.source.predict(X, raw_score=True), dtype=np.float64).reshape(n, -1)
        div = getattr(model, "lgb_average_divisor", None)
        return out / div if div else out
    if model.library == "sklearn":
        est = model.source.steps[-1][1] if hasattr(model.source, "steps") else model.source
        name = type(est).__name__
        if name.startswith("GradientBoosting") and hasattr(est, "classes_"):
            out = model.source.decision_function(X)
        elif hasattr(est, "classes_"):
            p = model.source.predict_proba(X)
            out = p[:, 1] if p.shape[1] == 2 else p
        else:
            out = model.source.predict(X)
        return np.asarray(out, dtype=np.float64).reshape(n, -1)
    if model.library == "onnx":
        return _onnx_run_raw(model, X)
    raise UnsupportedModelError(model.library)


def final_outputs(model: Model, X: np.ndarray) -> Dict[str, np.ndarray]:
    """What an application actually consumes: predictions / probabilities / labels."""
    n = X.shape[0]
    pipeline = getattr(model, "pipeline", None)
    if pipeline is not None:  # XGBoost / LightGBM inside a Pipeline: ask the Pipeline itself
        if model.task in ("binary", "multiclass"):
            p = np.asarray(pipeline.predict_proba(X), dtype=np.float64)
            return {"probability": p, "label": p.argmax(axis=1)}
        return {"prediction": np.asarray(pipeline.predict(X), dtype=np.float64).reshape(n, -1)}
    if model.library == "xgboost":
        import xgboost as xgb
        out = np.asarray(model.source.predict(xgb.DMatrix(X, missing=np.nan)), dtype=np.float64)
        if model.task == "binary":
            p = out.reshape(n)
            return {"probability": np.stack([1 - p, p], axis=1), "label": (p > 0.5).astype(np.int64)}
        if model.task == "multiclass":
            p = out.reshape(n, -1)
            return {"probability": p, "label": p.argmax(axis=1)}
        return {"prediction": out.reshape(n, -1)}
    if model.library == "lightgbm":
        out = np.asarray(model.source.predict(X), dtype=np.float64)
        if model.task == "binary":
            p = out.reshape(n)
            return {"probability": np.stack([1 - p, p], axis=1), "label": (p > 0.5).astype(np.int64)}
        if model.task == "multiclass":
            p = out.reshape(n, -1)
            return {"probability": p, "label": p.argmax(axis=1)}
        return {"prediction": out.reshape(n, -1)}
    if model.library == "sklearn":
        est = model.source
        if model.task in ("binary", "multiclass"):
            p = np.asarray(est.predict_proba(X), dtype=np.float64)
            return {"probability": p, "label": p.argmax(axis=1)}
        return {"prediction": np.asarray(est.predict(X), dtype=np.float64).reshape(n, -1)}
    if model.library == "onnx":
        return _onnx_run_final(model, X)
    raise UnsupportedModelError(model.library)


def _sk_split(model, X):
    src = model.source
    if hasattr(src, "steps"):
        Xt = X
        for _, tr in src.steps[:-1]:
            if tr is None or tr == "passthrough":
                continue
            Xt = tr.transform(Xt)
        return src.steps[-1][1], Xt
    return src, X


def _sk_members(est):
    name = type(est).__name__
    if name.startswith(("DecisionTree", "ExtraTree")) and not name.startswith("ExtraTrees"):
        return [est]
    if name.startswith("GradientBoosting"):
        st = est.estimators_
        return [st[i, k] for i in range(st.shape[0]) for k in range(st.shape[1])]
    return list(est.estimators_)


# =========================================================================== ONNX
def _session(model: Model, which: str, proto):
    """One onnxruntime session per (model, purpose), cached on the model object."""
    import onnxruntime as ort
    cache = model.__dict__.setdefault("_ort_sessions", {})
    if which not in cache:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        cache[which] = ort.InferenceSession(proto.SerializeToString(), so,
                                            providers=["CPUExecutionProvider"])
    return cache[which]


def _derived_regressor(model: Model, mode: str):
    """Copy of the ONNX graph where the tree node is replaced by a TreeEnsembleRegressor
    that outputs either leaf identities ('leaf') or the raw scores ('raw')."""
    import onnx
    from onnx import helper
    cache_name = f"_derived_{mode}"
    if hasattr(model, cache_name):
        return getattr(model, cache_name)
    proto = model.source
    g = proto.graph
    tn = [n for n in g.node if n.op_type in ("TreeEnsembleRegressor", "TreeEnsembleClassifier")][0]
    keep = {a.name: a for a in tn.attribute if a.name.startswith("nodes_")}
    is_clf = tn.op_type == "TreeEnsembleClassifier"
    attrs = {}
    from .loaders.onnx_loader import _attrs
    A = _attrs(tn)
    if mode == "leaf":
        treeids = list(A["nodes_treeids"])
        order = sorted(set(treeids))
        tt, tn_, ti, tw = [], [], [], []
        # leaf identity = position of the leaf in model.trees[t] (+1)
        for t_idx, tree in enumerate(model.trees):
            onnx_t = order[t_idx]
            for local in np.nonzero(tree.feature < 0)[0]:
                tt.append(int(onnx_t))
                tn_.append(int(tree.node_ids[local]))
                ti.append(t_idx)
                tw.append(float(local + 1))
        attrs.update(target_treeids=tt, target_nodeids=tn_, target_ids=ti, target_weights=tw,
                     n_targets=len(order), aggregate_function="SUM", post_transform="NONE")
    else:
        pre = "class_" if is_clf else "target_"
        attrs["target_treeids"] = list(A[pre + "treeids"])
        attrs["target_nodeids"] = list(A[pre + "nodeids"])
        attrs["target_ids"] = list(A[pre + "ids"])
        if pre + "weights_as_tensor" in A:
            attrs["target_weights_as_tensor"] = onnx.numpy_helper.from_array(
                np.asarray(A[pre + "weights_as_tensor"]))
        else:
            attrs["target_weights"] = [float(x) for x in A[pre + "weights"]]
        attrs["n_targets"] = model.n_outputs
        attrs["aggregate_function"] = A.get("aggregate_function", b"SUM")
        if isinstance(attrs["aggregate_function"], bytes):
            attrs["aggregate_function"] = attrs["aggregate_function"].decode()
        attrs["post_transform"] = "NONE"
        if "base_values_as_tensor" in A:
            attrs["base_values_as_tensor"] = onnx.numpy_helper.from_array(np.asarray(A["base_values_as_tensor"]))
        elif "base_values" in A and len(A["base_values"]):
            attrs["base_values"] = [float(x) for x in A["base_values"]]
    node = helper.make_node("TreeEnsembleRegressor", [tn.input[0]], ["lp_out"], domain="ai.onnx.ml",
                            **attrs)
    node.attribute.extend(keep.values())
    # keep every node needed to compute the tree input
    needed = set([tn.input[0]])
    kept = []
    for n in reversed(list(g.node)):
        if n is tn:
            continue
        if any(o in needed for o in n.output):
            kept.append(n)
            needed.update(n.input)
    kept = list(reversed(kept))
    out_vi = helper.make_tensor_value_info("lp_out", onnx.TensorProto.FLOAT, None)
    graph = helper.make_graph(kept + [node], "leafparity_derived", list(g.input), [out_vi],
                              initializer=list(g.initializer))
    ml_ver = max([o.version for o in proto.opset_import if o.domain == "ai.onnx.ml"] + [1])
    if mode == "raw" and ("target_weights_as_tensor" in attrs or "base_values_as_tensor" in attrs):
        ml_ver = max(ml_ver, 3)
    opsets = [o for o in proto.opset_import if o.domain not in ("ai.onnx.ml",)]
    opsets.append(helper.make_opsetid("ai.onnx.ml", min(ml_ver, 3)))
    m = helper.make_model(graph, opset_imports=opsets)
    m.ir_version = proto.ir_version
    setattr(model, cache_name, m)
    return m


def _feed(model: Model, X):
    info = model.onnx_info
    return {info["input_name"]: np.ascontiguousarray(X.astype(info["input_dtype"]))}


def _onnx_leaf_indices(model: Model, X):
    sess = _session(model, "leaf", _derived_regressor(model, "leaf"))
    out = sess.run(None, _feed(model, X))[0]
    return np.rint(np.asarray(out, dtype=np.float64)).astype(np.int64) - 1


def _onnx_run_raw(model: Model, X):
    sess = _session(model, "raw", _derived_regressor(model, "raw"))
    out = sess.run(None, _feed(model, X))[0]
    return np.asarray(out, dtype=np.float64).reshape(X.shape[0], -1)


def _onnx_run_final(model: Model, X):
    sess = _session(model, "final", model.source)
    outs = sess.run(None, _feed(model, X))
    n = X.shape[0]
    if model.task == "regression":
        return {"prediction": np.asarray(outs[0], dtype=np.float64).reshape(n, -1)}
    label = np.asarray(outs[0]).reshape(n)
    prob = outs[1] if len(outs) > 1 else None
    if isinstance(prob, list):  # ZipMap
        keys = sorted(prob[0].keys())
        prob = np.array([[row[k] for k in keys] for row in prob], dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64).reshape(n, -1)
    return {"probability": prob, "label_raw": label, "label": prob.argmax(axis=1)}
