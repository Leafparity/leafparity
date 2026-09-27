"""ONNX (ai.onnx.ml TreeEnsembleRegressor / TreeEnsembleClassifier) -> leafparity IR.

Supported graphs: one tree-ensemble node whose input is the graph input, possibly
through per-feature monotone elementwise nodes (Cast, Identity, Scaler, and
Add/Sub/Mul/Div by a constant).  Anything else is refused rather than guessed.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..ir import Model, Tree, UnsupportedModelError
from ..routers import Chain, ONNX_MODES, OnnxRouter

_TENSOR_DTYPES = {1: np.float32, 11: np.float64, 10: np.float16, 6: np.int32, 7: np.int64}
TREE_OPS = ("TreeEnsembleRegressor", "TreeEnsembleClassifier")


def _load_proto(obj):
    import onnx
    if isinstance(obj, onnx.ModelProto):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return onnx.load_from_string(bytes(obj))
    return onnx.load(str(obj))


def _attrs(node) -> Dict[str, Any]:
    import onnx
    out = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if a.type == onnx.AttributeProto.TENSOR:
            v = onnx.numpy_helper.to_array(v)
        out[a.name] = v
    return out


def _initializers(graph) -> Dict[str, np.ndarray]:
    import onnx
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            a = _attrs(n)
            if "value" in a:
                inits[n.output[0]] = np.asarray(a["value"])
    return inits


def _elem_type(graph, name):
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.name == name:
            return vi.type.tensor_type.elem_type
    return None


def _per_feature(c: np.ndarray, n_features: int) -> np.ndarray:
    c = np.asarray(c).reshape(-1)
    if c.size == 1:
        return np.full(n_features, c[0])
    if c.size != n_features:
        raise UnsupportedModelError("elementwise constant does not broadcast per feature")
    return c


def _build_chain(proto, tree_node, n_features):
    """Walk from the tree node's input back to the graph input, then replay the
    elementwise steps forward tracking the dtype each one computes in."""
    graph = proto.graph
    producers = {o: n for n in graph.node for o in n.output}
    inits = _initializers(graph)
    graph_inputs = {i.name for i in graph.input} - set(inits)
    name = tree_node.input[0]
    rev: List = []   # abstract steps, reverse execution order
    while name not in graph_inputs:
        node = producers.get(name)
        if node is None:
            raise UnsupportedModelError(f"cannot trace tree input '{name}' back to a graph input")
        op = node.op_type
        a = _attrs(node)
        if op == "Identity":
            name = node.input[0]
            continue
        if op == "Cast":
            dt = _TENSOR_DTYPES.get(int(a["to"]))
            if dt not in (np.float32, np.float64):
                raise UnsupportedModelError("Cast to a non-float type in front of the trees")
            rev.append(("cast", dt))
            name = node.input[0]
            continue
        if op == "Scaler":
            off = _per_feature(np.asarray(a.get("offset", [0.0]), np.float32), n_features)
            sc = _per_feature(np.asarray(a.get("scale", [1.0]), np.float32), n_features)
            rev.append(("scaler", off, sc))
            name = node.input[0]
            continue
        if op in ("Add", "Sub", "Mul", "Div"):
            opn = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div"}[op]
            if node.input[1] in inits and node.input[0] not in inits:
                rev.append(("arith", opn, _per_feature(inits[node.input[1]], n_features)))
                name = node.input[0]
                continue
            if op in ("Add", "Mul") and node.input[0] in inits and node.input[1] not in inits:
                rev.append(("arith", opn, _per_feature(inits[node.input[0]], n_features)))
                name = node.input[1]
                continue
        raise UnsupportedModelError(
            f"ONNX node '{op}' in front of the tree ensemble is not supported "
            "(only Cast, Identity, Scaler and elementwise Add/Sub/Mul/Div by constants)")
    et = _elem_type(graph, name)
    in_dtype = _TENSOR_DTYPES.get(et)
    if in_dtype not in (np.float32, np.float64):
        raise UnsupportedModelError("graph input must be a float or double tensor")
    steps = [("cast", None, in_dtype)]
    cur = np.dtype(in_dtype)
    for st in reversed(rev):
        if st[0] == "cast":
            cur = np.dtype(st[1])
            steps.append(("cast", None, cur))
        elif st[0] == "scaler":
            # onnxruntime ScalerOp<T>: y = float((x - offset) * scale) computed in T
            steps.append(("sub", st[1], cur))
            steps.append(("mul", st[2], cur))
            cur = np.dtype(np.float32)
            steps.append(("cast", None, cur))
        else:
            steps.append((st[1], np.asarray(st[2]).astype(cur), cur))
    return Chain(steps), name, np.dtype(in_dtype), cur


def load_onnx(obj: Any, user_dtype=np.float64) -> Model:
    proto = _load_proto(obj)
    graph = proto.graph
    tree_nodes = [n for n in graph.node if n.op_type in TREE_OPS]
    others = [n for n in graph.node if n.op_type == "TreeEnsemble"]
    if others:
        raise UnsupportedModelError("ai.onnx.ml opset-5 'TreeEnsemble' is not supported yet; "
                                    "convert with target ai.onnx.ml opset <= 3")
    if len(tree_nodes) != 1:
        raise UnsupportedModelError(f"expected exactly one tree-ensemble node, found {len(tree_nodes)}")
    tn = tree_nodes[0]
    a = _attrs(tn)
    is_clf = tn.op_type == "TreeEnsembleClassifier"

    treeids = np.asarray(a["nodes_treeids"], dtype=np.int64)
    nodeids = np.asarray(a["nodes_nodeids"], dtype=np.int64)
    featids = np.asarray(a["nodes_featureids"], dtype=np.int64)
    modes = [m.decode() if isinstance(m, bytes) else str(m) for m in a["nodes_modes"]]
    if "nodes_values_as_tensor" in a:
        values = np.asarray(a["nodes_values_as_tensor"], dtype=np.float64)
        thr_dtype = np.float64
    else:
        values = np.asarray(a.get("nodes_values", np.zeros(len(treeids))), dtype=np.float32)
        thr_dtype = np.float32
    truen = np.asarray(a["nodes_truenodeids"], dtype=np.int64)
    falsen = np.asarray(a["nodes_falsenodeids"], dtype=np.int64)
    tracks = np.asarray(a.get("nodes_missing_value_tracks_true", np.zeros(len(treeids))), dtype=bool)
    if tracks.size == 0:
        tracks = np.zeros(len(treeids), dtype=bool)

    pre = "class_" if is_clf else "target_"
    t_tree = np.asarray(a[pre + "treeids"], dtype=np.int64)
    t_node = np.asarray(a[pre + "nodeids"], dtype=np.int64)
    t_ids = np.asarray(a[pre + "ids"], dtype=np.int64)
    if pre + "weights_as_tensor" in a:
        t_w = np.asarray(a[pre + "weights_as_tensor"], dtype=np.float64)
    else:
        t_w = np.asarray(a[pre + "weights"], dtype=np.float32).astype(np.float64)
    if "base_values_as_tensor" in a:
        base_vals = np.asarray(a["base_values_as_tensor"], dtype=np.float64)
    else:
        base_vals = np.asarray(a.get("base_values", []), dtype=np.float32).astype(np.float64)
    post = a.get("post_transform", b"NONE")
    post = post.decode() if isinstance(post, bytes) else str(post)
    agg = a.get("aggregate_function", b"SUM")
    agg = agg.decode() if isinstance(agg, bytes) else str(agg)
    if agg not in ("SUM", "AVERAGE"):
        raise UnsupportedModelError(f"aggregate_function={agg} is not supported")

    if is_clf:
        labels = a.get("classlabels_int64s", None)
        if labels is None or len(labels) == 0:
            labels = a.get("classlabels_strings", [])
        n_labels = len(labels)
        n_out = int(t_ids.max()) + 1 if t_ids.size else 1
        if n_labels == 2 and n_out == 1:
            task = "binary"
        elif n_out == n_labels:
            task = "multiclass" if n_labels > 2 else "binary"
            if n_labels == 2:
                raise UnsupportedModelError("binary classifier with two score columns is not supported yet")
        else:
            raise UnsupportedModelError("unusual class/score layout in TreeEnsembleClassifier")
    else:
        n_out = int(a.get("n_targets", int(t_ids.max()) + 1 if t_ids.size else 1))
        task = "regression"

    n_features = int(featids.max()) + 1 if featids.size else 0
    # the graph input's declared width wins if known
    for gi in graph.input:
        dims = gi.type.tensor_type.shape.dim
        if len(dims) == 2 and dims[1].dim_value > 0:
            n_features = max(n_features, int(dims[1].dim_value))
    chain, input_name, in_dtype, tree_in = _build_chain(proto, tn, n_features)
    compare_dtype = np.float64 if tree_in == np.float64 else np.float32

    # ---- build trees, in increasing tree id order
    uniq = sorted(set(treeids.tolist()))
    n_trees = len(uniq)
    scale = 1.0 / n_trees if agg == "AVERAGE" else 1.0
    # leaf weights
    leaf_w: Dict[tuple, np.ndarray] = {}
    for tr, nd, k, w in zip(t_tree, t_node, t_ids, t_w):
        key = (int(tr), int(nd))
        if key not in leaf_w:
            leaf_w[key] = np.zeros(n_out, dtype=np.float64)
        leaf_w[key][int(k)] += w * scale

    trees: List[Tree] = []
    feats, thrs, mds, trk = [], [], [], []
    by_tree: Dict[int, List[int]] = {t: [] for t in uniq}
    for i, t in enumerate(treeids.tolist()):
        by_tree[t].append(i)
    for t in uniq:
        rows = by_tree[t]
        ids = [int(nodeids[r]) for r in rows]
        loc = {nid: j for j, nid in enumerate(ids)}
        n = len(rows)
        children = np.full((n, 2), -1, dtype=np.int64)
        feature = np.full(n, -1, dtype=np.int64)
        value = np.zeros((n, n_out), dtype=np.float64)
        thr = np.zeros(n, dtype=thr_dtype)
        md = np.zeros(n, dtype=np.int64)
        tk = np.zeros(n, dtype=bool)
        is_child = np.zeros(n, dtype=bool)
        for j, r in enumerate(rows):
            mode = modes[r]
            if mode == "LEAF":
                value[j] = leaf_w.get((t, ids[j]), np.zeros(n_out))
                continue
            if mode not in ONNX_MODES:
                raise UnsupportedModelError(f"node mode {mode} not supported")
            try:
                children[j, 0] = loc[int(truen[r])]
                children[j, 1] = loc[int(falsen[r])]
            except KeyError:
                raise UnsupportedModelError(f"tree {t}: child id missing")
            is_child[children[j]] = True
            feature[j] = int(featids[r])
            thr[j] = values[r]
            md[j] = ONNX_MODES[mode]
            tk[j] = tracks[r]
        roots = np.nonzero(~is_child)[0]
        if roots.size != 1:
            raise UnsupportedModelError(f"tree {t} does not have exactly one root")
        root = int(roots[0])
        if root != 0:  # re-index so that the root is node 0
            perm = [root] + [j for j in range(n) if j != root]
            inv = np.empty(n, dtype=np.int64)
            inv[perm] = np.arange(n)
            children = np.where(children >= 0, inv[np.maximum(children, 0)], -1)[perm]
            feature, value, thr, md, tk = feature[perm], value[perm], thr[perm], md[perm], tk[perm]
            ids = [ids[p] for p in perm]
        trees.append(Tree(children=children, feature=feature, value=value,
                          node_ids=np.asarray(ids, dtype=np.int64), tree_id=t))
        feats.append(feature)
        thrs.append(thr)
        mds.append(md)
        trk.append(tk)

    router = OnnxRouter(np.concatenate(feats), np.concatenate(thrs), np.concatenate(mds),
                        np.concatenate(trk), chain, compare_dtype)
    base = np.zeros(n_out, dtype=np.float64)
    if base_vals.size:
        if base_vals.size != n_out:
            raise UnsupportedModelError("base_values length does not match the number of scores")
        base = base_vals.copy()
    raw = {"NONE": "raw score", "LOGISTIC": "margin (before LOGISTIC)",
           "SOFTMAX": "margin (before SOFTMAX)"}.get(post, f"score before {post}")
    m = Model(library="onnx", task=task, n_features=n_features, n_outputs=n_out, trees=trees,
              base=base, router=router,
              description=f"ONNX {tn.op_type} ({len(trees)} trees, post_transform={post}, input {in_dtype.name})",
              accepts_nan=True, accepts_inf=True, accumulate_dtype=np.float32,  # float32 scores/output
              raw_meaning=raw, source=proto)
    m.onnx_info = {"input_name": input_name, "input_dtype": in_dtype, "tree_node": tn.name,
                   "op_type": tn.op_type, "post_transform": post, "aggregate": agg,
                   "chain": chain.describe()}
    return m
