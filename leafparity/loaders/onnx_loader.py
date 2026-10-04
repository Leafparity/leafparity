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
from .columns import Columns

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


_SUPPORTED_NODES = ("only Cast, Identity, Scaler, Add/Sub/Mul/Div by a constant, ArrayFeatureExtractor "
                    "and Gather of constant columns, and Concat along the columns")


def _input_width(graph, name) -> int:
    for gi in graph.input:
        if gi.name == name:
            dims = gi.type.tensor_type.shape.dim
            if len(dims) == 2 and dims[1].dim_value > 0:
                return int(dims[1].dim_value)
    return 0


def _const_indices(inits, name, node) -> np.ndarray:
    if name not in inits:
        raise UnsupportedModelError(f"ONNX node '{node.op_type}' ({node.name}) selects columns that are not constant")
    idx = np.asarray(inits[name])
    if idx.ndim != 1 or idx.dtype.kind not in "iu":
        raise UnsupportedModelError(f"ONNX node '{node.op_type}' ({node.name}) must select a 1-D list of columns")
    return idx.astype(np.int64)


def _build_chain(proto, tree_node, n_features):
    """Walk from the tree node's input back to the graph input and describe, for every
    column the trees see, the graph input column it comes from and each elementwise step
    on the way, in the dtype the node computes in.  Anything else is refused."""
    graph = proto.graph
    producers = {o: n for n in graph.node for o in n.output}
    inits = _initializers(graph)
    graph_inputs = {i.name for i in graph.input} - set(inits)
    used: set = set()
    memo: Dict[str, Columns] = {}

    def columns_of(name: str) -> Columns:
        if name in memo:
            return memo[name].copy()
        if name in graph_inputs:
            in_dtype = _TENSOR_DTYPES.get(_elem_type(graph, name))
            if in_dtype not in (np.float32, np.float64):
                raise UnsupportedModelError("graph input must be a float or double tensor")
            c = Columns(_input_width(graph, name) or n_features, in_dtype)
            c.apply([("cast", None, in_dtype)])  # the feed itself
            used.add(name)
            memo[name] = c
            return c.copy()
        node = producers.get(name)
        if node is None:
            raise UnsupportedModelError(f"cannot trace tree input '{name}' back to a graph input")
        op = node.op_type
        a = _attrs(node)
        if op == "Identity":
            c = columns_of(node.input[0])
        elif op == "Cast":
            dt = _TENSOR_DTYPES.get(int(a["to"]))
            if dt not in (np.float32, np.float64):
                raise UnsupportedModelError("Cast to a non-float type in front of the trees")
            c = columns_of(node.input[0])
            c.apply([("cast", None, dt)])
        elif op == "Scaler":
            # onnxruntime ScalerOp<T>: y = float((x - offset) * scale), float constants, in T
            c = columns_of(node.input[0])
            T = c.dtype
            off = np.asarray(a.get("offset", [0.0]), np.float32).astype(T)
            sc = np.asarray(a.get("scale", [1.0]), np.float32).astype(T)
            c.apply([("sub", off, T), ("mul", sc, T), ("cast", None, np.float32)])
        elif op in ("Add", "Sub", "Mul", "Div") and node.input[1] in inits and node.input[0] not in inits:
            c = columns_of(node.input[0])
            c.apply([(op.lower(), _row_constant(inits[node.input[1]], node), c.dtype)])
        elif op in ("Add", "Mul") and node.input[0] in inits and node.input[1] not in inits:
            c = columns_of(node.input[1])
            c.apply([(op.lower(), _row_constant(inits[node.input[0]], node), c.dtype)])
        elif op == "ArrayFeatureExtractor":
            c = columns_of(node.input[0]).select(_const_indices(inits, node.input[1], node))
        elif op == "Gather" and int(a.get("axis", 0)) in (1, -1):
            c = columns_of(node.input[0]).select(_const_indices(inits, node.input[1], node))
        elif op == "Concat" and int(a.get("axis", 0)) in (1, -1):
            parts = [columns_of(i) for i in node.input]
            if len({p.dtype for p in parts}) != 1:
                raise UnsupportedModelError(f"ONNX node 'Concat' ({node.name}) joins different dtypes")
            c = Columns.concat(parts, parts[0].dtype)
        else:
            raise UnsupportedModelError(
                f"ONNX node '{op}' ({node.name}) in front of the tree ensemble is not supported "
                f"({_SUPPORTED_NODES})")
        memo[name] = c
        return c.copy()

    cols = columns_of(tree_node.input[0])
    if len(used) != 1:
        raise UnsupportedModelError("the trees read more than one graph input")
    name = used.pop()
    if cols.n < n_features:
        raise UnsupportedModelError("the trees use more columns than their input provides")
    in_dtype = np.dtype(_TENSOR_DTYPES[_elem_type(graph, name)])
    return cols, name, in_dtype, _input_width(graph, name) or cols.n


def _row_constant(c, node) -> np.ndarray:
    """An elementwise constant that applies one value per column (or one for all)."""
    c = np.asarray(c)
    if c.ndim > 2 or (c.ndim == 2 and c.shape[0] != 1):
        raise UnsupportedModelError(
            f"ONNX node '{node.op_type}' ({node.name}): constant of shape {c.shape} is not one value per column")
    return c.reshape(-1)


def _verify_preprocessing(proto, tree_node, input_name, in_dtype, width, cols: Columns, chain: Chain) -> None:
    """Run the graph's own preprocessing in onnxruntime and require the modelled chain to
    reproduce it bit for bit, column by column, on values of every magnitude, signed
    zeros, infinities and missing values."""
    if cols.src == list(range(cols.n)) and all(len(o) == 1 for o in cols.ops):
        return  # the trees read the graph input directly
    import onnx
    import onnxruntime as ort
    from onnx import helper
    g = proto.graph
    needed, kept = {tree_node.input[0]}, []
    for n in reversed(list(g.node)):
        if n is not tree_node and any(o in needed for o in n.output):
            kept.append(n)
            needed.update(n.input)
    elem = {np.dtype(np.float32): onnx.TensorProto.FLOAT, np.dtype(np.float64): onnx.TensorProto.DOUBLE}
    out = helper.make_tensor_value_info(tree_node.input[0], elem[cols.dtype], None)
    graph = helper.make_graph(list(reversed(kept)), "leafparity_preprocessing",
                              [i for i in g.input if i.name == input_name], [out],
                              initializer=[i for i in g.initializer if i.name in needed])
    sub = helper.make_model(graph, opset_imports=list(proto.opset_import))
    sub.ir_version = proto.ir_version
    rng = np.random.default_rng(54321)
    X = rng.normal(size=(4096, width)) * rng.choice([1e-30, 1e-3, 1.0, 1e3, 1e8, 1e30], size=(4096, width))
    special = rng.random(X.shape)
    X[special < 0.04] = 0.0
    X[(special >= 0.04) & (special < 0.08)] = -0.0
    X[(special >= 0.08) & (special < 0.1)] = np.inf
    X[(special >= 0.1) & (special < 0.12)] = -np.inf
    X[special > 0.95] = np.nan
    X = X.astype(in_dtype)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(sub.SerializeToString(), so, providers=["CPUExecutionProvider"])
        real = np.asarray(sess.run(None, {input_name: X})[0])
    except Exception as exc:
        raise UnsupportedModelError(f"cannot run the preprocessing in front of the trees: {exc}")
    got = np.stack([chain.apply(X[:, cols.src[k]], np.full(len(X), k)) for k in range(cols.n)], axis=1)
    if real.shape != got.shape or real.dtype != got.dtype or not np.array_equal(real, got, equal_nan=True):
        raise UnsupportedModelError(
            "the ONNX preprocessing in front of the trees could not be reproduced bit for bit "
            "with this onnxruntime, so it cannot be analysed exactly")


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
    cols, input_name, in_dtype, n_inputs = _build_chain(proto, tn, n_features)
    chain = cols.to_chain()
    _verify_preprocessing(proto, tn, input_name, in_dtype, max(n_inputs, max(cols.src) + 1), cols, chain)
    reordered = cols.src != list(range(cols.n))
    # the graph input's declared width wins if known
    n_features = max(n_features, n_inputs)
    tree_in = cols.dtype
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
    if reordered:
        m.map_columns_to_inputs(cols.src, n_inputs)
    m.onnx_info = {"input_name": input_name, "input_dtype": in_dtype, "tree_node": tn.name,
                   "op_type": tn.op_type, "post_transform": post, "aggregate": agg,
                   "chain": chain.describe()}
    return m
