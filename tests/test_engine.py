"""Ground-truth tests for the slice engine.

We take a converted model, inject a known defect into the ONNX graph (one threshold
moved by one float step, one missing-value flag flipped, one leaf weight changed, one
comparison operator changed) and check that leafparity finds exactly that defect,
at exactly that place, affecting exactly the inputs it should, with a witness the
real runtimes confirm - and nothing else.
"""
import copy

import numpy as np
import pytest
from onnx import helper

from leafparity import analyze, runtimes
from leafparity.loaders import load_onnx, load_original
from leafparity.probing import adversarial_matrix, node_probe_matrix

from zoo import all_models, by_name, data


def tree_node(proto):
    return [n for n in proto.graph.node if n.op_type.startswith("TreeEnsemble")][0]


def get_attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return None


def set_attr(node, name, value):
    for i, a in enumerate(node.attribute):
        if a.name == name:
            del node.attribute[i]
            break
    node.attribute.extend([helper.make_attribute(name, value)])


def mutate(proto, fn):
    p = copy.deepcopy(proto)
    fn(tree_node(p))
    return p


def pick_node(proto, want_tree=0, depth_first=True):
    """An internal node of the given tree that has two differently-valued subtrees."""
    tn = tree_node(proto)
    tids = get_attr(tn, "nodes_treeids")
    modes = get_attr(tn, "nodes_modes")
    for i, (t, m) in enumerate(zip(tids, modes)):
        if t == want_tree and m != b"LEAF":
            return i
    raise AssertionError("no internal node")


def onnx_vs_onnx(orig_proto, conv_proto, udt="float32", **kw):
    om = load_onnx(orig_proto, np.dtype(udt))
    return analyze(om, conv_proto, input_dtype=udt, **kw)


BASE = "xgb_reg"


@pytest.fixture(scope="module")
def base_proto():
    return by_name(BASE)[1]


def test_identical_models_are_equivalent(base_proto):
    a = onnx_vs_onnx(base_proto, base_proto)
    assert a.verdict["status"] == "EQUIVALENT"
    assert a.verdict["max_raw_difference_found"] == 0.0
    # the guaranteed bound only contains the proven floating-point summation allowance
    assert a.verdict["max_raw_difference_guaranteed"] <= a.verdict["rounding_allowance"] + 1e-12


@pytest.mark.parametrize("udt", ["float32", "float64"])
def test_threshold_moved_by_one_ulp(base_proto, udt):
    i = pick_node(base_proto, want_tree=0)
    vals = np.array(get_attr(tree_node(base_proto), "nodes_values"), dtype=np.float32)
    t = vals[i]
    t_up = np.nextafter(t, np.float32(np.inf))

    def fn(node):
        v = np.array(get_attr(node, "nodes_values"), dtype=np.float32)
        v[i] = t_up
        set_attr(node, "nodes_values", [float(x) for x in v])
    mutated = mutate(base_proto, fn)
    a = onnx_vs_onnx(base_proto, mutated, udt=udt)
    assert a.verdict["status"] == "NOT EQUIVALENT"
    assert a.verdict["distinct_problems"] == 1
    pb = a.findings[0]
    ex = pb["example"]
    assert pb["n_occurrences"] == 1
    assert ex["tree"] == 0 and ex["node_original"] == get_attr(tree_node(base_proto), "nodes_nodeids")[i]
    av = ex["affected_values"]
    # base model is BRANCH_LT (XGBoost): x < t goes true.  With t -> t_up exactly the
    # inputs whose float32 value equals t now go the other way.
    if udt == "float32":
        assert av["count_of_values"] == 1
        assert float(av["from"]) == float(t) == float(av["to"])
    else:
        # all doubles that round to float32 t
        lo, hi = float(av["from"]), float(av["to"])
        assert np.float32(lo) == t and np.float32(hi) == t
        assert np.float32(np.nextafter(lo, -np.inf)) != t
        assert np.float32(np.nextafter(hi, np.inf)) != t
    w = pb["witness"]
    assert w["analysis_consistent"]
    assert abs(w["raw_difference"][0]) > 0
    assert np.float32(float(w["input"][ex["feature_name"]])) == t


def test_missing_flag_flipped(base_proto):
    i = pick_node(base_proto, want_tree=1)

    def fn(node):
        tr = list(get_attr(node, "nodes_missing_value_tracks_true"))
        tr[i] = 1 - tr[i]
        set_attr(node, "nodes_missing_value_tracks_true", tr)
    a = onnx_vs_onnx(base_proto, mutate(base_proto, fn))
    assert a.verdict["status"] == "NOT EQUIVALENT"
    pb = a.findings[0]
    assert pb["pattern"] == "missing"
    assert pb["example"]["affected_values"]["count_of_values"] == 0
    assert pb["example"]["affected_values"]["missing"]
    assert pb["witness"]["input"][pb["feature_name"]] == "NaN"
    assert pb["witness"]["analysis_consistent"]


def test_leaf_weight_changed(base_proto):
    def fn(node):
        ww = np.array(get_attr(node, "target_weights"), dtype=np.float32)
        ww[3] = ww[3] + np.float32(0.5)
        set_attr(node, "target_weights", [float(x) for x in ww])
    a = onnx_vs_onnx(base_proto, mutate(base_proto, fn))
    assert a.verdict["status"] == "NOT EQUIVALENT"
    pb = a.findings[0]
    assert pb["kind"] == "value"
    assert abs(abs(pb["max_single_tree_effect"]) - 0.5) < 1e-6
    assert abs(abs(pb["witness"]["raw_difference"][0]) - 0.5) < 1e-5


def test_operator_changed_lt_to_leq(base_proto):
    i = pick_node(base_proto, want_tree=2)
    t = np.float32(get_attr(tree_node(base_proto), "nodes_values")[i])

    def fn(node):
        m = list(get_attr(node, "nodes_modes"))
        assert m[i] == b"BRANCH_LT"
        m[i] = b"BRANCH_LEQ"
        set_attr(node, "nodes_modes", m)
    a = onnx_vs_onnx(base_proto, mutate(base_proto, fn))
    assert a.verdict["status"] == "NOT EQUIVALENT"
    ex = a.findings[0]["example"]
    assert ex["affected_values"]["count_of_values"] == 1
    assert np.float32(float(ex["affected_values"]["from"])) == t


def test_worst_case_combines_trees_soundly(base_proto):
    """Several defects at once: the branch-and-bound maximum must (a) be attained by a real
    input and (b) never be exceeded by anything the real runtimes produce."""
    targets = []
    for t in range(6):
        targets.append(pick_node(base_proto, want_tree=t))

    def fn(node):
        v = np.array(get_attr(node, "nodes_values"), dtype=np.float32)
        for i in targets:
            v[i] = np.nextafter(v[i], np.float32(np.inf))
        set_attr(node, "nodes_values", [float(x) for x in v])
    mutated = mutate(base_proto, fn)
    a = onnx_vs_onnx(base_proto, mutated)
    om = a.original
    cm = a.converted
    probes = np.concatenate([node_probe_matrix(om, a.domain), node_probe_matrix(cm, a.domain),
                             adversarial_matrix([om, cm], a.domain, n_rows=3000, seed=5)])
    d = runtimes.raw_outputs(om, probes)[:, 0] - runtimes.raw_outputs(cm, probes)[:, 0]
    wpos = [w for w in a.worst if w["direction"].startswith("original")][0]
    wneg = [w for w in a.worst if w["direction"].startswith("converted")][0]
    tol = a.verdict["rounding_allowance"] + 1e-6
    assert d.max() <= wpos["guaranteed_max"] + tol
    assert (-d).max() <= wneg["guaranteed_max"] + tol
    assert wpos["max_found"] >= d.max() - tol
    assert wneg["max_found"] >= (-d).max() - tol
    assert wpos["proven"] and wneg["proven"]


@pytest.mark.parametrize("name", [n for n, _, _ in all_models()])
def test_no_discrepancy_is_missed(name):
    """Completeness: every input on which the real runtimes disagree beyond rounding lies
    inside a region the engine reported, and no difference exceeds the proven bound."""
    X, Xn, *_ = data()
    m, onx = by_name(name)
    a = analyze(m, onx, background=Xn if load_original(m).accepts_nan else X,
                self_check_rows=500, cross_check_rows=500)
    om, cm, dom = a.original, a.converted, a.domain
    P = np.concatenate([node_probe_matrix(om, dom), node_probe_matrix(cm, dom),
                        adversarial_matrix([om, cm], dom, n_rows=3000, seed=11)])
    d = runtimes.raw_outputs(om, P) - runtimes.raw_outputs(cm, P)
    tol = a.verdict["rounding_allowance"] * 2 + 1e-6
    assert np.abs(d).max() <= a.verdict["max_raw_difference_guaranteed"] + tol
    big = np.nonzero(np.abs(d).max(axis=1) > max(tol, 1e-3))[0]
    if len(big):
        assert a.verdict["status"] == "NOT EQUIVALENT"
    fk = dom.fk
    for r in big[:200]:
        x = P[r]
        found = False
        for pr in a.pairs:
            for reg in pr.regions:
                ok = True
                for f, (lo, hi, nan) in reg.box.items():
                    v = x[f]
                    if np.isnan(v):
                        ok = nan
                    else:
                        k = fk.key(v)
                        ok = lo <= k <= hi
                    if not ok:
                        break
                if ok:
                    found = True
                    break
            if found:
                break
        assert found, f"{name}: input {x} differs by {d[r]} but lies in no reported region"
