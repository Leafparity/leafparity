"""Regression tests for issues found by an independent review (hand-built ONNX models)."""
import numpy as np
import pytest

from leafparity import analyze
from leafparity.engine import make_domain
from leafparity.loaders import load_onnx

from onnxbuild import build, run


def L(w):
    return ("leaf", w)


def eq(f, v, yes, no):  # x_f == v -> yes, else no (NaN -> no)
    return (f, "BRANCH_LT", v, 0, no, (f, "BRANCH_LEQ", v, 0, yes, no))


def test_verdict_counts_every_problem_even_if_none_listed():
    po = build([(0, "BRANCH_LT", 0.5, 0, L(0.0), L(100.0))], 1)
    pc = build([(0, "BRANCH_LT", 0.5, 1, L(0.0), L(100.0))], 1)  # NaN routed differently
    a = analyze(load_onnx(po, np.float32), pc, input_dtype="float32", max_findings=0)
    assert a.verdict["status"] == "NOT EQUIVALENT"
    assert a.findings == []


def test_worst_case_with_interval_plus_nan_regions():
    c = float(np.float32(0.5))
    orig, conv = [], []
    orig.append(eq(3, 3.0, (0, "BRANCH_LT", c, 0, L(0), L(10)), L(0)))
    conv.append(eq(3, 3.0, (0, "BRANCH_LEQ", c, 1, L(0), L(10)), L(0)))
    orig.append((0, "BRANCH_LT", c, 0, L(0), L(9)))
    conv.append((0, "BRANCH_LEQ", c, 0, L(0), L(9)))
    for f in (1, 2):
        orig.append((0, "BRANCH_LT", 0.0, 0, L(0), eq(f, 1.0, L(8), L(0))))
        conv.append((0, "BRANCH_LT", 0.0, 1, L(0), eq(f, 1.0, L(8), L(0))))
    for f, t in ((2, 1.0), (1, 2.0)):
        orig.append(eq(f, 2.0, eq(3, t, L(12), L(1)), L(1)))
        conv.append(eq(f, 2.0, eq(3, t, L(1), L(1)), L(1)))
    po, pc = build(orig, 4), build(conv, 4)
    a = analyze(load_onnx(po, np.float32), pc, input_dtype="float32")
    x = np.array([[np.nan, 1.0, 1.0, 3.0]], dtype=np.float32)
    real = (run(po, x) - run(pc, x))[0]
    wpos = [w for w in a.worst if w["direction"].startswith("original")][0]
    assert real == 26.0
    assert wpos["guaranteed_max"] >= real - 1e-6
    assert wpos["max_found"] >= real - 1e-6


def test_all_regions_negative_pair_does_not_hide_the_maximum():
    M, H = 4, 3e7

    def chain(i, small, big):
        if i > M:
            return L(small)
        return (i, "BRANCH_LT", 5.0, 0, L(big), (i, "BRANCH_LEQ", 5.0, 0, chain(i + 1, small, big), L(big)))
    orig, conv = [], []
    orig.append((0, "BRANCH_LT", 1.0, 0, L(0.0), L(10.0)))
    conv.append((0, "BRANCH_LT", 2.0, 1, L(0.0), L(10.0)))
    for k in range(1, 25):
        d = float(np.float32(1.0 + 0.03 * k))
        orig.append((0, "BRANCH_LT", d, 0, L(0.0), L(1.0)))
        conv.append((0, "BRANCH_LEQ", d, 0, L(0.0), L(1.0)))
    orig.append((0, "BRANCH_LT", 0.0, 0, chain(1, -8.0, H), chain(1, 0.0, H)))
    conv.append((0, "BRANCH_LT", 0.0, 1, chain(1, 12.0, H + 20), chain(1, 20.0, H + 20)))
    po, pc = build(orig, 1 + M), build(conv, 1 + M, base=-30.0)
    a = analyze(load_onnx(po, np.float32), pc, input_dtype="float32")
    x = np.array([[np.nan] + [5.0] * M], dtype=np.float32)
    real = (run(po, x) - run(pc, x))[0]
    wpos = [w for w in a.worst if w["direction"].startswith("original")][0]
    assert wpos["guaranteed_max"] >= real - 1e-6
    assert wpos["max_found"] >= real - 1e-6


@pytest.mark.parametrize("case", ["weights", "threshold"])
def test_small_genuine_changes_are_not_called_rounding(case):
    ulp = 2.0 ** -14
    if case == "weights":
        o = [(0, "BRANCH_LT", 0.5, 0, L(1000.0), L(-1000.0)) for _ in range(3)]
        c = [(0, "BRANCH_LT", 0.5, 0, L(1000.0 + 15 * ulp), L(-1000.0 - 15 * ulp)) for _ in range(3)]
    else:
        o = [(0, "BRANCH_LT", 0.5, 0, L(1000.0), L(1000.0 + 15 * ulp))]
        c = [(0, "BRANCH_LT", 7.0, 0, L(1000.0), L(1000.0 + 15 * ulp))]
    a = analyze(load_onnx(build(o, 1), np.float32), build(c, 1), input_dtype="float32")
    assert a.verdict["status"] == "NOT EQUIVALENT"


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError):
        make_domain(2, bounds={0: (10.0, 0.0)})
    with pytest.raises(ValueError):
        make_domain(2, bounds={5: (0.0, 1.0)})
    d = make_domain(2, bounds={0: (np.float32(np.nan), 3.0)})
    assert d.hi[0] == d.fk.key(3.0)


def test_double_input_onnx_rounding_allowance_is_float32():
    import lightgbm as lgb
    import onnxmltools
    from onnxmltools.convert.common.data_types import DoubleTensorType
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3)) * 10
    y = X[:, 0] + rng.normal(size=2000)
    m = lgb.LGBMRegressor(n_estimators=40, num_leaves=15, verbose=-1).fit(X, y)
    o = onnxmltools.convert_lightgbm(m, initial_types=[("X", DoubleTensorType([None, 3]))], target_opset=15)
    a = analyze(m, o, background=X[:200])
    assert a.verdict["status"] != "INCONCLUSIVE"
    assert a.self_check["converted"]["raw_within_bound"]
