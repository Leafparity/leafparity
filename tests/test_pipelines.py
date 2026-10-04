"""scikit-learn Pipelines as the original model, against their real ONNX conversion.

Every case is checked the way the rest of the suite checks a model family: leafparity's exact
model must reproduce both real runtimes leaf for leaf on inputs built to sit on and next to
every split boundary (self-check), every difference the real runtimes show on thousands of
boundary inputs must lie within the proven bound (cross-check), and every witness is run again
here on the real Pipeline and on onnxruntime."""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from leafparity import analyze
from leafparity.floats import F32


def _have(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


needs_sklearn = pytest.mark.skipif(not _have("sklearn", "skl2onnx", "joblib"),
                                   reason="needs scikit-learn, skl2onnx and joblib")
needs_xgboost = pytest.mark.skipif(not _have("sklearn", "skl2onnx", "joblib", "xgboost", "onnxmltools"),
                                   reason="needs xgboost, onnxmltools and skl2onnx")
needs_lightgbm = pytest.mark.skipif(not _have("sklearn", "skl2onnx", "joblib", "lightgbm", "onnxmltools"),
                                    reason="needs lightgbm, onnxmltools and skl2onnx")

EXE = [sys.executable, "-m", "leafparity.cli", "check"]


def _data(seed=0, n=1500, nan_frac=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)) * np.array([1.0, 10.0, 100.0, 0.1]) + np.array([0.0, 5.0, -50.0, 1.0])
    y = (X[:, 0] + X[:, 1] / 10 + (X[:, 3] > 1.02) > 0.6).astype(int)
    Xn = X.copy()
    Xn[rng.random(X.shape) < nan_frac] = np.nan
    return X, Xn, y


def register_booster_converters():
    """Let skl2onnx convert XGBoost / LightGBM steps with onnxmltools, the documented way."""
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    options = {"nocl": [True, False], "zipmap": [True, False, "columns"]}
    if _have("xgboost"):
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from xgboost import XGBClassifier
        update_registered_converter(XGBClassifier, "XGBoostXGBClassifier",
                                    calculate_linear_classifier_output_shapes, convert_xgboost, options=options)
    if _have("lightgbm"):
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
        update_registered_converter(LGBMClassifier, "LightGbmLGBMClassifier",
                                    calculate_linear_classifier_output_shapes, convert_lightgbm, options=options)


def _to_onnx(pipeline, X):
    from skl2onnx import to_onnx
    register_booster_converters()
    last = pipeline.steps[-1][1]
    options = {id(last): {"zipmap": False}} if hasattr(last, "classes_") else None
    return to_onnx(pipeline, np.asarray(X[:1], dtype=np.float32), options=options,
                   target_opset={"": 15, "ai.onnx.ml": 3})


def _ort(onx, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: np.asarray(X, dtype=np.float32)})


def _witness_rows(a, udt):
    ws = [f["witness"] for f in a.findings] + [w["witness"] for w in a.worst]
    return ws, np.array([[float(v) for v in w["input"].values()] for w in ws], dtype=udt)


def assert_exact_against_real_runtimes(a, pipeline, onx, udt):
    """What makes a result trustworthy, re-checked on the real Pipeline and onnxruntime."""
    for side in ("original", "converted"):
        assert a.self_check[side]["leaf_mismatches"] == 0
        assert a.self_check[side]["raw_within_bound"]
    assert a.self_check["all_nodes_probed"]
    assert a.cross_check["all_within_guaranteed_bounds"]
    assert a.verdict["status"] in ("EQUIVALENT", "NOT EQUIVALENT")
    ws, W = _witness_rows(a, udt)
    if not ws:
        return
    assert all(w["analysis_consistent"] for w in ws)
    classifier = hasattr(pipeline.steps[-1][1], "classes_")
    real_o = pipeline.predict_proba(W) if classifier else pipeline.predict(W).reshape(len(W), -1)
    outs = _ort(onx, W)
    real_c = outs[1] if classifier else outs[0]
    for i, w in enumerate(ws):
        if classifier:
            np.testing.assert_allclose(real_o[i], w["probability_original"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(real_c[i], w["probability_converted"], rtol=0, atol=1e-6)
        else:
            np.testing.assert_allclose(real_o[i], w["prediction_original"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(real_c[i], w["prediction_converted"], rtol=0, atol=1e-6)
    for f in a.findings:  # every reported problem changes the real output for its witness
        assert np.max(np.abs(f["witness"]["raw_difference"])) > 1e-6


# --------------------------------------------------------------------------- scikit-learn trees
SCALERS = ["StandardScaler", "MinMaxScaler", "RobustScaler"]


@needs_sklearn
@pytest.mark.parametrize("udt", ["float64", "float32"])
@pytest.mark.parametrize("scaler", SCALERS)
def test_forest_behind_scaler(scaler, udt):
    import sklearn.preprocessing as pre
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    X, Xn, y = _data()
    p = Pipeline([("scaler", getattr(pre, scaler)()),
                  ("model", RandomForestClassifier(8, max_depth=5, random_state=0))]).fit(Xn, y)
    onx = _to_onnx(p, X)
    a = analyze(p, onx, input_dtype=udt, background=Xn[:300])
    assert scaler in a.original.description
    assert_exact_against_real_runtimes(a, p, onx, udt)


def _one_split_pipeline(seed):
    """StandardScaler + a single split at the decimal threshold 0.1 (after scaling)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(500, 1)) * 3 + 1).astype(np.float32)
    y = (X[:, 0] > 1.3) * 10.0
    p = Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeRegressor(max_depth=1))]).fit(X, y)
    p.steps[-1][1].tree_.threshold[0] = 0.1
    return p, X


def _brute_force_disagreements(p, onx, window=2 ** 16):
    """Every float32 input within `window` steps of the split's raw boundary on which the
    real Pipeline and onnxruntime disagree."""
    sc = p.steps[0][1]
    approx = np.float32(0.1 * sc.scale_[0] + sc.mean_[0])  # only to place the window
    keys = np.arange(F32.key(approx) - window, F32.key(approx) + window)
    W = F32.values(keys).reshape(-1, 1)
    differ = np.float32(p.predict(W)) != _ort(onx, W)[0].ravel()
    return keys, keys[differ]


@needs_sklearn
def test_decimal_threshold_float32_one_input_disagrees(tmp_path):
    for seed in range(40):
        p, X = _one_split_pipeline(seed)
        onx = _to_onnx(p, X)
        window, bad = _brute_force_disagreements(p, onx)
        if len(bad):
            break
    else:
        pytest.fail("no seed gave a pipeline whose real runtimes disagree near the threshold")

    a = analyze(p, onx, input_dtype="float32", allow_missing=False)
    assert a.verdict["status"] == "NOT EQUIVALENT"
    assert_exact_against_real_runtimes(a, p, onx, "float32")
    # the analysis reports exactly the inputs on which the real runtimes disagree
    reported = set()
    for f in a.findings:
        av = f["example"]["affected_values"]
        assert f["n_occurrences"] == 1
        lo, hi = F32.key(np.float32(av["from"])), F32.key(np.float32(av["to"]))
        reported.update(range(lo, hi + 1))
    assert reported == set(bad.tolist())
    assert reported <= set(window.tolist())
    # the witness, run on the real Pipeline and the real onnxruntime: different outputs
    x = np.array([[float(a.findings[0]["witness"]["input"]["f0"])]], dtype=np.float32)
    assert np.float32(p.predict(x)[0]) != _ort(onx, x)[0].ravel()[0]

    import joblib
    import onnx
    joblib.dump(p, tmp_path / "p.pkl")
    onnx.save(onx, str(tmp_path / "p.onnx"))
    r = subprocess.run(EXE + [str(tmp_path / "p.pkl"), str(tmp_path / "p.onnx"), "--input-dtype",
                              "float32", "--no-missing", "--quiet"], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr


def _path_boxes(tree):
    """For every internal node: (node, {feature: (lo, hi]}) of the inputs that reach it."""
    out, todo = [], [(0, {})]
    while todo:
        node, box = todo.pop()
        f = tree.feature[node]
        if tree.children_left[node] == -1:
            continue
        out.append((node, box))
        t = tree.threshold[node]
        lo, hi = box.get(f, (-np.inf, np.inf))
        todo.append((tree.children_left[node], {**box, f: (lo, min(hi, t))}))
        todo.append((tree.children_right[node], {**box, f: (max(lo, t), hi)}))
    return out


@needs_sklearn
def test_equivalent_pipeline_brute_force_around_every_threshold(tmp_path):
    """MinMaxScaler fitted on [0, 4]: scale 0.25 and offset 0, so sklearn (float64, then
    float32) and onnxruntime (float32) compute the same scaled value, and skl2onnx rounds
    each threshold down to float32. leafparity must prove EQUIVALENT, and the real runtimes
    must agree on every float32 value next to every threshold of either model."""
    from onnx import helper
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 4, size=(2000, 2)).astype(np.float32)
    X[0], X[1] = 0.0, 4.0
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    p = Pipeline([("scaler", MinMaxScaler()), ("model", DecisionTreeRegressor(max_depth=5, random_state=0))])
    p.fit(X, y)
    sc, tree = p.steps[0][1], p.steps[-1][1].tree_
    assert np.all(sc.scale_ == 0.25) and np.all(sc.min_ == 0.0)
    onx = _to_onnx(p, X)

    a = analyze(p, onx, input_dtype="float32", allow_missing=False)
    assert a.verdict["status"] == "EQUIVALENT"
    assert_exact_against_real_runtimes(a, p, onx, "float32")

    tn = [n for n in onx.graph.node if n.op_type == "TreeEnsembleRegressor"][0]
    onnx_thr = {(int(nid)): float(v) for nid, v, m in zip(
        helper.get_attribute_value(next(x for x in tn.attribute if x.name == "nodes_nodeids")),
        helper.get_attribute_value(next(x for x in tn.attribute if x.name == "nodes_values")),
        helper.get_attribute_value(next(x for x in tn.attribute if x.name == "nodes_modes"))) if m != b"LEAF"}
    rows, nodes = [], []
    for node, box in _path_boxes(tree):
        f = tree.feature[node]
        base = np.empty(2, dtype=np.float32)
        for j in range(2):  # any value strictly inside the box (scaled space), mapped back
            lo, hi = box.get(j, (-np.inf, np.inf))
            lo, hi = max(lo, -1.0), min(hi, 2.0)
            base[j] = np.float32((lo + hi) / 2) * np.float32(4)
        for thr in (tree.threshold[node], onnx_thr[node]):  # both models' thresholds
            k = F32.key(np.float32(thr) * np.float32(4))
            for v in F32.values(np.arange(k - 64, k + 65)):
                x = base.copy()
                x[f] = v
                rows.append(x)
                nodes.append(node)
    W = np.array(rows, dtype=np.float32)
    reached = p.steps[-1][1].decision_path(p[:-1].transform(W)).toarray()[np.arange(len(W)), nodes]
    assert reached.mean() > 0.95  # the probes really sit at the node they are meant for
    differ = np.float32(p.predict(W)) != _ort(onx, W)[0].ravel()
    assert not differ.any(), W[differ][:5]

    import joblib
    import onnx
    joblib.dump(p, tmp_path / "p.pkl")
    onnx.save(onx, str(tmp_path / "p.onnx"))
    r = subprocess.run(EXE + [str(tmp_path / "p.pkl"), str(tmp_path / "p.onnx"), "--input-dtype",
                              "float32", "--no-missing", "--quiet"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------- XGBoost / LightGBM
def _booster(library):
    if library == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=12, max_depth=3)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=12, num_leaves=7, verbose=-1)


@pytest.mark.parametrize("udt", ["float64", "float32"])
@pytest.mark.parametrize("library", [pytest.param("xgboost", marks=needs_xgboost),
                                     pytest.param("lightgbm", marks=needs_lightgbm)])
def test_booster_behind_standard_scaler(library, udt):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, Xn, y = _data()
    p = Pipeline([("scaler", StandardScaler()), ("model", _booster(library))]).fit(Xn, y)
    onx = _to_onnx(p, X)
    a = analyze(p, onx, input_dtype=udt, background=Xn[:300])
    assert a.original.library == library
    assert "StandardScaler" in a.original.description
    assert_exact_against_real_runtimes(a, p, onx, udt)


@pytest.mark.parametrize("library", [pytest.param("xgboost", marks=needs_xgboost),
                                     pytest.param("lightgbm", marks=needs_lightgbm)])
def test_booster_pipeline_from_the_command_line(library, tmp_path):
    import joblib
    import onnx
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    X, Xn, y = _data(seed=1)
    p = Pipeline([("scaler", MinMaxScaler()), ("model", _booster(library))]).fit(Xn, y)
    joblib.dump(p, tmp_path / "p.joblib")
    onnx.save(_to_onnx(p, X), str(tmp_path / "p.onnx"))
    a = analyze(str(tmp_path / "p.joblib"), str(tmp_path / "p.onnx"))
    r = subprocess.run(EXE + [str(tmp_path / "p.joblib"), str(tmp_path / "p.onnx"), "--quiet"],
                       capture_output=True, text=True)
    assert r.returncode == {"EQUIVALENT": 0, "NOT EQUIVALENT": 1}[a.verdict["status"]], r.stdout + r.stderr


# --------------------------------------------------------------------------- ColumnTransformer
def _column_pipeline(kind):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    if kind == "scaler+passthrough":
        ct = ColumnTransformer([("num", StandardScaler(), [0, 2])], remainder="passthrough")
        return Pipeline([("columns", ct), ("model", RandomForestClassifier(8, max_depth=5, random_state=0))])
    # different arithmetic per column group, reordered columns, one column dropped
    ct = ColumnTransformer([("std", StandardScaler(), [3]), ("mm", MinMaxScaler(), [1]),
                            ("keep", "passthrough", [0])], remainder="drop")
    return Pipeline([("columns", ct), ("model", _booster("xgboost"))])


@pytest.mark.parametrize("udt", ["float64", "float32"])
@pytest.mark.parametrize("kind", [pytest.param("scaler+passthrough", marks=needs_sklearn),
                                  pytest.param("mixed+drop", marks=needs_xgboost)])
def test_column_transformer(kind, udt):
    X, Xn, y = _data(seed=2)
    p = _column_pipeline(kind).fit(Xn, y)
    onx = _to_onnx(p, X)
    a = analyze(p, onx, input_dtype=udt, background=Xn[:300])
    assert "ColumnTransformer" in a.original.description
    assert a.original.n_features == a.converted.n_features == 4
    used = {int(f) for t in a.original.trees for f in t.feature if f >= 0}
    if kind == "mixed+drop":
        assert 2 not in used  # the dropped input column reaches no tree
    assert_exact_against_real_runtimes(a, p, onx, udt)
    # a problem's feature is an input column of the Pipeline, and its witness shows it
    for f in a.findings:
        if f["feature"] is not None:
            assert f["feature_name"] == f"f{f['feature']}"


@needs_sklearn
def test_preprocessing_that_is_not_reproduced_bit_for_bit_is_refused(monkeypatch):
    """Both sides' modelled preprocessing is checked against the real thing (the Pipeline's
    transform steps, the ONNX preprocessing nodes run by onnxruntime). A model that does
    not match bit for bit is refused, never analysed: here a column-order bug is injected."""
    from leafparity.ir import UnsupportedModelError
    from leafparity.loaders import columns, load_onnx, load_original
    X, Xn, y = _data(seed=2)
    p = _column_pipeline("scaler+passthrough").fit(Xn, y)
    onx = _to_onnx(p, X)
    load_original(p)
    load_onnx(onx)
    concat = columns.Columns.concat

    def swapped(parts, dtype):
        c = concat(parts, dtype)
        c.src[0], c.src[1] = c.src[1], c.src[0]
        return c
    monkeypatch.setattr(columns.Columns, "concat", staticmethod(swapped))
    with pytest.raises(UnsupportedModelError, match="bit for bit"):
        load_original(p)
    with pytest.raises(UnsupportedModelError, match="bit for bit"):
        load_onnx(onx)
