"""Edge cases: tiny/degenerate trees, multi-output, string labels, pandas feature names,
unused features, ZipMap outputs, unsupported constructs, CLI behaviour."""
import json
import subprocess
import sys

import numpy as np
import pytest

from leafparity import UnsupportedModelError, analyze
from leafparity.report import to_json, to_text

from zoo import F, data


def test_single_leaf_trees():
    import lightgbm as lgb
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = np.ones(50)  # constant target -> stumps without splits
    m = lgb.LGBMRegressor(n_estimators=3, verbose=-1, min_child_samples=40).fit(X, y)
    o = onnxmltools.convert_lightgbm(m, initial_types=[("X", FloatTensorType([None, 3]))], target_opset=15)
    a = analyze(m, o)
    assert a.verdict["status"] == "EQUIVALENT"


def test_multi_output_regression():
    from skl2onnx import to_onnx
    from sklearn.ensemble import RandomForestRegressor
    X, Xn, y, *_ = data()
    Y = np.stack([y, -2 * y + 1], axis=1)
    m = RandomForestRegressor(5, max_depth=5, random_state=0).fit(X, Y)
    a = analyze(m, to_onnx(m, X[:1].astype(np.float32)), background=X[:100])
    assert a.original.n_outputs == 2
    assert a.verdict["status"] in ("EQUIVALENT", "NOT EQUIVALENT")
    assert a.cross_check["all_within_guaranteed_bounds"]
    assert len(a.worst) == 4


def test_string_labels_and_zipmap():
    from skl2onnx import to_onnx
    from sklearn.tree import DecisionTreeClassifier
    X, Xn, y, yb, ym = data()
    labels = np.array(["low", "mid", "high"])[ym]
    m = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, labels)
    o = to_onnx(m, X[:1].astype(np.float32))  # default: ZipMap output
    a = analyze(m, o, background=X[:100])
    assert a.cross_check["all_within_guaranteed_bounds"]
    json.loads(to_json(a))


def test_pandas_feature_names():
    import pandas as pd
    import xgboost as xgb
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    X, Xn, y, *_ = data()
    df = pd.DataFrame(Xn, columns=["age", "income", "count", "amount", "balance"])
    m = xgb.XGBRegressor(n_estimators=5, max_depth=3).fit(df, y)
    b = m.get_booster()
    b.feature_names = None  # onnxmltools requires f0.. names
    o = onnxmltools.convert_xgboost(b, initial_types=[("X", FloatTensorType([None, F]))], target_opset=15)
    a = analyze(m, o, feature_names=list(df.columns))
    assert a.feature_names[:2] == ["age", "income"]


def test_unused_features_and_wider_input():
    from skl2onnx import to_onnx
    from sklearn.tree import DecisionTreeRegressor
    X, Xn, y, *_ = data()
    Xw = np.concatenate([X, np.zeros((X.shape[0], 3))], axis=1)  # 3 features never used
    m = DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xw, y)
    a = analyze(m, to_onnx(m, Xw[:1].astype(np.float32)), allow_missing=False)
    assert a.domain.n_features == F + 3
    assert a.verdict["status"] == "EQUIVALENT"


def test_categorical_xgboost_is_refused():
    import pandas as pd
    import xgboost as xgb
    X, Xn, y, *_ = data()
    df = pd.DataFrame({"a": X[:, 0], "c": pd.Categorical(np.where(X[:, 1] > 0, "x", "y"))})
    m = xgb.XGBRegressor(n_estimators=3, max_depth=2, enable_categorical=True, tree_method="hist").fit(df, y)
    from leafparity.loaders import load_original
    with pytest.raises(UnsupportedModelError):
        load_original(m)


def test_unsupported_onnx_graph_is_refused():
    from skl2onnx import to_onnx
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.tree import DecisionTreeRegressor
    X, Xn, y, *_ = data()
    p = Pipeline([("n", Normalizer()), ("dt", DecisionTreeRegressor(max_depth=3))]).fit(X, y)
    o = to_onnx(p, X[:1].astype(np.float32))
    with pytest.raises(UnsupportedModelError):
        analyze(p, o)


def test_text_and_json_reports_render():
    from zoo import by_name
    m, o = by_name("lgb_reg_zero")
    X, *_ = data()
    a = analyze(m, o, background=X[:200])
    txt = to_text(a)
    assert "VERDICT: NOT EQUIVALENT" in txt
    assert "verified by running both real runtimes" in txt
    d = json.loads(to_json(a))
    assert d["verdict"]["status"] == "NOT EQUIVALENT"
    assert d["findings"][0]["witness"]["analysis_consistent"] is True


def test_cli_exit_codes(tmp_path):
    import joblib
    import onnx
    from zoo import by_name
    bad_m, bad_o = by_name("lgb_reg_zero")
    good_m, good_o = by_name("xgb_reg")
    X, Xn, *_ = data()
    joblib.dump(bad_m, tmp_path / "bad.pkl")
    onnx.save(bad_o, str(tmp_path / "bad.onnx"))
    good_m.get_booster().save_model(str(tmp_path / "good.json"))
    onnx.save(good_o, str(tmp_path / "good.onnx"))
    np.savetxt(tmp_path / "bg.csv", Xn[:100], delimiter=",", header="a,b,c,d,e", comments="")
    exe = [sys.executable, "-m", "leafparity.cli"]
    r = subprocess.run(exe + ["check", str(tmp_path / "good.json"), str(tmp_path / "good.onnx"), "--quiet"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("EQUIVALENT")
    r = subprocess.run(exe + ["check", str(tmp_path / "bad.pkl"), str(tmp_path / "bad.onnx"),
                              "--background", str(tmp_path / "bg.csv"), "--json", str(tmp_path / "r.json")],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["findings"][0]["feature_name"] in list("abcde")
    r = subprocess.run(exe + ["check", str(tmp_path / "bad.pkl"), str(tmp_path / "bad.onnx"),
                              "--fail-above", "1e9", "--quiet"], capture_output=True, text=True)
    assert r.returncode == 0
    r = subprocess.run(exe + ["check", str(tmp_path / "bg.csv"), str(tmp_path / "bad.onnx")],
                       capture_output=True, text=True)
    assert r.returncode == 2
