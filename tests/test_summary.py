"""`leafparity check --summary`: a result that can be shared without revealing the model.

With --summary neither stdout nor the --json file may contain a threshold, a feature name,
a witness input value, a leaf value, a node id or rule text. Exit codes do not change."""
import importlib.util
import json
import re
import subprocess
import sys

import numpy as np
import pytest

EXE = [sys.executable, "-m", "leafparity.cli", "check"]


def _have(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


needs_lightgbm = pytest.mark.skipif(not _have("lightgbm", "onnxmltools", "joblib"),
                                    reason="needs lightgbm, onnxmltools and joblib")
needs_sklearn = pytest.mark.skipif(not _have("sklearn", "skl2onnx", "joblib"),
                                   reason="needs scikit-learn, skl2onnx and joblib")
needs_xgboost = pytest.mark.skipif(not _have("xgboost", "onnxmltools"),
                                   reason="needs xgboost and onnxmltools")


# --------------------------------------------------------------------------- model pairs
def _lightgbm_zero_as_missing(d):
    """NOT EQUIVALENT: the converter ignores missing_type=Zero, and thresholds are rounded."""
    import joblib
    import lightgbm as lgb
    import onnx
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)) * 10
    y = X[:, 0] + 0.5 * X[:, 1]
    m = lgb.LGBMRegressor(n_estimators=5, num_leaves=4, zero_as_missing=True, verbose=-1).fit(X, y)
    o = onnxmltools.convert_lightgbm(m, initial_types=[("X", FloatTensorType([None, 3]))], target_opset=15)
    joblib.dump(m, d / "model.pkl")
    onnx.save(o, str(d / "model.onnx"))
    return d / "model.pkl", d / "model.onnx"


def _sklearn_nan_routing(d):
    """NOT EQUIVALENT: scikit-learn and the converted model send NaN to different branches."""
    import joblib
    import onnx
    from skl2onnx import to_onnx
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 4))
    y = (X[:, 0] + X[:, 1] ** 2 > 0.5).astype(int)
    m = RandomForestClassifier(5, max_depth=4, random_state=0).fit(X, y)  # no NaN in training
    o = to_onnx(m, X[:1].astype(np.float32), options={id(m): {"zipmap": False}})
    joblib.dump(m, d / "model.pkl")
    onnx.save(o, str(d / "model.onnx"))
    return d / "model.pkl", d / "model.onnx"


def _xgboost_equivalent(d):
    import onnx
    import onnxmltools
    import xgboost as xgb
    from onnxmltools.convert.common.data_types import FloatTensorType
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3)) * 10
    y = X[:, 0] - X[:, 2]
    m = xgb.XGBRegressor(n_estimators=5, max_depth=3).fit(X, y)
    o = onnxmltools.convert_xgboost(m, initial_types=[("X", FloatTensorType([None, 3]))], target_opset=15)
    m.get_booster().save_model(str(d / "model.json"))
    onnx.save(o, str(d / "model.onnx"))
    return d / "model.json", d / "model.onnx"


def _sklearn_unsupported(d):
    """Normalizer in front of the tree: refused, so the CLI cannot certify."""
    import joblib
    import onnx
    from skl2onnx import to_onnx
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 3))
    p = Pipeline([("n", Normalizer()), ("dt", DecisionTreeRegressor(max_depth=3))]).fit(X, X[:, 0])
    joblib.dump(p, d / "model.pkl")
    onnx.save(to_onnx(p, X[:1].astype(np.float32)), str(d / "model.onnx"))
    return d / "model.pkl", d / "model.onnx"


@pytest.fixture(scope="module")
def pair(request, tmp_path_factory):
    """The (original, converted) files written by one of the builders above."""
    build = request.param
    return build(tmp_path_factory.mktemp(build.__name__.strip("_")))


def _check(*args):
    return subprocess.run(EXE + [str(a) for a in args], capture_output=True, text=True)


# --------------------------------------------------------------------------- what must not leak
def _model_details(report):
    """Every string in a full JSON report that reveals part of the model, by category."""
    found = {"threshold": set(), "witness value": set(), "feature name": set(),
             "rule text": set(), "leaf value": set()}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("rule_original", "rule_converted"):
                    found["rule text"].add(v)
                    found["threshold"].update(re.findall(r"x\)? (?:<=|>=|==|!=|<|>) (\S+)", v))
                elif k == "affected_values":  # interval ending at a threshold
                    found["threshold"].update(s for s in (v["from"], v["to"]) if s)
                elif k == "feature_name" and v:
                    found["feature name"].add(v)
                elif k == "input" and isinstance(v, dict):  # a witness input
                    found["feature name"].update(v)
                    found["witness value"].update(v.values())
                elif k in ("leaf_value_original", "leaf_value_converted"):
                    found["leaf value"].update(repr(float(x)) for x in v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(report)
    # "NaN" is a kind of input, not a model detail, and the summary names the "NaN routing" kind.
    found["witness value"].discard("NaN")
    return found


def _appears(s, text):
    """`s` occurs in `text` as a whole token (so the witness value 0.0 is not found in 10.0)."""
    return re.search(r"(?<![\w.])" + re.escape(s) + r"(?![\w.])", text) is not None


SUMMARY_KEYS = {"tool", "version", "report", "verdict", "reason", "scope", "distinct_problems",
                "problem_kinds", "class_can_change", "max_raw_difference_found",
                "max_raw_difference_upper_bound", "max_raw_difference_is_proven_maximum",
                "trees_original", "trees_converted", "features", "joint_regions_examined",
                "run_seconds"}


@pytest.mark.parametrize("pair, kind", [
    pytest.param(_lightgbm_zero_as_missing, "zero value routing", marks=needs_lightgbm,
                 id="lightgbm_zero_as_missing"),
    pytest.param(_sklearn_nan_routing, "NaN routing", marks=needs_sklearn, id="sklearn_nan_routing"),
], indirect=["pair"])
def test_summary_reveals_no_model_details(pair, kind, tmp_path):
    original, converted = pair
    full = _check(original, converted, "--json", tmp_path / "full.json")
    assert full.returncode == 1, full.stdout + full.stderr
    report = json.loads((tmp_path / "full.json").read_text(encoding="utf-8"))
    details = _model_details(report)
    for category in ("threshold", "witness value", "feature name", "rule text"):
        assert details[category], f"the full report has no {category}: nothing would be tested"
    for s in details["threshold"] | details["witness value"]:
        float(s)  # every collected value really is a number
    for category, strings in details.items():  # the full report does show them
        assert any(_appears(s, full.stdout) for s in strings) or not strings, category

    r = _check(original, converted, "--summary", "--json", tmp_path / "summary.json")
    assert r.returncode == 1, r.stdout + r.stderr
    summary_file = (tmp_path / "summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_file)

    # The run time is the only number that does not come from the model or the verdict. It is
    # left out of the comparison so a run of 1.3 s cannot collide with a witness value of 1.3.
    stdout = re.sub(r"Run time.*", "", r.stdout)
    summary_file = re.sub(r'"run_seconds": [0-9.]+', "", summary_file)
    for where, text in (("stdout", stdout), ("summary JSON", summary_file), ("stderr", r.stderr)):
        for category, strings in details.items():
            leaked = sorted(s for s in strings if _appears(s, text))
            assert not leaked, f"{where} reveals {category}: {leaked}\n{text}"
        assert "witness" not in text.lower(), f"{where} mentions a witness:\n{text}"
        assert "x <=" not in text, f"{where} contains rule text:\n{text}"

    # a separate, reduced schema that still carries the verdict-level facts
    assert set(summary) == SUMMARY_KEYS
    assert set(summary["scope"]) == {"input_dtype", "missing_values", "infinity", "bounded"}
    v = report["verdict"]
    assert summary["report"] == "summary"
    assert summary["verdict"] == "NOT EQUIVALENT" and summary["reason"] is None
    assert summary["distinct_problems"] == v["distinct_problems"]
    assert kind in [k["kind"] for k in summary["problem_kinds"]]
    assert sum(k["problems"] for k in summary["problem_kinds"]) == v["distinct_problems"]
    assert summary["max_raw_difference_found"] == v["max_raw_difference_found"]
    assert summary["max_raw_difference_upper_bound"] == v["max_raw_difference_guaranteed"]
    assert summary["trees_original"] == report["original"]["n_trees"]
    assert summary["trees_converted"] == report["converted"]["n_trees"]
    assert summary["features"] == len(report["domain"]["features"])
    assert summary["joint_regions_examined"] == report["static_comparison"]["joint_regions_examined"]
    if report["original"]["task"] == "regression":
        assert summary["class_can_change"] == "not applicable"
    else:
        assert summary["class_can_change"] == ("yes" if v["label_flip_found"] else "not found")
    assert "VERDICT: NOT EQUIVALENT" in r.stdout
    assert kind in r.stdout
    assert "Run time" in r.stdout


@pytest.mark.parametrize("pair, expected, verdict", [
    pytest.param(_xgboost_equivalent, 0, "EQUIVALENT", marks=needs_xgboost, id="equivalent"),
    pytest.param(_lightgbm_zero_as_missing, 1, "NOT EQUIVALENT", marks=needs_lightgbm, id="not_equivalent"),
    pytest.param(_sklearn_unsupported, 2, "CANNOT CERTIFY", marks=needs_sklearn, id="unsupported"),
], indirect=["pair"])
def test_exit_codes_do_not_change_with_summary(pair, expected, verdict, tmp_path):
    original, converted = pair
    r = _check(original, converted)
    assert r.returncode == expected, r.stdout + r.stderr
    r = _check(original, converted, "--summary", "--json", tmp_path / "summary.json")
    assert r.returncode == expected, r.stdout + r.stderr
    assert f"VERDICT: {verdict}\n" in r.stdout
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == SUMMARY_KEYS
    assert summary["verdict"] == verdict
    if expected == 1:  # the CI gate still overrides NOT EQUIVALENT exactly as before
        assert _check(original, converted, "--fail-above", "1e9").returncode == 0
        assert _check(original, converted, "--summary", "--fail-above", "1e9").returncode == 0
    if expected == 2:
        assert summary["reason"]
        assert "Normalizer" not in r.stdout + r.stderr
