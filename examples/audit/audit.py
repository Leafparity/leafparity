"""Audit the documented conversion recipes of skl2onnx and onnxmltools with leafparity.

Builds original/ONNX pairs the way the skl2onnx and onnxmltools documentation shows (default
settings, float32 input), checks every pair with leafparity on finite inputs and on all inputs
including NaN, and prints a summary table.

    pip install leafparity[all] skl2onnx onnxmltools
    python audit.py            # full matrix: 126 pairs, 252 checks (about 12 minutes)
    python audit.py --quick    # iris and diabetes only, smallest size (under a minute)

Writes pairs/, reports/ (one leafparity JSON report per check) and results.json.
"""
import argparse, collections, json, os, subprocess, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np

warnings.filterwarnings("ignore")
from sklearn import datasets
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SkFloat
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OmFloat
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

PAIRS, REPORTS = "pairs", "reports"
CONFIGS = [(10, 3), (50, 6), (100, 8)]  # (number of trees, max depth)


def load_data():
    rs = np.random.RandomState(1)
    out = {
        "iris": ("clf", *datasets.load_iris(return_X_y=True)),
        "wine": ("clf", *datasets.load_wine(return_X_y=True)),
        "breast_cancer": ("clf", *datasets.load_breast_cancer(return_X_y=True)),
        "digits": ("clf", *datasets.load_digits(return_X_y=True)),
        "diabetes": ("reg", *datasets.load_diabetes(return_X_y=True)),
    }
    X, y = datasets.make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=0)
    out["friedman1"] = ("reg", X, y)
    # integer count features with many exact zeros
    Xc = rs.poisson(1.2, size=(2000, 8)).astype(float)
    yc = ((Xc[:, 0] > 1) ^ (Xc[:, 1] == 0) ^ (Xc[:, 2] + Xc[:, 3] > 3)).astype(int)
    out["counts_zero_heavy"] = ("clf", Xc, yc)
    return out


def make_models(task, n, d):
    clf = task == "clf"
    return {
        "sk_RandomForest": (RandomForestClassifier if clf else RandomForestRegressor)(n_estimators=n, max_depth=d, random_state=0),
        "sk_ExtraTrees": (ExtraTreesClassifier if clf else ExtraTreesRegressor)(n_estimators=n, max_depth=d, random_state=0),
        "sk_DecisionTree": (DecisionTreeClassifier if clf else DecisionTreeRegressor)(max_depth=d, random_state=0),
        "sk_GradientBoosting": (GradientBoostingClassifier if clf else GradientBoostingRegressor)(
            n_estimators=min(n, 50), max_depth=min(d, 4), random_state=0),
        "xgboost": (XGBClassifier if clf else XGBRegressor)(n_estimators=n, max_depth=d, random_state=0),
        "lightgbm": (LGBMClassifier if clf else LGBMRegressor)(
            n_estimators=n, max_depth=d, num_leaves=2 ** min(d, 6), random_state=0, verbose=-1),
    }


def build(quick):
    os.makedirs(PAIRS, exist_ok=True)
    manifest = []
    for dname, (task, X, y) in load_data().items():
        if quick and dname not in ("iris", "diabetes"):
            continue
        nf = X.shape[1]
        for n, d in CONFIGS[:1] if quick else CONFIGS:
            for lib, m in make_models(task, n, d).items():
                name = f"{dname}__{lib}__{n}x{d}"
                m.fit(X, y)
                if lib.startswith("sk_"):
                    orig = f"{PAIRS}/{name}.joblib"
                    joblib.dump(m, orig)
                    opts = {id(m): {"zipmap": False}} if task == "clf" else None
                    onx = convert_sklearn(m, initial_types=[("input", SkFloat([None, nf]))], options=opts)
                elif lib == "xgboost":
                    orig = f"{PAIRS}/{name}.json"
                    m.save_model(orig)
                    onx = convert_xgboost(m, initial_types=[("input", OmFloat([None, nf]))])
                else:
                    orig = f"{PAIRS}/{name}.txt"
                    m.booster_.save_model(orig)
                    kw = {"zipmap": False} if task == "clf" else {}
                    onx = convert_lightgbm(m, initial_types=[("input", OmFloat([None, nf]))], **kw)
                onnx_path = f"{PAIRS}/{name}.onnx"
                with open(onnx_path, "wb") as f:
                    f.write(onx.SerializeToString())
                manifest.append(dict(name=name, lib=lib, orig=orig, onnx=onnx_path))
    return manifest


def check(args):
    m, mode = args
    out = f"{REPORTS}/{m['name']}__{mode}.json"
    cmd = ["leafparity", "check", m["orig"], m["onnx"], "--input-dtype", "float32",
           "--json", out, "--quiet", "--worst-case-seconds", "5"]
    if mode == "finite":
        cmd.append("--no-missing")
    try:
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=300).returncode
    except subprocess.TimeoutExpired:
        rc = -9
    status = "NO REPORT"
    if os.path.exists(out):
        with open(out) as f:
            status = json.load(f)["verdict"]["status"]
    return dict(name=m["name"], lib=m["lib"], mode=mode, rc=rc, status=status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(REPORTS, exist_ok=True)
    t0 = time.time()
    manifest = build(a.quick)
    print(f"built {len(manifest)} pairs in {time.time() - t0:.0f}s, checking each on finite inputs and with NaN ...")
    jobs = [(m, mode) for m in manifest for mode in ("finite", "with_nan")]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(check, jobs))
    with open("results.json", "w") as f:
        json.dump(results, f, indent=1)
    table = collections.defaultdict(collections.Counter)
    for r in results:
        table[(r["lib"], r["mode"])][r["status"]] += 1
    print(f"\n{'model type':<22}{'inputs':<10}result")
    for (lib, mode), c in sorted(table.items()):
        print(f"{lib:<22}{mode:<10}" + ", ".join(f"{v} {k}" for k, v in sorted(c.items())))
    print(f"\nfinished in {time.time() - t0:.0f}s; reports in {REPORTS}/, summary in results.json")


if __name__ == "__main__":
    sys.exit(main())
