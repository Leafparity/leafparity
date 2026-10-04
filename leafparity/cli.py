"""Command line interface.

    leafparity check ORIGINAL CONVERTED.onnx [options]

Exit codes: 0 = EQUIVALENT, 1 = NOT EQUIVALENT, 2 = error / inconclusive.
This makes the command usable directly as a CI gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np

from . import __version__


def _load_background(path: Optional[str]):
    if not path:
        return None, None
    low = path.lower()
    if low.endswith(".npy"):
        return np.load(path), None
    if low.endswith((".csv", ".tsv", ".txt")):
        delim = "\t" if low.endswith(".tsv") else ","
        with open(path, "r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(delim)
        try:
            [float(h) for h in header]
            names = None
            skip = 0
        except ValueError:
            names = [h.strip().strip('"') for h in header]
            skip = 1
        data = np.genfromtxt(path, delimiter=delim, skip_header=skip, dtype=np.float64,
                             missing_values=("", "NA", "NaN", "nan", "null"), filling_values=np.nan)
        if data.ndim == 1:
            data = data[None, :]
        return data, names
    if low.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_numpy(dtype=np.float64), [str(c) for c in df.columns]
    raise SystemExit(f"unsupported background data format: {path} (use .csv, .npy or .parquet)")


def _parse_bounds(spec: Optional[str], names) -> Dict[int, Tuple[float, float]]:
    if not spec:
        return {}
    with open(spec, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: Dict[int, Tuple[float, float]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and not k.isdigit():
            if not names or k not in names:
                raise SystemExit(f"bounds: unknown feature name '{k}'")
            idx = names.index(k)
        else:
            idx = int(k)
        lo, hi = (v if isinstance(v, (list, tuple)) else (v.get("min"), v.get("max")))
        out[idx] = (None if lo is None else float(lo), None if hi is None else float(hi))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leafparity",
                                description="Exact equivalence checking for converted tree-ensemble models.")
    p.add_argument("--version", action="version", version=f"leafparity {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="compare an original model with its converted ONNX version")
    c.add_argument("original", help="original model: XGBoost .json/.ubj, LightGBM .txt, or pickled "
                                    "scikit-learn / XGBoost / LightGBM estimator (.pkl/.joblib)")
    c.add_argument("converted", help="converted model (.onnx)")
    c.add_argument("--input-dtype", choices=["float64", "float32"], default="float64",
                   help="precision of the feature values your application feeds in (default float64)")
    c.add_argument("--no-missing", action="store_true", help="assume inputs never contain missing values (NaN)")
    c.add_argument("--include-inf", action="store_true", help="also analyse +/-infinity inputs")
    c.add_argument("--bounds", help="JSON file {feature: [min, max]} restricting the analysed domain")
    c.add_argument("--background", help="sample of real inputs (.csv/.npy/.parquet) used to make "
                                        "witness inputs realistic and to name features")
    c.add_argument("--json", dest="json_out", help="also write the full machine-readable report here "
                                                   "(only the summary with --summary)")
    c.add_argument("--max-findings", type=int, default=25)
    c.add_argument("--worst-case-seconds", type=float, default=30.0,
                   help="time budget for tightening the worst-case bound (default 30)")
    c.add_argument("--fail-above", type=float, default=None,
                   help="CI gate: exit 1 only if the proven max raw difference exceeds this value")
    c.add_argument("--quiet", action="store_true", help="print only the verdict line")
    c.add_argument("--summary", action="store_true",
                   help="print, and write with --json, only the verdict, the number and kinds of "
                        "problems, the largest raw difference and what was examined; no thresholds, "
                        "feature names, input values, leaf values, node ids or rules, so the result "
                        "can be shared without revealing the model")
    return p


def _cannot_certify_reason(exc: Exception) -> str:
    from .analyze import SelfCheckError
    from .ir import UnsupportedModelError
    if isinstance(exc, UnsupportedModelError):
        return "the model pair uses a construct leafparity does not support"
    if isinstance(exc, SelfCheckError):
        return "self-check failed: leafparity's exact model disagreed with a real runtime"
    if isinstance(exc, ValueError):
        return "invalid input or options"
    return "unexpected error while analysing the model pair"


def _emit_summary(args, d) -> None:
    from .report import summary_json, summary_text
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(summary_json(d))
    print(d["verdict"] if args.quiet else summary_text(d))
    if d["verdict"] == "CANNOT CERTIFY":
        print("leafparity: run again without --summary to see why (the details may reveal the model)",
              file=sys.stderr)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd != "check":
        return 2
    from .analyze import SelfCheckError, analyze
    from .ir import UnsupportedModelError
    from .report import cannot_certify_summary, summary_dict, to_json, to_text
    bg, names = _load_background(args.background)
    t0 = time.time()
    try:
        bounds = _parse_bounds(args.bounds, names)
        a = analyze(args.original, args.converted, input_dtype=args.input_dtype,
                    allow_missing=not args.no_missing, include_inf=args.include_inf,
                    bounds=bounds, background=bg, feature_names=names,
                    max_findings=args.max_findings, worst_case_seconds=args.worst_case_seconds)
    except Exception as exc:  # anything that stops the analysis means: cannot certify (exit 2)
        if args.summary:  # the message itself may name model details
            _emit_summary(args, cannot_certify_summary(_cannot_certify_reason(exc), time.time() - t0))
        elif isinstance(exc, (UnsupportedModelError, SelfCheckError, ValueError)):
            print(f"leafparity: cannot certify this model pair: {exc}", file=sys.stderr)
        else:
            print(f"leafparity: cannot certify this model pair: unexpected error "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    if args.summary:
        _emit_summary(args, summary_dict(a))
    else:
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                fh.write(to_json(a))
        if args.quiet:
            print(f"{a.verdict['status']}: {a.verdict['headline']}")
        else:
            print(to_text(a))
    status = a.verdict["status"]
    if status == "EQUIVALENT":
        return 0
    if status == "NOT EQUIVALENT":
        if args.fail_above is not None and a.verdict["max_raw_difference_guaranteed"] <= args.fail_above:
            return 0
        return 1
    return 2


def entry() -> None:  # pragma: no cover - console script wrapper
    try:
        code = main()
    except BrokenPipeError:
        code = 0
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    entry()
