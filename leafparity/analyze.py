"""End-to-end analysis: load, normalise, self-check, compare, bound, verify."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import runtimes
from .engine import (Domain, PairResult, PointEvaluator, choose_point, compare_pair, ir_leaves, ir_raw, make_domain, node_rule_differences,
                     pair_trees, worst_case)
from .ir import Model
from .loaders import load_onnx, load_original
from .normalize import normalize_model
from .probing import adversarial_matrix, node_probe_matrix

__all__ = ["analyze", "Analysis"]


class SelfCheckError(Exception):
    """leafparity's exact model of a runtime disagreed with the real runtime."""


@dataclass
class Analysis:
    original: Model
    converted: Model
    domain: Domain
    pairs: List[PairResult]
    findings: List[Dict[str, Any]]
    worst: List[Dict[str, Any]]
    self_check: Dict[str, Any]
    cross_check: Dict[str, Any]
    static: Dict[str, Any]
    verdict: Dict[str, Any]
    timings: Dict[str, float]
    notes: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        from .report import analysis_to_dict
        return analysis_to_dict(self)


# --------------------------------------------------------------------------- helpers
def _fmt(v) -> str:
    v = float(v)
    if math.isnan(v):
        return "NaN"
    return repr(v)


def _acc_bound(model: Model) -> float:
    """Proven bound on floating-point summation error of the runtime's accumulation:
    |fl(sum) - sum| <= (n-1) * u * sum|terms|  (any summation order)."""
    u = float(np.finfo(np.dtype(model.accumulate_dtype)).eps) / 2
    n = len(model.trees) + 1
    s = float(np.max(np.abs(model.base))) if model.base.size else 0.0
    for t in model.trees:
        s += float(np.max(np.abs(t.value))) if t.value.size else 0.0
    return n * u * s


def _self_check(model: Model, X: np.ndarray, fk, chunk: int = 20000) -> Dict[str, Any]:
    mism, decisions, err = 0, 0, 0.0
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk]
        real = runtimes.leaf_indices(model, Xc)
        mine = ir_leaves(model, Xc, fk)
        mism += int((real != mine).sum())
        decisions += int(real.size)
        raw_real = runtimes.raw_outputs(model, Xc)
        raw_mine = ir_raw(model, mine)
        if raw_real.size:
            err = max(err, float(np.max(np.abs(raw_real - raw_mine))))
    acc = _acc_bound(model) + _acc_bound_f64(model)
    base_slack = float(np.max(np.abs(model.base))) * 2.0 ** -22 if model.base.size else 0.0
    return {"rows": int(X.shape[0]), "leaf_decisions": decisions, "leaf_mismatches": mism,
            "raw_max_abs_error": err, "acc_bound": acc,
            "raw_within_bound": bool(err <= 2 * acc + base_slack + 1e-12)}


def _acc_bound_f64(model: Model) -> float:
    # our own reference sum is float64
    u = 2.0 ** -53
    n = len(model.trees) + 1
    s = float(np.max(np.abs(model.base))) if model.base.size else 0.0
    for t in model.trees:
        s += float(np.max(np.abs(t.value))) if t.value.size else 0.0
    return n * u * s


def _classify(fk, lo, hi, nan) -> str:
    if lo > hi and nan:
        return "missing"
    if lo <= hi:
        vlo, vhi = float(fk.value(lo)), float(fk.value(hi))
        if vlo <= 0.0 <= vhi and max(abs(vlo), abs(vhi)) < 1e-30:
            return "zero"
        if vlo <= 0.0 <= vhi and (hi - lo) < 1 << 12:
            return "zero"
        width = hi - lo + 1
        if width <= 1 << 32:
            return "precision"
        return "threshold"
    return "missing"


_PATTERN_TEXT = {
    "missing": "missing values (NaN) are sent down a different branch",
    "zero": "zero / near-zero values are handled differently",
    "precision": "a narrow band of values next to a threshold is routed differently "
                 "(threshold or input rounded to lower precision)",
    "threshold": "a split threshold differs substantially",
    "value": "a leaf's output value differs (not a rounding difference)",
    "unmatched": "trees are structured differently and disagree",
}


# --------------------------------------------------------------------------- main entry
def analyze(original: Any, converted: Any, *, input_dtype="float64", allow_missing: bool = True,
            include_inf: bool = False, bounds: Optional[Dict[int, Tuple[float, float]]] = None,
            background: Optional[np.ndarray] = None, feature_names: Optional[Sequence[str]] = None,
            self_check_rows: int = 2000, cross_check_rows: int = 4000, max_findings: int = 25,
            max_probe_rows: int = 120000,
            node_limit: int = 200000, worst_case_seconds: float = 30.0, seed: int = 0) -> Analysis:
    t0 = time.time()
    timings: Dict[str, float] = {}
    udt = np.dtype(input_dtype)
    orig = original if isinstance(original, Model) else load_original(original, udt)
    conv = converted if isinstance(converted, Model) else load_onnx(converted, udt)
    timings["load"] = time.time() - t0

    t = time.time()
    normalize_model(orig, udt)
    normalize_model(conv, udt)
    timings["normalize"] = time.time() - t

    n_features = max(orig.n_features, conv.n_features)
    nan_ok = allow_missing and orig.accepts_nan and conv.accepts_nan
    inf_ok = include_inf and orig.accepts_inf and conv.accepts_inf
    dom = make_domain(n_features, udt, allow_nan=nan_ok, include_inf=inf_ok, bounds=bounds)
    notes: List[str] = list(orig.notes) + list(conv.notes)
    if allow_missing and not nan_ok:
        notes.append("missing values excluded from the domain: the original model rejects NaN input")
    if include_inf and not inf_ok:
        notes.append("infinite values excluded: one of the runtimes rejects them")
    names = list(feature_names) if feature_names else (orig.feature_names or [f"f{i}" for i in range(n_features)])
    if len(names) < n_features:
        names = names + [f"f{i}" for i in range(len(names), n_features)]
    bg = None
    if background is not None:
        bg = np.asarray(background, dtype=udt)
        if not nan_ok:
            bg = np.where(np.isnan(bg), 0, bg)

    # ---- 1. self-check: exact model == real runtime, leaf for leaf
    t = time.time()
    A = adversarial_matrix([orig, conv], dom, n_rows=self_check_rows, seed=seed + 1, background=bg)
    # rows constructed to reach every split node of both models, on / next to its boundary
    # keep the self-check's cost bounded: rows x trees (decisions) at most ~25 million
    n_trees_total = max(1, len(orig.trees) + len(conv.trees))
    probe_budget = int(min(max_probe_rows, max(20000, 25_000_000 // n_trees_total)))
    probe_res = [node_probe_matrix(m_, dom, bg, max_rows=probe_budget // 2, minimal=True,
                                   seed=seed, return_count=True) for m_ in (orig, conv)]
    probe_parts = [r_[0] for r_ in probe_res]
    A = np.concatenate([A] + probe_parts)
    n_probed = sum(r_[1] for r_ in probe_res)
    n_internal = sum(r_[2] for r_ in probe_res)
    sc = {"original": _self_check(orig, A, dom.fk), "converted": _self_check(conv, A, dom.fk),
          "boundary_probe_rows": int(sum(p_.shape[0] for p_ in probe_parts)),
          "nodes_probed": int(n_probed), "internal_nodes": int(n_internal),
          "all_nodes_probed": bool(n_probed >= n_internal)}
    for side in ("original", "converted"):
        r = sc[side]
        if not r["raw_within_bound"]:
            raise SelfCheckError(
                f"self-check failed for the {side} model: leaf values reproduced by leafparity differ "
                f"from the real runtime's raw output by {r['raw_max_abs_error']:.3g}, more than the proven "
                f"floating-point allowance ({r['acc_bound']:.3g}). The analysis was stopped.")
        if r["leaf_mismatches"]:
            raise SelfCheckError(
                f"self-check failed for the {side} model: leafparity's exact model disagreed with "
                f"the real runtime on {r['leaf_mismatches']} of {r['leaf_decisions']} tree decisions. "
                "The analysis would not be trustworthy, so it was stopped.")
    timings["self_check"] = time.time() - t

    # ---- 2. exhaustive pairwise comparison
    t = time.time()
    pairs_idx = pair_trees(orig, conv)
    pairs = [compare_pair(orig.trees[i], conv.trees[j], dom, i, j) for i, j in pairs_idx]
    timings["compare"] = time.time() - t

    # static rule comparison
    n_mapped, diff_nodes = 0, 0
    for pr in pairs:
        to, tc = orig.trees[pr.o_idx], conv.trees[pr.c_idx]
        n_mapped += int(((pr.map >= 0) & (to.feature >= 0)).sum())
        diff_nodes += len(node_rule_differences(to, tc, pr.map))
    static = {"internal_nodes_original": int(sum((tr.feature >= 0).sum() for tr in orig.trees)),
              "internal_nodes_converted": int(sum((tr.feature >= 0).sum() for tr in conv.trees)),
              "nodes_matched": n_mapped, "nodes_with_different_rules": diff_nodes,
              "joint_regions_examined": int(sum(p.n_regions for p in pairs)),
              "harmless_routing_regions": int(sum(p.harmless_routing for p in pairs))}

    # ---- 3. findings: occurrences (tree/node level) grouped into problems
    t = time.time()
    offs_o = np.cumsum([0] + [tr.n_nodes for tr in orig.trees])
    offs_c = np.cumsum([0] + [tr.n_nodes for tr in conv.trees])
    problems: Dict[Tuple, Dict[str, Any]] = {}

    def add_occurrence(pkey, kind, pat, item, witness_region):
        pb = problems.setdefault(pkey, {"kind": kind, "pattern": pat, "pattern_text": _PATTERN_TEXT[pat],
                                        "feature": pkey[2],
                                        "feature_name": (names[pkey[2]] if pkey[2] is not None else None),
                                        "occurrences": []})
        item["_region"] = witness_region
        pb["occurrences"].append(item)

    for pr in pairs:
        to, tc = orig.trees[pr.o_idx], conv.trees[pr.c_idx]
        # routing divergences, reported where they happen
        for dv in pr.divs:
            if not dv["harmful"]:
                continue
            best = dv["best"]
            if best is None:  # only bound-only regions: use any of them for a witness point
                best = next((r for r in pr.regions if r.div == pr.divs.index(dv)), None)
            item: Dict[str, Any] = {"tree": int(pr.o_idx), "tree_converted": int(pr.c_idx),
                                    "effect": float(dv["max_abs"]),
                                    "effect_is_bound": bool(dv["summarized"]),
                                    "regions": int(dv["n_regions"])}
            if best is not None and best.exact:
                item.update({"delta_raw_this_tree": [float(x) for x in best.delta],
                             "leaf_original": int(to.node_ids[best.o_leaf]),
                             "leaf_converted": int(tc.node_ids[best.c_leaf])})
            f = dv["feature"]
            if dv["kind"] == "routing" and f >= 0:
                o, c = dv["o_node"], dv["c_node"]
                lo, hi, nan = dv["interval"]
                pat = _classify(dom.fk, lo, hi, nan)
                item.update({"node_original": int(to.node_ids[o]), "node_converted": int(tc.node_ids[c]),
                             "feature": f, "feature_name": names[f] if f < len(names) else f"f{f}",
                             "rule_original": orig.router.describe_node(int(offs_o[pr.o_idx] + o)),
                             "rule_converted": conv.router.describe_node(int(offs_c[pr.c_idx] + c)),
                             "affected_values": _interval_dict(dom.fk, lo, hi, nan)})
                add_occurrence(("routing", pat, f), "routing", pat, item, best)
            else:
                add_occurrence(("unmatched", "unmatched", None), "unmatched", "unmatched", item, best)
        # leaf value mismatches between corresponding leaves
        for reg in pr.regions:
            if reg.kind != "value":
                continue
            item = {"tree": int(pr.o_idx), "tree_converted": int(pr.c_idx),
                    "effect": float(np.max(np.abs(reg.delta))), "effect_is_bound": False, "regions": 1,
                    "delta_raw_this_tree": [float(x) for x in reg.delta],
                    "leaf_original": int(to.node_ids[reg.o_leaf]),
                    "leaf_converted": int(tc.node_ids[reg.c_leaf]),
                    "leaf_value_original": [float(x) for x in to.value[reg.o_leaf]],
                    "leaf_value_converted": [float(x) for x in tc.value[reg.c_leaf]]}
            add_occurrence(("value", "value", None), "value", "value", item, reg)
    findings: List[Dict[str, Any]] = []
    witnesses = []
    for pb in problems.values():
        pb["occurrences"].sort(key=lambda it: -it["effect"])
        pb["max_single_tree_effect"] = pb["occurrences"][0]["effect"]
        pb["n_occurrences"] = len(pb["occurrences"])
        pb["trees_affected"] = len({it["tree"] for it in pb["occurrences"]})
    ordered = sorted(problems.values(), key=lambda pb: -pb["max_single_tree_effect"])
    for pb in ordered[:max_findings]:
        # witness: the occurrence with the largest *exact* effect
        wi = next((it for it in pb["occurrences"] if it["_region"] is not None and it["_region"].exact),
                  pb["occurrences"][0])
        reg = wi["_region"]
        box = reg.box if reg is not None else {}
        x = choose_point(box, dom, None if bg is None else _bg_row(bg, box, dom))
        witnesses.append(x)
        pb["example"] = {k: v for k, v in wi.items() if k != "_region"}
        for it in pb["occurrences"]:
            it.pop("_region", None)
        pb["occurrences"] = pb["occurrences"][:20]
        findings.append(pb)
    for pb in ordered[max_findings:]:
        for it in pb["occurrences"]:
            it.pop("_region", None)
    groups = problems
    timings["findings"] = time.time() - t

    # ---- 4. worst case over the whole domain (branch and bound)
    t = time.time()
    typical = None
    if bg is not None and len(bg):
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                typical = np.nanmedian(bg, axis=0)
    worst: List[Dict[str, Any]] = []
    evaluators = (PointEvaluator(orig, dom.fk), PointEvaluator(conv, dom.fk))
    for k in range(orig.n_outputs):
        for s in (1, -1):
            wc = worst_case(pairs, orig, conv, dom, k, s, background=typical, node_limit=node_limit,
                            time_budget=max(2.0, worst_case_seconds / (2 * orig.n_outputs)),
                            evaluators=evaluators)
            worst.append({"output": k, "direction": "original - converted" if s > 0 else "converted - original",
                          "max_found": wc.value, "guaranteed_max": wc.bound, "proven": wc.proven,
                          "search_nodes": wc.nodes, "_witness": wc.witness})
            witnesses.append(wc.witness)
    timings["worst_case"] = time.time() - t

    # ---- 5. verify every witness with the real runtimes
    t = time.time()
    acc = _acc_bound(orig) + _acc_bound(conv) + _acc_bound_f64(orig) + _acc_bound_f64(conv)
    if witnesses:
        W = np.stack(witnesses).astype(udt)
        ro, rc = runtimes.raw_outputs(orig, W), runtimes.raw_outputs(conv, W)
        fo, fcv = runtimes.final_outputs(orig, W), runtimes.final_outputs(conv, W)
        lo_ = ir_leaves(orig, W, dom.fk)
        lc_ = ir_leaves(conv, W, dom.fk)
        pred = ir_raw(orig, lo_) - ir_raw(conv, lc_)
        actual = ro - rc
        consistent = np.all(np.abs(pred - actual) <= acc * 4 + 1e-9, axis=1)
        for i, x in enumerate(W):
            wd = _witness_dict(x, names, ro[i], rc[i], fo, fcv, i, orig.task, bool(consistent[i]))
            if i < len(findings):
                findings[i]["witness"] = wd
            else:
                worst[i - len(findings)]["witness"] = wd
    for w in worst:
        w.pop("_witness", None)
    timings["verify"] = time.time() - t
    inconsistent = [f for f in findings if not f["witness"]["analysis_consistent"]] + \
                   [w for w in worst if not w["witness"]["analysis_consistent"]]

    # ---- 6. independent cross-check (no engine involved)
    t = time.time()
    B = adversarial_matrix([orig, conv], dom, n_rows=cross_check_rows, seed=seed + 7, background=bg)
    dB = runtimes.raw_outputs(orig, B) - runtimes.raw_outputs(conv, B)
    guaranteed_pos = np.array([w["guaranteed_max"] for w in worst if w["direction"].startswith("original")])
    guaranteed_neg = np.array([w["guaranteed_max"] for w in worst if w["direction"].startswith("converted")])
    within = bool(np.all(dB <= guaranteed_pos[None, :] + acc + 1e-9) and
                  np.all(-dB <= guaranteed_neg[None, :] + acc + 1e-9))
    cross = {"rows": int(B.shape[0]), "max_abs_raw_difference_observed": float(np.max(np.abs(dB))) if dB.size else 0.0,
             "all_within_guaranteed_bounds": within}
    timings["cross_check"] = time.time() - t

    # ---- 7. verdict
    harmful = bool(groups)          # every problem, not just the ones listed in the report
    gmax = max([abs(w["guaranteed_max"]) for w in worst] + [0.0])
    fmax = max([abs(w["max_found"]) for w in worst] + [0.0])
    label_flip = any(f.get("witness", {}).get("label_changed") for f in findings) or \
        any(w.get("witness", {}).get("label_changed") for w in worst)
    if inconsistent or not within:
        status = "INCONCLUSIVE"
        headline = ("Internal consistency check failed - do not rely on this report; "
                    "please send it to support.")
    elif not harmful:
        status = "EQUIVALENT"
        if static["harmless_routing_regions"]:
            paths = (f"for every possible input the two models reach leaves with the same value "
                     f"({static['harmless_routing_regions']} input region(s) take a different path to an "
                     f"equal-valued leaf)")
        else:
            paths = "the two models take corresponding paths for every possible input"
        headline = (f"Equivalent on the whole analysed input domain: {paths}, and their raw outputs "
                    f"differ by at most {gmax + acc:.3g} (floating-point rounding of leaf values and sums).")
    else:
        status = "NOT EQUIVALENT"
        n_occ = sum(pb["n_occurrences"] for pb in groups.values())
        tight = (gmax + acc) <= fmax * 1.001 + acc
        bound_txt = ("proven to be the largest possible difference" if tight else
                     f"proven upper bound for any input: {gmax + acc:.6g}")
        headline = (f"Not equivalent: {len(groups)} distinct problem(s) at {n_occ} place(s) in the trees. "
                    f"For some inputs the raw outputs differ by {fmax:.6g} ({bound_txt})"
                    + ("; the predicted class changes for at least one input." if label_flip else "."))
    verdict = {"status": status, "headline": headline, "max_raw_difference_found": fmax,
               "max_raw_difference_guaranteed": gmax + acc, "rounding_allowance": acc,
               "distinct_problems": len(groups),
               "places_in_trees": int(sum(pb["n_occurrences"] for pb in groups.values())),
               "label_flip_found": bool(label_flip),
               "worst_case_proven": all(w["proven"] for w in worst)}
    timings["total"] = time.time() - t0
    return Analysis(original=orig, converted=conv, domain=dom, pairs=pairs, findings=findings,
                    worst=worst, self_check=sc, cross_check=cross, static=static, verdict=verdict,
                    timings=timings, notes=notes, feature_names=names)


def _bg_row(bg, box, dom):
    """The background row that already satisfies most of the box's constraints."""
    if bg is None or not len(bg):
        return None
    score = np.zeros(len(bg))
    fk = dom.fk
    for f, (lo, hi, nan) in box.items():
        col = bg[:, f]
        isn = np.isnan(col)
        keys = fk.keys(np.where(isn, 0, col))
        inside = np.where(isn, nan, (keys >= lo) & (keys <= hi))
        score += inside
    return bg[int(np.argmax(score))]


def _interval_dict(fk, lo, hi, nan) -> Dict[str, Any]:
    d: Dict[str, Any] = {"missing": bool(nan)}
    if lo <= hi:
        d.update({"from": _fmt(fk.value(lo)), "to": _fmt(fk.value(hi)),
                  "count_of_values": int(hi - lo + 1)})
    else:
        d.update({"from": None, "to": None, "count_of_values": 0})
    return d


def _witness_dict(x, names, ro, rc, fo, fc, i, task, consistent) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "input": {names[j]: _fmt(x[j]) for j in range(len(x))},
        "raw_original": [float(v) for v in ro], "raw_converted": [float(v) for v in rc],
        "raw_difference": [float(a - b) for a, b in zip(ro, rc)],
        "analysis_consistent": consistent,
    }
    if task == "regression":
        po, pc = fo["prediction"][i], fc["prediction"][i]
        d.update({"prediction_original": [float(v) for v in po], "prediction_converted": [float(v) for v in pc],
                  "label_changed": False})
    else:
        po, pc = fo["probability"][i], fc["probability"][i]
        d.update({"probability_original": [float(v) for v in po],
                  "probability_converted": [float(v) for v in pc],
                  "label_original": int(fo["label"][i]), "label_converted": int(fc["label"][i]),
                  "label_changed": bool(int(fo["label"][i]) != int(fc["label"][i]))})
    return d
