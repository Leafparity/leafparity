"""Human-readable and machine-readable reports."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from . import __version__


def analysis_to_dict(a) -> Dict[str, Any]:
    return {
        "tool": "leafparity",
        "version": __version__,
        "verdict": a.verdict,
        "original": a.original.summary(),
        "converted": a.converted.summary(),
        "domain": a.domain.describe(),
        "self_check": a.self_check,
        "static_comparison": a.static,
        "findings": a.findings,
        "worst_case": a.worst,
        "cross_check": a.cross_check,
        "notes": a.notes,
        "timings_seconds": {k: round(v, 3) for k, v in a.timings.items()},
    }


def to_json(a, indent: int = 2) -> str:
    return json.dumps(analysis_to_dict(a), indent=indent, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _g(v) -> str:
    return f"{v:.6g}"


def to_text(a, max_findings: int = 10) -> str:
    v = a.verdict
    L: List[str] = []
    bar = "=" * 78
    L.append(bar)
    L.append(f"leafparity {__version__} - equivalence report")
    L.append(bar)
    L.append(f"Original : {a.original.description}")
    L.append(f"Converted: {a.converted.description}")
    dom = a.domain
    miss = "including missing values (NaN)" if any(dom.nan) else "no missing values"
    L.append(f"Domain   : every {dom.fk.dtype.name} input vector ({dom.n_features} features), {miss}")
    L.append("")
    L.append(f"VERDICT: {v['status']}")
    L.append(_wrap(v["headline"], 78))
    L.append("")
    sc = a.self_check
    if sc.get("all_nodes_probed", True):
        cover = "inputs constructed to reach every split node, on and beside its boundary"
    else:
        cover = (f"inputs constructed to reach {sc['nodes_probed']} randomly chosen of "
                 f"{sc['internal_nodes']} split nodes, on and beside their boundaries")
    L.append(f"Self-check (exact model vs. real runtimes; {cover}):")
    for side in ("original", "converted"):
        r = sc[side]
        L.append(f"  {side:9s}: {r['leaf_decisions']} tree decisions, {r['leaf_mismatches']} mismatches")
    st = a.static
    L.append(f"Examined : {st['joint_regions_examined']} joint regions; "
             f"{st['nodes_with_different_rules']} of {st['nodes_matched']} matched split nodes have "
             f"different exact rules")
    cc = a.cross_check
    L.append(f"Cross-check: {cc['rows']} boundary inputs run through both real runtimes, "
             f"max |raw diff| {_g(cc['max_abs_raw_difference_observed'])}, "
             f"{'all within proven bounds' if cc['all_within_guaranteed_bounds'] else 'BOUND VIOLATED'}")
    L.append("")
    if a.findings:
        L.append(f"PROBLEMS (largest effect first; {v['distinct_problems']} problem(s), "
                 f"{v['places_in_trees']} place(s) in the trees)")
        L.append("-" * 78)
        for i, f in enumerate(a.findings[:max_findings], 1):
            L.extend(_finding_text(i, f, a))
            L.append("")
    elif st["nodes_with_different_rules"]:
        L.append(_wrap(f"Note: {st['nodes_with_different_rules']} split node(s) have slightly different "
                       "exact rules, but for every possible input they lead to leaves with the same "
                       "output (or cannot be reached), so no output changes.", 78))
        L.append("")
    L.append("WORST CASE over the whole domain (raw output = " + a.original.raw_meaning + ")")
    L.append("-" * 78)
    for w in a.worst:
        proven = "proven maximum" if w["proven"] else "true maximum lies between these two values"
        L.append(f"  output {w['output']}, {w['direction']}: attained {_g(w['max_found'])}, "
                 f"guaranteed <= {_g(w['guaranteed_max'])} ({proven})")
    big = max(a.worst, key=lambda w: abs(w["max_found"])) if a.worst else None
    if big is not None and a.findings and big.get("witness"):
        L.append("  input attaining the largest difference:")
        L.extend(_witness_lines(big["witness"], None, indent="    "))
    L.append("")
    if a.notes:
        L.append("")
        L.append("NOTES")
        for n in a.notes:
            L.append(_wrap("  - " + n, 78))
    L.append("")
    L.append(f"Completed in {a.timings.get('total', 0):.1f}s.")
    return "\n".join(L)


def _finding_text(i, f, a) -> List[str]:
    L = []
    head = f"#{i}  {f['pattern_text']}"
    if f.get("feature_name"):
        head += f"  [feature '{f['feature_name']}']"
    L.append(_wrap(head, 78))
    L.append(f"    occurs at {f['n_occurrences']} split node(s) in {f['trees_affected']} tree(s); "
             f"largest effect of a single tree: {_g(f['max_single_tree_effect'])}")
    ex = f["example"]
    if "rule_original" in ex:
        L.append(f"    example - tree {ex['tree']}:")
        L.append(f"      original  node {ex['node_original']}: {ex['rule_original']}")
        L.append(f"      converted node {ex['node_converted']}: {ex['rule_converted']}")
        av = ex["affected_values"]
        if av["count_of_values"]:
            cnt = av["count_of_values"]
            size = f" ({cnt} representable value{'s' if cnt != 1 else ''})" if cnt < 10 ** 6 else ""
            rng = f"[{av['from']}, {av['to']}]{size}"
            if av["missing"]:
                rng += " and missing (NaN)"
        else:
            rng = "missing (NaN) only"
        L.append(f"      inputs routed differently here: {rng}")
    else:
        L.append(f"    example - tree {ex['tree']}, leaf {ex['leaf_original']} vs {ex['leaf_converted']}: "
                 f"value {ex['leaf_value_original']} vs {ex['leaf_value_converted']}")
    w = f.get("witness")
    if w:
        L.extend(_witness_lines(w, f.get("feature_name"), indent="    "))
    return L


def _witness_lines(w, focus, indent="    ") -> List[str]:
    L = [f"{indent}witness: {_short_input(w['input'], focus)}"]
    if "prediction_original" in w:
        L.append(f"{indent}  original predicts {', '.join(_g(x) for x in w['prediction_original'])}; "
                 f"converted predicts {', '.join(_g(x) for x in w['prediction_converted'])}  "
                 f"(verified by running both real runtimes)")
    else:
        po, pc = w["probability_original"], w["probability_converted"]
        L.append(f"{indent}  original P = [{', '.join(_g(x) for x in po)}] -> class {w['label_original']}; "
                 f"converted P = [{', '.join(_g(x) for x in pc)}] -> class {w['label_converted']}"
                 + ("   << CLASS FLIPS" if w["label_changed"] else ""))
        L.append(f"{indent}  (verified by running both real runtimes)")
    if not w["analysis_consistent"]:
        L.append(f"{indent}  !! real runtimes did not reproduce the predicted difference")
    return L


def _short_input(inp: Dict[str, str], focus=None, limit: int = 8) -> str:
    items = list(inp.items())
    if focus is not None:
        items.sort(key=lambda kv: kv[0] != focus)
    s = ", ".join(f"{k}={v}" for k, v in items[:limit])
    if len(items) > limit:
        s += f", ... ({len(items) - limit} more)"
    return "{" + s + "}"


def _wrap(s: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(s, width=width, subsequent_indent="    ")) or s
