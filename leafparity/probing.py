"""Adversarial input generation: values sitting exactly on, and one float step either
side of, every decision boundary of one or more models.

Used for (a) the self-test that proves leafparity's exact model matches the real
runtimes leaf for leaf, and (b) an independent, engine-free cross-check that the
engine did not miss any discrepancy.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .engine import Domain


def boundary_values(models, dom: Domain) -> List[np.ndarray]:
    """Per feature: every value adjacent to a routing change in any of ``models``."""
    fk = dom.fk
    per_f: List[set] = [set() for _ in range(dom.n_features)]
    for m in models:
        for tree in m.trees:
            for i, seg in enumerate(tree.segments):
                if seg is None:
                    continue
                f = int(tree.feature[i])
                if f >= dom.n_features:
                    continue
                for e in seg[0][:-1]:
                    for k in (e - 1, e, e + 1, e + 2):
                        if dom.lo[f] <= k <= dom.hi[f]:
                            per_f[f].add(k)
    out = []
    for f in range(dom.n_features):
        ks = np.array(sorted(per_f[f]), dtype=np.int64)
        out.append(fk.values(ks) if ks.size else np.zeros(0, dtype=fk.dtype))
    return out


def adversarial_matrix(models, dom: Domain, n_rows: int = 4000, seed: int = 0,
                       background: Optional[np.ndarray] = None, p_boundary: float = 0.7,
                       p_nan: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fk = dom.fk
    bvals = boundary_values(models, dom)
    F = dom.n_features
    if background is not None and len(background):
        X = np.asarray(background, dtype=fk.dtype)[rng.integers(0, len(background), n_rows)].copy()
    else:
        X = rng.normal(size=(n_rows, F)).astype(fk.dtype)
    for f in range(F):
        bv = bvals[f]
        if bv.size:
            sel = rng.random(n_rows) < p_boundary
            X[sel, f] = bv[rng.integers(0, bv.size, sel.sum())]
        if dom.nan[f]:
            seln = rng.random(n_rows) < p_nan
            X[seln, f] = np.nan
    # special values
    specials = np.array([0.0, -0.0, 1e-36, -1e-36, 1e-35, 1.0000000180025095e-35, -1e-35], dtype=np.float64)
    specials = specials.astype(fk.dtype)
    sel = rng.random((n_rows, F)) < 0.02
    X[sel] = specials[rng.integers(0, specials.size, sel.sum())]
    # clip to domain
    lo = fk.values(np.array(dom.lo))
    hi = fk.values(np.array(dom.hi))
    with np.errstate(invalid="ignore"):
        X = np.where(np.isnan(X), X, np.clip(X, lo, hi))
    return X.astype(fk.dtype)


def node_probe_matrix(model, dom: Domain, background: Optional[np.ndarray] = None,
                      max_rows: int = 200000, minimal: bool = False, seed: int = 0,
                      return_count: bool = False):
    """Rows that *reach* split nodes and sit exactly on / next to their decision
    boundary: for each probed node, a point inside the node's path region, with the node's
    feature set to (a) every boundary key and its neighbours per the normalised rule and
    (b) (full mode) values derived directly from the raw threshold parameter, independent
    of the normalisation, plus zero, tiny values and NaN.

    If probing every node would exceed ``max_rows``, a uniformly random sample of nodes
    (across all trees) is probed instead; ``return_count`` also returns how many nodes
    were probed."""
    from .engine import box_get, choose_point, split_box
    fk = dom.fk
    thr_all = getattr(model.router, "thr", None)
    per_node = 3 if minimal else 16
    internal = [(t, i) for t, tree in enumerate(model.trees) for i in np.nonzero(tree.feature >= 0)[0]]
    if len(internal) * per_node > max_rows:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(internal), size=max(1, max_rows // per_node), replace=False)
        wanted = {internal[j] for j in pick}
    else:
        wanted = None
    rows = []
    probed = 0
    offset = 0
    specials = [0.0, 1e-36, -1e-36, 1e-35, -1e-35, 1.0000000180025095e-35, -1.0000000180025095e-35]
    for t, tree in enumerate(model.trees):
        stack = [(0, {})]
        while stack:
            node, box = stack.pop()
            if tree.feature[node] < 0:
                continue
            f = int(tree.feature[node])
            seg = tree.segments[node]
            for ch, nb in split_box(box, dom, f, seg):
                stack.append((int(tree.children[node][ch]), nb))
            if wanted is not None and (t, int(node)) not in wanted:
                continue
            probed += 1
            base = choose_point(box, dom, None if background is None else background[0])
            lo, hi, nan = box_get(box, dom, f)
            cands = set()
            for e in seg[0][:-1]:
                for k in ((e, e + 1) if minimal else (e - 2, e - 1, e, e + 1, e + 2)):
                    if lo <= k <= hi:
                        cands.add(float(fk.value(k)))
            if not minimal and thr_all is not None:
                tval = float(thr_all[offset + node])
                if np.isfinite(tval):
                    with np.errstate(over="ignore"):
                        tv = fk.dtype.type(tval)
                    if np.isfinite(tv):
                        kt = fk.key(tv)
                        for k in range(kt - 3, kt + 4):
                            if lo <= k <= hi:
                                cands.add(float(fk.value(k)))
                for sv in specials:
                    with np.errstate(over="ignore", under="ignore"):
                        k = fk.key(fk.dtype.type(sv))
                    if lo <= k <= hi:
                        cands.add(float(fk.dtype.type(sv)))
            for v in cands:
                r = base.copy()
                r[f] = v
                rows.append(r)
            if nan:
                r = base.copy()
                r[f] = np.nan
                rows.append(r)
        offset += tree.n_nodes
    X = (np.asarray(rows, dtype=fk.dtype) if rows else np.zeros((0, dom.n_features), dtype=fk.dtype))
    return (X, probed, len(internal)) if return_count else X
