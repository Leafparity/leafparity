"""The slice engine: exact, exhaustive comparison of two normalised tree ensembles.

For every pair of corresponding trees (original tree t, converted tree t) the engine
enumerates the *joint regions* of input space - boxes on which both trees are
constant - by walking both trees over exact float-key intervals.  Every input of
the domain lies in exactly one joint region of every pair, so nothing is sampled
and nothing can be missed: a region where the two trees reach non-corresponding
leaves is a *routing discrepancy*, and its box is the exact set of inputs affected.

A branch-and-bound search then combines the per-pair regions into the global
worst case: the largest possible difference between the two models' raw outputs
anywhere in the domain, with a witness input that attains it.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .floats import FloatKeys, keys_for
from .ir import Model, Tree, UnsupportedModelError

F32_MAX = float(np.finfo(np.float32).max)


# =========================================================================== domain
@dataclass
class Domain:
    """The set of inputs analysed: per feature an inclusive key interval, plus NaN."""

    fk: FloatKeys
    lo: List[int]
    hi: List[int]
    nan: List[bool]

    @property
    def n_features(self) -> int:
        return len(self.lo)

    def full(self, f: int) -> Tuple[int, int, bool]:
        return (self.lo[f], self.hi[f], self.nan[f])

    def describe(self) -> Dict[str, Any]:
        vals = lambda k: float(self.fk.value(k))
        return {
            "input_dtype": self.fk.dtype.name,
            "features": [
                {"min": vals(self.lo[f]), "max": vals(self.hi[f]), "missing_allowed": bool(self.nan[f])}
                for f in range(self.n_features)
            ],
        }


def make_domain(n_features: int, user_dtype=np.float64, allow_nan: bool = True,
                include_inf: bool = False, bounds: Optional[Dict[int, Tuple[float, float]]] = None,
                nan_features: Optional[Sequence[bool]] = None) -> Domain:
    fk = keys_for(user_dtype)
    if include_inf:
        lo0, hi0 = fk.key_min, fk.key_max
    else:
        m = fk.dtype.type(F32_MAX)
        lo0, hi0 = fk.key(-m), fk.key(m)
    lo = [lo0] * n_features
    hi = [hi0] * n_features
    nan = [bool(allow_nan)] * n_features
    if nan_features is not None:
        nan = [bool(allow_nan and x) for x in nan_features]
    for f, (a, b) in (bounds or {}).items():
        if not 0 <= int(f) < n_features:
            raise ValueError(f"bounds given for feature {f}, but the model has {n_features} features")
        a = None if a is None or np.isnan(float(a)) else float(a)
        b = None if b is None or np.isnan(float(b)) else float(b)
        if a is not None and b is not None and a > b:
            raise ValueError(f"bounds for feature {f}: min {a} is greater than max {b}")
        if a is not None:
            ka = fk.key(a)
            if float(fk.value(ka)) < a:
                ka += 1
            lo[f] = max(lo[f], ka)
        if b is not None:
            kb = fk.key(b)
            if float(fk.value(kb)) > b:
                kb -= 1
            hi[f] = min(hi[f], kb)
    for f in range(n_features):
        if lo[f] > hi[f] and not nan[f]:
            raise ValueError(f"feature {f}: the bounds leave no representable value")
    return Domain(fk=fk, lo=lo, hi=hi, nan=nan)


# =========================================================================== boxes
Box = Dict[int, Tuple[int, int, bool]]


def box_get(box: Box, dom: Domain, f: int):
    r = box.get(f)
    return r if r is not None else dom.full(f)


def box_intersect(a: Box, b: Box, dom: Domain) -> Optional[Box]:
    out = dict(a)
    for f, (lo2, hi2, n2) in b.items():
        lo1, hi1, n1 = box_get(a, dom, f)
        lo, hi, nn = max(lo1, lo2), min(hi1, hi2), (n1 and n2)
        if lo > hi and not nn:
            return None
        out[f] = (lo, hi, nn)
    return out


def box_intersects(a: Box, b: Box, dom: Domain) -> bool:
    for f, (lo2, hi2, n2) in b.items():
        lo1, hi1, n1 = box_get(a, dom, f)
        if max(lo1, lo2) > min(hi1, hi2) and not (n1 and n2):
            return False
    return True


def split_box(box: Box, dom: Domain, f: int, seg) -> List[Tuple[int, Box]]:
    """Split ``box`` by a node's normalised rule on feature ``f``."""
    lo, hi, nan = box_get(box, dom, f)
    ends, childs, nanc = seg
    parts = []
    start = dom.fk.key_min
    for e, c in zip(ends, childs):
        a = lo if lo > start else start
        b = hi if hi < e else e
        if a <= b:
            parts.append([c, a, b, False])
        start = e + 1
        if start > hi:
            break
    if nan:
        for p in parts:
            if p[0] == nanc:
                p[3] = True
                break
        else:
            parts.append([nanc, 1, 0, True])  # NaN only
    out = []
    for c, a, b, n in parts:
        nb = dict(box)
        nb[f] = (a, b, n)
        out.append((c, nb))
    return out


# =========================================================================== fast IR evaluation
def _seg_arrays(tree: Tree, fk: FloatKeys):
    if getattr(tree, "_segarr", None) is not None:
        return tree._segarr
    n = tree.n_nodes
    P = max([len(s[0]) for s in tree.segments if s is not None] + [1])
    E = np.full((n, P), fk.key_max, dtype=np.int64)
    C = np.zeros((n, P), dtype=np.int64)
    N = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(tree.segments):
        if s is None:
            continue
        ends, childs, nanc = s
        E[i, :len(ends)] = ends
        C[i, :len(childs)] = childs
        C[i, len(childs):] = childs[-1]
        N[i] = nanc
    tree._segarr = (E, C, N)
    return tree._segarr


def ir_leaves(model: Model, X: np.ndarray, fk: FloatKeys) -> np.ndarray:
    """Leaves reached, computed purely from the normalised description."""
    X = np.asarray(X, dtype=fk.dtype)
    nanm = np.isnan(X)
    keys = fk.keys(np.where(nanm, 0, X))
    n = X.shape[0]
    out = np.empty((n, len(model.trees)), dtype=np.int64)
    rows = np.arange(n)
    for t, tree in enumerate(model.trees):
        E, C, N = _seg_arrays(tree, fk)
        node = np.zeros(n, dtype=np.int64)
        while True:
            f = tree.feature[node]
            act = f >= 0
            if not act.any():
                break
            r = rows[act]
            nd = node[act]
            ff = f[act]
            k = keys[r, ff]
            isn = nanm[r, ff]
            piece = np.argmax(k[:, None] <= E[nd], axis=1)
            ch = np.where(isn, N[nd], C[nd, piece])
            node[act] = tree.children[nd, ch]
        out[:, t] = node
    return out


def ir_raw(model: Model, leaves: np.ndarray) -> np.ndarray:
    raw = np.tile(model.base, (leaves.shape[0], 1)).astype(np.float64)
    for t, tree in enumerate(model.trees):
        raw += tree.value[leaves[:, t]]
    return raw


# =========================================================================== pairing
def _low_high(seg):
    ends, childs, nanc = seg
    return childs[0], childs[-1]


def pair_trees(orig: Model, conv: Model) -> List[Tuple[int, int]]:
    if orig.n_outputs != conv.n_outputs:
        raise UnsupportedModelError(
            f"output layouts differ: original has {orig.n_outputs} raw outputs, "
            f"converted has {conv.n_outputs}")
    no, nc = len(orig.trees), len(conv.trees)
    if no != nc:
        raise UnsupportedModelError(
            f"tree counts differ ({no} original vs {nc} converted): the converter merged, "
            "dropped or added trees (e.g. early stopping / best_iteration)")

    def sig(tr: Tree):
        cols = tuple(np.nonzero(np.any(tr.value != 0, axis=0))[0].tolist())
        return (cols, int((tr.feature < 0).sum()), int(tr.feature[0]))

    so = [sig(t) for t in orig.trees]
    sc = [sig(t) for t in conv.trees]
    same = sum(1 for a, b in zip(so, sc) if a == b)
    if same >= 0.9 * no:
        return [(i, i) for i in range(no)]
    # fall back: match by signature, preserving order within each signature
    buckets: Dict[Any, List[int]] = {}
    for j, s in enumerate(sc):
        buckets.setdefault(s, []).append(j)
    pairs = []
    for i, s in enumerate(so):
        lst = buckets.get(s)
        if not lst:
            raise UnsupportedModelError("could not match original and converted trees one-to-one")
        pairs.append((i, lst.pop(0)))
    return pairs


def structure_map(to: Tree, tc: Tree) -> np.ndarray:
    """Map original nodes to corresponding converted nodes (-1 = no correspondence)."""
    m = np.full(to.n_nodes, -1, dtype=np.int64)
    stack = [(0, 0)]
    while stack:
        o, c = stack.pop()
        lo_, lc_ = to.feature[o] < 0, tc.feature[c] < 0
        if lo_ and lc_:
            m[o] = c
            continue
        if lo_ or lc_ or to.feature[o] != tc.feature[c]:
            continue
        m[o] = c
        so, sc = to.segments[o], tc.segments[c]
        lo_o, hi_o = _low_high(so)
        lo_c, hi_c = _low_high(sc)
        if lo_o == hi_o or lo_c == hi_c:
            continue
        stack.append((int(to.children[o][lo_o]), int(tc.children[c][lo_c])))
        stack.append((int(to.children[o][hi_o]), int(tc.children[c][hi_c])))
    return m


# =========================================================================== joint regions
@dataclass
class Region:
    box: Box
    o_leaf: int
    c_leaf: int
    delta: Optional[np.ndarray]   # exact per-output difference (None for bound-only)
    kind: str                     # 'routing' | 'value' | 'unmatched'
    harmful: bool
    div: int = -1                 # index into PairResult.divs (-1: leaf-value mismatch)
    bound_hi: Optional[np.ndarray] = None   # bound-only regions: delta <= bound_hi
    bound_lo: Optional[np.ndarray] = None   #                     delta >= bound_lo

    @property
    def exact(self) -> bool:
        return self.delta is not None


@dataclass
class PairResult:
    o_idx: int
    c_idx: int
    n_regions: int = 0
    noise_max: np.ndarray = None
    noise_min: np.ndarray = None
    regions: List[Region] = field(default_factory=list)
    truncated: bool = False
    big_max: np.ndarray = None
    big_min: np.ndarray = None
    harmless_routing: int = 0
    map: Optional[np.ndarray] = None
    best_pos: List[Optional[Region]] = field(default_factory=list)   # per output: max delta
    best_neg: List[Optional[Region]] = field(default_factory=list)   # per output: min delta
    divs: List[Dict[str, Any]] = field(default_factory=list)
    has_noise: bool = False


def _tol(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """What two leaf values may differ by and still be "the same value": one float32
    rounding step (converters store leaf weights as float32) plus float32 underflow."""
    return np.maximum(np.abs(a), np.abs(b)) * 2.0 ** -23 + 1.5e-45


def subtree_extremes(tree: Tree):
    """Per node: min and max leaf value (per output) in its subtree."""
    cached = getattr(tree, "_extremes", None)
    if cached is not None:
        return cached
    mn = tree.value.copy()
    mx = tree.value.copy()
    order = []
    stack = [0]
    while stack:
        i = stack.pop()
        order.append(i)
        if tree.feature[i] >= 0:
            stack.extend(int(c) for c in tree.children[i])
    for i in reversed(order):
        if tree.feature[i] >= 0:
            a, b = tree.children[i]
            mn[i] = np.minimum(mn[a], mn[b])
            mx[i] = np.maximum(mx[a], mx[b])
    tree._extremes = (mn, mx)
    return mn, mx


def split_joint(box: Box, dom: Domain, f: int, seg_o, seg_c):
    """Split a box by two rules on the same feature at once: yields
    (child_orig, child_conv, lo, hi, nan) pieces covering the box's set on f."""
    lo, hi, nan = box_get(box, dom, f)
    eo, co_, no = seg_o
    ec, cc_, nc = seg_c
    out = []
    if lo <= hi:
        pts = sorted({e for e in eo if lo <= e < hi} | {e for e in ec if lo <= e < hi})
        start = lo
        io = ic = 0
        for e in pts + [hi]:
            while eo[io] < start:
                io += 1
            while ec[ic] < start:
                ic += 1
            a, b = co_[io], cc_[ic]
            if out and out[-1][0] == a and out[-1][1] == b and out[-1][3] == start - 1:
                out[-1][3] = e
            else:
                out.append([a, b, start, e, False])
            start = e + 1
    if nan:
        for p in out:
            if p[0] == no and p[1] == nc:
                p[4] = True
                break
        else:
            out.append([no, nc, 1, 0, True])
    return out


def compare_pair(to: Tree, tc: Tree, dom: Domain, oi: int, ci: int,
                 max_regions: int = 100000, div_budget: int = 4000) -> PairResult:
    """Exhaustive joint walk of one original tree and its converted counterpart.

    Corresponding subtrees are walked in lock-step; where the two rules send inputs to
    non-corresponding children a *divergence* is recorded (the exact node pair, feature
    and interval of disagreement) and the two diverging subtrees are then walked as a
    product.  A divergence whose product exceeds ``div_budget`` regions is completed
    with sound bound-only regions (min/max leaf values of the remaining subtrees)."""
    n_out = to.value.shape[1]
    res = PairResult(o_idx=oi, c_idx=ci)
    res.noise_max = np.full(n_out, -np.inf)
    res.noise_min = np.full(n_out, np.inf)
    res.big_max = np.full(n_out, -np.inf)
    res.big_min = np.full(n_out, np.inf)
    res.best_pos = [None] * n_out
    res.best_neg = [None] * n_out
    smap = structure_map(to, tc)
    res.map = smap
    mn_o, mx_o = subtree_extremes(to)
    mn_c, mx_c = subtree_extremes(tc)
    fo, fc = to.feature, tc.feature
    div_count: List[int] = []

    def new_div(o, c, f, lo, hi, nan, kind):
        res.divs.append({"o_node": int(o), "c_node": int(c), "feature": int(f),
                         "interval": (int(lo), int(hi), bool(nan)), "kind": kind,
                         "n_regions": 0, "summarized": False, "harmful": False,
                         "max_abs": 0.0, "best": None})
        div_count.append(0)
        return len(res.divs) - 1

    def store(reg: Region, magnitude_vec_hi, magnitude_vec_lo):
        for k in range(n_out):
            if magnitude_vec_hi[k] > res.big_max[k]:
                res.big_max[k] = magnitude_vec_hi[k]
                if reg.exact:
                    res.best_pos[k] = reg
            if magnitude_vec_lo[k] < res.big_min[k]:
                res.big_min[k] = magnitude_vec_lo[k]
                if reg.exact:
                    res.best_neg[k] = reg
        if len(res.regions) < max_regions:
            res.regions.append(reg)
        else:
            res.truncated = True

    if smap[0] == 0:
        stack: List[Tuple[int, int, Box, int]] = [(0, 0, {}, -1)]
    else:
        stack = [(0, 0, {}, new_div(0, 0, -1, 1, 0, False, "unmatched"))]
    while stack:
        o, c, box, d = stack.pop()
        if d == -1:
            # lock-step mode: o and c correspond
            if fo[o] < 0 and fc[c] < 0:
                res.n_regions += 1
                vo, vc = to.value[o], tc.value[c]
                delta = vo - vc
                if np.all(np.abs(delta) <= _tol(vo, vc)):
                    np.maximum(res.noise_max, delta, out=res.noise_max)
                    np.minimum(res.noise_min, delta, out=res.noise_min)
                else:
                    store(Region(box=box, o_leaf=o, c_leaf=c, delta=delta, kind="value", harmful=True),
                          delta, delta)
                continue
            f = int(fo[o])
            for a, b, lo, hi, nan in split_joint(box, dom, f, to.segments[o], tc.segments[c]):
                nb = dict(box)
                nb[f] = (lo, hi, nan)
                oc, cc = int(to.children[o][a]), int(tc.children[c][b])
                if smap[oc] == cc:
                    stack.append((oc, cc, nb, -1))
                else:
                    stack.append((oc, cc, nb, new_div(o, c, f, lo, hi, nan, "routing")))
            continue
        # product mode inside divergence d
        if div_count[d] >= div_budget:
            dv = res.divs[d]
            dv["summarized"] = True
            hi_v = mx_o[o] - mn_c[c]
            lo_v = mn_o[o] - mx_c[c]
            reg = Region(box=box, o_leaf=o, c_leaf=c, delta=None, kind=dv["kind"], harmful=True, div=d,
                         bound_hi=hi_v, bound_lo=lo_v)
            dv["harmful"] = True
            dv["max_abs"] = max(dv["max_abs"], float(np.max(np.maximum(np.abs(hi_v), np.abs(lo_v)))))
            store(reg, hi_v, lo_v)
            continue
        if fo[o] >= 0:
            for ch, nb in split_box(box, dom, int(fo[o]), to.segments[o]):
                stack.append((int(to.children[o][ch]), c, nb, d))
            continue
        if fc[c] >= 0:
            for ch, nb in split_box(box, dom, int(fc[c]), tc.segments[c]):
                stack.append((o, int(tc.children[c][ch]), nb, d))
            continue
        res.n_regions += 1
        div_count[d] += 1
        dv = res.divs[d]
        dv["n_regions"] += 1
        vo, vc = to.value[o], tc.value[c]
        delta = vo - vc
        if np.all(np.abs(delta) <= _tol(vo, vc)):
            if dv["kind"] == "routing":
                res.harmless_routing += 1
            np.maximum(res.noise_max, delta, out=res.noise_max)
            np.minimum(res.noise_min, delta, out=res.noise_min)
            continue
        reg = Region(box=box, o_leaf=o, c_leaf=c, delta=delta, kind=dv["kind"], harmful=True, div=d)
        mag = float(np.max(np.abs(delta)))
        dv["harmful"] = True
        if dv["best"] is None or mag > dv["max_abs"]:
            dv["best"] = reg
        dv["max_abs"] = max(dv["max_abs"], mag)
        store(reg, delta, delta)
    res.has_noise = bool(np.all(np.isfinite(res.noise_max)))
    res.noise_max = np.where(np.isfinite(res.noise_max), res.noise_max, 0.0)
    res.noise_min = np.where(np.isfinite(res.noise_min), res.noise_min, 0.0)
    return res


def _pair_floor(pr: PairResult, k: int, s: int) -> float:
    """A value every *unlisted* region of the pair is known not to exceed (for s*delta[k]).
    If the pair has no noise regions at all, every region is listed, so the floor can sit
    just below the smallest listed value (never realised, but a sound floor)."""
    if pr.has_noise:
        return float(pr.noise_max[k]) if s > 0 else float(-pr.noise_min[k])
    vals = []
    for r in pr.regions:
        if r.exact:
            vals.append(s * float(r.delta[k]))
        else:
            vals.append(float(r.bound_lo[k]) if s > 0 else float(-r.bound_hi[k]))
    if not vals:
        return 0.0
    m = min(vals)
    return m - 1e-9 * max(1.0, abs(m))


def divergence_node(to: Tree, tc: Tree, smap: np.ndarray, o_leaf: int, c_leaf: int) -> Optional[int]:
    """Deepest original node whose corresponding converted node is on the converted
    leaf's path while its path-child's correspondent is not: where the trees split up."""
    if to.parent is None:
        to.compute_parents()
    if tc.parent is None:
        tc.compute_parents()
    anc_c = set()
    x = c_leaf
    while x >= 0:
        anc_c.add(int(x))
        x = int(tc.parent[x])
    path = []
    x = o_leaf
    while x >= 0:
        path.append(int(x))
        x = int(to.parent[x])
    path.reverse()
    last = None
    for node in path:
        if smap[node] >= 0 and int(smap[node]) in anc_c:
            last = node
        else:
            break
    return last


# =========================================================================== static node comparison
def node_rule_differences(to: Tree, tc: Tree, smap: np.ndarray) -> List[Tuple[int, int]]:
    """Corresponding internal nodes whose exact routing rules differ anywhere."""
    out = []
    for o in range(to.n_nodes):
        c = int(smap[o])
        if c < 0 or to.feature[o] < 0 or tc.feature[c] < 0:
            continue
        so, sc = to.segments[o], tc.segments[c]
        # express the converted rule in terms of the original's children
        lo_o, hi_o = _low_high(so)
        lo_c, hi_c = _low_high(sc)
        cmap = {lo_c: lo_o, hi_c: hi_o}
        if len(cmap) < 2:
            if so != sc:
                out.append((o, c))
            continue
        sc2 = (sc[0], tuple(cmap[x] for x in sc[1]), cmap.get(sc[2], sc[2]))
        if so != sc2:
            out.append((o, c))
    return out


# =========================================================================== branch and bound
@dataclass
class WorstCase:
    output: int
    sign: int
    value: float                  # best (largest) s*delta found, attained by witness
    bound: float                  # proven upper bound on s*delta
    witness: Optional[np.ndarray]
    proven: bool
    nodes: int


def choose_point(box: Box, dom: Domain, background: Optional[np.ndarray] = None) -> np.ndarray:
    fk = dom.fk
    x = np.zeros(dom.n_features, dtype=fk.dtype)
    for f in range(dom.n_features):
        lo, hi, nan = box_get(box, dom, f)
        pref = None if background is None else background[f]
        x[f] = pick_value(fk, lo, hi, nan, pref)
    return x


def pick_value(fk: FloatKeys, lo: int, hi: int, nan: bool, preferred=None):
    """A value inside [lo, hi] (keys) that a human would recognise as a plausible input:
    the preferred (background) value if it is allowed, otherwise the allowed value
    nearest to it (or to 0) with the fewest significant digits."""
    if lo > hi:
        return fk.dtype.type(np.nan)
    if preferred is not None and np.isnan(preferred) and nan:
        return fk.dtype.type(np.nan)
    target = 0.0 if preferred is None or np.isnan(preferred) else float(preferred)
    with np.errstate(over="ignore"):
        tv = fk.dtype.type(target)
    if not np.isfinite(tv):
        tv = fk.dtype.type(np.sign(target) * float(np.finfo(fk.dtype).max))
    tk = fk.key(tv)
    if lo <= tk <= hi:
        return tv
    if lo <= 0 <= hi:          # zero is the most recognisable value of all
        return fk.dtype.type(0.0)
    if lo == hi:
        return fk.value(lo)
    if tk < lo:
        return _simplest(fk, lo, hi, up=True)
    return _simplest(fk, lo, hi, up=False)


def _simplest(fk: FloatKeys, lo: int, hi: int, up: bool):
    """Fewest-significant-digit value in [lo, hi], searching from the lower end (up=True)
    or from the upper end (up=False)."""
    edge = float(fk.value(lo if up else hi))
    for digits in range(1, 18):
        c = _round_sig(edge, digits, up)
        if c is None:
            continue
        with np.errstate(over="ignore"):
            cv = fk.dtype.type(c)
        if not np.isfinite(cv):
            continue
        k = fk.key(cv)
        # accept only values close to the edge (within 1%), so the witness stays near
        # the realistic part of the interval
        if lo <= k <= hi and abs(float(cv) - edge) <= 0.01 * abs(edge):
            return cv
    return fk.value(lo if up else hi)


def _round_sig(x: float, digits: int, up: bool):
    if x == 0.0:
        return 0.0
    if not math.isfinite(x):
        return None
    from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
    d = Decimal(repr(x))
    e = d.adjusted()
    q = Decimal(1).scaleb(e - digits + 1)
    r = d.quantize(q, rounding=ROUND_CEILING if up else ROUND_FLOOR)
    return float(r)


class PointEvaluator:
    """Fast exact evaluation of single points from the normalised descriptions."""

    def __init__(self, model: Model, fk: FloatKeys):
        self.fk = fk
        self.base = model.base.astype(np.float64)
        self.trees = []
        for t in model.trees:
            segs = [None if sg is None else (list(sg[0]), list(sg[1]), sg[2]) for sg in t.segments]
            self.trees.append((t.feature.tolist(), t.children.tolist(), segs, t.value))

    def leaves(self, x) -> List[int]:
        fk = self.fk
        isn = [bool(np.isnan(v)) for v in x]
        keys = [0 if n else fk.key(v) for v, n in zip(x, isn)]
        out = []
        for feat, ch, segs, _ in self.trees:
            node = 0
            while feat[node] >= 0:
                f = feat[node]
                ends, childs, nanc = segs[node]
                if isn[f]:
                    c = nanc
                else:
                    kk = keys[f]
                    c = childs[-1]
                    for e, cc in zip(ends, childs):
                        if kk <= e:
                            c = cc
                            break
                node = ch[node][c]
            out.append(node)
        return out

    def raw(self, x) -> np.ndarray:
        r = self.base.copy()
        for (_, _, _, val), leaf in zip(self.trees, self.leaves(x)):
            r += val[leaf]
        return r


def worst_case(pairs: List[PairResult], orig: Model, conv: Model, dom: Domain, k: int, s: int,
               background=None, node_limit: int = 20000, time_budget: float = 20.0,
               max_bigs: int = 200000, evaluators=None) -> WorstCase:
    """Maximise s * (raw_orig[k] - raw_conv[k]) over the domain.

    Sound branch and bound over the per-pair discrepancy regions ("bigs"), vectorised:
    every region is a row of key-interval arrays, so testing a box against all regions
    is a handful of numpy operations.  The returned ``bound`` is always a valid upper
    bound; ``value`` is attained by ``witness`` (checked by exact evaluation).
    """
    import time as _time
    t_start = _time.time()
    F = dom.n_features
    const = s * float(orig.base[k] - conv.base[k])
    nh_list, big_rows = [], []
    for pi, pr in enumerate(pairs):
        nh = _pair_floor(pr, k, s) if not pr.truncated else (
            float(pr.noise_max[k]) if s > 0 else float(-pr.noise_min[k]))
        if pr.truncated:
            bm = float(pr.big_max[k]) if s > 0 else float(-pr.big_min[k])
            nh = max(nh, bm)          # sound: fold the whole pair into a constant bound
        else:
            for r in pr.regions:
                if r.exact:
                    v = s * float(r.delta[k])
                else:
                    v = float(r.bound_hi[k]) if s > 0 else float(-r.bound_lo[k])
                if v > nh:
                    big_rows.append((v, len(nh_list), r.box))
        nh_list.append(nh)
    nh_arr = np.asarray(nh_list, dtype=np.float64)
    big_rows.sort(key=lambda t: -t[0])
    if len(big_rows) > max_bigs:      # fold the smallest regions into their pair's constant
        for v, p, _ in big_rows[max_bigs:]:
            nh_arr[p] = max(nh_arr[p], v)
        big_rows = [row for row in big_rows[:max_bigs] if row[0] > nh_arr[row[1]]]
    n = len(big_rows)
    base_sum = const + float(nh_arr.sum())

    ev_o, ev_c = evaluators if evaluators is not None else (PointEvaluator(orig, dom.fk), PointEvaluator(conv, dom.fk))

    def evaluate(x):
        return s * float(ev_o.raw(x)[k] - ev_c.raw(x)[k])

    dlo = np.asarray(dom.lo, dtype=np.int64)
    dhi = np.asarray(dom.hi, dtype=np.int64)
    dnan = np.asarray(dom.nan, dtype=bool)
    x0 = choose_point({}, dom, background)
    best_val, best_x = evaluate(x0), x0
    if n == 0:
        return WorstCase(output=k, sign=s, value=best_val, bound=max(best_val, base_sum),
                         witness=best_x, proven=True, nodes=0)

    # A region whose box allows both an interval *and* NaN on its *thin* feature (the one
    # the cluster logic below treats as exclusive) is split into a numeric-only row and a
    # NaN-only row, so that every row lies in exactly one cluster of its thin feature.
    def _thin(box):
        # exactly the rule used for `phi` below: smallest relative width among constrained
        # features, ties broken by the lowest feature index
        best_f, best_w = None, None
        for f in sorted(box):
            lo, hi, nn = box[f]
            if lo == dom.lo[f] and hi == dom.hi[f] and nn == dom.nan[f]:
                continue
            w = 0.0 if lo > hi else (hi - lo) / max(dom.hi[f] - dom.lo[f], 1)
            if best_w is None or w < best_w:
                best_f, best_w = f, w
        return best_f, best_w
    for _round in range(4):
        split_rows = []
        changed = False
        for v, p, box in big_rows:
            f, w = _thin(box)
            if f is not None and w <= 0.25:
                lo, hi, nn = box[f]
                if nn and lo <= hi:
                    a = dict(box); a[f] = (lo, hi, False)
                    b = dict(box); b[f] = (1, 0, True)
                    split_rows.append((v, p, a))
                    split_rows.append((v, p, b))
                    changed = True
                    continue
            split_rows.append((v, p, box))
        big_rows = split_rows
        if not changed:
            break
    n = len(big_rows)
    V = np.array([r[0] for r in big_rows], dtype=np.float64)
    P = np.array([r[1] for r in big_rows], dtype=np.int64)
    LO = np.tile(dlo, (n, 1))
    HI = np.tile(dhi, (n, 1))
    NN = np.tile(dnan, (n, 1))
    for i, (_, _, box) in enumerate(big_rows):
        for f, (lo, hi, nn) in box.items():
            LO[i, f], HI[i, f], NN[i, f] = lo, hi, nn

    def intersects(idx, blo, bhi, bnan):
        if idx.size == 0:
            return idx
        lo = np.maximum(LO[idx], blo)
        hi = np.minimum(HI[idx], bhi)
        ok = np.all((lo <= hi) | (NN[idx] & bnan), axis=1)
        return idx[ok]

    def intersects_cols(idx, blo, bhi, bnan, cols):
        """Like intersects, but only re-checks the columns that changed."""
        if idx.size == 0:
            return idx
        cols = np.asarray(cols, dtype=np.int64)
        lo = np.maximum(LO[np.ix_(idx, cols)], blo[cols])
        hi = np.minimum(HI[np.ix_(idx, cols)], bhi[cols])
        ok = np.all((lo <= hi) | (NN[np.ix_(idx, cols)] & bnan[cols]), axis=1)
        return idx[ok]

    def to_box(blo, bhi, bnan) -> Box:
        return {f: (int(blo[f]), int(bhi[f]), bool(bnan[f])) for f in range(F)
                if blo[f] != dlo[f] or bhi[f] != dhi[f] or bnan[f] != dnan[f]}

    # ---- greedy incumbents: start from the strongest regions, add compatible ones
    tried = set()
    for i0 in range(min(n, 400)):
        if len(tried) >= 24:
            break
        if int(P[i0]) in tried:
            continue
        tried.add(int(P[i0]))
        blo, bhi, bnan = LO[i0].copy(), HI[i0].copy(), NN[i0].copy()
        used = {int(P[i0])}
        alive = intersects(np.arange(n), blo, bhi, bnan)
        while True:
            cand = alive[~np.isin(P[alive], list(used))]
            if cand.size == 0:
                break
            j = int(cand[0])      # alive stays sorted by value
            nlo, nhi = np.maximum(blo, LO[j]), np.minimum(bhi, HI[j])
            nnan = bnan & NN[j]
            if not np.all((nlo <= nhi) | nnan):
                alive = alive[alive != j]
                continue
            blo, bhi, bnan = nlo, nhi, nnan
            used.add(int(P[j]))
            alive = intersects(cand, blo, bhi, bnan)
        x = choose_point(to_box(blo, bhi, bnan), dom, background)
        v = evaluate(x)
        if v > best_val:
            best_val, best_x = v, x
        if _time.time() - t_start > time_budget / 3:
            break

    # ---- branch and bound
    # Each discrepancy region is usually a *sliver*: very thin on one feature (its "thin
    # feature") and wide elsewhere.  An input can only sit in one sliver cluster per
    # feature, so the search first decides, feature by feature, which cluster (or none)
    # the input's value falls in; remaining ties are resolved by branching on pairs.
    width = HI - LO
    dom_w = np.maximum(dhi - dlo, 1)
    rel = np.where(LO > HI, 0.0, width / dom_w)          # NaN-only interval -> 0 (thinnest)
    constrained = (LO != dlo) | (HI != dhi) | (NN != dnan)
    rel = np.where(constrained, rel, np.inf)
    phi = np.argmin(rel, axis=1)
    phi = np.where(np.min(rel, axis=1) <= 0.25, phi, -1)
    # safety: a row that still mixes an interval and NaN on its thin feature is excluded from
    # the exclusive-cluster logic (phi = -1 keeps every bound sound, just less tight)
    rows_ = np.arange(n)
    mixed = (phi >= 0) & (LO[rows_, np.maximum(phi, 0)] <= HI[rows_, np.maximum(phi, 0)]) & \
        NN[rows_, np.maximum(phi, 0)]
    phi = np.where(mixed, -1, phi)
    # clusters per feature: merge overlapping thin intervals (NaN-only is its own cluster)
    clusters: Dict[int, List[Tuple[int, int, bool, np.ndarray]]] = {}
    for f in np.unique(phi[phi >= 0]):
        f = int(f)
        idx = np.nonzero(phi == f)[0]
        nan_only = idx[LO[idx, f] > HI[idx, f]]
        rest = idx[LO[idx, f] <= HI[idx, f]]
        cl = []
        if nan_only.size:
            cl.append((1, 0, True, nan_only))
        if rest.size:
            order = rest[np.argsort(LO[rest, f], kind="stable")]
            cur = [order[0]]
            clo, chi = int(LO[order[0], f]), int(HI[order[0], f])
            for j in order[1:]:
                if LO[j, f] <= chi:
                    chi = max(chi, int(HI[j, f]))
                    cur.append(j)
                else:
                    cl.append((clo, chi, False, np.array(cur)))
                    cur = [j]
                    clo, chi = int(LO[j, f]), int(HI[j, f])
            cl.append((clo, chi, False, np.array(cur)))
        clusters[f] = cl
    cid = np.full(n, -1, dtype=np.int64)
    cfeat = []
    for f, cl in clusters.items():
        for (_, _, _, members) in cl:
            cid[members] = len(cfeat)
            cfeat.append(f)
    cfeat = np.asarray(cfeat, dtype=np.int64)

    def lp_bound() -> float:
        """LP relaxation: choose at most one sliver cluster per feature (y), let each pair
        profit from at most one feature (u); ignores path constraints, so it is a sound
        upper bound that is never looser than the naive or the feature-wise bound."""
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:  # pragma: no cover
            return math.inf
        C = len(cfeat)
        g = V - nh_arr[P]
        # best gain per (pair, cluster) and per (pair, non-sliver)
        gpc: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            key = (int(P[i]), int(cid[i]))
            if g[i] > gpc.get(key, 0.0):
                gpc[key] = float(g[i])
        pf: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        gmax: Dict[int, float] = {}
        for (p_, c_), val in gpc.items():
            f_ = int(cfeat[c_]) if c_ >= 0 else -1
            pf.setdefault((p_, f_), []).append((c_, val))
            gmax[p_] = max(gmax.get(p_, 0.0), val)
        keys = list(pf.keys())
        nz = len(keys)
        # variable layout: y[0:C], z[C:C+nz], u[C+nz:C+2nz]
        nv = C + 2 * nz
        rows, cols, vals, ub = [], [], [], []
        r = 0
        feats = {}
        for c_ in range(C):
            feats.setdefault(int(cfeat[c_]), []).append(c_)
        for f_, cl in feats.items():
            for c_ in cl:
                rows.append(r); cols.append(c_); vals.append(1.0)
            ub.append(1.0); r += 1
        pair_u: Dict[int, List[int]] = {}
        for j, (p_, f_) in enumerate(keys):
            zc, uc = C + j, C + nz + j
            rows.append(r); cols.append(zc); vals.append(1.0)
            if f_ >= 0:
                for c_, val in pf[(p_, f_)]:
                    rows.append(r); cols.append(c_); vals.append(-val)
                ub.append(0.0)
            else:
                ub.append(max(v for _, v in pf[(p_, f_)]))
            r += 1
            rows.append(r); cols.append(zc); vals.append(1.0)
            rows.append(r); cols.append(uc); vals.append(-gmax[p_])
            ub.append(0.0); r += 1
            pair_u.setdefault(p_, []).append(uc)
        for p_, ucs in pair_u.items():
            for uc in ucs:
                rows.append(r); cols.append(uc); vals.append(1.0)
            ub.append(1.0); r += 1
        A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
        cvec = np.zeros(nv)
        cvec[C:C + nz] = -1.0
        bounds = [(0, 1)] * C + [(0, None)] * nz + [(0, 1)] * nz
        try:
            res = linprog(cvec, A_ub=A, b_ub=np.asarray(ub), bounds=bounds, method="highs",
                          options={"time_limit": max(1.0, time_budget / 4)})
        except Exception:  # pragma: no cover
            return math.inf
        if res.status != 0 or res.fun is None:
            return math.inf
        # the dual bound of an LP solved to optimality equals -res.fun; add a tiny safety margin
        return base_sum + float(-res.fun) * (1 + 1e-9) + 1e-12

    def gain_fw(alive):
        """Feature-wise bound: an input sits in at most one sliver cluster per feature, so
        sum over features of the best cluster's total gain (+ non-sliver regions)."""
        if alive.size == 0:
            return 0.0
        pp = P[alive]
        cc = cid[alive]
        key = pp * (len(cfeat) + 1) + (cc + 1)
        _, first = np.unique(key, return_index=True)          # max per (pair, cluster)
        g = V[alive[first]] - nh_arr[pp[first]]
        c1 = cc[first]
        tot = 0.0
        thin = c1 >= 0
        if (~thin).any():
            # non-sliver regions: per pair max
            pp2 = pp[first][~thin]
            g2 = g[~thin]
            order = np.argsort(-g2, kind="stable")
            _, f2 = np.unique(pp2[order], return_index=True)
            tot += float(g2[order][f2].sum())
        if thin.any():
            Gc = np.zeros(len(cfeat))
            np.add.at(Gc, c1[thin], g[thin])
            best = {}
            for c in np.unique(c1[thin]):
                f = int(cfeat[c])
                best[f] = max(best.get(f, 0.0), float(Gc[c]))
            tot += sum(best.values())
        return tot

    def gain(alive):
        """Upper bound on the extra gain available in a state, and the best pair to branch on."""
        if alive.size == 0:
            return 0.0, -1
        pp = P[alive]
        _, first = np.unique(pp, return_index=True)
        g = V[alive[first]] - nh_arr[pp[first]]
        j = int(np.argmax(g))
        naive = float(g.sum())
        return min(naive, gain_fw(alive)), int(pp[first][j])

    # ---- local search: coordinate ascent over sliver clusters (batch-evaluated exactly)
    cand_vals: Dict[int, List[float]] = {}
    for f, cl in clusters.items():
        vals_f = []
        for clo, chi, cnan, _ in cl:
            vals_f.append(float(pick_value(dom.fk, clo, chi, cnan, None)))
        cand_vals[f] = vals_f

    def batch_eval(Xb):
        lo_ = ir_leaves(orig, Xb, dom.fk)
        lc_ = ir_leaves(conv, Xb, dom.fk)
        return s * (ir_raw(orig, lo_)[:, k] - ir_raw(conv, lc_)[:, k])

    def local_search(x, rounds=12):
        cur = x.copy()
        cur_v = float(batch_eval(cur[None, :])[0])
        for _ in range(rounds):
            if _time.time() - t_start > time_budget * 0.6:
                break
            rows_ = []
            for f, vals_f in cand_vals.items():
                for val in vals_f:
                    row = cur.copy()
                    row[f] = val
                    rows_.append(row)
            if not rows_:
                break
            Xb = np.asarray(rows_, dtype=dom.fk.dtype)
            vb = batch_eval(Xb)
            j = int(np.argmax(vb))
            if vb[j] > cur_v + eps0(cur_v):
                cur, cur_v = Xb[j].copy(), float(vb[j])
            else:
                break
        return cur, cur_v

    eps0 = lambda v: 1e-12 * max(1.0, abs(v))
    if cand_vals:
        starts = [best_x] + ([x0] if x0 is not best_x else [])
        for st in starts:
            xl, vl = local_search(st)
            if vl > best_val:
                best_val, best_x = vl, xl

    counter = itertools.count()
    nodes = 0
    proven = True
    leaf_ub = -math.inf
    eps = lambda v: 1e-12 * max(1.0, abs(v))
    heap: list = []

    def push(blo, bhi, bnan, alive, fixed_sum, decided):
        g, p = gain(alive)
        ub = base_sum + fixed_sum + g
        if ub > best_val + eps(best_val):
            heapq.heappush(heap, (-ub, next(counter), blo, bhi, bnan, alive, fixed_sum, decided))
        return ub

    push(dlo.copy(), dhi.copy(), dnan.copy(), np.arange(n), 0.0, frozenset())
    while heap:
        top = -heap[0][0]
        if top <= max(best_val, leaf_ub) + eps(best_val):
            break
        if nodes >= node_limit or _time.time() - t_start > time_budget:
            proven = False
            break
        neg_ub, _, blo, bhi, bnan, alive, fixed_sum, decided = heapq.heappop(heap)
        ub = -neg_ub
        nodes += 1
        if nodes % 10 == 1 or alive.size == 0:
            x = choose_point(to_box(blo, bhi, bnan), dom, background)
            v = evaluate(x)
            if v > best_val:
                best_val, best_x = v, x
        if alive.size == 0:
            leaf_ub = max(leaf_ub, ub)
            continue
        # pick the undecided thin feature with the largest potential
        ph = phi[alive]
        cand_f = [int(f) for f in np.unique(ph[ph >= 0]) if int(f) not in decided]
        if cand_f:
            best_f, best_pot = None, -1.0
            for f in cand_f:
                sub = alive[ph == f]
                pot, _ = gain(sub)
                if pot > best_pot:
                    best_f, best_pot = f, pot
            f = best_f
            nd = decided | {f}
            thin_f = alive[ph == f]
            keep = alive[ph != f]
            thin_set = set(thin_f.tolist())
            for clo, chi, cnan, members in clusters[f]:
                live = [m for m in members.tolist() if m in thin_set]
                if not live:
                    continue
                nlo, nhi, nnan = blo.copy(), bhi.copy(), bnan.copy()
                if cnan:
                    if not bnan[f]:
                        continue
                    nlo[f], nhi[f], nnan[f] = 1, 0, True
                else:
                    a, b = max(blo[f], clo), min(bhi[f], chi)
                    if a > b:
                        continue
                    nlo[f], nhi[f], nnan[f] = a, b, False
                push(nlo, nhi, nnan, intersects_cols(alive, nlo, nhi, nnan, [f]), fixed_sum, nd)
            # "none": the value avoids every sliver cluster on f
            push(blo, bhi, bnan, keep, fixed_sum, nd)
            continue
        # no thin feature left: branch on the pair with the largest gain
        _, p = gain(alive)
        mine = alive[P[alive] == p]
        others = alive[P[alive] != p]
        if mine.size > 64:
            mine = mine[:64]       # alive is sorted by value; the rest stay as a relaxation
            others = np.concatenate([others, alive[P[alive] == p][64:]])
            # (keeping them in 'others' keeps the bound sound)
        for j in mine:
            nlo, nhi = np.maximum(blo, LO[j]), np.minimum(bhi, HI[j])
            nnan = bnan & NN[j]
            if not np.all((nlo <= nhi) | nnan):
                continue
            changed = np.nonzero((nlo != blo) | (nhi != bhi) | (nnan != bnan))[0]
            push(nlo, nhi, nnan, intersects_cols(others[P[others] != p], nlo, nhi, nnan, changed),
                 fixed_sum + float(V[j] - nh_arr[p]), decided)
        push(blo, bhi, bnan, others[P[others] != p] if mine.size == (P[alive] == p).sum() else others,
             fixed_sum, decided)
    if cand_vals and not (proven and not heap):
        xl, vl = local_search(best_x)
        if vl > best_val:
            best_val, best_x = vl, xl
    ub_final = max(best_val, leaf_ub, (-heap[0][0]) if heap else -math.inf)
    if not proven:
        ub_final = max(best_val, min(ub_final, lp_bound()))
    tight = ub_final <= best_val + 1e-9 * max(1.0, abs(best_val))
    return WorstCase(output=k, sign=s, value=best_val, bound=ub_final,
                     witness=best_x, proven=bool(tight), nodes=nodes)

