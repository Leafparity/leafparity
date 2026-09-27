"""Exact routing semantics of each runtime.

A router answers one question, exactly as the real runtime would: *given the
raw value ``u`` the user feeds in for this node's feature, does the input take
the node's first ("true"/left) branch?*  It reproduces every cast, every
preprocessing step and the precise comparison operator, in the precise float
dtype, including missing-value rules.

Each router also exposes *atoms*: monotone boolean functions of ``u`` whose
transitions are the only places the routing decision can change.  Normalisation
(see normalize.py) finds those transitions exactly by binary search over float
keys, then verifies the resulting piecewise description against ``route``.

All functions are vectorised: ``u`` is an array of user-dtype values and ``g``
an equally long array of *global node indices* (node i of tree t has global
index ``offsets[t] + i``), so one call evaluates many nodes at once.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np



# --------------------------------------------------------------------------- transforms
class Chain:
    """Per-feature elementwise preprocessing applied before the tree sees a value.

    ``steps`` is a list of ``(op, consts, dtype)`` with op in
    {'cast', 'sub', 'add', 'mul', 'div'}; ``consts`` is a per-feature array (or
    None for 'cast').  Each step is computed in ``dtype`` with IEEE semantics, exactly
    like numpy (and, verified by the test-suite, like the C/C++ runtimes).
    """

    def __init__(self, steps: Sequence[Tuple[str, Optional[np.ndarray], object]] = ()):
        self.steps = [(op, None if c is None else np.asarray(c), np.dtype(dt)) for op, c, dt in steps]

    def apply(self, u: np.ndarray, feat: np.ndarray) -> np.ndarray:
        v = u
        for op, consts, dt in self.steps:
            with np.errstate(over="ignore", invalid="ignore"):
                v = np.asarray(v).astype(dt, copy=False)
            if op == "cast":
                continue
            c = consts.astype(dt)[feat]
            with np.errstate(all="ignore"):
                if op == "sub":
                    v = v - c
                elif op == "add":
                    v = v + c
                elif op == "mul":
                    v = v * c
                elif op == "div":
                    v = v / c
                else:  # pragma: no cover - guarded at construction
                    raise ValueError(op)
            v = np.asarray(v, dtype=dt)
        return v

    def describe(self) -> str:
        if not self.steps:
            return "identity"
        parts = []
        for op, c, dt in self.steps:
            parts.append(f"cast->{dt.name}" if op == "cast" else f"{op}[{dt.name}]")
        return " , ".join(parts)

    def then(self, other: "Chain") -> "Chain":
        ch = Chain()
        ch.steps = list(self.steps) + list(other.steps)
        return ch


# --------------------------------------------------------------------------- base class
class Router:
    """Base class; subclasses fill the per-node parameter arrays."""

    name = "router"

    def __init__(self, feature: np.ndarray, chain: Chain):
        self.feature = np.asarray(feature, dtype=np.int64)   # global node -> feature (-1 leaf)
        self.chain = chain

    def values(self, u, g):
        """Value the runtime actually compares, after all casts/preprocessing."""
        return self.chain.apply(np.asarray(u), self.feature[g])

    def route(self, u, g) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def atoms(self) -> List[Callable]:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe_node(self, g: int) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


def _nan_mask(v):
    return np.isnan(v)


# --------------------------------------------------------------------------- XGBoost
class XGBoostRouter(Router):
    """XGBoost CPU predictor: input cast to float32; ``fvalue < split_cond`` -> yes (left);
    missing (NaN) -> default child."""

    name = "xgboost"

    def __init__(self, feature, threshold_f32, default_left, chain: Chain):
        super().__init__(feature, chain)
        self.thr = np.asarray(threshold_f32, dtype=np.float32)
        self.default_left = np.asarray(default_left, dtype=bool)

    def route(self, u, g):
        v = self.values(u, g)
        nan = np.isnan(v)
        with np.errstate(invalid="ignore"):
            cmp = v < self.thr[g]
        return np.where(nan, self.default_left[g], cmp)

    def atoms(self):
        def a(u, g):
            v = self.values(u, g)
            with np.errstate(invalid="ignore"):
                return v < self.thr[g]
        return [a]

    def describe_node(self, g):
        return (f"x < {float(self.thr[g])!r} (float32) -> left, "
                f"missing -> {'left' if self.default_left[g] else 'right'}")


# --------------------------------------------------------------------------- LightGBM
LGB_ZERO_THRESHOLD = float(np.float32(1e-35))  # kZeroThreshold = 1e-35f stored in a double
MISSING_NONE, MISSING_ZERO, MISSING_NAN = 0, 1, 2


class LightGBMRouter(Router):
    """LightGBM ``Tree::NumericalDecision`` on double inputs.

    * NaN with missing_type != NaN is replaced by 0.0
    * missing_type Zero: |v| <= kZeroThreshold goes to the default child
    * missing_type NaN : NaN goes to the default child
    * otherwise ``v <= threshold`` (double) -> left
    """

    name = "lightgbm"

    def __init__(self, feature, threshold_f64, default_left, missing_type, chain: Chain):
        super().__init__(feature, chain)
        self.thr = np.asarray(threshold_f64, dtype=np.float64)
        self.default_left = np.asarray(default_left, dtype=bool)
        self.missing_type = np.asarray(missing_type, dtype=np.int64)

    def values(self, u, g):
        # LightGBM's predictor turns each dense row into (index, value) pairs and keeps
        # only entries with |v| > kZeroThreshold (or NaN); everything else is read back
        # as exactly 0.0.  So |v| <= 1e-35f is snapped to zero before any comparison.
        v = self.chain.apply(np.asarray(u), self.feature[g]).astype(np.float64)
        with np.errstate(invalid="ignore"):
            snap = np.abs(v) <= LGB_ZERO_THRESHOLD
        return np.where(snap, 0.0, v)

    def route(self, u, g):
        v = self.values(u, g).astype(np.float64)
        mt = self.missing_type[g]
        nan = np.isnan(v)
        v = np.where(nan & (mt != MISSING_NAN), 0.0, v)
        with np.errstate(invalid="ignore"):
            is_zero = (v >= -LGB_ZERO_THRESHOLD) & (v <= LGB_ZERO_THRESHOLD)
            use_default = ((mt == MISSING_ZERO) & is_zero) | ((mt == MISSING_NAN) & nan)
            cmp = v <= self.thr[g]
        return np.where(use_default, self.default_left[g], cmp)

    def atoms(self):
        def cmp(u, g):
            v = self.values(u, g).astype(np.float64)
            with np.errstate(invalid="ignore"):
                return v <= self.thr[g]

        def zlo(u, g):
            v = self.values(u, g).astype(np.float64)
            with np.errstate(invalid="ignore"):
                return v >= -LGB_ZERO_THRESHOLD

        def zhi(u, g):
            v = self.values(u, g).astype(np.float64)
            with np.errstate(invalid="ignore"):
                return v <= LGB_ZERO_THRESHOLD
        return [cmp, zlo, zhi]

    def describe_node(self, g):
        mt = {0: "None", 1: "Zero", 2: "NaN"}[int(self.missing_type[g])]
        return (f"x <= {float(self.thr[g])!r} (double) -> left, missing_type={mt}, "
                f"default -> {'left' if self.default_left[g] else 'right'}")


# --------------------------------------------------------------------------- scikit-learn
class SklearnRouter(Router):
    """scikit-learn trees: X validated to float32, compared as double against a double
    threshold: ``float64(float32(x)) <= threshold`` -> left; NaN follows
    ``missing_go_to_left`` (scikit-learn >= 1.3)."""

    name = "sklearn"

    def __init__(self, feature, threshold_f64, missing_left, chain: Chain):
        super().__init__(feature, chain)
        self.thr = np.asarray(threshold_f64, dtype=np.float64)
        self.missing_left = np.asarray(missing_left, dtype=bool)

    def route(self, u, g):
        v = self.values(u, g).astype(np.float64)
        nan = np.isnan(v)
        with np.errstate(invalid="ignore"):
            cmp = v <= self.thr[g]
        return np.where(nan, self.missing_left[g], cmp)

    def atoms(self):
        def a(u, g):
            v = self.values(u, g).astype(np.float64)
            with np.errstate(invalid="ignore"):
                return v <= self.thr[g]
        return [a]

    def describe_node(self, g):
        return (f"float32(x) <= {float(self.thr[g])!r} (double) -> left, missing -> "
                f"{'left' if self.missing_left[g] else 'right'}")


# --------------------------------------------------------------------------- ONNX
ONNX_MODES = {"BRANCH_LEQ": 0, "BRANCH_LT": 1, "BRANCH_GTE": 2, "BRANCH_GT": 3, "BRANCH_EQ": 4, "BRANCH_NEQ": 5}
ONNX_MODE_NAMES = {v: k for k, v in ONNX_MODES.items()}
_MODE_SYM = {0: "<=", 1: "<", 2: ">=", 3: ">", 4: "==", 5: "!="}


class OnnxRouter(Router):
    """onnxruntime TreeEnsemble{Regressor,Classifier}: the true branch is taken iff
    ``(x MODE threshold) || (missing_value_tracks_true && isnan(x))`` evaluated in the
    threshold dtype (float32 for float models)."""

    name = "onnx"

    def __init__(self, feature, threshold, mode, tracks_true, chain: Chain, compare_dtype):
        super().__init__(feature, chain)
        self.compare_dtype = np.dtype(compare_dtype)
        self.thr = np.asarray(threshold).astype(self.compare_dtype)
        self.mode = np.asarray(mode, dtype=np.int64)
        self.tracks_true = np.asarray(tracks_true, dtype=bool)

    def _cmp(self, v, g):
        t = self.thr[g]
        m = self.mode[g]
        with np.errstate(invalid="ignore"):
            out = np.zeros(v.shape, dtype=bool)
            for code, fn in ((0, np.less_equal), (1, np.less), (2, np.greater_equal),
                             (3, np.greater), (4, np.equal), (5, np.not_equal)):
                sel = m == code
                if sel.any():
                    out[sel] = fn(v[sel], t[sel])
        return out

    def _v(self, u, g):
        return self.values(u, g).astype(self.compare_dtype)

    def route(self, u, g):
        v = self._v(u, g)
        return self._cmp(v, g) | (self.tracks_true[g] & np.isnan(v))

    def atoms(self):
        def le(u, g):
            v = self._v(u, g)
            with np.errstate(invalid="ignore"):
                return v <= self.thr[g]

        def lt(u, g):
            v = self._v(u, g)
            with np.errstate(invalid="ignore"):
                return v < self.thr[g]
        # every mode is a boolean combination of these two monotone atoms
        return [le, lt]

    def describe_node(self, g):
        if self.tracks_true[g]:
            miss = "true branch"
        else:  # IEEE comparisons with NaN are false, except '!='
            miss = "true branch" if int(self.mode[g]) == 5 else "false branch"
        return (f"x {_MODE_SYM[int(self.mode[g])]} {float(self.thr[g])!r} ({self.compare_dtype.name}) "
                f"-> true branch, missing -> {miss}")
