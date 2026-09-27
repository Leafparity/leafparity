"""Exact, order-preserving integer keys for IEEE-754 floating point values.

Every non-NaN value of a float dtype is mapped to an integer *key* such that

    a < b  (IEEE comparison)   <=>   key(a) < key(b)

and -0.0 / +0.0 share key 0 (they compare equal in every runtime we model).
Consecutive keys are consecutive representable values, so "the next float up"
is simply ``key + 1``.  All interval logic in leafparity is done on these keys,
which removes every source of floating point fuzziness from the analysis
itself: intervals are sets of *representable values*, not real numbers.

NaN has no key; it is always handled separately.
"""
from __future__ import annotations

import numpy as np

_INT = {np.dtype(np.float64): np.int64, np.dtype(np.float32): np.int32}
_MAG = {np.dtype(np.float64): 0x7FFFFFFFFFFFFFFF, np.dtype(np.float32): 0x7FFFFFFF}
_INF = {np.dtype(np.float64): 0x7FF0000000000000, np.dtype(np.float32): 0x7F800000}
_SIGN = {np.dtype(np.float64): -0x8000000000000000, np.dtype(np.float32): -0x80000000}


class FloatKeys:
    """Key conversion for one float dtype (float32 or float64)."""

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        if self.dtype not in _INT:
            raise ValueError(f"unsupported float dtype {dtype}")
        self.itype = np.dtype(_INT[self.dtype])
        self.inf_key = _INF[self.dtype]
        # key(-inf) .. key(+inf)
        self.key_min = -self.inf_key
        self.key_max = self.inf_key
        self.max_finite_key = self.inf_key - 1
        self.min_finite_key = -self.inf_key + 1

    # ------------------------------------------------------------------ vector
    def keys(self, values) -> np.ndarray:
        """Vectorised value -> key (int64). NaN input is undefined; callers must mask it."""
        v = np.asarray(values, dtype=self.dtype)
        bits = v.view(self.itype).astype(np.int64)
        mag = bits & _MAG[self.dtype]
        return np.where(bits < 0, -mag, mag)

    def values(self, keys) -> np.ndarray:
        """Vectorised key -> value in this dtype."""
        k = np.asarray(keys, dtype=np.int64)
        mag = np.abs(k)
        if self.dtype == np.float64:
            bits = np.where(k < 0, mag | np.int64(_SIGN[self.dtype]), mag).astype(np.int64)
            return bits.view(np.float64)
        bits = np.where(k < 0, mag | _SIGN[self.dtype], mag).astype(np.int32)
        return bits.view(np.float32)

    # ------------------------------------------------------------------ scalar
    def key(self, value) -> int:
        v = self.dtype.type(value)
        if np.isnan(v):
            raise ValueError("NaN has no key")
        return int(self.keys(np.array([v]))[0])

    def value(self, key: int):
        return self.values(np.array([key], dtype=np.int64))[0]

    def clamp(self, key: int) -> int:
        return max(self.key_min, min(self.key_max, key))


F64 = FloatKeys(np.float64)
F32 = FloatKeys(np.float32)


def keys_for(dtype) -> FloatKeys:
    d = np.dtype(dtype)
    if d == np.float64:
        return F64
    if d == np.float32:
        return F32
    raise ValueError(f"unsupported dtype {dtype}")


def find_transitions(fk: FloatKeys, func, n: int) -> np.ndarray:
    """For ``n`` monotone boolean functions evaluated jointly, find each transition.

    ``func(values)`` receives an array of ``n`` values of dtype ``fk.dtype`` (one per
    function) and returns ``n`` booleans.  Each function must be monotone over the
    key range [key_min, key_max] (non-increasing *or* non-decreasing).

    Returns an int64 array ``t`` where ``t[i]`` is the last key at which function i
    still has the value it has at ``key_min``; ``t[i] == key_max`` means function i
    is constant over the whole range (no transition).
    """
    lo = np.full(n, fk.key_min, dtype=np.int64)
    hi = np.full(n, fk.key_max, dtype=np.int64)
    v_lo = np.asarray(func(fk.values(lo)), dtype=bool)
    v_hi = np.asarray(func(fk.values(hi)), dtype=bool)
    const = v_lo == v_hi
    result = np.where(const, fk.key_max, fk.key_min).astype(np.int64)
    active = ~const
    if not active.any():
        return result
    idx = np.nonzero(active)[0]
    lo = lo[idx].copy()
    hi = hi[idx].copy()
    target = v_lo[idx]
    # Invariant: f(lo) == target, f(hi) != target.  For float64 the span between
    # key(-inf) and key(+inf) exceeds the int64 range, so distances are computed in
    # wrap-around uint64 arithmetic (exact, since every true distance is < 2**64).
    while True:
        gap_u = hi.view(np.uint64) - lo.view(np.uint64)
        todo = gap_u > np.uint64(1)
        if not todo.any():
            break
        mid = (lo.view(np.uint64) + (gap_u >> np.uint64(1))).view(np.int64)
        mid = np.where(todo, mid, lo)
        f_mid = np.asarray(func_sub(func, fk, mid, idx, n), dtype=bool)
        same = f_mid == target
        lo = np.where(todo & same, mid, lo)
        hi = np.where(todo & ~same, mid, hi)
    result[idx] = lo
    return result


def func_sub(func, fk: FloatKeys, keys_sub: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """Evaluate ``func`` on a subset: fill a full-length value vector, return the subset."""
    full = np.zeros(n, dtype=fk.dtype)
    full[idx] = fk.values(keys_sub)
    out = np.asarray(func(full), dtype=bool)
    return out[idx]
