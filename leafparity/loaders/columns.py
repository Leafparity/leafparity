"""Per-column preprocessing in front of the trees.

A :class:`Columns` describes the data the trees see: for every column ``k``, the user's
input column ``src[k]`` it is computed from, and the exact elementwise steps applied to
it on the way, in order, each in its own dtype.  Selecting, reordering and concatenating
columns (a scikit-learn ColumnTransformer, ONNX ArrayFeatureExtractor / Gather / Concat)
only moves these descriptions around; arithmetic is never merged or simplified.

:meth:`Columns.to_chain` turns the description into a :class:`~leafparity.routers.Chain`
that evaluates exactly those steps for each column.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..ir import UnsupportedModelError
from ..routers import Chain

Step = Tuple[str, Optional[float], np.dtype]   # (op, constant or None, dtype)


class Columns:
    def __init__(self, n: int, dtype, src=None, ops=None):
        self.dtype = np.dtype(dtype)
        self.src = list(range(n)) if src is None else list(src)
        self.ops: List[Tuple[Step, ...]] = [()] * n if ops is None else list(ops)

    @property
    def n(self) -> int:
        return len(self.src)

    def copy(self) -> "Columns":
        return Columns(self.n, self.dtype, self.src, self.ops)

    def select(self, idx: Sequence[int]) -> "Columns":
        idx = [int(i) for i in idx]
        if any(not 0 <= i < self.n for i in idx):
            raise UnsupportedModelError(f"column index out of range (have {self.n} columns)")
        return Columns(len(idx), self.dtype, [self.src[i] for i in idx], [self.ops[i] for i in idx])

    def apply(self, steps) -> None:
        """Append chain-style steps ``(op, per-column constants or None, dtype)``."""
        for op, consts, dt in steps:
            dt = np.dtype(dt)
            if consts is None:
                self.ops = [o + ((op, None, dt),) for o in self.ops]
            else:
                c = np.asarray(consts).reshape(-1)
                if c.size == 1:
                    c = np.full(self.n, c[0])
                if c.size != self.n:
                    raise UnsupportedModelError(
                        f"elementwise constant of length {c.size} for {self.n} columns")
                self.ops = [o + ((op, c[k], dt),) for k, o in enumerate(self.ops)]
            self.dtype = dt

    @staticmethod
    def concat(parts: Sequence["Columns"], dtype) -> "Columns":
        """Columns side by side, as one array of ``dtype``; a part in another dtype must
        widen to it exactly (float32 -> float64), like numpy's hstack."""
        dtype = np.dtype(dtype)
        out = Columns(0, dtype)
        for p in parts:
            p = p.copy()
            if p.dtype != dtype:
                if not np.can_cast(p.dtype, dtype, casting="safe"):
                    raise UnsupportedModelError(f"cannot join {p.dtype} columns into {dtype} exactly")
                p.apply([("cast", None, dtype)])
            out.src += p.src
            out.ops += p.ops
        return out

    def is_identity(self) -> bool:
        return self.src == list(range(self.n)) and all(not o for o in self.ops)

    def to_chain(self) -> Chain:
        """One chain step per column step; columns whose steps differ (different
        transformers of a ColumnTransformer, different ONNX branches) get a group each."""
        groups = {}
        for k, o in enumerate(self.ops):
            groups.setdefault(tuple((op, dt) for op, _, dt in o), []).append(k)

        def chain_of(signature, cols):
            steps = []
            for j, (op, dt) in enumerate(signature):
                if op == "cast":
                    steps.append((op, None, dt))
                else:
                    c = np.zeros(self.n, dtype=dt)
                    c[cols] = [self.ops[k][j][1] for k in cols]
                    steps.append((op, c, dt))
            return Chain(steps)

        if len(groups) == 1:
            ((signature, cols),) = groups.items()
            return chain_of(signature, cols)
        parts = []
        for signature, cols in groups.items():
            sub = chain_of(signature, cols)
            if not signature or signature[-1][1] != self.dtype:
                sub = sub.then(Chain([("cast", None, self.dtype)]))
            parts.append((np.asarray(cols, dtype=np.int64), sub))
        return Chain([("split", parts, self.dtype)])
