"""Library-independent representation of a tree ensemble.

A :class:`Model` is a list of :class:`Tree` objects plus a :class:`~leafparity.routers.Router`
that knows the *exact* arithmetic the originating runtime uses to decide, for a
raw user input value, which child of a node an input goes to.

Outputs are expressed in a *canonical raw space* shared by both sides of a
comparison (margins / raw scores for boosted models, probabilities for
forests), so that a leaf's contribution in the original model and in the
converted model can be subtracted directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class UnsupportedModelError(Exception):
    """The model uses a feature leafparity cannot analyse exactly.

    leafparity refuses rather than guesses: an analysis that silently skipped an
    unsupported construct would produce a certificate that is not true.
    """


@dataclass
class Tree:
    """One decision tree.

    ``children[i]`` is ``(child_if_true, child_if_false)`` for internal node ``i`` in
    the runtime's own terms (the runtime's "yes/left" branch first), ``-1`` for leaves.
    ``value[i]`` is leaf ``i``'s contribution to the canonical raw outputs (already
    scaled by averaging / learning-rate factors).  ``node_ids`` are the node ids as the
    originating library names them, used only for reporting.
    """

    children: np.ndarray          # (n_nodes, 2) int64
    feature: np.ndarray           # (n_nodes,) int64, -1 for leaves
    value: np.ndarray             # (n_nodes, n_outputs) float64
    node_ids: np.ndarray          # (n_nodes,) int64
    tree_id: Any = None
    # filled in by normalisation (see normalize.py)
    segments: Optional[List[Any]] = None
    parent: Optional[np.ndarray] = None

    @property
    def n_nodes(self) -> int:
        return int(self.feature.shape[0])

    def is_leaf(self, i: int) -> bool:
        return self.feature[i] < 0

    def leaves(self) -> np.ndarray:
        return np.nonzero(self.feature < 0)[0]

    def compute_parents(self) -> np.ndarray:
        p = np.full(self.n_nodes, -1, dtype=np.int64)
        for i in range(self.n_nodes):
            if self.feature[i] >= 0:
                for c in self.children[i]:
                    p[c] = i
        self.parent = p
        return p


@dataclass
class Model:
    """A tree ensemble plus everything needed to reproduce its routing exactly."""

    library: str                      # 'xgboost' | 'lightgbm' | 'sklearn' | 'onnx'
    task: str                         # 'regression' | 'binary' | 'multiclass'
    n_features: int
    n_outputs: int                    # size of the canonical raw output vector
    trees: List[Tree]
    base: np.ndarray                  # (n_outputs,) raw offset added to every prediction
    router: Any                       # leafparity.routers.Router
    description: str = ""
    feature_names: Optional[List[str]] = None
    accepts_nan: bool = True          # whether the runtime accepts NaN inputs at all
    accepts_inf: bool = True
    # float dtype in which the runtime accumulates leaf values (for rounding bounds)
    accumulate_dtype: Any = np.float64
    notes: List[str] = field(default_factory=list)
    raw_meaning: str = "raw score"    # human description of the canonical raw output
    source: Any = None                # original python object / path, for runtimes

    @property
    def n_nodes(self) -> int:
        return sum(t.n_nodes for t in self.trees)

    def summary(self) -> Dict[str, Any]:
        return {
            "library": self.library,
            "task": self.task,
            "n_trees": len(self.trees),
            "n_nodes": self.n_nodes,
            "n_features": self.n_features,
            "n_outputs": self.n_outputs,
            "raw_meaning": self.raw_meaning,
            "description": self.description,
        }
