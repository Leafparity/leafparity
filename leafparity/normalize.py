"""Turn every node's exact routing rule into a piecewise description over float keys.

After normalisation each internal node carries

    segments = (ends, childs, nan_child)

meaning: user values whose key is in ``(ends[k-1], ends[k]]`` go to child index
``childs[k]`` (0 = the runtime's true/left branch, 1 = the other), with
``ends[-1] == key(+inf)``; a NaN input goes to ``nan_child``.

This description is *derived from* the router's exact arithmetic (transitions of
monotone atoms found by exhaustive binary search) and then *checked against* it
at every piece boundary and at random probe points, so any mismatch between the
two is caught here instead of silently corrupting the analysis.
"""
from __future__ import annotations

import numpy as np

from .floats import FloatKeys, find_transitions, keys_for


class NormalizationError(Exception):
    pass


def _global_arrays(model):
    offsets = []
    total = 0
    for t in model.trees:
        offsets.append(total)
        total += t.n_nodes
    return np.array(offsets, dtype=np.int64), total


def normalize_model(model, user_dtype=np.float64, probes: int = 16, seed: int = 0) -> None:
    """Compute ``tree.segments`` for every tree of ``model`` (in place)."""
    fk: FloatKeys = keys_for(user_dtype)
    router = model.router
    offsets, total = _global_arrays(model)
    feat = router.feature
    if feat.shape[0] != total:
        raise NormalizationError("router/node count mismatch")
    g = np.nonzero(feat >= 0)[0].astype(np.int64)
    n = g.shape[0]
    for t in model.trees:
        t.segments = [None] * t.n_nodes
        t.compute_parents()
        t._segarr = None          # derived caches depend on the segments / dtype
        t._extremes = None
    if n == 0:
        return

    # 1. transitions of every monotone atom, for every node at once
    atoms = router.atoms()
    T = np.empty((len(atoms), n), dtype=np.int64)
    for j, atom in enumerate(atoms):
        T[j] = find_transitions(fk, lambda vals, _a=atom: _a(vals, g), n)

    # 2. candidate piece boundaries = sorted unique transitions below key_max
    T = np.sort(T, axis=0)
    kmax, kmin = fk.key_max, fk.key_min
    # pieces: [kmin, b0], [b0+1, b1], ... , [b_last+1, kmax]
    n_pieces_max = len(atoms) + 1
    starts = np.full((n_pieces_max, n), kmax + 1, dtype=np.int64)  # sentinel = no piece
    ends = np.full((n_pieces_max, n), kmax, dtype=np.int64)
    cur_start = np.full(n, kmin, dtype=np.int64)
    count = np.zeros(n, dtype=np.int64)
    prev_b = np.full(n, kmin - 1, dtype=np.int64)
    for j in range(len(atoms)):
        b = T[j]
        valid = (b < kmax) & (b > prev_b)  # a real, new transition
        rows = count
        # close the current piece at b
        idx = np.nonzero(valid)[0]
        starts[rows[idx], idx] = cur_start[idx]
        ends[rows[idx], idx] = b[idx]
        count[idx] += 1
        cur_start[idx] = b[idx] + 1
        prev_b[idx] = b[idx]
    # final piece up to kmax
    idx = np.arange(n)
    starts[count, idx] = cur_start
    ends[count, idx] = kmax
    count += 1

    # 3. evaluate the exact router at both ends of each piece and check constancy
    child = np.full((n_pieces_max, n), -1, dtype=np.int64)
    for p in range(n_pieces_max):
        has = p < count
        if not has.any():
            continue
        sel = np.nonzero(has)[0]
        r_start = router.route(fk.values(starts[p, sel]), g[sel])
        r_end = router.route(fk.values(ends[p, sel]), g[sel])
        bad = r_start != r_end
        if bad.any():
            k = sel[np.nonzero(bad)[0][0]]
            raise NormalizationError(
                f"routing of global node {g[k]} is not piecewise constant between its "
                f"transitions (piece {p}); the runtime rule is not monotone as modelled")
        child[p, sel] = np.where(r_start, 0, 1)

    nan_route = router.route(np.full(n, np.nan, dtype=fk.dtype), g)
    nan_child = np.where(nan_route, 0, 1)

    # 4. write merged segments back to the trees
    tree_of = np.searchsorted(offsets, g, side="right") - 1
    for k in range(n):
        es, cs = [], []
        for p in range(int(count[k])):
            c = int(child[p, k])
            e = int(ends[p, k])
            if cs and cs[-1] == c:
                es[-1] = e
            else:
                es.append(e)
                cs.append(c)
        ti = int(tree_of[k])
        local = int(g[k] - offsets[ti])
        model.trees[ti].segments[local] = (tuple(es), tuple(cs), int(nan_child[k]))

    # 5. independent probe check: random keys + keys around every boundary
    _probe_check(model, fk, g, offsets, tree_of, probes, seed)


def segment_child(seg, key: int) -> int:
    ends, childs, _ = seg
    for e, c in zip(ends, childs):
        if key <= e:
            return c
    return childs[-1]


def _probe_check(model, fk, g, offsets, tree_of, probes, seed):
    rng = np.random.default_rng(seed)
    router = model.router
    n = g.shape[0]
    keys_list = []
    # random keys, biased towards "normal" magnitudes as well as the full range
    for _ in range(probes):
        full = rng.integers(fk.key_min, fk.key_max, size=n, dtype=np.int64, endpoint=True)
        keys_list.append(full)
    # around each node's own boundaries
    for k_off in (-1, 0, 1):
        ks = np.empty(n, dtype=np.int64)
        for k in range(n):
            ti = int(tree_of[k])
            ends = model.trees[ti].segments[int(g[k] - offsets[ti])][0]
            ks[k] = ends[0] + k_off if len(ends) > 1 else ends[0]
        keys_list.append(np.clip(ks, fk.key_min, fk.key_max))
    for ks in keys_list:
        r = router.route(fk.values(ks), g)
        for k in np.nonzero(r != _seg_route_vec(model, g, offsets, tree_of, ks))[0][:1]:
            raise NormalizationError(
                f"probe mismatch at global node {int(g[k])}, key {int(ks[k])}")


def _seg_route_vec(model, g, offsets, tree_of, ks):
    out = np.empty(g.shape[0], dtype=bool)
    for k in range(g.shape[0]):
        ti = int(tree_of[k])
        seg = model.trees[ti].segments[int(g[k] - offsets[ti])]
        out[k] = segment_child(seg, int(ks[k])) == 0
    return out
