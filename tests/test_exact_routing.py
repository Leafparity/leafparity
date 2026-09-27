"""The core guarantee: leafparity's exact model of each runtime reaches the same leaf as
the real runtime for every tested input, including values exactly on and one float
step either side of every threshold, zeros, tiny values, and missing values."""
import numpy as np
import pytest

from leafparity import runtimes
from leafparity.engine import ir_leaves, ir_raw, make_domain
from leafparity.loaders import load_onnx, load_original
from leafparity.normalize import normalize_model
from leafparity.probing import adversarial_matrix

from zoo import F, all_models, data

NAMES = [n for n, _, _ in all_models()]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("udt", [np.float64, np.float32])
def test_leaf_for_leaf(name, udt):
    X, Xn, *_ = data()
    m, onx = next((m, o) for n, m, o in all_models() if n == name)
    om = load_original(m, udt)
    cm = load_onnx(onx, udt)
    normalize_model(om, udt)
    normalize_model(cm, udt)
    dom = make_domain(F, udt, allow_nan=om.accepts_nan)
    A = adversarial_matrix([om, cm], dom, n_rows=2500, seed=3,
                           background=(Xn if om.accepts_nan else X))
    for mod in (om, cm):
        real = runtimes.leaf_indices(mod, A)
        mine = ir_leaves(mod, A, dom.fk)
        assert np.array_equal(real, mine), f"{name}/{mod.library}: {(real != mine).sum()} mismatches"
        raw_real = runtimes.raw_outputs(mod, A)
        raw_mine = ir_raw(mod, mine)
        scale = 1 + np.abs(raw_real).max()
        assert np.max(np.abs(raw_real - raw_mine)) <= 1e-4 * scale, name


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("udt", [np.float64, np.float32])
def test_every_node_boundary_reached(name, udt):
    """Stronger: every internal node of both models is *reached* by constructed inputs sitting
    on / next to its boundary (from the normalised rule and, independently, from the raw
    threshold parameter)."""
    from leafparity.probing import node_probe_matrix
    X, Xn, *_ = data()
    m, onx = next((m, o) for n, m, o in all_models() if n == name)
    om = load_original(m, udt)
    cm = load_onnx(onx, udt)
    normalize_model(om, udt)
    normalize_model(cm, udt)
    dom = make_domain(F, udt, allow_nan=om.accepts_nan)
    for probe_src in (om, cm):
        A = node_probe_matrix(probe_src, dom)
        assert A.shape[0] > 0
        for mod in (om, cm):
            real = runtimes.leaf_indices(mod, A)
            mine = ir_leaves(mod, A, dom.fk)
            bad = real != mine
            assert not bad.any(), f"{name}/{mod.library}: {bad.sum()} mismatches, e.g. row {A[np.argwhere(bad)[0][0]]}"
