import numpy as np
import pytest

from leafparity.floats import F32, F64, find_transitions


@pytest.mark.parametrize("fk", [F32, F64])
def test_keys_preserve_order_and_roundtrip(fk):
    rng = np.random.default_rng(0)
    vals = np.concatenate([
        rng.normal(size=5000) * 10.0 ** rng.integers(-40, 38, size=5000),
        [0.0, -0.0, np.inf, -np.inf, np.finfo(fk.dtype).tiny, -np.finfo(fk.dtype).tiny,
         np.finfo(fk.dtype).max, -np.finfo(fk.dtype).max, 1e-45, -1e-45],
    ]).astype(fk.dtype)
    vals = vals[np.isfinite(vals) | np.isinf(vals)]
    keys = fk.keys(vals)
    order_v = np.argsort(vals, kind="stable")
    sv = vals[order_v]
    sk = keys[order_v]
    # key order == value order (ties only for equal values)
    assert np.all(np.diff(sk) >= 0)
    eq = sv[1:] == sv[:-1]
    assert np.all((np.diff(sk) == 0) == eq)
    back = fk.values(keys)
    assert np.all((back == vals))
    assert fk.key(-0.0) == fk.key(0.0) == 0
    assert fk.key(np.inf) == fk.key_max and fk.key(-np.inf) == fk.key_min


@pytest.mark.parametrize("fk", [F32, F64])
def test_consecutive_keys_are_consecutive_floats(fk):
    for v in [1.0, -1.0, 1e-30, -3.5, 12345.678, fk.dtype.type(np.finfo(fk.dtype).tiny)]:
        k = fk.key(v)
        up = fk.value(k + 1)
        dn = fk.value(k - 1)
        assert up == np.nextafter(fk.dtype.type(v), fk.dtype.type(np.inf))
        assert dn == np.nextafter(fk.dtype.type(v), fk.dtype.type(-np.inf))


@pytest.mark.parametrize("fk", [F32, F64])
def test_find_transitions_matches_bruteforce(fk):
    rng = np.random.default_rng(1)
    t = (rng.normal(size=200) * 10.0 ** rng.integers(-5, 6, size=200)).astype(fk.dtype)
    # monotone decreasing: v <= t ; monotone increasing: v > t ; constant: always True
    f_le = lambda v: v <= t
    tr = find_transitions(fk, f_le, t.shape[0])
    assert np.array_equal(fk.values(tr), t)          # last key with v <= t is t itself
    f_gt = lambda v: v > t
    tr2 = find_transitions(fk, f_gt, t.shape[0])
    assert np.array_equal(tr2, tr)
    const = find_transitions(fk, lambda v: np.ones(v.shape, bool), t.shape[0])
    assert np.all(const == fk.key_max)
