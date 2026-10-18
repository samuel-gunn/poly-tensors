import math

import pytest
import torch
import torch.nn.functional as F

from polytensors import PolyTensor
from polytensors import _fused_attention

pytestmark = pytest.mark.skipif(not _fused_attention.available(), reason="needs CUDA and Triton")


def _series(seed, shape, dtype, degree=1, requires_grad=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    coeffs = tuple(torch.randn(shape, generator=g, device="cuda", dtype=dtype) for _ in range(degree + 1))
    return PolyTensor(coeffs, requires_grad=requires_grad)


def _window_mask(L, window, device="cuda"):
    i = torch.arange(L, device=device)
    return (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] <= window)


def _run(fused, dtype, shape, causal, mask, seed=0):
    q, k, v = (_series(seed + i, shape, dtype) for i in range(3))
    g = _series(seed + 3, shape, dtype, requires_grad=False)
    calls = []
    original = _fused_attention.attention

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    _fused_attention.attention = spy
    try:
        with PolyTensor.retain_wrappers(), PolyTensor.plain_range(), PolyTensor.fused_attention(fused):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
            out.backward(g)
    finally:
        _fused_attention.attention = original
    return out.coeffs, [x.grad.coeffs for x in (q, k, v)], bool(calls)


def _relative(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
@pytest.mark.parametrize("causal,window", [(True, None), (False, 37), (False, None)])
def test_fused_matches_general_series_rules(dtype, tol, causal, window):
    torch.backends.cuda.matmul.allow_tf32 = False
    shape = (2, 3, 150, 64)
    mask = None if window is None else _window_mask(150, window)
    out_f, grads_f, used_f = _run(True, dtype, shape, causal, mask)
    out_g, grads_g, used_g = _run(False, dtype, shape, causal, mask)
    assert used_f and not used_g
    for a, b in zip(out_f, out_g):
        assert _relative(a, b) < tol
    for gf, gg in zip(grads_f, grads_g):
        for a, b in zip(gf, gg):
            assert _relative(a, b) < tol


def test_bfloat16_close_to_float32():
    shape = (1, 2, 256, 128)
    out_b, grads_b, _ = _run(True, torch.bfloat16, shape, True, None)
    out_f, grads_f, _ = _run(True, torch.float32, shape, True, None)
    for a, b in zip(out_b, out_f):
        assert _relative(a.float(), b) < 2e-2
    for gb, gf in zip(grads_b, grads_f):
        for a, b in zip(gb, gf):
            assert _relative(a.float(), b) < 5e-2


def test_coefficient_autograd_mode_matches_wrapper_autograd():
    shape = (1, 2, 96, 32)
    dtype = torch.float64
    coefficients = [tuple(c.requires_grad_() for c in _series(i, shape, dtype, requires_grad=False).coeffs)
                    for i in range(3)]
    q, k, v = (PolyTensor(c, requires_grad=False) for c in coefficients)
    with PolyTensor.coefficient_autograd(), PolyTensor.plain_range():
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # Loss = sum of the order-1 coefficient: its gradient w.r.t. each order-0 coefficient is order 1 of the
    # gradient series of sum(out), i.e. what wrapper autograd returns for G = (1, 0).
    first = torch.autograd.grad(out.coeffs[1].sum(), [c[0] for c in coefficients])
    q2, k2, v2 = (PolyTensor(tuple(x.detach() for x in c), requires_grad=True) for c in coefficients)
    with PolyTensor.retain_wrappers(), PolyTensor.plain_range():
        out2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        out2.backward(PolyTensor.constant(torch.ones_like(out2.value), 1))
    for x, g in zip((q2, k2, v2), first):
        assert _relative(g, x.grad.coeffs[1]) < 1e-12
    assert _relative(out.coeffs[1], out2.coeffs[1]) < 1e-12


@pytest.mark.parametrize("case", ["degree2", "float_mask", "dropout", "disabled"])
def test_ineligible_calls_use_general_rules(case):
    shape = (1, 1, 32, 16)
    degree = 2 if case == "degree2" else 1
    q, k, v = (_series(i, shape, torch.float32, degree=degree, requires_grad=False) for i in range(3))
    kwargs = {}
    if case == "float_mask":
        kwargs["attn_mask"] = torch.zeros(32, 32, device="cuda")
    if case == "dropout":
        kwargs["dropout_p"] = 0.5
    assert not _fused_attention.eligible(q, k, v, kwargs.get("attn_mask"), kwargs.get("dropout_p", 0.0),
                                         False, False) or case == "disabled"
    if case == "disabled":
        with PolyTensor.plain_range(), PolyTensor.fused_attention(False):
            assert not _fused_attention.eligible(q, k, v, None, 0.0, False, False)


def test_no_grad_forward_and_plain_operands():
    shape = (2, 2, 64, 32)
    q = _series(0, shape, torch.float64, requires_grad=False)
    k = torch.randn(shape, device="cuda", dtype=torch.float64)      # an ordinary tensor: zero tangent
    v = _series(2, shape, torch.float64, requires_grad=False)
    with torch.no_grad(), PolyTensor.plain_range():
        fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        with PolyTensor.fused_attention(False):
            general = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    for a, b in zip(fused.coeffs, general.coeffs):
        assert _relative(a, b) < 1e-12


def test_scale_argument():
    shape = (1, 2, 40, 64)
    q, k, v = (_series(i, shape, torch.float64, requires_grad=False) for i in range(3))
    with torch.no_grad(), PolyTensor.plain_range():
        fused = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.3)
        with PolyTensor.fused_attention(False):
            general = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.3)
    for a, b in zip(fused.coeffs, general.coeffs):
        assert _relative(a, b) < 1e-12
    assert math.isclose(_fused_attention.default_scale(64), 0.125)
