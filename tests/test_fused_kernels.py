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


# --- fused cross entropy -------------------------------------------------------------------------------

from polytensors import _fused_cross_entropy  # noqa: E402


def _ce_run(fused, dtype, reduction, N=300, C=5000, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = PolyTensor(tuple(3 * torch.randn(N, C, generator=g, device="cuda", dtype=dtype) for _ in range(2)),
                        requires_grad=True)
    target = torch.randint(0, C, (N,), generator=g, device="cuda")
    target[::7] = -1
    with PolyTensor.retain_wrappers(), PolyTensor.plain_range(), PolyTensor.fused_cross_entropy(fused):
        assert _fused_cross_entropy.eligible(logits, target, None, reduction, 0.0) == fused
        loss = F.cross_entropy(logits, target, ignore_index=-1, reduction=reduction)
        weights = PolyTensor(tuple(torch.randn(loss.shape, generator=g, device="cuda", dtype=dtype)
                                   for _ in range(2)))
        (loss * weights).sum().backward()
    return loss.coeffs, logits.grad.coeffs


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_fused_cross_entropy_matches_general_rules(reduction):
    out_f, grad_f = _ce_run(True, torch.float32, reduction)
    out_g, grad_g = _ce_run(False, torch.float32, reduction)
    for a, b in zip(out_f, out_g):
        assert _relative(a, b) < 1e-5
    for a, b in zip(grad_f, grad_g):
        assert _relative(a, b) < 1e-5


def test_fused_cross_entropy_coefficient_map_gradcheck_like():
    N, C = 17, 300
    z = [torch.randn(N, C, device="cuda", dtype=torch.float32, requires_grad=True) for _ in range(2)]
    y = torch.randint(0, C, (N,), device="cuda")
    l0, l1 = _fused_cross_entropy.CoefficientCrossEntropy.apply(z[0], z[1], y, -100)
    zr = [t.detach().double().requires_grad_() for t in z]
    p = torch.softmax(zr[0], -1)
    r0 = torch.logsumexp(zr[0], -1) - zr[0].gather(1, y[:, None])[:, 0]
    r1 = (p * zr[1]).sum(-1) - zr[1].gather(1, y[:, None])[:, 0]
    c = [torch.randn(N, device="cuda") for _ in range(2)]
    g = torch.autograd.grad((l0, l1), z, c)
    gr = torch.autograd.grad((r0, r1), zr, [t.double() for t in c])
    assert _relative(l0.double(), r0) < 1e-6 and _relative(l1.double(), r1) < 1e-6
    for a, b in zip(g, gr):
        assert _relative(a.double(), b) < 1e-5


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("strided", [False, True])
def test_softcapped_cross_entropy_matches_composite(dtype, tol, strided):
    import polytensors

    N, C, pad, cap = 200, 3000, 64 if strided else 0, 15.0
    g = torch.Generator(device="cuda").manual_seed(1)
    full = [40 * torch.randn(N, C + pad, generator=g, device="cuda", dtype=dtype) for _ in range(2)]
    target = torch.randint(0, C, (N,), generator=g, device="cuda")
    target[::5] = -1
    results = []
    for use_fused in (True, False):
        logits = PolyTensor(tuple(f.clone() for f in full), requires_grad=True)
        with PolyTensor.retain_wrappers(), PolyTensor.plain_range(), PolyTensor.fused_cross_entropy(use_fused):
            sliced = logits[:, :C]
            if use_fused:
                loss = PolyTensor.cross_entropy(sliced, target, softcap=cap, ignore_index=-1, reduction="none")
            else:
                z = sliced.float()
                loss = F.cross_entropy(cap * torch.tanh(z / cap), target, ignore_index=-1, reduction="none")
            w = PolyTensor((torch.linspace(-1, 1, N, device="cuda"), torch.linspace(1, 2, N, device="cuda")))
            (loss * w).sum().backward()
        results.append((loss.coeffs, logits.grad.coeffs))
    (out_f, grad_f), (out_g, grad_g) = results
    for a, b in zip(out_f, out_g):
        assert a.dtype == torch.float32 and _relative(a, b.float()) < tol
    for a, b in zip(grad_f, grad_g):
        assert _relative(a.float(), b.float()) < tol


def test_public_cross_entropy_on_plain_tensors():
    import polytensors

    z = torch.randn(10, 50, device="cuda")
    y = torch.randint(0, 50, (10,), device="cuda")
    expected = F.cross_entropy(15 * torch.tanh(z / 15), y)
    assert torch.allclose(PolyTensor.cross_entropy(z, y, softcap=15.0), expected)
