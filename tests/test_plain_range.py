"""Plain-range mode must agree with the default extended-range rules at ordinary magnitudes."""

import math

import pytest
import torch
import torch.nn.functional as F

from polytensors import PolyTensor


DEGREE = 5


def _series(shape, seed, complex_directions=True):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(shape, dtype=torch.float64, generator=generator)
    directions = []
    for _ in range(DEGREE):
        real = torch.randn(shape, dtype=torch.float64, generator=generator) * 0.5
        if complex_directions:
            imag = torch.randn(shape, dtype=torch.float64, generator=generator) * 0.5
            directions.append(torch.complex(real, imag))
        else:
            directions.append(real)
    return base, directions


def _poly(base, directions, requires_grad=False):
    return PolyTensor((base.clone(), *(d.clone() for d in directions)), degree=DEGREE, requires_grad=requires_grad)


OPERATIONS = {
    "exp": lambda x: torch.exp(x),
    "log": lambda x: torch.log(x * x + 1),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "gelu": F.gelu,
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "log_softmax": lambda x: torch.log_softmax(x, dim=-1),
    "logsumexp": lambda x: torch.logsumexp(x, dim=-1),
    "cross_entropy": lambda x: F.cross_entropy(x, torch.tensor([1, 3, 0]), reduction="none"),
    "rms_norm": lambda x: F.rms_norm(x, (x.shape[-1],), eps=1e-5),
}


@pytest.mark.parametrize("name", sorted(OPERATIONS))
@pytest.mark.parametrize("complex_directions", (False, True))
def test_plain_forward_matches_default(name, complex_directions):
    base, directions = _series((3, 5), seed=1, complex_directions=complex_directions)
    operation = OPERATIONS[name]
    default = operation(_poly(base, directions))
    with PolyTensor.plain_range():
        plain = operation(_poly(base, directions))
    for a, b in zip(default.coeffs, plain.coeffs):
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", sorted(OPERATIONS))
def test_plain_backward_matches_default(name):
    base, directions = _series((3, 5), seed=2)
    operation = OPERATIONS[name]
    upstream_base, upstream_directions = _series(operation(base).shape, seed=3)

    def gradient(plain):
        x = _poly(base, directions, requires_grad=True)
        upstream = _poly(upstream_base, upstream_directions)
        with PolyTensor.retain_wrappers():
            if plain:
                with PolyTensor.plain_range():
                    (operation(x) * upstream).sum().backward()
            else:
                (operation(x) * upstream).sum().backward()
        return x.grad.coeffs

    for a, b in zip(gradient(False), gradient(True)):
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("is_causal", (False, True))
def test_plain_attention_matches_default(is_causal):
    q_base, q_dirs = _series((1, 2, 4, 3), seed=4)
    k_base, k_dirs = _series((1, 2, 4, 3), seed=5)
    v_base, v_dirs = _series((1, 2, 4, 3), seed=6)

    def run(plain):
        q, k, v = (_poly(b, d, requires_grad=True) for b, d in ((q_base, q_dirs), (k_base, k_dirs), (v_base, v_dirs)))
        with PolyTensor.retain_wrappers():
            if plain:
                with PolyTensor.plain_range():
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                    out.sum().backward()
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                out.sum().backward()
        return out.coeffs, q.grad.coeffs, k.grad.coeffs, v.grad.coeffs

    for group_default, group_plain in zip(run(False), run(True)):
        for a, b in zip(group_default, group_plain):
            torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)


def test_rms_norm_matches_independent_higher_derivatives():
    # Real direction: coefficient k equals (d/dt)^k rms_norm(x + t v) / k! at t = 0.
    base, directions = _series((4,), seed=7, complex_directions=False)
    x = PolyTensor((base, directions[0]), degree=4)
    with PolyTensor.plain_range():
        result = F.rms_norm(x, (4,), eps=1e-5)
    t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    curve = F.rms_norm(base + t * directions[0], (4,), eps=1e-5)
    for k in range(5):
        expected = torch.stack([curve[i] for i in range(4)]) / math.factorial(k)
        torch.testing.assert_close(result.coeffs[k], expected.detach(), rtol=1e-10, atol=1e-12)
        if k < 4:
            curve = torch.stack([torch.autograd.grad(curve[i], t, create_graph=True)[0] for i in range(4)])


def test_plain_range_is_context_local_and_restored():
    from polytensors._plain import plain_range_enabled

    assert not plain_range_enabled()
    with PolyTensor.plain_range():
        assert plain_range_enabled()
    assert not plain_range_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA NLL kernels reject complex inputs")
@pytest.mark.parametrize("plain", (False, True))
def test_cross_entropy_with_complex_directions_on_cuda_matches_cpu(plain):
    base, directions = _series((3, 5), seed=8)
    target = torch.tensor([1, 3, 0])

    def run(device):
        x = PolyTensor((base.to(device), *(d.to(device) for d in directions)), degree=DEGREE, requires_grad=True)
        with PolyTensor.retain_wrappers():
            if plain:
                with PolyTensor.plain_range():
                    loss = F.cross_entropy(x, target.to(device))
                    loss.backward()
            else:
                loss = F.cross_entropy(x, target.to(device))
                loss.backward()
        return [c.cpu() for c in loss.coeffs], [c.cpu() for c in x.grad.coeffs]

    for group_cpu, group_cuda in zip(run("cpu"), run("cuda")):
        for a, b in zip(group_cpu, group_cuda):
            torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)
