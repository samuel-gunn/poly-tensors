import gc

import pytest
import torch

from polytensors import PolyTensor


def _poly(seed, shape=(4, 3), degree=2, requires_grad=False, complex_directions=False):
    g = torch.Generator().manual_seed(seed)
    coeffs = [torch.randn(shape, generator=g, dtype=torch.float64) for _ in range(degree + 1)]
    if complex_directions:
        coeffs[1:] = [c + 1j * torch.randn(shape, generator=g, dtype=torch.float64) for c in coeffs[1:]]
    return PolyTensor(tuple(coeffs), degree=degree, requires_grad=requires_grad)


CONSTANT_OPS = [
    lambda x: x * 2.5,
    lambda x: 2.5 * x,
    lambda x: x + 1.5,
    lambda x: 1.5 + x,
    lambda x: x - 1.5,
    lambda x: 1.5 - x,
    lambda x: x / 4.0,
    lambda x: torch.add(x, 2.0, alpha=3),
    lambda x: torch.sub(x, 2.0, alpha=3),
    lambda x: x * torch.arange(3.0, dtype=torch.float64),
    lambda x: torch.arange(3.0, dtype=torch.float64) + x,
    lambda x: torch.arange(3.0, dtype=torch.float64) - x,
    lambda x: x - torch.ones(2, 1, 3, dtype=torch.float64),   # broadcasts to a larger shape
    lambda x: x / (1 + torch.arange(3.0, dtype=torch.float64)),
]


@pytest.mark.parametrize("complex_directions", [False, True])
@pytest.mark.parametrize("op", range(len(CONSTANT_OPS)))
def test_plain_constant_operands_match_extended_range(op, complex_directions):
    x = _poly(0, complex_directions=complex_directions)
    expected = CONSTANT_OPS[op](x)
    with PolyTensor.plain_range():
        result = CONSTANT_OPS[op](x)
    assert result.shape == expected.shape
    for a, b in zip(result.coeffs, expected.coeffs):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=1e-15, atol=0)
        # results own their storage: no order aliases an input coefficient
        assert not any(torch._C._overlaps(a, c) for c in x.coeffs)


@pytest.mark.parametrize("update", ["add_", "sub_", "mul_", "div_"])
@pytest.mark.parametrize("constant", [2.0, "tensor"])
def test_plain_in_place_constant_updates(update, constant):
    c = torch.arange(1.0, 4.0, dtype=torch.float64) if constant == "tensor" else constant
    x, y = _poly(1), _poly(1)
    getattr(x, update)(c)
    with PolyTensor.plain_range():
        storage = y.coeffs
        getattr(y, update)(c)
    assert all(a is b for a, b in zip(y.coeffs, storage))
    for a, b in zip(y.coeffs, x.coeffs):
        torch.testing.assert_close(a, b, rtol=1e-15, atol=0)


def test_forward_backward_creates_no_reference_cycles():
    """Operands and results must be freed by reference counting alone: a cycle would keep coefficient storage
    alive until a cyclic collection, which can then run in the middle of a backward pass."""
    w = _poly(2, shape=(3, 3), degree=2, requires_grad=True)
    v = _poly(3, shape=(5, 3), degree=2)

    def step():
        with PolyTensor.retain_wrappers(), PolyTensor.plain_range():
            h = torch.tanh(v @ w) * 0.5 + 1.0
            loss = torch.log_softmax(torch.cat([h, h.exp()], dim=1), dim=1).mean()
            loss.backward()
        w.grad = None

    step()
    enabled = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        step()
        assert gc.collect() == 0
    finally:
        if enabled:
            gc.enable()


def test_well_formed_coefficients_are_kept_and_overlaps_cloned():
    a, b = torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.complex128)
    p = PolyTensor((a, b))
    assert p.coeffs[0] is a and p.coeffs[1] is b
    q = PolyTensor((a, a))
    assert q.coeffs[0] is a and not torch._C._overlaps(q.coeffs[0], q.coeffs[1])
