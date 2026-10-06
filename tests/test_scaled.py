"""Independent checks of internal extended-exponent arithmetic."""
import math

import pytest
import torch

from polytensors._scaled import ScaledTensor as S


def test_product_and_division_keep_finite_final_result():
    a = torch.tensor(1e300, dtype=torch.float64)
    b = torch.tensor(1e-300, dtype=torch.float64)
    torch.testing.assert_close((S(a) * S(a) * S(b)).to_tensor(), a)
    torch.testing.assert_close((S(b) * S(b) / S(b)).to_tensor(), b, atol=0, rtol=1e-14)


def test_small_exponential_can_be_amplified_before_conversion():
    decay = S.from_log(torch.tensor(-1000., dtype=torch.float64))
    direction = S(torch.tensor(1e300, dtype=torch.float64))
    torch.testing.assert_close((decay * direction).to_tensor(),
                               torch.tensor(5.075958897549457e-135, dtype=torch.float64), rtol=1e-12, atol=0)
    torch.testing.assert_close((decay * direction * direction / 2).to_tensor(),
                               torch.tensor(2.5379794487747284e165, dtype=torch.float64), rtol=1e-12, atol=0)


def test_complex_sum_keeps_small_nonzero_values():
    value = torch.tensor([[1e-200+2e-200j, 3e-200-1e-200j]], dtype=torch.complex128)
    result = (S(value) * S(value)).sum(dim=-1).to_tensor()
    assert result == 0  # True result is below the ordinary dtype's range.
    restored = ((S(value) * S(value)).sum(dim=-1) * 1e200).to_tensor()
    expected = torch.tensor([5e-200-2e-200j], dtype=torch.complex128)
    torch.testing.assert_close(restored, expected, atol=0, rtol=1e-14)


def test_mantissa_scaling_keeps_ordinary_autograd():
    x = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    result = (S.from_log(-x) * S(x) * S(x)).to_tensor()
    derivative, = torch.autograd.grad(result, x, create_graph=True)
    second, = torch.autograd.grad(derivative, x)
    torch.testing.assert_close(derivative, torch.zeros_like(x), atol=1e-15, rtol=0)
    torch.testing.assert_close(second, torch.tensor(-2*math.exp(-2), dtype=torch.float64))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_exponential_inputs_keep_their_meaning(value):
    x = torch.tensor(value)
    torch.testing.assert_close(S.from_log(x).to_tensor(), x.exp(), equal_nan=True)


def test_sum_empty_axis():
    value = torch.empty(2, 0, dtype=torch.float64)
    torch.testing.assert_close(S(value).sum(dim=1).to_tensor(), value.sum(dim=1))


@pytest.mark.parametrize('value', [1e300, 1e-300])
def test_float64_identity_retains_finite_reverse_derivative(value):
    x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    y = S(x).to_tensor()
    gradient, = torch.autograd.grad(y, x)
    torch.testing.assert_close(y, x)
    torch.testing.assert_close(gradient, torch.ones_like(x))


def test_complex_components_have_independent_exponent_ranges():
    value = torch.tensor(1e300 + 1e-300j, dtype=torch.complex128)
    torch.testing.assert_close(S(value).to_tensor(), value, atol=0, rtol=0)
    imaginary = (S(value) - S(value.real)).to_tensor()
    torch.testing.assert_close(imaginary, torch.tensor(1e-300j, dtype=torch.complex128), atol=0, rtol=0)
    torch.testing.assert_close(S(value).conj().to_tensor(), value.conj(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_binary_scale_never_materializes_out_of_range_powers_of_two(dtype):
    # torch.ldexp may be implemented as value * 2**power; 2**power alone can be
    # inf/0 for exponents that still give representable (or exactly zero) results.
    from polytensors._scaled import _safe_ldexp
    info = torch.finfo(dtype)
    max_exponent = math.frexp(info.max)[1]
    value = torch.tensor([0.0, 1.5, -1.0], dtype=dtype)
    huge = torch.full((3,), 3 * max_exponent, dtype=torch.int32)
    out = _safe_ldexp(value, huge)
    assert out[0] == 0 and torch.isinf(out[1]) and out[2] == -math.inf
    # Results that are representable (here subnormal) must match an exact ldexp.
    for shift in (-(max_exponent + 10), -(2 * max_exponent - 20)):
        out = _safe_ldexp(torch.tensor([1.5], dtype=dtype), torch.tensor([shift], dtype=torch.int32))
        assert out.item() == torch.tensor(math.ldexp(1.5, shift), dtype=dtype).item()
