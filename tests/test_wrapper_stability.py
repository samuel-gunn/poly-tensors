"""Stable native-style reverse AD through the public PolyTensor interface."""

import math

import pytest
import torch

from polytensors import PolyTensor


EXP_CALLS = (
    torch.exp,
    lambda x: x.exp(),
    lambda x: torch.exp(input=x),
    torch.ops.aten.exp.default,
)

LOGSUMEXP_CALLS = (
    lambda x, dim, keepdim: torch.logsumexp(x, dim, keepdim),
    lambda x, dim, keepdim: x.logsumexp(dim, keepdim),
    lambda x, dim, keepdim: torch.logsumexp(input=x, dim=dim, keepdim=keepdim),
    lambda x, dim, keepdim: torch.special.logsumexp(x, dim=dim, keepdim=keepdim),
    lambda x, dim, keepdim: torch.ops.aten.logsumexp.default(x, [dim], keepdim),
)


@pytest.mark.parametrize("function", EXP_CALLS)
@pytest.mark.parametrize("direction", (1e50, (0.6 + 0.8j) * 1e50))
def test_exp_wrapper_backward_recovers_underflow_through_three_reverse_derivatives(function, direction):
    degree = 3
    base = torch.tensor(-1000.0, dtype=torch.float64)
    tangent = torch.tensor(direction, dtype=torch.complex128 if isinstance(direction, complex) else torch.float64)
    value = PolyTensor((base, tangent), degree=degree, requires_grad=True)
    upstream = PolyTensor.constant(torch.tensor(1e300, dtype=torch.float64), degree)
    with PolyTensor.retain_wrappers():
        result = function(value)
        assert result.value == 0
        derivative = torch.autograd.grad(result, value, upstream, create_graph=True)[0]
        for reverse_order in range(3):
            for order, actual in enumerate(derivative.coeffs):
                log_size = -1000 + math.log(1e300) + order * math.log(abs(direction)) - math.lgamma(order + 1)
                phase = (direction / abs(direction)) ** order if order else 1.0
                expected = torch.tensor(math.exp(log_size) * phase, dtype=actual.dtype)
                torch.testing.assert_close(actual, expected, rtol=2e-12, atol=0)
            if reverse_order < 2:
                derivative = torch.autograd.grad(derivative, value, create_graph=True)[0]


@pytest.mark.parametrize("function", LOGSUMEXP_CALLS)
@pytest.mark.parametrize("keepdim", (False, True))
@pytest.mark.parametrize("direction", (0.4, 0.4 + 0.3j))
def test_logsumexp_wrapper_backward_normalizes_huge_common_offset(function, keepdim, direction):
    base = torch.full((2, 2), 1e20, dtype=torch.float64)
    tangent = torch.tensor([[direction, 0], [0, direction]], dtype=torch.complex128 if isinstance(direction, complex) else torch.float64)
    value = PolyTensor((base, tangent), degree=3, requires_grad=True)
    with PolyTensor.retain_wrappers():
        result = function(value, -1, keepdim)
        derivative = torch.autograd.grad(result, value, torch.ones_like(result), create_graph=True)[0]
        expected = (
            torch.full_like(base, 0.5),
            (tangent - tangent.mean(-1, keepdim=True)) / 2,
            torch.zeros_like(tangent),
            -(tangent - tangent.mean(-1, keepdim=True)) ** 3 / 6,
        )
        for actual, reference in zip(derivative.coeffs, expected):
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-15)
        # Differentiate one component per row to check the wrapper Hessian.
        hessian = torch.autograd.grad(derivative[:, 0].sum(), value)[0]
        sign = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float64)
        expected_hessian = (sign / 4, torch.zeros_like(tangent), -sign * direction ** 2 / 16, torch.zeros_like(tangent))
        for actual, reference in zip(hessian.coeffs, expected_hessian):
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-15)


def test_logsumexp_wrapper_backward_preserves_a_tiny_probability_through_higher_reverse():
    value = PolyTensor((torch.tensor([0.0, -1000.0], dtype=torch.float64),), requires_grad=True)
    upstream = PolyTensor.constant(torch.tensor(1e300, dtype=torch.float64), 0)
    small = math.exp(-1000.0 + math.log(1e300))
    with PolyTensor.retain_wrappers():
        result = torch.logsumexp(value, 0)
        first = torch.autograd.grad(result, value, upstream, create_graph=True)[0]
        second = torch.autograd.grad(first[1], value, create_graph=True)[0]
        third = torch.autograd.grad(second[1], value)[0]
    torch.testing.assert_close(first.value, torch.tensor([1e300, small], dtype=torch.float64), rtol=1e-12, atol=0)
    for derivative in (second, third):
        torch.testing.assert_close(derivative.value, torch.tensor([-small, small], dtype=torch.float64), rtol=1e-12, atol=0)


@pytest.mark.parametrize("dim", (0, -1, (0,)))
def test_logsumexp_scalar_wrapper_backward(dim):
    value = PolyTensor((torch.tensor(1e20, dtype=torch.float64), torch.tensor(0.4, dtype=torch.float64)), requires_grad=True)
    with PolyTensor.retain_wrappers():
        derivative = torch.autograd.grad(torch.logsumexp(value, dim), value)[0]
    torch.testing.assert_close(derivative.value, torch.ones_like(value.value))
    torch.testing.assert_close(derivative.tangent, torch.zeros_like(value.tangent))


@pytest.mark.parametrize("function", (torch.exp, lambda x: torch.logsumexp(x, 0)))
def test_wrapper_overrides_preserve_coefficient_autograd_and_no_grad(function):
    base = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    with PolyTensor.coefficient_autograd():
        result = function(PolyTensor((base,)))
    assert not result.requires_grad
    actual = torch.autograd.grad(result.value.sum(), base)[0]
    reference = torch.autograd.grad(function(base).sum(), base)[0]
    torch.testing.assert_close(actual, reference)
    with torch.no_grad():
        result = function(PolyTensor((base.detach(),), requires_grad=True))
    assert not result.requires_grad


def test_exp_wrapper_autograd_preserves_complex_base_conjugation():
    base = torch.tensor(0.3 + 0.2j, dtype=torch.complex128)
    tangent = torch.tensor(0.1 - 0.4j, dtype=torch.complex128)
    value = PolyTensor((base, tangent), degree=2, requires_grad=True)
    upstream = PolyTensor.constant(torch.ones_like(base), 2)
    with PolyTensor.retain_wrappers():
        result = torch.exp(value)
        derivative = torch.autograd.grad(result, value, upstream, create_graph=True)[0]
        for reverse_order in range(3):
            for order, actual in enumerate(derivative.coeffs):
                expected = (torch.exp(base) * tangent ** order / math.factorial(order)).conj()
                torch.testing.assert_close(actual, expected)
            if reverse_order < 2:
                derivative = torch.autograd.grad(derivative, value, upstream, create_graph=True)[0]
