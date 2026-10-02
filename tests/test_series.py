"""Analytic directional coefficients and avoidable-overflow regressions."""

import math

import pytest
import torch

from polytensors import PolyTensor


def _line(base, direction, degree):
    return PolyTensor((base, direction), degree=degree)


@pytest.mark.parametrize("degree", (1, 3, 12))
@pytest.mark.parametrize("complex_direction", (False, True))
@pytest.mark.parametrize("operation", ("exp", "log", "reciprocal", "rsqrt", "sin", "cos"))
def test_line_coefficients_match_analytic_derivatives(operation, complex_direction, degree):
    base = torch.tensor(1.7, dtype=torch.float64)
    direction = torch.tensor(
        0.2 + 0.3j if complex_direction else 0.2,
        dtype=torch.complex128 if complex_direction else torch.float64,
    )
    function = getattr(torch, operation)
    result = function(_line(base, direction, degree))
    assert torch.equal(result.coeffs[0], function(base))

    binomial = 1.0
    for order in range(1, degree + 1):
        if operation == "exp":
            expected = torch.exp(base) * direction**order / math.factorial(order)
        elif operation == "log":
            expected = ((-1) ** (order - 1) / order) * (direction / base)**order
        elif operation == "reciprocal":
            expected = torch.reciprocal(base) * (-direction / base)**order
        elif operation == "rsqrt":
            binomial *= (-0.5 - (order - 1)) / order
            expected = binomial * torch.rsqrt(base) * (direction / base)**order
        else:
            derivative = (
                (torch.sin(base), torch.cos(base), -torch.sin(base), -torch.cos(base))
                if operation == "sin" else
                (torch.cos(base), -torch.sin(base), -torch.cos(base), torch.sin(base))
            )[order % 4]
            expected = derivative * direction**order / math.factorial(order)
        torch.testing.assert_close(result.coeffs[order], expected, rtol=2e-13, atol=1e-28)


@pytest.mark.parametrize("operation,base_value", (("reciprocal", 1e-20), ("rsqrt", 1e-30)))
@pytest.mark.parametrize("constant", (False, True))
def test_inverse_recurrences_avoid_overflow_at_small_base(operation, base_value, constant):
    base = torch.tensor(base_value, dtype=torch.float32)
    direction = torch.zeros_like(base) if constant else base.clone()
    result = getattr(torch, operation)(_line(base, direction, 8))
    for order, coefficient in enumerate(result.coeffs):
        assert torch.isfinite(coefficient)
        if constant and order:
            expected = torch.zeros_like(base)
        elif operation == "reciprocal":
            expected = torch.reciprocal(base) * (-1)**order
        else:
            binomial = math.prod((-0.5 - i) / (i + 1) for i in range(order))
            expected = torch.rsqrt(base) * binomial
        torch.testing.assert_close(coefficient, expected, rtol=1e-6, atol=0)


def test_log_does_not_multiply_large_coefficients_by_order():
    base = torch.tensor(2e38, dtype=torch.float32)
    value = PolyTensor((base, torch.zeros_like(base), base.clone()), degree=6)
    result = torch.log(value)
    for order in range(1, 7):
        expected = 0.0 if order % 2 else (-1)**(order // 2 - 1) / (order // 2)
        torch.testing.assert_close(result.coeffs[order], torch.tensor(expected), rtol=1e-6, atol=0)


def test_exp_scales_order_before_multiplication():
    zero = torch.tensor(0.0, dtype=torch.float32)
    large = torch.tensor(2e38, dtype=torch.float32)
    result = torch.exp(PolyTensor((zero, zero.clone(), large)))
    torch.testing.assert_close(result.coeffs[2], large, rtol=0, atol=0)


def test_nonlinear_input_series_round_trips():
    coefficients = tuple(
        torch.tensor(value, dtype=torch.float64)
        for value in (1.5, 0.2, -0.1, 0.03, 0.0, -0.002)
    )
    value = PolyTensor(coefficients)
    for result in (torch.exp(torch.log(value)), torch.reciprocal(torch.reciprocal(value))):
        for actual, expected in zip(result.coeffs, coefficients):
            torch.testing.assert_close(actual, expected, rtol=2e-13, atol=1e-15)

    square_root = torch.sqrt(value).square()
    identity = torch.rsqrt(value).square() * value
    for order in range(value.degree + 1):
        torch.testing.assert_close(square_root.coeffs[order], coefficients[order], rtol=2e-13, atol=1e-15)
        expected = torch.ones_like(coefficients[0]) if order == 0 else torch.zeros_like(coefficients[0])
        torch.testing.assert_close(identity.coeffs[order], expected, rtol=2e-13, atol=1e-15)


def test_softmax_accepts_complex_direction_at_real_base():
    base = torch.tensor([0.3, -0.4, 1.2], dtype=torch.float64)
    direction = torch.tensor([0.2 + 0.1j, -0.3 + 0.4j, 0.1 - 0.2j], dtype=torch.complex128)
    result = torch.softmax(_line(base, direction, 5), dim=0)
    probability = torch.softmax(base, dim=0)
    expected_first = probability * (direction - (probability * direction).sum())
    torch.testing.assert_close(result.coeffs[1], expected_first)
    for coefficient in result.coeffs[1:]:
        torch.testing.assert_close(coefficient.sum(), torch.zeros((), dtype=torch.complex128), rtol=0, atol=1e-15)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_logsumexp_derivatives_ignore_large_common_offset(dtype):
    base = torch.full((2,), 1e20, dtype=dtype)
    result = torch.logsumexp(_line(base, torch.ones_like(base), 5), dim=0)
    assert torch.equal(result.value, torch.logsumexp(base, dim=0))
    assert result.coeffs[1].item() == 1.0
    for coefficient in result.coeffs[2:]:
        assert coefficient.item() == 0.0


@pytest.mark.parametrize("function", (torch.exp, torch.log, torch.reciprocal, torch.sqrt, torch.rsqrt, torch.sin, torch.cos))
def test_degree_zero_matches_native(function):
    base = torch.tensor([0.3, 1.2], dtype=torch.float64)
    result = function(PolyTensor((base,)))
    assert result.degree == 0
    assert torch.equal(result.value, function(base))
