"""Regressions where the result fits the dtype but an intermediate does not."""

import math

import pytest
import torch

from polytensors import PolyTensor


@pytest.mark.parametrize("log_probabilities", (False, True))
def test_scaled_normalization_boundary_preserves_complex_coefficient_gradients(log_probabilities):
    from polytensors._normalization import normalization_scaled_series

    coefficients = (
        torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.1 + 0.2j, -0.2 + 0.1j], dtype=torch.complex128, requires_grad=True),
    )

    def function(*values):
        probabilities, normalizers = normalization_scaled_series(values, 0, log_probabilities)
        return tuple(value.to_tensor() for value in probabilities + normalizers)

    assert torch.autograd.gradcheck(function, coefficients)
    assert torch.autograd.gradgradcheck(function, coefficients)


@pytest.mark.parametrize("function", (torch.softmax, torch.log_softmax))
def test_normalization_explicit_dtype_preserves_complex_directions(function):
    base = torch.tensor([0.2, -0.7], dtype=torch.float32)
    direction = torch.tensor([0.3 + 0.4j, -0.5 + 0.1j], dtype=torch.complex64)
    result = function(PolyTensor((base, direction)), dim=0, dtype=torch.float64)
    probabilities = torch.softmax(base.double(), dim=0)
    centered = direction.to(torch.complex128) - (probabilities * direction).sum()
    expected = centered if function is torch.log_softmax else probabilities * centered
    assert result.value.dtype == torch.float64
    torch.testing.assert_close(result.tangent, expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("complex_direction", (False, True))
def test_normalization_removes_large_common_higher_coefficients(dtype, complex_direction):
    magnitude = 2e38 if dtype == torch.float32 else 1e308
    base = torch.tensor([[0.0, 1.0, -2.0]], dtype=dtype)
    common = torch.full_like(base, magnitude)
    if complex_direction:
        common = torch.complex(common, -common)
    zero = torch.zeros_like(common)
    value = PolyTensor((base, zero, common, zero.clone(), common.clone()), degree=6)

    for function in (torch.softmax, torch.log_softmax):
        result = function(value, dim=-1)
        assert torch.equal(result.value, function(base, dim=-1))
        for coefficient in result.coeffs[1:]:
            assert torch.count_nonzero(coefficient) == 0

    result = torch.logsumexp(value, dim=-1)
    assert torch.equal(result.value, torch.logsumexp(base, dim=-1))
    for order in range(1, value.degree + 1):
        torch.testing.assert_close(result.coeffs[order], value.coeffs[order][..., 0])


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_softmax_large_opposing_directions(dtype):
    magnitude = 3e38 if dtype == torch.float32 else 1.5e308
    base = torch.zeros(2, dtype=dtype)
    direction = torch.tensor([magnitude, -magnitude], dtype=dtype)
    result = torch.softmax(PolyTensor((base, direction)), dim=0)
    torch.testing.assert_close(result.coeffs[1], 0.5 * direction)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("complex_direction", (False, True))
def test_sqrt_divides_before_overflowing_intermediate_square(dtype, complex_direction):
    base_value, direction_value = ((1e20, 1e31) if dtype == torch.float32 else (1e200, 1e300))
    base = torch.tensor(base_value, dtype=dtype)
    direction = torch.tensor(direction_value, dtype=dtype)
    if complex_direction:
        direction = torch.complex(torch.zeros_like(direction), direction)
    result = torch.sqrt(PolyTensor((base, direction), degree=2))
    expected_first = 0.5 * (direction / torch.sqrt(base))
    expected_second = -0.25 * (direction / base) * expected_first
    torch.testing.assert_close(result.coeffs[1], expected_first)
    torch.testing.assert_close(result.coeffs[2], expected_second)


@pytest.mark.parametrize("dtype,base_value,direction_value", (
    (torch.float32, -120.0, 1e20),
    (torch.float64, -1000.0, 1e150),
))
@pytest.mark.parametrize("complex_direction", (False, True))
def test_exp_recovers_coefficients_after_value_underflows(dtype, base_value, direction_value, complex_direction):
    base = torch.tensor(base_value, dtype=dtype)
    direction = torch.tensor(direction_value, dtype=dtype)
    if complex_direction:
        direction = torch.complex(torch.zeros_like(direction), direction)
    result = torch.exp(PolyTensor((base, direction), degree=2))
    assert result.value.item() == 0
    for order in (1, 2):
        log_magnitude = base_value + order * math.log(direction_value) - math.lgamma(order + 1)
        phase = (1j ** order) if complex_direction else 1
        expected = torch.tensor(math.exp(log_magnitude) * phase, dtype=result.coeffs[order].dtype)
        torch.testing.assert_close(result.coeffs[order], expected, rtol=3e-5, atol=0)


@pytest.mark.parametrize("dtype,base_value,direction_value", (
    (torch.float32, -120.0, 1e20),
    (torch.float64, -1000.0, 1e150),
))
def test_softmax_and_logsumexp_recover_small_probability_times_large_direction(dtype, base_value, direction_value):
    base = torch.tensor([0.0, base_value], dtype=dtype)
    direction = torch.tensor([0.0, direction_value], dtype=dtype)
    value = PolyTensor((base, direction), degree=2)
    probability = torch.softmax(value, dim=0)
    normalizer = torch.logsumexp(value, dim=0)
    for order in (1, 2):
        expected = math.exp(base_value + order * math.log(direction_value) - math.lgamma(order + 1))
        expected_probability = torch.tensor([-expected, expected], dtype=dtype)
        # Terms proportional to exp(2*base_value) are below these tolerances.
        torch.testing.assert_close(probability.coeffs[order], expected_probability, rtol=3e-5, atol=0)
        torch.testing.assert_close(normalizer.coeffs[order], torch.tensor(expected, dtype=dtype), rtol=3e-5, atol=0)


@pytest.mark.parametrize("function", (torch.exp, torch.softmax, torch.logsumexp))
@pytest.mark.parametrize("base_value,direction_value", ((-120.0, 1e30), (-1000.0, 1e150)))
def test_scaled_coefficients_support_second_order_coefficient_autograd(function, base_value, direction_value):
    base = torch.tensor([0.0, base_value], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.0, direction_value], dtype=torch.float64)
    with PolyTensor.coefficient_autograd():
        value = PolyTensor((base, direction))
        if function is torch.exp:
            coefficient = function(value).coeffs[1][1]
        elif function is torch.softmax:
            coefficient = function(value, dim=0).coeffs[1][1]
        else:
            coefficient = function(value, dim=0).coeffs[1]
    first_gradient = torch.autograd.grad(coefficient, base, create_graph=True)[0]
    second_gradient = torch.autograd.grad(first_gradient[1], base)[0]
    expected = math.exp(base_value + math.log(direction_value))
    for gradient in (first_gradient, second_gradient):
        torch.testing.assert_close(gradient[1], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0)


@pytest.mark.parametrize("function", (torch.softmax, torch.log_softmax, torch.logsumexp))
def test_normalization_coefficient_gradients_preserve_small_direction_dependence(function):
    base = torch.tensor([0.0, -1000.0], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.0, 1e150], dtype=torch.float64, requires_grad=True)
    with PolyTensor.coefficient_autograd():
        result = function(PolyTensor((base, direction), degree=2), dim=0)
    coefficient = result.coeffs[2] if function is torch.logsumexp else result.coeffs[2][1]
    base_gradient, direction_gradient = torch.autograd.grad(coefficient, (base, direction), create_graph=True)
    second_base_gradient = torch.autograd.grad(base_gradient[1], base)[0]
    sign = -1 if function is torch.log_softmax else 1
    first = sign * math.exp(-1000.0 + math.log(1e150))
    second = sign * math.exp(-1000.0 + 2 * math.log(1e150) - math.log(2))
    torch.testing.assert_close(direction_gradient, torch.tensor([-first, first], dtype=torch.float64), rtol=1e-12, atol=0)
    for gradient in (base_gradient, second_base_gradient):
        torch.testing.assert_close(gradient, torch.tensor([-second, second], dtype=torch.float64), rtol=1e-12, atol=0)


def test_exp_zero_order_gradient_combines_underflowing_derivative_with_upstream_first():
    base = torch.tensor(-1000.0, dtype=torch.float64, requires_grad=True)
    with PolyTensor.coefficient_autograd():
        result = torch.exp(PolyTensor((base,)))
    assert result.value.item() == 0
    gradient = torch.autograd.grad(result.value, base, grad_outputs=torch.tensor(1e300, dtype=torch.float64), create_graph=True)[0]
    second = torch.autograd.grad(gradient, base)[0]
    expected = torch.tensor(math.exp(-1000.0 + math.log(1e300)), dtype=torch.float64)
    for derivative in (gradient, second):
        torch.testing.assert_close(derivative, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("function", (torch.softmax, torch.log_softmax, torch.logsumexp))
def test_normalization_second_reverse_derivative_preserves_saturated_probability_complement(function):
    base = torch.tensor([0.0, -1000.0], dtype=torch.float64, requires_grad=True)
    with PolyTensor.coefficient_autograd():
        result = function(PolyTensor((base,)), dim=0)
    value = result.value if function is torch.logsumexp else result.value[0]
    first = torch.autograd.grad(value, base, grad_outputs=torch.tensor(1e300, dtype=torch.float64), create_graph=True)[0]
    second = torch.autograd.grad(first[0], base)[0]
    small = math.exp(-1000.0 + math.log(1e300))
    sign = 1 if function is torch.logsumexp else -1
    expected = torch.tensor([sign * small, -sign * small], dtype=torch.float64)
    torch.testing.assert_close(second, expected, rtol=1e-12, atol=0)


def test_logsumexp_coefficient_autograd_does_not_subtract_rounded_normalizer():
    base = torch.tensor([1e20, 1e20], dtype=torch.float32, requires_grad=True)
    with PolyTensor.coefficient_autograd():
        result = torch.logsumexp(PolyTensor((base,)), dim=0)
    gradient = torch.autograd.grad(result.value, base)[0]
    torch.testing.assert_close(gradient, torch.full_like(base, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("function", (torch.softmax, torch.log_softmax, torch.logsumexp))
def test_normalization_scalar_and_empty_axes(function):
    for base in (torch.tensor(2.0), torch.empty(2, 0)):
        value = PolyTensor((base, torch.ones_like(base)), degree=2)
        result = function(value, dim=-1)
        expected = function(base, dim=-1)
        torch.testing.assert_close(result.value, expected)
        assert all(torch.isfinite(coefficient).all() for coefficient in result.coeffs[1:])


@pytest.mark.parametrize("function", (torch.softmax, torch.log_softmax, torch.logsumexp))
def test_normalization_keeps_undefined_unmasked_rows_visible(function):
    base = torch.full((3,), -torch.inf, dtype=torch.float64)
    value = PolyTensor((base, torch.ones_like(base)), degree=2)
    result = function(value, dim=0)
    assert torch.isnan(result.coeffs[1]).all()


def test_log_and_root_domain_singularities_remain_nonfinite():
    value = PolyTensor((torch.tensor(0.0), torch.tensor(1.0)), degree=2)
    for function in (torch.log, torch.reciprocal, torch.sqrt, torch.rsqrt):
        result = function(value)
        assert not torch.isfinite(result.coeffs[1])
