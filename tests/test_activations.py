import math
import contextlib

import pytest
import torch

from polytensors import PolyTensor
from polytensors._activations import activation_backward_series, activation_series


KINDS = ("gelu", "gelu_tanh", "sigmoid", "tanh", "silu")


def _native(value, kind):
    if kind.startswith("gelu"):
        return torch.nn.functional.gelu(value, approximate="tanh" if kind.endswith("tanh") else "none")
    if kind == "silu":
        return torch.nn.functional.silu(value)
    return getattr(torch, kind)(value)


def _native_derivatives(base, kind, order):
    base = base.detach().requires_grad_()
    result = [_native(base, kind)]
    for _ in range(order):
        result.append(torch.autograd.grad(result[-1], base, create_graph=True)[0])
    return [value.detach() for value in result]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("base", (-1.3, 0.0, 0.7))
def test_activation_order_seven_matches_independent_native_derivatives(kind, base):
    base = torch.tensor(base, dtype=torch.float64)
    direction = torch.tensor(0.4 + 0.3j, dtype=torch.complex128)
    coefficients = [base, direction] + [torch.zeros_like(direction) for _ in range(6)]
    result = activation_series(coefficients, kind)
    derivatives = _native_derivatives(base, kind, 7)
    for n, value in enumerate(result):
        expected = derivatives[n] * direction**n / math.factorial(n)
        torch.testing.assert_close(value, expected.real if n == 0 else expected, rtol=3e-11, atol=2e-14)


@pytest.mark.parametrize("kind", KINDS)
def test_activation_composes_nonlinear_input_polynomial(kind):
    coefficients = [torch.tensor(value, dtype=torch.float64) for value in (0.4, 0.3, -0.2, 0.1, 0.05)]
    result = activation_series(coefficients, kind)
    t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    curve = sum(coefficient * t**n for n, coefficient in enumerate(coefficients))
    derivative = _native(curve, kind)
    for n, value in enumerate(result):
        torch.testing.assert_close(value, derivative.detach() / math.factorial(n), rtol=2e-12, atol=2e-14)
        if n < len(result) - 1:
            derivative = torch.autograd.grad(derivative, t, create_graph=True)[0]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype, magnitude", ((torch.float32, 1e20), (torch.float64, 1e200)))
def test_extreme_activation_tails_are_finite_in_forward_and_coefficient_backward(kind, dtype, magnitude):
    coefficients = [
        torch.tensor([-magnitude, magnitude], dtype=dtype, requires_grad=True),
        torch.full((2,), magnitude, dtype=dtype, requires_grad=True),
        torch.zeros(2, dtype=dtype, requires_grad=True),
        torch.zeros(2, dtype=dtype, requires_grad=True),
    ]
    result = activation_series(coefficients, kind)
    torch.testing.assert_close(result[0], _native(coefficients[0], kind), rtol=0, atol=0)
    linear_tail = kind in ("gelu", "gelu_tanh", "silu")
    for n, value in enumerate(result[1:], 1):
        expected = coefficients[n] * torch.tensor([0, 1], dtype=dtype) if linear_tail else torch.zeros_like(value)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(sum(value.sum() for value in result), coefficients)
    for gradient in gradients:
        expected = torch.tensor([0, 1], dtype=dtype) if linear_tail else torch.zeros_like(gradient)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sign", (-1, 1))
def test_gelu_keeps_gaussian_tail_until_large_directions_are_multiplied(sign):
    base = torch.tensor(sign * 15.0, dtype=torch.float32)
    direction = torch.tensor(1e20, dtype=torch.float32)
    result = activation_series([base, direction, torch.zeros_like(base), torch.zeros_like(base)], "gelu")
    x = base.double()
    v = direction.double()
    density = torch.exp(-x.square() / 2) / math.sqrt(2 * math.pi)
    cdf = torch.erfc(-x / math.sqrt(2)) / 2
    expected = [(cdf + x * density) * v, (2 - x.square()) * density * v.square() / 2,
                (x**3 - 4 * x) * density * v**3 / 6]
    for actual, reference in zip(result[1:], expected):
        torch.testing.assert_close(actual, reference.float(), rtol=2e-6, atol=0)


def test_tanh_gelu_keeps_saturated_tail_and_avoids_cubic_intermediate_overflow():
    base = torch.tensor(10.0, dtype=torch.float32)
    direction = torch.tensor(1e20, dtype=torch.float32)
    result = activation_series([base, direction, torch.zeros_like(base), torch.zeros_like(base)], "gelu_tanh")
    # Independent 180-decimal-digit differentiation, rescaled for the actual
    # float32 direction rather than the decimal value before rounding.
    ratio = direction.double() / 1e20
    reference = [ratio * 1e20, ratio**2 * -313196.531301143927, ratio**3 * 2.3516041053358169e26]
    for actual, expected in zip(result[1:], reference):
        torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=0)


@pytest.mark.parametrize("kind, base, multiplier", (("sigmoid", 120.0, 1), ("tanh", 60.0, 4)))
def test_saturated_activations_keep_direction_amplified_tail(kind, base, multiplier):
    x = torch.tensor(base, dtype=torch.float32)
    v = torch.tensor(1e20, dtype=torch.float32)
    result = activation_series([x, v], kind)
    tail = torch.exp(torch.tensor(-120.0, dtype=torch.float64))
    expected = multiplier * tail / (1 + tail).square() * v.double()
    assert result[1] != 0
    torch.testing.assert_close(result[1], expected.float(), rtol=2e-6, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_activation_coefficient_autograd_handles_complex_directions(kind):
    coefficients = (
        torch.tensor(0.4, dtype=torch.float64, requires_grad=True),
        torch.tensor(0.2 + 0.1j, dtype=torch.complex128, requires_grad=True),
        torch.tensor(-0.1 + 0.3j, dtype=torch.complex128, requires_grad=True),
    )
    function = lambda *values: tuple(activation_series(values, kind))
    assert torch.autograd.gradcheck(function, coefficients)
    assert torch.autograd.gradgradcheck(function, coefficients)


@pytest.mark.parametrize("kind", KINDS)
def test_activation_second_reverse_derivative_at_zero(kind):
    base = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    result = activation_series([base], kind)[0]
    first = torch.autograd.grad(result, base, create_graph=True)[0]
    second = torch.autograd.grad(first, base)[0]
    expected = _native_derivatives(base, kind, 2)[2]
    torch.testing.assert_close(second, expected, rtol=2e-12, atol=1e-14)


def test_activation_coefficient_backward_keeps_gaussian_underflow_scaled():
    coefficients = [torch.tensor(40.0, dtype=torch.float64, requires_grad=True),
                    torch.tensor(1e300, dtype=torch.float64, requires_grad=True)]
    result = activation_series(coefficients, "gelu")
    derivative = torch.autograd.grad(result[1], coefficients[0])[0]
    expected = -torch.exp(torch.tensor(-800 + math.log(1598) + math.log(1e300) - math.log(2 * math.pi) / 2,
                                       dtype=torch.float64))
    assert derivative != 0
    torch.testing.assert_close(derivative, expected, rtol=2e-12, atol=0)


def test_underflowed_activation_derivative_is_recovered_by_large_upstream_gradient():
    base = torch.tensor(-40.0, dtype=torch.float64, requires_grad=True)
    gradient = torch.tensor(1e300, dtype=torch.float64)
    # erfcx supplies Phi(-40)/phi(40) without forming either tiny quantity.
    factor = math.sqrt(math.pi / 2) * torch.special.erfcx(-base.detach() / math.sqrt(2)) + base.detach()
    expected = factor * torch.exp(torch.tensor(-800 + math.log(1e300) - math.log(2 * math.pi) / 2,
                                              dtype=torch.float64))
    output = activation_series([base], "gelu")[0]
    coefficient_gradient = torch.autograd.grad(output, base, gradient)[0]
    wrapper_gradient = activation_backward_series([base], [gradient], "gelu")[0]
    for actual in (coefficient_gradient, wrapper_gradient):
        assert actual != 0
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=0)

    wrapped = PolyTensor([base.detach()], requires_grad=True)
    with PolyTensor.retain_wrappers():
        torch.nn.functional.gelu(wrapped).backward(gradient)
    torch.testing.assert_close(wrapped.grad.coeffs[0], expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("kind, base", (("gelu", -40.0), ("sigmoid", -1000.0), ("tanh", -500.0)))
def test_second_reverse_derivative_of_rescued_tail_remains_finite(kind, base):
    base = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    output = activation_series([base], kind)[0]
    first = torch.autograd.grad(output, base, torch.tensor(1e300, dtype=torch.float64), create_graph=True)[0]
    second = torch.autograd.grad(first, base)[0]
    expected = activation_backward_series([base], [torch.tensor(1e300, dtype=torch.float64)], kind)[0]
    assert torch.isfinite(first) and torch.isfinite(second)
    assert first != 0 and second != 0
    torch.testing.assert_close(first, expected, rtol=2e-12, atol=0)
    if kind == "gelu":
        second_expected = (2 - base.detach().square()) * torch.exp(
            -base.detach().square() / 2 + math.log(1e300) - math.log(2 * math.pi) / 2)
    else:
        second_expected = torch.exp(torch.tensor(-1000 + math.log(1e300), dtype=torch.float64))
        if kind == "tanh":
            second_expected = 8 * second_expected
    torch.testing.assert_close(second, second_expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("kind, base", (("gelu", -40.0), ("sigmoid", -1000.0), ("tanh", -500.0)))
def test_later_reverse_gradient_can_recover_a_previously_underflowed_tail(kind, base):
    base = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    output = activation_series([base], kind)[0]
    first = torch.autograd.grad(output, base, create_graph=True)[0]
    assert first == 0
    second = torch.autograd.grad(first, base, torch.tensor(1e300, dtype=torch.float64))[0]
    if kind == "gelu":
        expected = (2 - base.detach().square()) * torch.exp(
            -base.detach().square() / 2 + math.log(1e300) - math.log(2 * math.pi) / 2)
    else:
        expected = torch.exp(torch.tensor(-1000 + math.log(1e300), dtype=torch.float64))
        if kind == "tanh":
            expected = 8 * expected
    torch.testing.assert_close(second, expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_public_activation_supports_complex_directions_and_backward(kind):
    coefficients = [torch.tensor(0.4, dtype=torch.float64),
                    torch.tensor(0.2 + 0.3j, dtype=torch.complex128),
                    torch.tensor(0.0j, dtype=torch.complex128)]
    x = PolyTensor(coefficients, requires_grad=True)
    with PolyTensor.retain_wrappers():
        y = _native(x, kind)
        y.backward()
    expected = activation_series(coefficients, kind, derivative_order=1)
    for actual, reference in zip(x.grad.coeffs, expected):
        torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("kind, base_value", (("sigmoid", -1000.0), ("gelu", -40.0)))
@pytest.mark.parametrize("mode", ("wrapper", "coefficient"))
@pytest.mark.parametrize("multiplications", (1, 2))
def test_activation_tails_survive_separate_products_and_higher_reverse(kind, base_value, mode, multiplications):
    base = torch.tensor(base_value, dtype=torch.float64, requires_grad=mode == "coefficient")
    direction = torch.tensor(0.2 + 0.3j, dtype=torch.complex128)
    value = PolyTensor((base, direction), requires_grad=mode == "wrapper")
    log_scale = multiplications * math.log(1e300)
    if kind == "sigmoid":
        # Corrections are of relative size exp(-1000), below float64 precision.
        references = [math.exp(base_value + log_scale)] * 4
    else:
        density = math.exp(-base_value ** 2 / 2 - math.log(2 * math.pi) / 2 + log_scale)
        tail_factor = math.sqrt(math.pi / 2) * torch.special.erfcx(-base.detach() / math.sqrt(2)).item()
        references = [base_value * tail_factor * density,
                      (tail_factor + base_value) * density,
                      (2 - base_value ** 2) * density,
                      (base_value ** 3 - 4 * base_value) * density]
    with PolyTensor.retain_wrappers():
        scope = PolyTensor.coefficient_autograd() if mode == "coefficient" else contextlib.nullcontext()
        with scope:
            result = _native(value, kind)
            for _ in range(multiplications):
                # The two-product backward reaches the activation with a
                # gradient of 1e600, represented by the hidden scaled payload.
                result = result * 1e300
        torch.testing.assert_close(result.value, torch.tensor(references[0], dtype=torch.float64), rtol=2e-12, atol=0)
        torch.testing.assert_close(result.tangent, references[1] * direction, rtol=2e-12, atol=0)
        if mode == "wrapper":
            first = torch.autograd.grad(result, value, create_graph=True)[0]
            second = torch.autograd.grad(first, value)[0]
            for order, derivative in enumerate((first, second), 1):
                torch.testing.assert_close(derivative.value, torch.tensor(references[order], dtype=torch.float64), rtol=2e-12, atol=0)
                torch.testing.assert_close(derivative.tangent, references[order + 1] * direction, rtol=2e-12, atol=0)
        else:
            first = torch.autograd.grad(result.value, base, create_graph=True)[0]
            second = torch.autograd.grad(first, base)[0]
            for actual, expected in zip((first, second), references[1:]):
                torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=2e-12, atol=0)


def test_activation_backward_accepts_extended_range_complex_upstream():
    from polytensors._scaled import ScaledTensor

    base = torch.tensor(-1000.0, dtype=torch.float64)
    upstream = ScaledTensor.from_tensor(torch.tensor(1e300 + 1e300j, dtype=torch.complex128)) * 1e300
    result = activation_backward_series((base,), (upstream,), "sigmoid", _return_scaled=True)[0]
    expected = math.exp(-1000 + 2 * math.log(1e300)) * (1 + 1j)
    torch.testing.assert_close(result.to_tensor(), torch.tensor(expected, dtype=torch.complex128), rtol=2e-12, atol=0)
