"""Numerical regressions spanning more than one dispatched operation."""

import math

import pytest
import torch

from polytensors import PolyTensor


@pytest.mark.parametrize("dtype,base,multiplier", (
    (torch.float32, -120.0, 1e30),
    (torch.float64, -1000.0, 1e300),
))
@pytest.mark.parametrize("complex_direction", (False, True))
def test_exponential_tail_survives_later_multiplication(dtype, base, multiplier, complex_direction):
    constant = torch.tensor(base, dtype=dtype)
    direction = torch.tensor(2.0, dtype=dtype)
    if complex_direction:
        direction = torch.complex(torch.zeros_like(direction), direction)
    intermediate = torch.exp(PolyTensor((constant, direction), degree=3))
    assert intermediate.value == 0
    result = intermediate * multiplier
    magnitude = math.exp(base + math.log(multiplier))
    for order, coefficient in enumerate(result.coeffs):
        phase = (2j if complex_direction else 2.0) ** order
        if not coefficient.is_complex() and isinstance(phase, complex):
            phase = phase.real
        expected = torch.tensor(magnitude * phase / math.factorial(order), dtype=coefficient.dtype)
        torch.testing.assert_close(coefficient, expected, rtol=2e-5 if dtype == torch.float32 else 1e-12, atol=0)


@pytest.mark.parametrize("magnitude", (1e-300, 1e300))
def test_product_outside_dtype_range_recovers_after_division(magnitude):
    base = torch.tensor(magnitude, dtype=torch.float64)
    value = PolyTensor((base, base / 4))
    intermediate = value * value
    assert intermediate.value == 0 if magnitude < 1 else torch.isinf(intermediate.value)
    result = intermediate / value
    for actual, expected in zip(result.coeffs, value.coeffs):
        torch.testing.assert_close(actual, expected, atol=0, rtol=1e-12)


def test_tiny_exponentials_survive_sum_before_amplification():
    base = torch.tensor([-1000.0, -1001.0], dtype=torch.float64)
    result = torch.exp(PolyTensor((base, torch.ones_like(base)))).sum() * 1e300
    expected = torch.tensor(math.exp(-1000 + math.log(1e300)) * (1 + math.exp(-1)), dtype=torch.float64)
    for coefficient in result.coeffs:
        torch.testing.assert_close(coefficient, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("layout", ("dot", "matrix", "batch"))
def test_softmax_tail_survives_matrix_product(layout):
    base = torch.tensor([0.0, -1000.0], dtype=torch.float64)
    direction = torch.tensor([0.0, 2.0], dtype=torch.float64)
    values = torch.tensor([0.0, 1e300], dtype=torch.float64)
    if layout == "matrix":
        base, direction, values = base[None], direction[None], values[:, None]
    elif layout == "batch":
        base, direction, values = base[None, None], direction[None, None], values[None, :, None]
    probabilities = torch.softmax(PolyTensor((base, direction), degree=2), dim=-1)
    result = probabilities @ values
    magnitude = math.exp(-1000 + math.log(1e300))
    for order, coefficient in enumerate(result.coeffs):
        expected = torch.full_like(coefficient, magnitude * 2**order / math.factorial(order))
        torch.testing.assert_close(coefficient, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("operation", (
    lambda x: x.clone(),
    lambda x: x.detach(),
    lambda x: x.reshape(1, 2).transpose(0, 1).reshape(2),
    lambda x: x[:1].expand(2),
))
def test_range_survives_clone_detach_and_shape_operations(operation):
    base = torch.full((2,), -1000.0, dtype=torch.float64)
    result = operation(torch.exp(PolyTensor((base, torch.ones_like(base))))) * 1e300
    expected = torch.full_like(base, math.exp(-1000 + math.log(1e300)))
    for coefficient in result.coeffs:
        torch.testing.assert_close(coefficient, expected, rtol=1e-12, atol=0)


def test_inplace_multiplication_preserves_range_on_destination():
    base = torch.tensor(-1000.0, dtype=torch.float64)
    value = torch.exp(PolyTensor((base, torch.ones_like(base))))
    value.mul_(1e300)
    expected = torch.tensor(math.exp(-1000 + math.log(1e300)), dtype=torch.float64)
    for coefficient in value.coeffs:
        torch.testing.assert_close(coefficient, expected, rtol=1e-12, atol=0)


def test_public_coefficient_mutation_invalidates_saved_range():
    value = torch.exp(PolyTensor((torch.tensor(-1000.0, dtype=torch.float64),)))
    value.value.fill_(2.0)
    torch.testing.assert_close((value * 3).value, torch.tensor(6.0, dtype=torch.float64))


def test_public_coefficient_replacement_invalidates_saved_range():
    value = torch.exp(PolyTensor((torch.tensor(-1000.0, dtype=torch.float64),)))
    value.coeffs = (torch.tensor(2.0, dtype=torch.float64),)
    torch.testing.assert_close((value * 3).value, torch.tensor(6.0, dtype=torch.float64))


def test_alias_coefficient_mutation_invalidates_saved_range():
    value = torch.exp(PolyTensor((torch.full((2,), -1000.0, dtype=torch.float64),)))
    alias = value.view(1, 2)
    value.value.fill_(2.0)
    torch.testing.assert_close((alias * 3).value, torch.full((1, 2), 6.0, dtype=torch.float64))


def test_mutation_through_alias_invalidates_original_saved_range():
    value = torch.exp(PolyTensor((torch.full((2,), -1000.0, dtype=torch.float64),)))
    alias = value.view(1, 2)
    alias.value.fill_(2.0)
    torch.testing.assert_close((value * 3).value, torch.full((2,), 6.0, dtype=torch.float64))


def test_clone_has_independent_coefficient_mutation():
    value = torch.exp(PolyTensor((torch.tensor(-1000.0, dtype=torch.float64),)))
    copy = value.clone()
    value.value.fill_(2.0)
    expected = torch.tensor(math.exp(-1000 + math.log(1e300)), dtype=torch.float64)
    torch.testing.assert_close((copy * 1e300).value, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("operation", (
    lambda x: x.index_select(0, torch.tensor([1, 0])),
    lambda x: x.gather(0, torch.tensor([[1, 0], [0, 1]])),
    lambda x: x[torch.tensor([1, 0])],
    lambda x: x.transpose(0, 1).reshape(4),
))
def test_copying_index_and_shape_operations_have_independent_storage(operation):
    value = torch.exp(PolyTensor((torch.full((2, 2), -1000.0, dtype=torch.float64),)))
    copied = operation(value)
    value.value.fill_(2.0)
    expected = torch.full_like(copied.value, math.exp(-1000 + math.log(1e300)))
    torch.testing.assert_close((copied * 1e300).value, expected, rtol=1e-12, atol=0)


def test_broadcast_copy_keeps_hidden_shape_consistent_for_views():
    value = torch.exp(PolyTensor((torch.tensor(-1000.0, dtype=torch.float64),)))
    destination = PolyTensor.constant(torch.zeros(2, 2, dtype=torch.float64), degree=0)
    destination.copy_(value)
    result = destination.reshape(4) * 1e300
    expected = torch.full_like(result.value, math.exp(-1000 + math.log(1e300)))
    torch.testing.assert_close(result.value, expected, rtol=1e-12, atol=0)


def test_python_complex_scalar_uses_native_dtype_promotion():
    value = PolyTensor((torch.tensor(2.0, dtype=torch.float64),
                        torch.tensor(3.0, dtype=torch.float64)))
    result = value * 1j
    for actual, original in zip(result.coeffs, value.coeffs):
        torch.testing.assert_close(actual, original * 1j)


@pytest.mark.parametrize("operation", (torch.add, torch.mul, torch.div))
def test_scalar_tensor_uses_native_dtype_promotion(operation):
    base = torch.tensor([1.0, 2.0], dtype=torch.float32)
    scalar = torch.tensor(2.0, dtype=torch.float64)
    result = operation(PolyTensor((base, torch.ones_like(base))), scalar)
    torch.testing.assert_close(result.value, operation(base, scalar))


@pytest.mark.parametrize("operation", (torch.sum, torch.mean))
def test_reduction_dtype_preserves_imaginary_directions(operation):
    value = PolyTensor((torch.tensor([1.0, 2.0]), torch.tensor([1j, 2j])))
    result = operation(value, dtype=torch.float64)
    torch.testing.assert_close(result.value, operation(value.value.double()))
    torch.testing.assert_close(result.tangent, operation(value.tangent.to(torch.complex128)))


def test_empty_matrix_contraction_retains_coefficient_autograd_connection():
    base = torch.empty(2, 0, dtype=torch.float64, requires_grad=True)
    with PolyTensor.coefficient_autograd():
        result = PolyTensor((base,)) @ torch.empty(0, 3, dtype=torch.float64)
    derivative, = torch.autograd.grad(result.value.sum(), base)
    torch.testing.assert_close(result.value, torch.zeros(2, 3, dtype=torch.float64))
    torch.testing.assert_close(derivative, torch.empty_like(base))


def test_dtype_conversion_preserves_range_and_complex_directions():
    value = PolyTensor((torch.tensor(-120.0, dtype=torch.float64),
                        torch.tensor(2j, dtype=torch.complex128)))
    result = torch.exp(value).float() * 1e30
    magnitude = math.exp(-120 + math.log(1e30))
    assert result.value.dtype == torch.float32
    assert result.tangent.dtype == torch.complex64
    torch.testing.assert_close(result.value, torch.tensor(magnitude), rtol=2e-5, atol=0)
    torch.testing.assert_close(result.tangent, torch.tensor(2j * magnitude), rtol=2e-5, atol=0)


def test_detach_also_detaches_saved_range_graph():
    base = torch.tensor(-1000.0, dtype=torch.float64, requires_grad=True)
    with PolyTensor.coefficient_autograd():
        value = torch.exp(PolyTensor((base, torch.ones_like(base))))
        result = value.detach() * 1e300
    assert all(not coefficient.requires_grad for coefficient in result.coeffs)


def test_cross_operation_tail_supports_coefficient_reverse_derivatives():
    base = torch.tensor(-1000.0, dtype=torch.float64, requires_grad=True)
    with PolyTensor.coefficient_autograd():
        result = torch.exp(PolyTensor((base, torch.ones_like(base)))) * 1e300
    first, = torch.autograd.grad(result.tangent, base, create_graph=True)
    second, = torch.autograd.grad(first, base)
    expected = torch.tensor(math.exp(-1000 + math.log(1e300)), dtype=torch.float64)
    for derivative in (first, second):
        torch.testing.assert_close(derivative, expected, rtol=1e-12, atol=0)


def test_cross_operation_tail_supports_wrapper_reverse_derivatives():
    base = torch.tensor(-1000.0, dtype=torch.float64)
    value = PolyTensor((base, torch.ones_like(base)), requires_grad=True)
    with PolyTensor.retain_wrappers():
        result = torch.exp(value) * 1e300
        first, = torch.autograd.grad(result, value, create_graph=True)
        second, = torch.autograd.grad(first, value)
        expected = torch.tensor(math.exp(-1000 + math.log(1e300)), dtype=torch.float64)
        for derivative in (first, second):
            for coefficient in derivative.coeffs:
                torch.testing.assert_close(coefficient, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("coefficient_mode", (False, True))
def test_saturated_softmax_matrix_product_gradients(coefficient_mode):
    base = torch.tensor([0.0, -1000.0], dtype=torch.float64)
    values = torch.tensor([0.0, 1e300], dtype=torch.float64)
    if coefficient_mode:
        base.requires_grad_()
        values.requires_grad_()
        with PolyTensor.coefficient_autograd():
            logits = PolyTensor((base,))
            value = PolyTensor((values,))
            result = (torch.softmax(logits, dim=0) @ value) * 1e300
        gradients = torch.autograd.grad(result.value, (base, values))
    else:
        logits = PolyTensor((base,), requires_grad=True)
        value = PolyTensor((values,), requires_grad=True)
        with PolyTensor.retain_wrappers():
            result = (torch.softmax(logits, dim=0) @ value) * 1e300
            gradients = tuple(g.value for g in torch.autograd.grad(result, (logits, value)))
    first = math.exp(-1000 + math.log(1e300))
    second = math.exp(-1000 + 2 * math.log(1e300))
    torch.testing.assert_close(gradients[0], torch.tensor([-second, second], dtype=torch.float64), rtol=1e-12, atol=0)
    torch.testing.assert_close(gradients[1], torch.tensor([1e300, first], dtype=torch.float64), rtol=1e-12, atol=0)


@pytest.mark.parametrize("coefficient_mode", (False, True))
def test_broadcast_gradient_sum_keeps_range_until_activation_backward(coefficient_mode):
    base = torch.tensor(-1000.0, dtype=torch.float64)
    multipliers = torch.full((3,), 1e308, dtype=torch.float64)
    if coefficient_mode:
        base.requires_grad_()
        with PolyTensor.coefficient_autograd():
            value = PolyTensor((base, torch.ones_like(base)))
            result = (torch.exp(value) * multipliers).sum()
        derivatives = torch.autograd.grad(result.tangent, base)
    else:
        value = PolyTensor((base, torch.ones_like(base)), requires_grad=True)
        with PolyTensor.retain_wrappers():
            result = (torch.exp(value) * multipliers).sum()
            derivatives = torch.autograd.grad(result, value)[0].coeffs
    expected = torch.tensor(3 * math.exp(-1000 + math.log(1e308)), dtype=torch.float64)
    for derivative in derivatives:
        torch.testing.assert_close(derivative, expected, rtol=1e-12, atol=0)
