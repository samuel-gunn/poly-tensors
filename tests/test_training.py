"""Small, independently checked examples of differentiating through updates."""

import math

import pytest
import torch

from polytensors import PolyTensor


def _reference_training_coefficients(initial, direction, degree, steps=2, rate=0.1):
    """Differentiate ordinary functional SGD with PyTorch's reverse AD."""
    initial = initial.detach().requires_grad_()
    weight = initial
    for _ in range(steps):
        loss = (weight - 1).pow(4) / 4
        gradient = torch.autograd.grad(loss, weight, create_graph=True)[0]
        weight = weight - rate * gradient

    coefficients = [weight.detach()]
    derivative = weight
    for order in range(1, degree + 1):
        derivative = torch.autograd.grad(
            derivative, initial, create_graph=True
        )[0]
        coefficients.append(
            derivative.detach() * direction.pow(order) / math.factorial(order)
        )
    return coefficients


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_through_sgd_matches_independent_higher_derivatives(dtype):
    initial = torch.tensor(0.5, dtype=dtype)
    direction = torch.tensor(0.3, dtype=dtype)
    degree = 4
    weight = PolyTensor(
        (initial.clone(), direction.clone()), degree=degree, requires_grad=True
    )
    optimizer = torch.optim.SGD([weight], lr=0.1, foreach=False)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with PolyTensor.retain_wrappers():
            loss = (weight - 1).pow(4) / 4
            torch.autograd.backward(loss, torch.ones_like(loss))
            assert isinstance(weight.grad, PolyTensor)
            optimizer.step()

    expected = _reference_training_coefficients(initial, direction, degree)
    for actual, reference in zip(weight.coeffs, expected):
        torch.testing.assert_close(actual, reference)


def test_forward_through_sgd_with_real_base_and_complex_direction():
    initial = torch.tensor(0.5, dtype=torch.float64)
    direction = torch.tensor(0.3 + 0.2j, dtype=torch.complex128)
    degree = 4
    weight = PolyTensor(
        (initial.clone(), direction.clone()), degree=degree, requires_grad=True
    )
    optimizer = torch.optim.SGD([weight], lr=0.1, foreach=False)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with PolyTensor.retain_wrappers():
            loss = (weight - 1).pow(4) / 4
            torch.autograd.backward(loss, torch.ones_like(loss))
            optimizer.step()

    expected = _reference_training_coefficients(initial, direction, degree)
    for actual, reference in zip(weight.coeffs, expected):
        torch.testing.assert_close(actual, reference)


def test_training_input_direction_matches_closed_form_quadratic():
    initial = torch.tensor(0.2, dtype=torch.float64)
    target_value = torch.tensor(1.0, dtype=torch.float64)
    direction = torch.tensor(0.4, dtype=torch.float64)
    target = PolyTensor((target_value, direction), degree=3)
    weight = PolyTensor((initial.clone(),), degree=3, requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=0.1, foreach=False)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with PolyTensor.retain_wrappers():
            loss = (weight - target).square() / 2
            torch.autograd.backward(loss, torch.ones_like(loss))
            optimizer.step()

    decay = (1 - 0.1) ** 3
    torch.testing.assert_close(
        weight.coeffs[0], decay * initial + (1 - decay) * target_value
    )
    torch.testing.assert_close(weight.coeffs[1], (1 - decay) * direction)
    for coefficient in weight.coeffs[2:]:
        torch.testing.assert_close(coefficient, torch.zeros_like(coefficient))


@pytest.mark.parametrize("complex_direction", [False, True])
def test_coefficient_autograd_training_matches_independent_higher_derivatives(
    complex_direction,
):
    initial = torch.tensor(0.5, dtype=torch.float64)
    direction = torch.tensor(
        0.3 + 0.2j if complex_direction else 0.3,
        dtype=torch.complex128 if complex_direction else torch.float64,
    )
    degree = 4
    coefficients = (initial.clone(), direction) + tuple(
        torch.zeros_like(direction) for _ in range(degree - 1)
    )

    for _ in range(2):
        # Each coefficient is an independent leaf for this update's gradient.
        coefficients = tuple(c.detach().requires_grad_() for c in coefficients)
        weight = PolyTensor(coefficients)
        with PolyTensor.coefficient_autograd():
            loss = (weight - 1).pow(4) / 4
        assert not loss.requires_grad

        gradients = []
        for coefficient in loss.coeffs:
            gradient = torch.autograd.grad(
                coefficient.real, coefficients[0], retain_graph=True
            )[0]
            if coefficient.is_complex():
                # The base is real, so differentiate both output components.
                imaginary = torch.autograd.grad(
                    coefficient.imag, coefficients[0], retain_graph=True
                )[0]
                gradient = torch.complex(gradient, imaginary)
            gradients.append(gradient)
        coefficients = tuple(
            coefficient - 0.1 * gradient
            for coefficient, gradient in zip(coefficients, gradients)
        )

    expected = _reference_training_coefficients(initial, direction, degree)
    for actual, reference in zip(coefficients, expected):
        torch.testing.assert_close(actual, reference)


def test_coefficient_autograd_treats_taylor_coefficients_as_independent_inputs():
    coefficients = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (0.2, -0.3, 0.4, 0.1)
    )
    with PolyTensor.coefficient_autograd():
        result = torch.exp(PolyTensor(coefficients))

    base, first, second, third = (c.detach() for c in coefficients)
    expected = (
        base.exp(),
        base.exp() * first,
        base.exp() * (second + first.square() / 2),
        base.exp() * (third + first * second + first.pow(3) / 6),
    )
    for order, output in enumerate(result.coeffs):
        torch.testing.assert_close(output, expected[order])
        gradients = torch.autograd.grad(
            output, coefficients, retain_graph=True, allow_unused=True
        )
        for input_order, gradient in enumerate(gradients):
            # d[exp(P)]_k / d[P]_j = [exp(P)]_(k-j), or zero for j > k.
            reference = (
                expected[order - input_order]
                if input_order <= order
                else torch.zeros_like(base)
            )
            if gradient is None:
                gradient = torch.zeros_like(base)
            torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize("direction_scale", [1.0, 1.0 + 0.5j])
def test_forward_through_module_training_matches_functional_reference(direction_scale):
    degree = 3
    inputs = torch.tensor([[0.4, -0.1], [0.7, 0.3]], dtype=torch.float64)
    direction = torch.tensor([[0.1, 0.2], [-0.2, 0.3]], dtype=torch.float64)
    evaluation = torch.tensor([[0.6, -0.2]], dtype=torch.float64)
    initial_weight = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    initial_bias = torch.tensor([0.1], dtype=torch.float64)
    direction_dtype = (
        torch.complex128 if isinstance(direction_scale, complex) else torch.float64
    )
    module = torch.nn.Linear(2, 1, dtype=torch.float64)
    module.weight = torch.nn.Parameter(
        PolyTensor(
            (initial_weight.clone(), torch.zeros_like(initial_weight, dtype=direction_dtype)),
            degree=degree,
        )
    )
    module.bias = torch.nn.Parameter(
        PolyTensor(
            (initial_bias.clone(), torch.zeros_like(initial_bias, dtype=direction_dtype)),
            degree=degree,
        )
    )
    polynomial_inputs = PolyTensor((inputs, direction * direction_scale), degree=degree)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1, foreach=False)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with PolyTensor.retain_wrappers():
            loss = torch.sin(module(polynomial_inputs)).square().mean()
            torch.autograd.backward(loss, torch.ones_like(loss))
            optimizer.step()
    actual = torch.sin(module(evaluation)).sum()

    perturbation = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    perturbed_inputs = inputs + perturbation * direction
    weight = initial_weight.requires_grad_()
    bias = initial_bias.requires_grad_()
    for _ in range(2):
        prediction = torch.nn.functional.linear(perturbed_inputs, weight, bias)
        loss = torch.sin(prediction).square().mean()
        weight_gradient, bias_gradient = torch.autograd.grad(
            loss, (weight, bias), create_graph=True
        )
        weight = weight - 0.1 * weight_gradient
        bias = bias - 0.1 * bias_gradient
    reference = torch.sin(torch.nn.functional.linear(evaluation, weight, bias)).sum()
    for order, coefficient in enumerate(actual.coeffs):
        if order:
            reference = torch.autograd.grad(
                reference, perturbation, create_graph=True
            )[0]
        coefficient_reference = reference / math.factorial(order)
        if order:
            coefficient_reference = coefficient_reference * direction_scale**order
        torch.testing.assert_close(
            coefficient, coefficient_reference, rtol=1e-12, atol=1e-14
        )
    torch.testing.assert_close(module.weight.coeffs[0], weight)
    torch.testing.assert_close(module.bias.coeffs[0], bias)


def test_training_requires_lifting_parameters_before_polynomial_updates():
    parameter = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    target = PolyTensor((1.0, 0.2), degree=2)
    optimizer = torch.optim.SGD([parameter], lr=0.1, foreach=False)
    with PolyTensor.retain_wrappers():
        loss = (parameter - target).square() / 2
        torch.autograd.backward(loss, torch.ones_like(loss))
        with pytest.raises(RuntimeError, match="regular Tensor in-place with a PolyTensor"):
            optimizer.step()
