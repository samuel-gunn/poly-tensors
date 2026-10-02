import gc

import pytest
import torch

from polytensors import PolyTensor


def _degree_three_line(x0, direction, *, requires_grad=False):
    zeros = torch.zeros_like(x0)
    return PolyTensor(
        (x0, direction, zeros, zeros),
        requires_grad=requires_grad,
    )


def _finite_difference_coefficients(function, x0, direction, step=2e-3):
    f0 = function(x0)
    plus = function(x0 + step * direction)
    minus = function(x0 - step * direction)
    plus_two = function(x0 + 2 * step * direction)
    minus_two = function(x0 - 2 * step * direction)
    return (
        f0,
        (plus - minus) / (2 * step),
        (plus - 2 * f0 + minus) / (2 * step**2),
        (plus_two - 2 * plus + 2 * minus - minus_two) / (12 * step**3),
    )


def test_retention_scope_keeps_intermediate_wrappers_through_backward():
    coefficients = tuple(
        torch.tensor([0.2, -0.3], dtype=torch.float64)
        for _ in range(4)
    )
    x = PolyTensor(coefficients, requires_grad=True)

    with PolyTensor.retain_wrappers() as retained:
        loss = torch.nn.functional.silu((x * x + 0.3).sum())
        gc.collect()
        loss.backward()

        assert len(retained) > 1
        assert isinstance(x.grad, PolyTensor)
        assert all(torch.isfinite(coefficient).all() for coefficient in x.grad.coeffs)

    # A training step cannot accidentally retain the whole preceding graph.
    assert retained == []


def test_nested_retention_scope_also_retains_in_outer_scope():
    x = PolyTensor.constant(torch.tensor([0.1, -0.2]), degree=3)
    with PolyTensor.retain_wrappers() as outer:
        with PolyTensor.retain_wrappers() as inner:
            y = torch.nn.functional.silu(x + 0.4)
            assert any(item is y for item in inner)
            assert any(item is y for item in outer)
        assert inner == []
        assert any(item is y for item in outer)
    assert outer == []


def test_coefficient_autograd_gradient_jet_matches_finite_differences():
    x0_value = torch.tensor(
        [[0.2, -1.3, 2.1], [3.0, -4.0, 0.1]],
        dtype=torch.float64,
    )
    direction = torch.tensor(
        [[0.7, -0.2, 0.4], [-0.8, 0.5, 0.3]],
        dtype=torch.float64,
    )
    coefficients = (
        x0_value.clone().requires_grad_(),
        direction.clone().requires_grad_(),
        torch.zeros_like(x0_value, requires_grad=True),
        torch.zeros_like(x0_value, requires_grad=True),
    )
    x = PolyTensor(coefficients, requires_grad=False)

    with PolyTensor.coefficient_autograd():
        probability = torch.softmax(x, dim=-1)
        loss = probability.square().sum()

    assert not loss.requires_grad
    assert all(coefficient.grad_fn is not None for coefficient in loss.coeffs)
    gradient_coefficients = tuple(
        torch.autograd.grad(
            loss_coefficient,
            coefficients[0],
            retain_graph=order < 3,
        )[0]
        for order, loss_coefficient in enumerate(loss.coeffs)
    )

    def scalar_gradient(value):
        scalar_x = value.detach().requires_grad_()
        scalar_loss = torch.softmax(scalar_x, dim=-1).square().sum()
        return torch.autograd.grad(scalar_loss, scalar_x)[0]

    finite_difference = _finite_difference_coefficients(
        scalar_gradient,
        x0_value,
        direction,
    )
    assert torch.equal(gradient_coefficients[0], finite_difference[0])
    for order in range(1, 4):
        torch.testing.assert_close(
            gradient_coefficients[order],
            finite_difference[order],
            atol=8e-8,
            rtol=0.0,
        )


def test_cross_entropy_jet_uses_native_constant_and_gradient_path():
    logits = torch.tensor(
        [[0.2, -1.3, 2.1], [3.0, -4.0, 0.1]],
        dtype=torch.float64,
    )
    direction = torch.tensor(
        [[0.7, -0.2, 0.4], [-0.8, 0.5, 0.3]],
        dtype=torch.float64,
    )
    targets = torch.tensor([2, 0])

    def loss_function(value):
        return torch.nn.functional.cross_entropy(
            value,
            targets,
            reduction="sum",
        )

    actual = loss_function(_degree_three_line(logits, direction))
    finite_difference = _finite_difference_coefficients(
        loss_function,
        logits,
        direction,
    )
    assert torch.equal(actual.coeffs[0], finite_difference[0])
    for order in range(1, 4):
        torch.testing.assert_close(
            actual.coeffs[order],
            finite_difference[order],
            atol=5e-8,
            rtol=0.0,
        )


def test_softmax_and_logsumexp_jets_are_rooted_at_native_float32_values():
    generator = torch.Generator().manual_seed(481516234)
    x0 = 5 * torch.randn((4, 4096), generator=generator, dtype=torch.float32)
    direction = torch.randn((4, 4096), generator=generator, dtype=torch.float32)
    x = _degree_three_line(x0, direction)

    probability = torch.softmax(x, dim=-1)
    log_probability = torch.log_softmax(x, dim=-1)
    normalizer = torch.logsumexp(x, dim=-1)
    native_probability = torch.softmax(x0, dim=-1)
    native_log_probability = torch.log_softmax(x0, dim=-1)
    native_normalizer = torch.logsumexp(x0, dim=-1)
    native_normalizer_probability = torch.softmax(x0, dim=-1)
    expected_first_normalizer = (
        native_normalizer_probability * direction
    ).sum(dim=-1)
    expected_first_probability_normalizer = (
        native_probability * direction
    ).sum(dim=-1)
    expected_first_probability = native_probability * (
        direction - expected_first_probability_normalizer.unsqueeze(-1)
    )

    # Exact equality is intentional: coefficient zero is the native kernel and
    # coefficient one is generated directly from that stored value.  The old
    # exp/sum implementation differed by roughly 1e-6 for this float32 case.
    assert torch.equal(probability.coeffs[0], native_probability)
    assert torch.equal(normalizer.coeffs[0], native_normalizer)
    assert torch.equal(probability.coeffs[1], expected_first_probability)
    assert torch.equal(normalizer.coeffs[1], expected_first_normalizer)
    expected_first_log_probability = direction - (
        torch.exp(native_log_probability) * direction
    ).sum(dim=-1, keepdim=True)
    assert torch.equal(log_probability.coeffs[0], native_log_probability)
    assert torch.equal(
        log_probability.coeffs[1],
        expected_first_log_probability,
    )


def test_elementwise_jets_are_rooted_at_native_float32_values():
    generator = torch.Generator().manual_seed(73)
    variance = 0.01 + 3 * torch.rand(
        (32, 1536), generator=generator, dtype=torch.float32
    )
    variance_direction = torch.randn(
        variance.shape, generator=generator, dtype=torch.float32
    )
    variance_jet = _degree_three_line(variance, variance_direction)

    inverse = torch.reciprocal(variance_jet)
    inverse_zero = torch.reciprocal(variance)
    inverse_first = -(inverse_zero * inverse_zero) * variance_direction
    assert torch.equal(inverse.coeffs[0], inverse_zero)
    # Stable recurrences can differ from the native backward's evaluation order.
    torch.testing.assert_close(inverse.coeffs[1], inverse_first, rtol=3e-7, atol=0)

    inverse_root = torch.rsqrt(variance_jet)
    inverse_root_zero = torch.rsqrt(variance)
    inverse_root_first = -(
        inverse_root_zero * inverse_root_zero * inverse_root_zero
    ) * variance_direction / 2
    assert torch.equal(inverse_root.coeffs[0], inverse_root_zero)
    torch.testing.assert_close(inverse_root.coeffs[1], inverse_root_first, rtol=4e-7, atol=0)

    activation = 5 * torch.randn(
        (32, 1536), generator=generator, dtype=torch.float32
    )
    activation_direction = torch.randn(
        activation.shape, generator=generator, dtype=torch.float32
    )
    activation_jet = _degree_three_line(activation, activation_direction)
    sigmoid = torch.sigmoid(activation_jet)
    sigmoid_zero = torch.sigmoid(activation)
    sigmoid_first = (
        sigmoid_zero - sigmoid_zero * sigmoid_zero
    ) * activation_direction
    assert torch.equal(sigmoid.coeffs[0], sigmoid_zero)
    assert torch.equal(sigmoid.coeffs[1], sigmoid_first)

    silu = torch.nn.functional.silu(activation_jet)
    silu_first = activation * sigmoid_first + activation_direction * sigmoid_zero
    assert torch.equal(silu.coeffs[0], torch.nn.functional.silu(activation))
    assert torch.equal(silu.coeffs[1], silu_first)


@pytest.mark.parametrize(
    "function",
    (
        lambda x: torch.softmax(x, dim=-1),
        lambda x: torch.log_softmax(x, dim=-1),
        lambda x: torch.logsumexp(x, dim=-1),
    ),
)
def test_degree_three_nonlinear_jets_match_finite_differences(function):
    x0 = torch.tensor(
        [[0.2, -1.3, 2.1], [3.0, -4.0, 0.1]],
        dtype=torch.float64,
    )
    direction = torch.tensor(
        [[0.7, -0.2, 0.4], [-0.8, 0.5, 0.3]],
        dtype=torch.float64,
    )
    actual = function(_degree_three_line(x0, direction))
    finite_difference = _finite_difference_coefficients(function, x0, direction)

    assert torch.equal(actual.coeffs[0], finite_difference[0])
    for order in range(1, 4):
        torch.testing.assert_close(
            actual.coeffs[order],
            finite_difference[order],
            atol=5e-8,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    "function",
    (
        torch.reciprocal,
        torch.rsqrt,
        torch.sigmoid,
        torch.nn.functional.silu,
    ),
)
def test_degree_three_elementwise_jets_match_finite_differences(function):
    x0 = torch.tensor(
        [[0.7, 1.3, 2.1], [3.0, 4.0, 1.1]],
        dtype=torch.float64,
    )
    direction = torch.tensor(
        [[0.7, -0.2, 0.4], [-0.8, 0.5, 0.3]],
        dtype=torch.float64,
    )
    actual = function(_degree_three_line(x0, direction))
    finite_difference = _finite_difference_coefficients(
        function,
        x0,
        direction,
        step=5e-4,
    )

    assert torch.equal(actual.coeffs[0], finite_difference[0])
    for order in range(1, 4):
        torch.testing.assert_close(
            actual.coeffs[order],
            finite_difference[order],
            atol=2.2e-6,
            rtol=0.0,
        )
