"""Attention values and gradient coefficients against ordinary reverse AD."""

import contextlib
import math

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from polytensors import PolyTensor


def _data():
    generator = torch.Generator().manual_seed(27)
    values = tuple(torch.randn(1, 1, 3, 2, generator=generator, dtype=torch.float64) / 3 for _ in range(3))
    directions = tuple(torch.randn(c.shape, generator=generator, dtype=c.dtype) / 4 for c in values)
    return values, directions


def _plain_attention(query, key, value, mask, *, causal=False, scale=None):
    scores = query @ key.transpose(-2, -1) * (1 / math.sqrt(query.shape[-1]) if scale is None else scale)
    excluded = ~mask if mask.dtype == torch.bool else torch.isneginf(mask)
    if causal:
        excluded = excluded | ~torch.ones_like(excluded).tril()
    if mask.dtype != torch.bool:
        scores = scores + mask.masked_fill(excluded, 0)
    scores = scores.masked_fill(excluded, float("-inf"))
    empty = excluded.all(dim=-1, keepdim=True)
    weights = torch.softmax(scores.masked_fill(empty, 0), dim=-1).masked_fill(excluded, 0)
    return weights @ value


def _reference(values, directions, mask, degree=3):
    def function(t):
        inputs = tuple(value + t * direction for value, direction in zip(values, directions))
        output = _plain_attention(*inputs, mask)
        gradients = torch.autograd.grad(output.square().sum(), inputs, create_graph=True)
        return torch.stack((output, *gradients))

    result = []
    t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    for order in range(degree + 1):
        result.append(function(t).detach() / math.factorial(order))
        previous = function
        function = lambda t, previous=previous: torch.autograd.functional.jacobian(previous, t, create_graph=True)
    return result


@pytest.fixture(scope="module", params=["boolean", "additive"])
def reference_case(request):
    values, directions = _data()
    mask = torch.tensor([[False, False, False], [True, False, True], [True, True, True]])
    if request.param == "additive":
        mask = torch.tensor([[float("-inf")] * 3, [0.2, float("-inf"), -0.1], [0.1, -0.3, 0.0]], dtype=torch.float64)
    return values, directions, mask, _reference(values, directions, mask)


@pytest.mark.parametrize("backend", [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
@pytest.mark.parametrize("mode", ["wrapper", "coefficient"])
@pytest.mark.parametrize("phase", [1.0, 1.0 + 0.5j])
def test_masked_attention_values_and_gradient_coefficients(backend, mode, phase, reference_case):
    values, directions, mask, reference = reference_case
    inputs = tuple(
        PolyTensor((value.clone().requires_grad_(), phase * direction), degree=3, requires_grad=mode == "wrapper")
        for value, direction in zip(values, directions)
    )
    with PolyTensor.retain_wrappers(), sdpa_kernel(backend):
        scope = PolyTensor.coefficient_autograd() if mode == "coefficient" else contextlib.nullcontext()
        with scope:
            output = torch.nn.functional.scaled_dot_product_attention(*inputs, attn_mask=mask)
            loss = output.square().sum()
        if mode == "wrapper":
            loss.backward()
            gradients = [tuple(x.grad.coeffs[k] for x in inputs) for k in range(4)]
        else:
            gradients = []
            for coefficient in loss.coeffs:
                real = torch.autograd.grad(coefficient.real, tuple(x.value for x in inputs), retain_graph=True)
                if coefficient.is_complex():
                    imaginary = torch.autograd.grad(coefficient.imag, tuple(x.value for x in inputs), retain_graph=True)
                    real = tuple(torch.complex(r, i) for r, i in zip(real, imaginary))
                gradients.append(real)

    for order, coefficient in enumerate(output.coeffs):
        assert torch.equal(coefficient[..., 0, :], torch.zeros_like(coefficient[..., 0, :]))
        factor = phase**order if order else 1.0
        torch.testing.assert_close(coefficient, reference[order][0] * factor, atol=2e-14, rtol=2e-12)
        for index, gradient in enumerate(gradients[order]):
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(gradient, reference[order][index + 1] * factor, atol=2e-14, rtol=2e-12)


@pytest.mark.parametrize("causal", [False, True])
def test_cpu_attention_auxiliary_normalizer_is_zero_for_masked_rows(causal):
    values, directions = _data()
    inputs = tuple(PolyTensor((value, direction), degree=3) for value, direction in zip(values, directions))
    mask = torch.tensor([[float("-inf")] * 3, [float("-inf"), 0, 0], [0, 0, 0]], dtype=torch.float64)
    actual, normalizer = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(*inputs, is_causal=causal, attn_mask=mask)
    expected, expected_normalizer = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(*values, is_causal=causal, attn_mask=mask)
    torch.testing.assert_close(actual.value, expected)
    torch.testing.assert_close(normalizer.value, expected_normalizer)
    for coefficient in normalizer.coeffs:
        assert torch.isfinite(coefficient).all()
        assert torch.equal(coefficient[..., 0], torch.zeros_like(coefficient[..., 0]))


@pytest.mark.parametrize("dropout", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("mode", ["wrapper", "coefficient"])
def test_fully_masked_attention_remains_zero_through_dropout(dropout, mode):
    values, directions = _data()
    inputs = tuple(PolyTensor((v.clone().requires_grad_(), d), degree=3, requires_grad=mode == "wrapper") for v, d in zip(values, directions))
    with PolyTensor.retain_wrappers(), sdpa_kernel(SDPBackend.MATH):
        scope = PolyTensor.coefficient_autograd() if mode == "coefficient" else contextlib.nullcontext()
        with scope:
            output = torch.nn.functional.scaled_dot_product_attention(*inputs, attn_mask=torch.zeros(3, 3, dtype=torch.bool), dropout_p=dropout)
            loss = output.sum()
        if mode == "wrapper":
            loss.backward()
            gradients = [c for x in inputs for c in x.grad.coeffs]
        else:
            gradients = [g for c in loss.coeffs for g in torch.autograd.grad(c, tuple(x.value for x in inputs), retain_graph=True)]
    for coefficient in (*output.coeffs, *gradients):
        assert torch.equal(coefficient, torch.zeros_like(coefficient))


def test_safe_softmax_does_not_conceal_invalid_unmasked_scores():
    value = torch.tensor([[float("nan"), 0.0], [float("inf"), 0.0], [float("-inf"), float("-inf")]], dtype=torch.float64)
    output = torch.ops.aten._safe_softmax.default(PolyTensor((value, torch.ones_like(value)), degree=3), -1)
    for coefficient in output.coeffs:
        assert torch.isnan(coefficient[:2]).all()
        assert torch.equal(coefficient[2], torch.zeros_like(coefficient[2]))


def test_attention_scales_before_dot_product_to_avoid_intermediate_overflow():
    value = torch.full((1, 1, 2, 1), 1e20, dtype=torch.float32)
    query = PolyTensor((value, torch.ones_like(value)), degree=2)
    key = PolyTensor((value, -torch.ones_like(value)), degree=2)
    output = torch.nn.functional.scaled_dot_product_attention(query, key, torch.tensor([[[[2.0], [4.0]]]]), scale=1e-20)
    torch.testing.assert_close(output.value, torch.full_like(value, 3.0))
    for coefficient in output.coeffs[1:]:
        torch.testing.assert_close(coefficient, torch.zeros_like(coefficient))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["wrapper", "coefficient"])
def test_attention_low_precision_accumulates_scores_in_float32(dtype, mode):
    value = torch.full((1, 1, 2, 1), 1000.0, dtype=dtype)
    base = value.clone().requires_grad_()
    query = PolyTensor((base, torch.ones_like(value)), degree=3, requires_grad=mode == "wrapper")
    with PolyTensor.retain_wrappers():
        scope = PolyTensor.coefficient_autograd() if mode == "coefficient" else contextlib.nullcontext()
        with scope:
            output = torch.nn.functional.scaled_dot_product_attention(query, query, torch.tensor([[[[2.0], [4.0]]]], dtype=dtype))
            loss = output.sum()
        if mode == "wrapper":
            loss.backward()
            gradients = query.grad.coeffs
        else:
            gradients = tuple(torch.autograd.grad(c, base, retain_graph=True)[0] for c in loss.coeffs)
    assert output.value.dtype == dtype
    torch.testing.assert_close(output.value, torch.full_like(value, 3.0))
    for coefficient in output.coeffs[1:]:
        torch.testing.assert_close(coefficient, torch.zeros_like(coefficient))
    signs = torch.tensor([[[[-1.0], [1.0]]]], dtype=dtype)
    for order, gradient in enumerate(gradients):
        expected = signs * 1000 if order == 0 else signs if order == 1 else torch.zeros_like(signs)
        torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("backend", [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
@pytest.mark.parametrize("mode", ["wrapper", "coefficient"])
@pytest.mark.parametrize("phase", [1.0, 1.0 + 0.5j])
def test_attention_backward_preserves_tiny_probabilities_amplified_by_values(backend, mode, phase):
    base = torch.ones((1, 1, 2, 1), dtype=torch.float64, requires_grad=True)
    query = PolyTensor((base, torch.full_like(base, 0.001) * phase), degree=3, requires_grad=mode == "wrapper")
    key = torch.tensor([[[[0.0], [-1000.0]]]], dtype=torch.float64)
    value = torch.tensor([[[[0.0], [1e300]]]], dtype=torch.float64)
    mask = torch.tensor([[False, False], [True, True]])
    with PolyTensor.retain_wrappers(), sdpa_kernel(backend):
        scope = PolyTensor.coefficient_autograd() if mode == "coefficient" else contextlib.nullcontext()
        with scope:
            output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
            loss = output.sum()
        if mode == "wrapper":
            loss.backward()
            gradients = query.grad.coeffs
        else:
            gradients = []
            for coefficient in loss.coeffs:
                gradient = torch.autograd.grad(coefficient.real, base, retain_graph=True)[0]
                if coefficient.is_complex():
                    imaginary = torch.autograd.grad(coefficient.imag, base, retain_graph=True)[0]
                    gradient = torch.complex(gradient, imaginary)
                gradients.append(gradient)

    # exp(-1000) rounds to zero in float64, but multiplication by 1e300
    # and the score derivative -1000 yields this representable derivative.
    # Corrections from the sigmoid denominator are below float64 precision.
    first_derivative = -5.075958897549457e-132
    for order, gradient in enumerate(gradients):
        expected = torch.zeros_like(gradient)
        expected[..., 1, :] = first_derivative * (-phase)**order / math.factorial(order)
        torch.testing.assert_close(gradient, expected, atol=0, rtol=2e-12)


@pytest.mark.parametrize("backend", [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
@pytest.mark.parametrize("mode", ["wrapper", "coefficient"])
@pytest.mark.parametrize("phase", [1.0, 1.0 + 0.5j])
def test_attention_preserves_tails_through_value_products_and_large_upstream_gradients(backend, mode, phase):
    query_base = torch.ones((1, 1, 1, 1), dtype=torch.float64, requires_grad=True)
    key = torch.tensor([[[[0.0], [-1000.0]]]], dtype=torch.float64)
    value_base = torch.tensor([[[[0.0], [1e300]]]], dtype=torch.float64, requires_grad=True)
    query = PolyTensor((query_base, torch.full_like(query_base, 0.001) * phase), degree=2, requires_grad=mode == "wrapper")
    value = PolyTensor((value_base, torch.zeros_like(value_base)), degree=2, requires_grad=mode == "wrapper")
    with PolyTensor.retain_wrappers(), sdpa_kernel(backend):
        scope = PolyTensor.coefficient_autograd() if mode == "coefficient" else contextlib.nullcontext()
        with scope:
            output = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=1.0)
            loss = output.sum() * 1e300
        if mode == "wrapper":
            loss.backward()
            gradients = tuple(zip(query.grad.coeffs, value.grad.coeffs))
        else:
            gradients = []
            for coefficient in loss.coeffs:
                real = torch.autograd.grad(coefficient.real, (query_base, value_base), retain_graph=True)
                if coefficient.is_complex():
                    imaginary = torch.autograd.grad(coefficient.imag, (query_base, value_base), retain_graph=True)
                    real = tuple(torch.complex(r, i) for r, i in zip(real, imaginary))
                gradients.append(real)
    for order, (query_gradient, value_gradient) in enumerate(gradients):
        factor = (-phase)**order / math.factorial(order)
        expected_output = torch.full_like(output.coeffs[order], 5.075958897549457e-135 * factor)
        torch.testing.assert_close(output.coeffs[order], expected_output, atol=0, rtol=2e-12)
        torch.testing.assert_close(query_gradient, torch.full_like(query_gradient, -5.075958897549457e168 * factor), atol=0, rtol=2e-12)
        expected_value = torch.empty_like(value_gradient)
        expected_value[..., 0, :] = 1e300 if order == 0 else -5.075958897549457e-135 * factor
        expected_value[..., 1, :] = 5.075958897549457e-135 * factor
        torch.testing.assert_close(value_gradient, expected_value, atol=0, rtol=2e-12)


def test_attention_coefficient_autograd_includes_additive_mask_and_normalizer():
    from polytensors._attention import attention

    generator = torch.Generator().manual_seed(782)
    inputs = tuple((torch.randn(1, 1, 2, 1, dtype=torch.float64, generator=generator) / 4).requires_grad_() for _ in range(6))
    mask = tuple((torch.randn(2, 2, dtype=torch.float64, generator=generator) / 4).requires_grad_() for _ in range(2))

    def function(*coefficients):
        output, normalizer = attention(coefficients[:2], coefficients[2:4], coefficients[4:6], attn_mask=coefficients[6:])
        return (*output, *normalizer)

    assert torch.autograd.gradcheck(function, (*inputs, *mask), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (*inputs, *mask), fast_mode=True)


@pytest.mark.parametrize("mode", ["wrapper", "coefficient"])
def test_attention_score_products_cancel_before_conversion(mode):
    query_base = torch.full((1, 1, 1, 2), 1e200, dtype=torch.float64, requires_grad=True)
    key = torch.tensor([[[[1e200, -1e200], [0.0, 0.0]]]], dtype=torch.float64)
    value = torch.tensor([[[[2.0, 0.0], [4.0, 0.0]]]], dtype=torch.float64)
    query = PolyTensor((query_base, torch.zeros_like(query_base)), degree=2, requires_grad=mode == "wrapper")
    with PolyTensor.retain_wrappers(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        scope = PolyTensor.coefficient_autograd() if mode == "coefficient" else contextlib.nullcontext()
        with scope:
            output = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=1.0)
            loss = output.sum()
        if mode == "wrapper":
            loss.backward()
            gradients = query.grad.coeffs
        else:
            gradients = tuple(torch.autograd.grad(c, query_base, retain_graph=True)[0] for c in loss.coeffs)
    torch.testing.assert_close(output.value, torch.tensor([[[[3.0, 0.0]]]], dtype=torch.float64))
    torch.testing.assert_close(gradients[0], torch.tensor([[[[-5e199, 5e199]]]], dtype=torch.float64))
    for coefficient in (*output.coeffs[1:], *gradients[1:]):
        torch.testing.assert_close(coefficient, torch.zeros_like(coefficient))


@pytest.mark.parametrize("backend", [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
@pytest.mark.parametrize("mode", ["wrapper", "coefficient"])
@pytest.mark.parametrize("phase", [1.0, 1.0 + 0.5j])
@pytest.mark.parametrize("multiplications", [1, 2])
def test_attention_output_tail_survives_later_multiplication(backend, mode, phase, multiplications):
    base = torch.ones((1, 1, 1, 1), dtype=torch.float64, requires_grad=True)
    query = PolyTensor((base, torch.full_like(base, 0.001) * phase), degree=2, requires_grad=mode == "wrapper")
    key = torch.tensor([[[[0.0], [-1000.0]]]], dtype=torch.float64)
    value = torch.tensor([[[[0.0], [1.0]]]], dtype=torch.float64)
    with PolyTensor.retain_wrappers(), sdpa_kernel(backend):
        scope = PolyTensor.coefficient_autograd() if mode == "coefficient" else contextlib.nullcontext()
        with scope:
            tiny = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=1.0)
            amplified = tiny
            for _ in range(multiplications):
                amplified = amplified * 1e300
            loss = amplified.sum()
        if mode == "wrapper":
            loss.backward()
            gradients = query.grad.coeffs
        else:
            gradients = []
            for coefficient in loss.coeffs:
                first = torch.autograd.grad(coefficient.real, base, create_graph=True, retain_graph=True)[0]
                if coefficient.is_complex():
                    imaginary = torch.autograd.grad(coefficient.imag, base, create_graph=True, retain_graph=True)[0]
                    first = torch.complex(first, imaginary)
                gradients.append(first)
    amplitude = 5.075958897549457e-135 if multiplications == 1 else 5.075958897549457e165
    for order, (output, gradient) in enumerate(zip(amplified.coeffs, gradients)):
        factor = (-phase)**order / math.factorial(order)
        torch.testing.assert_close(output, torch.full_like(output, amplitude * factor), atol=0, rtol=2e-12)
        torch.testing.assert_close(gradient, torch.full_like(gradient, -1000 * amplitude * factor), atol=0, rtol=2e-12)
        assert torch.equal(tiny.coeffs[order], torch.zeros_like(tiny.coeffs[order]))
        if mode == "coefficient":
            second = torch.autograd.grad(gradient.real, base, retain_graph=True)[0]
            if gradient.is_complex():
                imaginary = torch.autograd.grad(gradient.imag, base, retain_graph=True)[0]
                second = torch.complex(second, imaginary)
            torch.testing.assert_close(second, torch.full_like(second, 1e6 * amplitude * factor), atol=0, rtol=2e-12)


def test_polynomial_attention_full_dropout_has_zero_value_and_gradient():
    from polytensors._attention import attention

    base = torch.ones((1, 1, 2, 2), dtype=torch.float64, requires_grad=True)
    coefficients = (base, torch.ones_like(base))
    output, _ = attention(coefficients, coefficients, coefficients, dropout_p=1.0)
    gradient, = torch.autograd.grad(sum(c.sum() for c in output), base)
    assert all(torch.count_nonzero(c) == 0 for c in output)
    torch.testing.assert_close(gradient, torch.zeros_like(base))
