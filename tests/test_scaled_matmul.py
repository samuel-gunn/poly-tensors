"""Scaled matrix products retain adjoints across extreme exponent gaps."""

import pytest
import torch

from polytensors._normalization import normalization_scaled_series
from polytensors._scaled import ScaledTensor as S


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("shapes", [((3,), (3,)), ((2, 3), (3,)), ((3,), (2, 3, 2)), ((2, 1, 2, 3), (1, 4, 3, 2))])
def test_scaled_matmul_adjoint_matches_native_with_broadcasting_and_vectors(dtype, shapes):
    generator = torch.Generator().manual_seed(193)
    left, right = tuple(torch.randn(shape, generator=generator, dtype=dtype).requires_grad_() for shape in shapes)

    def function(a, b):
        return S(a).matmul(S(b)).to_tensor()

    torch.testing.assert_close(function(left, right), left @ right)
    assert torch.autograd.gradcheck(function, (left, right), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (left, right), fast_mode=True)


def test_scaled_matmul_zero_entry_does_not_contaminate_extreme_probability_gradient():
    logits = torch.tensor([[0.0, -1000.0]], dtype=torch.float64, requires_grad=True)
    probabilities, _ = normalization_scaled_series((logits,), -1)
    output = probabilities[0].matmul(S(torch.tensor([[0.0], [1.0]], dtype=torch.float64)))
    amplified = (output * 1e300 * 1e300).to_tensor()
    gradient = torch.autograd.grad(amplified, logits, create_graph=True)[0]
    second = torch.autograd.grad(gradient[0, 1], logits)[0]
    expected = torch.tensor([[-5.075958897549457e165, 5.075958897549457e165]], dtype=torch.float64)
    torch.testing.assert_close(gradient, expected, atol=0, rtol=2e-12)
    torch.testing.assert_close(second, expected, atol=0, rtol=2e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("matrix", [False, True])
def test_scaled_polynomial_product_adjoint_matches_second_order_coefficients(dtype, matrix):
    from polytensors._scaled_matmul import scaled_polynomial_product

    generator = torch.Generator().manual_seed(826)
    coefficients = tuple((torch.randn(2, 2, generator=generator, dtype=dtype) / 3).requires_grad_() for _ in range(6))

    def function(*values):
        left, right = tuple(S(value) for value in values[:3]), tuple(S(value) for value in values[3:])
        return tuple(c.to_tensor() for c in scaled_polynomial_product(left, right, matrix=matrix))

    outputs = function(*coefficients)
    for k, output in enumerate(outputs):
        expected = sum(coefficients[i] @ coefficients[3 + k - i] if matrix
                       else coefficients[i] * coefficients[3 + k - i] for i in range(k + 1))
        torch.testing.assert_close(output, expected)
    assert torch.autograd.gradcheck(function, coefficients, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, coefficients, fast_mode=True)
