import pytest
import torch

from polytensors import PolyTensor


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("complex_direction", [False, True])
@pytest.mark.parametrize("operation", [torch.square, torch.sqrt])
def test_plain_first_order_matches_independent_jvp_and_backward(device, complex_direction, operation):
    generator = torch.Generator(device=device).manual_seed(14)
    x = torch.rand((4, 7), dtype=torch.float64, device=device, generator=generator) + 0.5
    v = torch.randn(x.shape, dtype=x.dtype, device=device, generator=generator)
    if complex_direction:
        v = v + 1j * torch.randn(x.shape, dtype=x.dtype, device=device, generator=generator)
    expected_value, expected_real = torch.func.jvp(operation, (x,), (v.real,))
    expected = expected_real
    if complex_direction:
        expected = expected + 1j * torch.func.jvp(operation, (x,), (v.imag,))[1]
    p = PolyTensor((x, v), requires_grad=True)
    with PolyTensor.retain_wrappers(), PolyTensor.plain_range():
        y = operation(p)
        y.sum().backward()
    torch.testing.assert_close(y.coeffs[0], expected_value)
    torch.testing.assert_close(y.coeffs[1], expected)
    derivative = torch.func.grad(lambda t: operation(t).sum())
    expected_gradient, expected_tangent = torch.func.jvp(derivative, (x,), (v.real,))
    if complex_direction:
        expected_tangent = expected_tangent + 1j * torch.func.jvp(derivative, (x,), (v.imag,))[1]
    torch.testing.assert_close(p.grad.coeffs[0], expected_gradient)
    torch.testing.assert_close(p.grad.coeffs[1], expected_tangent)


@pytest.mark.parametrize("degree", [0, 1, 4])
def test_plain_square_matches_extended_range_all_orders(degree):
    torch.manual_seed(16)
    cs = tuple(torch.randn((4, 3), dtype=torch.float64) for _ in range(degree + 1))
    p = PolyTensor(cs)
    expected = p.square()
    with PolyTensor.plain_range():
        actual = p.square()
    for a, b in zip(actual.coeffs, expected.coeffs):
        torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-14)


def test_plain_sqrt_extreme_finite_values_and_coefficient_autograd():
    x = torch.tensor([1e-300, 1e300], dtype=torch.float64)
    v = torch.tensor([1e-150, 1e150], dtype=torch.float64)
    with PolyTensor.plain_range():
        y = PolyTensor((x, v)).sqrt()
    torch.testing.assert_close(y.tangent, torch.full_like(x, 0.5))
    x = torch.tensor([0.7, 2.0], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    def evaluate(a, b):
        with PolyTensor.coefficient_autograd(), PolyTensor.plain_range():
            return PolyTensor((a, b)).sqrt().tangent
    assert torch.autograd.gradcheck(evaluate, (x, v))
