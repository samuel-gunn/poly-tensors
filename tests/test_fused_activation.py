import pytest
import torch
import torch.nn.functional as F
from polytensors import PolyTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('function', [F.gelu, torch.tanh, torch.sigmoid])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_activation_value_and_gradient_jvp(function, dtype):
    torch.manual_seed(18)
    data = [torch.randn(3, 1027, device='cuda', dtype=dtype).T for _ in range(4)]
    x0, x1, g0, g1 = data
    value_ref = torch.func.jvp(function, (x0.double(),), (x1.double(),))
    def vjp(x, g):
        return torch.func.vjp(function, x)[1](g)[0]
    gradient_ref = torch.func.jvp(vjp, (x0.double(), g0.double()), (x1.double(), g1.double()))
    x = PolyTensor((x0, x1), requires_grad=True)
    with PolyTensor.plain_range(), PolyTensor.retain_wrappers():
        y = function(x)
        dx, = torch.autograd.grad(y, x, PolyTensor((g0, g1)))
    tol = {torch.float64: 1e-11, torch.float32: 3e-6, torch.float16: .004, torch.bfloat16: .04}[dtype]
    for got, want in zip((*y.coeffs, *dx.coeffs), (*value_ref, *gradient_ref)):
        torch.testing.assert_close(got, want.to(dtype), rtol=tol, atol=tol)


@pytest.mark.parametrize('function', [F.gelu, torch.tanh, torch.sigmoid])
def test_second_reverse_and_coefficient_autograd(function):
    x0 = torch.randn(2, 3, device='cuda', dtype=torch.float64, requires_grad=True)
    x1 = torch.randn_like(x0, requires_grad=True)
    with PolyTensor.plain_range(), PolyTensor.coefficient_autograd():
        y = function(PolyTensor((x0, x1)))
        actual = torch.autograd.grad(y.coeffs[1].sum(), (x0, x1))
    _, expected_y = torch.func.jvp(function, (x0,), (x1,))
    expected = torch.autograd.grad(expected_y.sum(), (x0, x1))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    x = PolyTensor((x0.detach(), x1.detach()), requires_grad=True)
    with PolyTensor.plain_range(), PolyTensor.retain_wrappers():
        dy, = torch.autograd.grad(function(x).sum(), x, create_graph=True)
        ddy, = torch.autograd.grad(dy.sum(), x)
    def second(z):
        return torch.func.grad(lambda v: torch.func.grad(lambda w: function(w).sum())(v).sum())(z)
    expected = torch.func.jvp(second, (x0.detach(),), (x1.detach(),))
    for a, b in zip(ddy.coeffs, expected):
        torch.testing.assert_close(a, b)
