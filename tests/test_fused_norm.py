import pytest
import torch
import torch.nn.functional as F
from polytensors import PolyTensor
from polytensors._decompositions import _rms_norm
from polytensors._fused_norm import eligible

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('shape,normalized', [((3, 17), (17,)), ((2, 3, 4), (3, 4)), ((2, 4, 3), (3,))])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('eps', [None, 0.1, 1e-5])
def test_norm_jvp_and_gradient(shape, normalized, dtype, eps):
    torch.manual_seed(3)
    x0 = torch.randn(shape, device='cuda', dtype=dtype)
    x1 = torch.randn_like(x0) * .2
    g0, g1 = torch.randn_like(x0), torch.randn_like(x0)
    floor = torch.finfo(dtype).eps if eps is None else eps
    # Compare to independent JVPs in double precision; half inputs/outputs round once.
    def function(x):
        dims = tuple(range(-len(normalized), 0))
        return x * (x.square().mean(dims, keepdim=True) + floor).rsqrt()
    ref = torch.func.jvp(function, (x0.double(),), (x1.double(),))
    def vjp(x, g):
        return torch.func.vjp(function, x)[1](g)[0]
    grad_ref = torch.func.jvp(vjp, (x0.double(), g0.double()), (x1.double(), g1.double()))
    x = PolyTensor((x0, x1), requires_grad=True)
    with PolyTensor.plain_range(), PolyTensor.retain_wrappers():
        assert eligible(x, normalized, eps)
        y = F.rms_norm(x, normalized, eps=eps)
        dx, = torch.autograd.grad(y, x, PolyTensor((g0, g1)))
    tol = {torch.float64: 1e-12, torch.float32: 2e-6, torch.float16: .002, torch.bfloat16: .02}[dtype]
    for got, want in zip((*y.coeffs, *dx.coeffs), (*ref, *grad_ref)):
        torch.testing.assert_close(got, want.to(dtype), atol=tol, rtol=tol)


def test_weight_and_noncontiguous_and_second_reverse():
    torch.manual_seed(7)
    data = [torch.randn(2, 7, 3, device='cuda', dtype=torch.float64).transpose(1, 2) for _ in range(2)]
    weight_data = [torch.randn(7, device='cuda', dtype=torch.float64) for _ in range(2)]
    results = []
    for fn in (F.rms_norm, _rms_norm):
        x = PolyTensor(data, requires_grad=True)
        w = PolyTensor(weight_data, requires_grad=True)
        with PolyTensor.plain_range(), PolyTensor.retain_wrappers():
            loss = fn(x, (7,), weight=w, eps=.1).square().sum()
            dx, dw = torch.autograd.grad(loss, (x, w), create_graph=True)
            dxx, = torch.autograd.grad(dx.sum(), x)
            results.append([*loss.coeffs, *dx.coeffs, *dw.coeffs, *dxx.coeffs])
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


def test_coefficient_autograd_falls_back():
    x0 = torch.randn(2, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    x1 = torch.randn_like(x0, requires_grad=True)
    with PolyTensor.plain_range(), PolyTensor.coefficient_autograd():
        x = PolyTensor((x0, x1))
        assert not eligible(x, (4,), .1)
        y = F.rms_norm(x, (4,), eps=.1)
        torch.autograd.grad(y.coeffs[1].sum(), (x0, x1))
