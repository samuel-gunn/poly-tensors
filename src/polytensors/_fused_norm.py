"""Fused plain-range first-order RMSNorm, with the general path as fallback.

Half and bfloat16 inputs accumulate in float32, as native normalization kernels
normally do. Float64 keeps float64 accumulation. Weight multiplication stays in
ordinary PolyTensor operations, so arbitrary broadcast and weight gradients work.
"""
import math
import torch


def eligible(input, normalized_shape, eps):
    from ._plain import plain_range_enabled
    from ._tensor import PolyTensor, _coefficient_autograd_enabled
    from ._fused_attention import available
    return (plain_range_enabled() and not _coefficient_autograd_enabled.get()
            and isinstance(input, PolyTensor) and input.degree == 1
            and input.is_cuda and input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            and all(c.dtype == input.dtype for c in input.coeffs)
            and len(normalized_shape) > 0 and tuple(input.shape[-len(normalized_shape):]) == normalized_shape
            and 0 < math.prod(normalized_shape) <= 8192 and input.numel() > 0
            and (eps is None or isinstance(eps, (float, int))) and available())


def _run(input, shape, eps, gradient=None):
    import triton
    from ._norm_kernels import _forward, _backward
    from ._tensor import PolyTensor
    x0, x1 = (c.contiguous() for c in input.coeffs)
    out = (torch.empty_like(x0), torch.empty_like(x1))
    n = math.prod(shape)
    block = triton.next_power_of_2(n)
    options = dict(N=n, EPS=eps, BLOCK=block, DOUBLE=x0.dtype == torch.float64,
                   num_warps=4 if block < 2048 else 8, enable_fp_fusion=False)
    if gradient is None:
        _forward[(x0.numel() // n,)](x0, x1, *out, **options)
    else:
        coeffs = gradient.coeffs if isinstance(gradient, PolyTensor) else (gradient, torch.zeros_like(gradient))
        g0, g1 = (c.expand(x0.shape).contiguous() for c in coeffs)
        _backward[(x0.numel() // n,)](x0, x1, g0, g1, *out, **options)
    return PolyTensor(out)


class _SeriesNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, shape, eps):
        ctx.save_for_backward(input)
        ctx.shape, ctx.eps = shape, eps
        return _run(input, shape, eps)

    @staticmethod
    def backward(ctx, gradient):
        (input,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Preserve further reverse derivatives rather than returning a detached kernel result.
            dims = tuple(range(-len(ctx.shape), 0))
            r = torch.rsqrt(input.square().mean(dim=dims, keepdim=True) + ctx.eps)
            result = gradient * r - input * r.pow(3) * (input * gradient).mean(dim=dims, keepdim=True)
        else:
            result = _run(input, ctx.shape, ctx.eps, gradient)
        return result, None, None


def rms_norm(input, normalized_shape, weight=None, eps=None):
    eps = torch.finfo(input.dtype).eps if eps is None else eps
    if input.requires_grad and torch.is_grad_enabled():
        result = _SeriesNorm.apply(input, normalized_shape, eps)
    else:
        result = _run(input, normalized_shape, eps)
    return result if weight is None else result * weight
