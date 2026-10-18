"""Fused cross entropy for real degree-1 series of logits.

For logits ``z0 + t z1`` (rows of length C) and integer targets ``y``:

    loss0 = logsumexp(z0) - z0[y],     loss1 = <p, z1> - z1[y],     p = softmax(z0).

One Triton program per row streams over the classes (online log-sum-exp), so
no (N, C) intermediates are formed besides the gradients. The backward computes
the exact vector-Jacobian product of ``(z0, z1) -> (loss0, loss1)``:

    d z1 = c1 (p - e_y),     d z0 = c0 (p - e_y) + c1 p (z1 - <p, z1>),

and, as for attention, the PolyTensor gradient series is that VJP with the
cotangents ``(G1, G0)`` and the input orders swapped. Rows whose target equals
``ignore_index`` have zero loss and gradient; reductions follow
``torch.nn.functional.cross_entropy`` (``mean`` divides by the number of
non-ignored rows).
"""

import contextvars

import torch

from ._fused_attention import available

_enabled = contextvars.ContextVar("polytensor_fused_cross_entropy", default=True)

_kernels = None


def _build_kernels():
    from ._cross_entropy_kernels import _forward, _backward

    return _forward, _backward


def _get_kernels():
    global _kernels
    if _kernels is None:
        _kernels = _build_kernels()
    return _kernels


_BLOCK = 1024


def _rows(z0, z1):
    """(N, C) logits whose rows are contiguous (a common row stride), e.g. a slice of padded logits."""
    if z0.stride(-1) != 1 or z1.stride() != z0.stride():
        z0, z1 = z0.contiguous(), z1.contiguous()
    return z0, z1


def forward(z0, z1, y, ignore_index, softcap=None):
    """Per-row (loss0, loss1, lse, <p, z1>) of (N, C) logits (optionally capped as softcap tanh(z / softcap))."""
    fwd, _ = _get_kernels()
    z0, z1 = _rows(z0, z1)
    N, C = z0.shape
    out = [torch.empty(N, dtype=torch.float32, device=z0.device) for _ in range(4)]
    if N:
        fwd[(N,)](z0, z1, y, *out, C, z0.stride(0), ignore_index, float(softcap or 1.0),
                  HAS_CAP=softcap is not None, BLOCK=_BLOCK, num_warps=4)
    return out


def backward(z0, z1, y, lse, pz1, c0, c1, ignore_index, softcap=None):
    """VJP of (z0, z1) -> (loss0, loss1) with per-row cotangents (c0, c1); contiguous (N, C) results."""
    _, bwd = _get_kernels()
    z0, z1 = _rows(z0, z1)
    N, C = z0.shape
    dz0 = torch.empty(N, C, dtype=z0.dtype, device=z0.device)
    dz1 = torch.empty_like(dz0)
    if N:
        bwd[(N,)](z0, z1, y, lse, pz1, c0.float().contiguous(), c1.float().contiguous(), dz0, dz1, C, z0.stride(0),
                  ignore_index, float(softcap or 1.0), HAS_CAP=softcap is not None, BLOCK=_BLOCK, num_warps=4)
    return dz0, dz1


class CoefficientCrossEntropy(torch.autograd.Function):
    """(z0, z1) -> per-row (loss0, loss1) on ordinary tensors."""

    @staticmethod
    def forward(ctx, z0, z1, y, ignore_index, softcap=None):
        y = y.contiguous()
        loss0, loss1, lse, pz1 = forward(z0, z1, y, ignore_index, softcap)
        ctx.save_for_backward(z0, z1, y, lse, pz1)
        ctx.ignore_index, ctx.softcap, ctx.shape = ignore_index, softcap, z0.shape
        dtype = torch.promote_types(z0.dtype, torch.float32)
        return loss0.to(dtype), loss1.to(dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, c0, c1):
        z0, z1, y, lse, pz1 = ctx.saved_tensors
        c0 = torch.zeros_like(lse) if c0 is None else c0
        c1 = torch.zeros_like(lse) if c1 is None else c1
        dz0, dz1 = backward(z0, z1, y, lse, pz1, c0, c1, ctx.ignore_index, ctx.softcap)
        return dz0, dz1, None, None, None


class _SeriesCrossEntropy(torch.autograd.Function):
    """Per-row degree-1 cross entropy on a PolyTensor of logits (wrapper autograd)."""

    @staticmethod
    def forward(ctx, logits, y, ignore_index, softcap=None):
        from ._tensor import PolyTensor

        z0, z1 = logits.coeffs
        y = y.contiguous()
        loss0, loss1, lse, pz1 = forward(z0, z1, y, ignore_index, softcap)
        ctx.save_for_backward(z0, z1, y, lse, pz1)
        ctx.ignore_index, ctx.softcap = ignore_index, softcap
        ctx.strided = z0.stride(-1) != 1 or z1.stride() != z0.stride() or not z0.is_contiguous()
        ctx.shape = z0.shape
        dtype = torch.promote_types(z0.dtype, torch.float32)
        return PolyTensor((loss0.to(dtype), loss1.to(dtype)))

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        from ._tensor import PolyTensor

        z0, z1, y, lse, pz1 = ctx.saved_tensors
        g0, g1 = gradient.coeffs if isinstance(gradient, PolyTensor) else (gradient, torch.zeros_like(gradient))
        g0, g1 = (g.expand(lse.shape) for g in (g0, g1))
        # Gradient series coefficient j = VJP w.r.t. input coefficient 1 - j, with cotangents (G1, G0).
        dz0, dz1 = backward(z0, z1, y, lse, pz1, g1, g0, ctx.ignore_index, ctx.softcap)
        return PolyTensor((dz1, dz0)), None, None, None


def eligible(input, target, weight, reduction, label_smoothing, softcap=None):
    from ._plain import plain_range_enabled
    from ._tensor import PolyTensor

    if not _enabled.get() or not plain_range_enabled() or not available():
        return False
    if not isinstance(input, PolyTensor) or input.degree != 1 or input.ndim != 2:
        return False
    if isinstance(target, PolyTensor) or not isinstance(target, torch.Tensor) or target.dtype != torch.int64:
        return False
    if target.shape != input.shape[:1] or weight is not None or label_smoothing != 0.0:
        return False
    if reduction not in ("none", "mean", "sum"):
        return False
    z0, z1 = input.coeffs
    return (z0.is_cuda and z0.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and z1.dtype == z0.dtype and target.is_cuda)


def fused(input, target, ignore_index=-100, reduction="mean", softcap=None):
    """Fused degree-1 cross entropy for an eligible call (see ``eligible``)."""
    from ._tensor import PolyTensor, _coefficient_autograd_enabled

    if _coefficient_autograd_enabled.get():
        loss0, loss1 = CoefficientCrossEntropy.apply(*input.coeffs, target, ignore_index, softcap)
        losses = PolyTensor((loss0, loss1), requires_grad=False)
    elif torch.is_grad_enabled() and input.requires_grad:
        losses = _SeriesCrossEntropy.apply(input, target, ignore_index, softcap)
    else:
        z0, z1 = input.coeffs
        loss0, loss1, _, _ = forward(z0, z1, target.contiguous(), ignore_index, softcap)
        dtype = torch.promote_types(z0.dtype, torch.float32)
        losses = PolyTensor((loss0.to(dtype), loss1.to(dtype)))
    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    count = (target != ignore_index).sum().to(losses.value.dtype)
    return total / count


def cross_entropy(logits, target, softcap=None, ignore_index=-100, reduction="mean"):
    """Cross entropy of (N, C) logits, optionally capped as ``softcap * tanh(logits / softcap)`` first.

    Works for ordinary tensors and PolyTensors. Degree-1 real CUDA series use one
    fused Triton pass (forward and backward) that applies the cap on the fly, so
    neither capped nor float32 copies of the logits are formed; other inputs use
    the composite operations (the cap in at least float32, then
    ``torch.nn.functional.cross_entropy``).
    """
    if eligible(logits, target, None, reduction, 0.0, softcap):
        return fused(logits, target, ignore_index=ignore_index, reduction=reduction, softcap=softcap)
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return torch.nn.functional.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)
