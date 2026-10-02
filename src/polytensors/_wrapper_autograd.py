"""Autograd rules that need an operation's input before native rounding.

``__torch_dispatch__`` runs below PyTorch's Autograd key, so replacing an
operator's forward coefficients there cannot replace its native backward
formula. These small functions enter above Autograd through
``__torch_function__``. Their backwards use ordinary PolyTensor operations,
which also gives subsequent reverse derivatives an ordinary autograd graph.
"""

import torch


class _Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        # Function.forward runs without gradient recording, so this follows
        # the usual dispatch rule instead of recursively applying _Exp.
        return torch.exp(value)

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        if value.is_complex():
            value = value.conj_physical()
        # The scaled arithmetic payload survives the exp/mul boundary, so a
        # tiny exponential can be combined with a large incoming gradient.
        return torch.exp(value) * gradient


class _Logsumexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, dim, keepdim):
        ctx.save_for_backward(value)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return torch.logsumexp(value, dim=dim, keepdim=keepdim)

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        if not ctx.keepdim and value.ndim:
            gradient = gradient.unsqueeze(ctx.dim)
        # Subtracting the rounded logsumexp from large equal inputs can lose
        # log(n), giving n copies of 1 instead of a normalized distribution.
        return torch.softmax(value, dim=ctx.dim) * gradient, None, None


_EXP_FUNCTIONS = frozenset((torch.exp, torch.Tensor.exp, torch.ops.aten.exp.default,
                           torch.ops.aten.exp))
_LOGSUMEXP_FUNCTIONS = frozenset((torch.logsumexp, torch.Tensor.logsumexp,
                                 torch.special.logsumexp,
                                 torch.ops.aten.logsumexp.default,
                                 torch.ops.aten.logsumexp))


def maybe_apply(function, args, kwargs):
    """Return a stable wrapper result, or NotImplemented to dispatch normally."""
    from ._tensor import PolyTensor

    if function not in _EXP_FUNCTIONS and function not in _LOGSUMEXP_FUNCTIONS:
        return NotImplemented
    allowed = {"input", "self", "out"}
    if function in _LOGSUMEXP_FUNCTIONS:
        allowed.update(("dim", "keepdim"))
    if kwargs.keys() - allowed or len(args) > (1 if function in _EXP_FUNCTIONS else 3):
        return NotImplemented
    value = args[0] if args else kwargs.get("input", kwargs.get("self"))
    if not isinstance(value, PolyTensor) or not value.requires_grad:
        return NotImplemented
    if kwargs.get("out") is not None:
        # Preserve PyTorch's normal prohibition on out= with autograd.
        return NotImplemented
    if function in _EXP_FUNCTIONS:
        return _Exp.apply(value)
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    if isinstance(dim, (tuple, list)):
        if len(dim) != 1:
            return NotImplemented
        dim = dim[0]
    if dim is None:
        return NotImplemented
    return _Logsumexp.apply(value, dim, keepdim)
