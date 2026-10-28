"""Decompositions of composite operations into supported PolyTensor operations.

These run in ``__torch_function__`` (above Autograd), so PyTorch records the
decomposed operations and their existing forward and backward rules apply.
Each decomposition follows PyTorch's documented definition of the operation.
"""

import math

import torch
import torch.nn.functional as F


def _rms_norm(input, normalized_shape, weight=None, eps=None):
    dims = tuple(range(-len(normalized_shape), 0))
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    # x / sqrt(mean(x^2) + eps), the definition used by torch.nn.functional.rms_norm.
    out = input * torch.rsqrt((input * input).mean(dim=dims, keepdim=True) + eps)
    if weight is not None:
        out = out * weight
    return out


_RMS_NORM = frozenset((F.rms_norm, torch.rms_norm))


def _attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    # The reference definition in torch.nn.functional.scaled_dot_product_attention.
    if dropout_p != 0.0 or enable_gqa:
        raise NotImplementedError("plain-range PolyTensor attention supports dropout_p=0 without GQA")
    L, S = query.size(-2), key.size(-2)
    scale = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = (query * scale) @ key.transpose(-2, -1)
    if is_causal:
        keep = torch.ones(L, S, dtype=torch.bool, device=scores.device).tril(diagonal=0)
        scores = scores.masked_fill(~keep, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    return torch.softmax(scores, dim=-1) @ value


_ATTENTION = frozenset((F.scaled_dot_product_attention,))
_CROSS_ENTROPY = frozenset((F.cross_entropy,))
_CROSS_ENTROPY_ARGS = ("input", "target", "weight", "size_average", "ignore_index", "reduce", "reduction",
                       "label_smoothing")


def maybe_decompose(function, args, kwargs):
    """Return the decomposed result, or NotImplemented to continue normally."""
    from ._tensor import PolyTensor

    if function in _RMS_NORM:
        bound = dict(zip(("input", "normalized_shape", "weight", "eps"), args))
        bound.update(kwargs)
        if not any(isinstance(bound.get(k), PolyTensor) for k in ("input", "weight")):
            return NotImplemented
        from . import _fused_norm
        shape = tuple(bound["normalized_shape"])
        operation = _fused_norm.rms_norm if _fused_norm.eligible(bound["input"], shape, bound.get("eps")) else _rms_norm
        return operation(bound["input"], shape, bound.get("weight"), bound.get("eps"))
    if function in _CROSS_ENTROPY:
        # Fused degree-1 rule (otherwise PyTorch decomposes into log_softmax and nll_loss as usual).
        bound = dict(zip(_CROSS_ENTROPY_ARGS, args))
        bound.update(kwargs)
        if bound.get("size_average") is not None or bound.get("reduce") is not None:
            return NotImplemented
        from . import _fused_cross_entropy

        if not _fused_cross_entropy.eligible(bound.get("input"), bound.get("target"), bound.get("weight"),
                                             bound.get("reduction", "mean"), bound.get("label_smoothing", 0.0)):
            return NotImplemented
        loss = _fused_cross_entropy.fused(bound["input"], bound["target"], ignore_index=bound.get("ignore_index", -100),
                                          reduction=bound.get("reduction", "mean"))
        # torch.nn.functional.cross_entropy returns the input's dtype (the kernel accumulates in float32)
        return loss if loss.dtype == bound["input"].dtype else loss.to(bound["input"].dtype)
    if function in _ATTENTION:
        from ._plain import plain_range_enabled

        if not plain_range_enabled():
            return NotImplemented
        names = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
        bound = dict(zip(names, args))
        bound.update(kwargs)
        if not any(isinstance(v, PolyTensor) for v in bound.values()):
            return NotImplemented
        from . import _fused_attention

        if _fused_attention.eligible(bound["query"], bound["key"], bound["value"], bound.get("attn_mask"),
                                     bound.get("dropout_p", 0.0), bound.get("is_causal", False),
                                     bound.get("enable_gqa", False)):
            return _fused_attention.attention(bound["query"], bound["key"], bound["value"],
                                              attn_mask=bound.get("attn_mask"), is_causal=bound.get("is_causal", False),
                                              scale=bound.get("scale"))
        return _attention(**bound)
    return NotImplemented
