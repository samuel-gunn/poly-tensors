"""Fused first-order activation arithmetic below wrapper Autograd."""
import torch


def maybe_activation(X, kind, G=None):
    from ._tensor import _coefficient_autograd_enabled
    from ._fused_attention import available
    if _coefficient_autograd_enabled.get() or len(X) != 2:
        return None
    values = list(X) + ([] if G is None else list(G))
    if (any(c.dtype != X[0].dtype or c.shape != X[0].shape or not c.is_cuda for c in values)
            or X[0].dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or not available()):
        return None
    import triton
    from ._activation_kernels import _activation
    x0, x1 = (c.contiguous() for c in X)
    g0, g1 = (x0, x1) if G is None else (c.contiguous() for c in G)
    out = torch.empty_like(x0), torch.empty_like(x1)
    if x0.numel():
        _activation[(triton.cdiv(x0.numel(), 256),)](x0, x1, g0, g1, *out, x0.numel(),
            KIND=kind, BACKWARD=G is not None, DOUBLE=x0.dtype == torch.float64,
            BLOCK=256, enable_fp_fusion=False)
    return out
