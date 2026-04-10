"""
Polynomial arithmetic on coefficient tensors.

Representation:
  coeffs: torch.Tensor with shape (deg+1, *primal_shape)
  P(z) = sum_{k=0..deg} coeffs[k] * z^k

This file intentionally knows nothing about PolyTensor or dispatch; it only
implements arithmetic on coefficient tensors (plain torch.Tensors).
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch


def _check_coeffs(coeffs: torch.Tensor) -> None:
    if not isinstance(coeffs, torch.Tensor):
        raise TypeError("coeffs must be a torch.Tensor")
    if coeffs.ndim < 1:
        raise ValueError("coeffs must have shape (deg+1, *shape); got a scalar tensor")
    if coeffs.shape[0] < 1:
        raise ValueError("coeffs leading dimension must be >= 1")


def degree(coeffs: torch.Tensor) -> int:
    _check_coeffs(coeffs)
    return int(coeffs.shape[0] - 1)


def make_constant_coeffs(
    value: torch.Tensor | int | float,
    *,
    deg: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create coefficient tensor for a constant polynomial P(z)=value.
    """
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise RuntimeError(f"Device mismatch: value on {value.device}, expected {device}")
        v = value.to(dtype=dtype)
        shape = tuple(v.shape)
    else:
        v = torch.tensor(value, device=device, dtype=dtype)
        shape = tuple(v.shape)  # typically ()

    out = torch.zeros((deg + 1,) + shape, device=device, dtype=dtype)
    out[0] = v
    return out


def _broadcast_shape(a0: torch.Tensor, b0: torch.Tensor) -> Tuple[int, ...]:
    a0b, _ = torch.broadcast_tensors(a0, b0)
    return tuple(a0b.shape)


def scale(coeffs: torch.Tensor, alpha: int | float) -> torch.Tensor:
    _check_coeffs(coeffs)
    if alpha == 1:
        return coeffs
    return coeffs * alpha


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _check_coeffs(a)
    _check_coeffs(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Degree mismatch: {degree(a)} vs {degree(b)}")

    deg = a.shape[0] - 1
    out_shape = _broadcast_shape(a[0], b[0])
    out_dtype = torch.result_type(a, b)
    out = torch.zeros((deg + 1,) + out_shape, device=a.device, dtype=out_dtype)
    for k in range(deg + 1):
        out[k] = a[k] + b[k]
    return out


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _check_coeffs(a)
    _check_coeffs(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Degree mismatch: {degree(a)} vs {degree(b)}")

    deg = a.shape[0] - 1
    out_shape = _broadcast_shape(a[0], b[0])
    out_dtype = torch.result_type(a, b)
    out = torch.zeros((deg + 1,) + out_shape, device=a.device, dtype=out_dtype)
    for k in range(deg + 1):
        out[k] = a[k] - b[k]
    return out


def neg(a: torch.Tensor) -> torch.Tensor:
    _check_coeffs(a)
    return -a


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Truncated Cauchy product (O(deg^2)).

    out[n] = sum_{k=0..n} a[k] * b[n-k]
    """
    _check_coeffs(a)
    _check_coeffs(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Degree mismatch: {degree(a)} vs {degree(b)}")

    deg = a.shape[0] - 1
    out_shape = _broadcast_shape(a[0], b[0])
    out_dtype = torch.result_type(a, b)
    out = torch.zeros((deg + 1,) + out_shape, device=a.device, dtype=out_dtype)

    for n in range(deg + 1):
        acc = None
        for k in range(n + 1):
            term = a[k] * b[n - k]
            acc = term if acc is None else (acc + term)
        out[n] = acc if acc is not None else torch.zeros(out_shape, device=a.device, dtype=out_dtype)

    return out


def add_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _check_coeffs(a)
    _check_coeffs(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Degree mismatch: {degree(a)} vs {degree(b)}")
    for k in range(a.shape[0]):
        a[k].add_(b[k])
    return a


def sub_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _check_coeffs(a)
    _check_coeffs(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Degree mismatch: {degree(a)} vs {degree(b)}")
    for k in range(a.shape[0]):
        a[k].sub_(b[k])
    return a


def mul_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    In-place ring multiplication requires a temporary.
    """
    prod = mul(a, b)
    a.copy_(prod)
    return a


def neg_(a: torch.Tensor) -> torch.Tensor:
    _check_coeffs(a)
    a.neg_()
    return a


def sum_to_shape(x: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    """
    Reduce x by summing over broadcasted dims so that result has shape target_shape.

    This mimics the common "sum_to_size" logic used in broadcasting backwards.
    """
    tgt = tuple(int(s) for s in target_shape)
    if tuple(x.shape) == tgt:
        return x

    x_shape = tuple(x.shape)
    if len(tgt) > len(x_shape):
        raise ValueError(f"target_shape {tgt} has more dims than x.shape {x_shape}")

    # Align target shape to x by left-padding with ones.
    pad = len(x_shape) - len(tgt)
    aligned = (1,) * pad + tgt

    # Identify dims to sum over.
    dims = []
    for i, (xs, ts) in enumerate(zip(x_shape, aligned)):
        if ts == 1 and xs != 1:
            dims.append(i)
        elif i < pad:
            dims.append(i)

    if dims:
        x = x.sum(dim=tuple(dims), keepdim=True)

    # Now x has same ndim as before, with 1s in reduced dims.
    x = x.reshape(aligned)
    if pad:
        x = x.reshape(tgt)
    return x


def sum_to_shape_coeffs(coeffs: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    """
    Apply sum_to_shape to each coefficient slice.
    """
    _check_coeffs(coeffs)
    deg = coeffs.shape[0] - 1
    tgt = tuple(int(s) for s in target_shape)
    out = []
    for k in range(deg + 1):
        out.append(sum_to_shape(coeffs[k], tgt))
    return torch.stack(out, dim=0)
