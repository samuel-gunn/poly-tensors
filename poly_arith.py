"""
Polynomial arithmetic on coefficient tensors.

Representation:
  coeffs: torch.Tensor with shape (deg+1, *primal_shape)
  P(z) = sum_{k=0..deg} coeffs[k] * z^k

This file knows nothing about __torch_dispatch__ or PolyTensor; it only implements
operations on coefficient tensors (plain torch.Tensor objects).

Adds:
  - cauchy_bilinear: generic bilinear Cauchy product for ops like matmul/conv2d
  - taylor_unary: compose an elementwise analytic function using its Taylor series
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

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
    Truncated Cauchy product (O(deg^2)) for elementwise multiplication.

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


def cauchy_bilinear(
    a: torch.Tensor,
    b: torch.Tensor,
    bilinear: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Truncated Cauchy product for a generic bilinear op ⊗:

        out[n] = sum_{k=0..n} bilinear(a[k], b[n-k])
    """
    _check_coeffs(a)
    _check_coeffs(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Degree mismatch: {degree(a)} vs {degree(b)}")

    deg = a.shape[0] - 1
    out_slices = []
    for n in range(deg + 1):
        acc = None
        for k in range(n + 1):
            term = bilinear(a[k], b[n - k])
            acc = term if acc is None else (acc + term)
        if acc is None:
            raise RuntimeError("cauchy_bilinear: internal error (acc=None)")
        out_slices.append(acc)
    return torch.stack(out_slices, dim=0)


def taylor_unary(coeffs: torch.Tensor, taylor_over_fact: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Compose an elementwise analytic function using its Taylor series around the
    constant term.

    Input:
      coeffs: (deg+1, *shape), x(z)
      taylor_over_fact: list length deg+1 with entries:
         taylor_over_fact[n] = f^{(n)}(x0) / n!   (elementwise tensors)
    """
    _check_coeffs(coeffs)
    deg = degree(coeffs)
    if len(taylor_over_fact) != deg + 1:
        raise ValueError(f"Expected {deg+1} Taylor coefficients, got {len(taylor_over_fact)}")

    # δ = x - x0
    delta = coeffs.clone()
    delta[0] = torch.zeros_like(delta[0])

    # pow = δ^0 = 1
    pow_coeffs = torch.zeros_like(coeffs)
    pow_coeffs[0] = torch.ones_like(coeffs[0])

    out = torch.zeros_like(coeffs)
    for n in range(deg + 1):
        out = out + pow_coeffs * taylor_over_fact[n]
        if n < deg:
            pow_coeffs = mul(pow_coeffs, delta)
    return out
