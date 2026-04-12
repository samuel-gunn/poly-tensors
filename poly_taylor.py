"""
Derivative helpers for univariate elementwise Taylor expansion.

All functions here return the list:
  c[n] = f^{(n)}(x0) / n!   for n=0..deg

These are intended to be passed into poly_arith.taylor_unary().
"""

from __future__ import annotations

import math
from typing import List

import torch


def _inv_factorials(deg: int) -> List[float]:
    out = [1.0]
    acc = 1.0
    for n in range(1, deg + 1):
        acc *= float(n)
        out.append(1.0 / acc)
    return out


def _binom(alpha: float, n: int) -> float:
    # generalized binomial coefficient (alpha choose n)
    if n == 0:
        return 1.0
    num = 1.0
    den = 1.0
    for k in range(n):
        num *= float(alpha - k)
        den *= float(k + 1)
    return num / den


def taylor_coeffs_exp(x0: torch.Tensor, deg: int):
    base = torch.exp(x0)
    inv_fact = _inv_factorials(deg)
    return [base * inv_fact[n] for n in range(deg + 1)]


def taylor_coeffs_log(x0: torch.Tensor, deg: int):
    # f(x)=log x, f^{(n)}(x)/n! = (-1)^{n-1}/(n*x^n) for n>=1
    out = [torch.log(x0)]
    for n in range(1, deg + 1):
        coef = ((-1.0) ** (n - 1)) / float(n)
        out.append(coef * torch.pow(x0, -n))
    return out


def taylor_coeffs_pow(x0: torch.Tensor, alpha: float, deg: int):
    # f(x)=x^alpha -> f^{(n)}(x)/n! = (alpha choose n) x^{alpha-n}
    out = []
    for n in range(deg + 1):
        out.append(_binom(alpha, n) * torch.pow(x0, alpha - n))
    return out


def _hermite_physicists(x: torch.Tensor, n_max: int):
    # H0=1, H1=2x, H_{n+1}=2x H_n - 2n H_{n-1}
    H = []
    H0 = torch.ones_like(x)
    H.append(H0)
    if n_max == 0:
        return H
    H1 = 2 * x
    H.append(H1)
    for n in range(1, n_max):
        Hn1 = 2 * x * H[n] - 2 * n * H[n - 1]
        H.append(Hn1)
    return H


def taylor_coeffs_erf(x0: torch.Tensor, deg: int):
    # d^n erf(x)/dx^n = (2/sqrt(pi)) (-1)^{n-1} H_{n-1}(x) exp(-x^2) for n>=1
    erf_fn = torch.special.erf if hasattr(torch, "special") and hasattr(torch.special, "erf") else torch.erf
    out = [erf_fn(x0)]
    if deg == 0:
        return out

    inv_fact = _inv_factorials(deg)
    gauss = torch.exp(-(x0 * x0))
    H = _hermite_physicists(x0, deg - 1)
    const = 2.0 / math.sqrt(math.pi)

    for n in range(1, deg + 1):
        sign = -1.0 if ((n - 1) % 2 == 1) else 1.0
        out.append(const * sign * H[n - 1] * gauss * inv_fact[n])
    return out
