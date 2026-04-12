"""
PolyTensor: truncated univariate Taylor polynomials living inside a PyTorch Tensor subclass.

Representation:
  coeffs: torch.Tensor of shape (deg+1, *shape)
  P(z) = sum_{k=0..deg} coeffs[k] * z^k

This implementation is forward-mode only: it never uses torch.autograd.

- Elementwise analytic ops f are composed by Taylor expansion around the constant term.
- Bilinear ops (elementwise mul, matmul, conv2d) use truncated Cauchy products.
- MaxPool2d is deliberately unsupported (non-analytic). Use LSEPool2d as a smooth replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

import poly_arith
import poly_taylor

aten = torch.ops.aten


@dataclass(frozen=True)
class _Spec:
    deg: int
    device: torch.device
    dtype: torch.dtype


def _find_first_poly(x: Any) -> Optional["PolyTensor"]:
    if isinstance(x, PolyTensor):
        return x
    if isinstance(x, (list, tuple)):
        for xi in x:
            p = _find_first_poly(xi)
            if p is not None:
                return p
    if isinstance(x, dict):
        for xi in x.values():
            p = _find_first_poly(xi)
            if p is not None:
                return p
    return None


def _to_int_tuple(x: Any) -> Tuple[int, ...]:
    if x is None:
        return tuple()
    if isinstance(x, int):
        return (int(x),)
    if isinstance(x, (list, tuple)):
        return tuple(int(i) for i in x)
    raise TypeError(f"Expected int or list/tuple of ints; got {type(x)}")


def _canonical_dims(dims: Sequence[int], ndim: int) -> Tuple[int, ...]:
    out = []
    for d in dims:
        d = int(d)
        if d < 0:
            d += ndim
        if d < 0 or d >= ndim:
            raise ValueError(f"dim={d} out of range for ndim={ndim}")
        out.append(d)
    return tuple(sorted(set(out)))


def _reduction_from_int(i: int) -> str:
    # Matches torch.nn.functional._Reduction.get_enum(...)
    if int(i) == 0:
        return "none"
    if int(i) == 1:
        return "mean"
    if int(i) == 2:
        return "sum"
    raise ValueError(f"Unknown reduction enum: {i}")


class PolyTensor(torch.Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[attr-defined]

    # ---------------- construction ----------------

    @staticmethod
    def _wrap(coeffs: torch.Tensor) -> "PolyTensor":
        poly_arith._check_coeffs(coeffs)
        coeffs = coeffs.detach()
        coeffs.requires_grad_(False)

        shape = tuple(coeffs.shape[1:])
        out = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            PolyTensor,
            shape,
            dtype=coeffs.dtype,
            device=coeffs.device,
            requires_grad=False,
        )
        out._coeffs = coeffs
        return out

    @classmethod
    def from_coeffs(cls, coeffs: torch.Tensor) -> "PolyTensor":
        return cls._wrap(coeffs)

    @classmethod
    def from_primal(
        cls,
        primal: torch.Tensor,
        degree: int,
        tangent: Optional[torch.Tensor] = None,
    ) -> "PolyTensor":
        if not isinstance(primal, torch.Tensor):
            raise TypeError("primal must be a torch.Tensor")
        if degree < 0:
            raise ValueError("degree must be >= 0")

        coeffs = torch.zeros((degree + 1,) + tuple(primal.shape), device=primal.device, dtype=primal.dtype)
        coeffs[0] = primal.detach()
        if degree >= 1:
            if tangent is None:
                coeffs[1] = torch.zeros_like(primal)
            else:
                if tangent.shape != primal.shape:
                    raise ValueError(f"tangent shape {tangent.shape} must match primal shape {primal.shape}")
                coeffs[1] = tangent.detach().to(device=primal.device, dtype=primal.dtype)
        return cls._wrap(coeffs)

    # ---------------- introspection ----------------

    @property
    def degree(self) -> int:
        return int(self._coeffs.shape[0] - 1)

    def coeff(self, i: int) -> torch.Tensor:
        if i < 0 or i > self.degree:
            raise IndexError(f"Coefficient index {i} out of range for degree {self.degree}")
        return self._coeffs[i]

    def primal(self) -> torch.Tensor:
        return self._coeffs[0]

    @property
    def coeffs(self) -> torch.Tensor:
        return self._coeffs

    # ---------------- convenience ----------------

    def to(self, *args: Any, **kwargs: Any) -> "PolyTensor":
        return PolyTensor.from_coeffs(self._coeffs.to(*args, **kwargs))

    def detach(self) -> "PolyTensor":
        return PolyTensor.from_coeffs(self._coeffs.detach())

    def clone(self) -> "PolyTensor":
        return PolyTensor.from_coeffs(self._coeffs.clone())

    def __repr__(self) -> str:
        return f"PolyTensor(degree={self.degree}, shape={tuple(self.shape)}, device={self.device}, dtype={self.dtype})"

    # ---------------- internal helpers ----------------

    @staticmethod
    def _spec_from_call(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> _Spec:
        # IMPORTANT: do NOT write `_find_first_poly(args) or _find_first_poly(kwargs)`
        # because `or` forces truth-value testing (bool(...)) on the first result.
        p = _find_first_poly(args)
        if p is None:
            p = _find_first_poly(kwargs)
        if p is None:
            raise RuntimeError("PolyTensor dispatch called without PolyTensor inputs (unexpected).")
        return _Spec(deg=p.degree, device=p.device, dtype=p.dtype)

    @staticmethod
    def _promote(x: Any, *, spec: _Spec) -> "PolyTensor":
        if isinstance(x, PolyTensor):
            if x.degree != spec.deg:
                raise ValueError(f"Degree mismatch: {x.degree} vs {spec.deg}")
            if x.device != spec.device:
                raise RuntimeError(f"Device mismatch: {x.device} vs {spec.device}")
            return x

        if isinstance(x, torch.Tensor):
            if x.device != spec.device:
                raise RuntimeError(f"Device mismatch: {x.device} vs {spec.device}")
            coeffs = poly_arith.make_constant_coeffs(x.detach(), deg=spec.deg, device=spec.device, dtype=spec.dtype)
            return PolyTensor.from_coeffs(coeffs)

        coeffs = poly_arith.make_constant_coeffs(x, deg=spec.deg, device=spec.device, dtype=spec.dtype)
        return PolyTensor.from_coeffs(coeffs)

    @staticmethod
    def _wrap_linear_op(func, a: "PolyTensor", args_tail: Tuple[Any, ...], kwargs: Dict[str, Any]) -> "PolyTensor":
        deg = a.degree
        out_slices = []
        for k in range(deg + 1):
            out_slices.append(func(a._coeffs[k], *args_tail, **kwargs))
        return PolyTensor.from_coeffs(torch.stack(out_slices, dim=0))

    @staticmethod
    def _unary_taylor(a: "PolyTensor", taylor_over_fact) -> "PolyTensor":
        out_coeffs = poly_arith.taylor_unary(a._coeffs, taylor_over_fact)
        return PolyTensor.from_coeffs(out_coeffs)

    @staticmethod
    def _elementwise_erf(a: "PolyTensor") -> "PolyTensor":
        return PolyTensor._unary_taylor(a, poly_taylor.taylor_coeffs_erf(a.primal(), a.degree))

    @staticmethod
    def _elementwise_exp(a: "PolyTensor") -> "PolyTensor":
        return PolyTensor._unary_taylor(a, poly_taylor.taylor_coeffs_exp(a.primal(), a.degree))

    @staticmethod
    def _elementwise_log(a: "PolyTensor") -> "PolyTensor":
        return PolyTensor._unary_taylor(a, poly_taylor.taylor_coeffs_log(a.primal(), a.degree))

    @staticmethod
    def _elementwise_pow(a: "PolyTensor", alpha: float) -> "PolyTensor":
        return PolyTensor._unary_taylor(a, poly_taylor.taylor_coeffs_pow(a.primal(), alpha, a.degree))

    @staticmethod
    def _logsumexp_coeffs(x: "PolyTensor", dim: Union[int, Sequence[int]], keepdim: bool) -> torch.Tensor:
        dims = _canonical_dims(_to_int_tuple(dim), x.dim())

        x0 = x.primal()
        shift = x0.amax(dim=dims, keepdim=True)
        shift_poly = PolyTensor._promote(shift, spec=_Spec(x.degree, x.device, x.dtype))

        z = PolyTensor.from_coeffs(poly_arith.sub(x.coeffs, shift_poly.coeffs))
        e = PolyTensor._elementwise_exp(z)

        s_coeffs = [e.coeffs[k].sum(dim=dims, keepdim=True) for k in range(x.degree + 1)]
        s_poly = PolyTensor.from_coeffs(torch.stack(s_coeffs, dim=0))

        l = PolyTensor._elementwise_log(s_poly)
        out = PolyTensor.from_coeffs(poly_arith.add(l.coeffs, shift_poly.coeffs))

        if keepdim:
            return out.coeffs

        squeezed = []
        for k in range(x.degree + 1):
            t = out.coeffs[k]
            for d in sorted(dims, reverse=True):
                t = t.squeeze(d)
            squeezed.append(t)
        return torch.stack(squeezed, dim=0)

    # ---------------- dispatch ----------------

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # noqa: C901
        if kwargs is None:
            kwargs = {}

        spec = cls._spec_from_call(args, kwargs)
        fstr = str(func)

        # ---- view-like ops (linear in first arg) ----
        if (
            fstr.startswith("aten.view.")
            or fstr.startswith("aten.reshape.")
            or fstr.startswith("aten.permute.")
            or fstr.startswith("aten.transpose.")
            or fstr.startswith("aten.t.")
            or fstr.startswith("aten.squeeze.")
            or fstr.startswith("aten.unsqueeze.")
            or fstr.startswith("aten.contiguous.")
            or fstr.startswith("aten.clone.")
            or fstr.startswith("aten.detach.")
            or fstr.startswith("aten._to_copy.")
        ):
            a = cls._promote(args[0], spec=spec)
            return cls._wrap_linear_op(func, a, args[1:], kwargs)

        # ---- arithmetic ----
        if fstr.startswith("aten.neg."):
            a = cls._promote(args[0], spec=spec)
            return PolyTensor.from_coeffs(poly_arith.neg(a.coeffs))

        if fstr.startswith("aten.add."):
            a = cls._promote(args[0], spec=spec)
            b = cls._promote(args[1], spec=spec)
            alpha = kwargs.get("alpha", 1)
            b_coeffs = b.coeffs
            if alpha != 1:
                b_coeffs = poly_arith.scale(b_coeffs, alpha)
            return PolyTensor.from_coeffs(poly_arith.add(a.coeffs, b_coeffs))

        if fstr.startswith("aten.sub."):
            a = cls._promote(args[0], spec=spec)
            b = cls._promote(args[1], spec=spec)
            alpha = kwargs.get("alpha", 1)
            b_coeffs = b.coeffs
            if alpha != 1:
                b_coeffs = poly_arith.scale(b_coeffs, alpha)
            return PolyTensor.from_coeffs(poly_arith.sub(a.coeffs, b_coeffs))

        if fstr.startswith("aten.mul."):
            a = cls._promote(args[0], spec=spec)
            b = cls._promote(args[1], spec=spec)
            return PolyTensor.from_coeffs(poly_arith.mul(a.coeffs, b.coeffs))

        if fstr.startswith("aten.div.") or fstr.startswith("aten.true_divide."):
            a = cls._promote(args[0], spec=spec)
            b = cls._promote(args[1], spec=spec)
            inv_b = cls._elementwise_pow(b, -1.0)
            return PolyTensor.from_coeffs(poly_arith.mul(a.coeffs, inv_b.coeffs))

        # ---- elementwise analytic funcs ----
        if fstr.startswith("aten.exp."):
            a = cls._promote(args[0], spec=spec)
            return cls._elementwise_exp(a)

        if fstr.startswith("aten.log."):
            a = cls._promote(args[0], spec=spec)
            return cls._elementwise_log(a)

        if fstr.startswith("aten.sqrt."):
            a = cls._promote(args[0], spec=spec)
            return cls._elementwise_pow(a, 0.5)

        if fstr.startswith("aten.rsqrt."):
            a = cls._promote(args[0], spec=spec)
            return cls._elementwise_pow(a, -0.5)

        if fstr.startswith("aten.erf.") or fstr.startswith("aten.special_erf."):
            a = cls._promote(args[0], spec=spec)
            return cls._elementwise_erf(a)

        # ---- GELU ----
        if fstr.startswith("aten.gelu."):
            a = cls._promote(args[0], spec=spec)
            approximate = kwargs.get("approximate", "none")
            if len(args) >= 2 and isinstance(args[1], str):
                approximate = args[1]
            if approximate not in ("none", None):
                raise NotImplementedError("Only GELU(approximate='none') is supported (erf-based).")

            c = float(1.0 / (2.0 ** 0.5))
            a_scaled = PolyTensor.from_coeffs(poly_arith.scale(a.coeffs, c))
            erf_term = cls._elementwise_erf(a_scaled)

            one_coeffs = poly_arith.make_constant_coeffs(
                torch.ones_like(a.primal()), deg=a.degree, device=a.device, dtype=a.dtype
            )
            one = PolyTensor.from_coeffs(one_coeffs)
            inner = PolyTensor.from_coeffs(poly_arith.add(one.coeffs, erf_term.coeffs))

            out = PolyTensor.from_coeffs(poly_arith.mul(a.coeffs, inner.coeffs))
            out = PolyTensor.from_coeffs(poly_arith.scale(out.coeffs, 0.5))
            return out

        # ---- reductions ----
        if fstr.startswith("aten.sum.") or fstr.startswith("aten.mean."):
            a = cls._promote(args[0], spec=spec)
            return cls._wrap_linear_op(func, a, args[1:], kwargs)

        if fstr.startswith("aten.logsumexp."):
            x = cls._promote(args[0], spec=spec)
            dim = args[1]
            keepdim = bool(args[2])
            return PolyTensor.from_coeffs(cls._logsumexp_coeffs(x, dim=dim, keepdim=keepdim))

        # ---- softmax / log_softmax ----
        if fstr.startswith("aten.softmax.") or fstr.startswith("aten._softmax."):
            x = cls._promote(args[0], spec=spec)
            dim = int(args[1] if len(args) >= 2 else kwargs["dim"])
            lse_coeffs = cls._logsumexp_coeffs(x, dim=(dim,), keepdim=True)
            shifted = PolyTensor.from_coeffs(poly_arith.sub(x.coeffs, lse_coeffs))
            return cls._elementwise_exp(shifted)

        if fstr.startswith("aten.log_softmax.") or fstr.startswith("aten._log_softmax."):
            x = cls._promote(args[0], spec=spec)
            dim = int(args[1] if len(args) >= 2 else kwargs["dim"])
            lse_coeffs = cls._logsumexp_coeffs(x, dim=(dim,), keepdim=True)
            return PolyTensor.from_coeffs(poly_arith.sub(x.coeffs, lse_coeffs))

        # ---- nll_loss_forward (used by F.cross_entropy path) ----
        if fstr.startswith("aten.nll_loss_forward."):
            x = cls._promote(args[0], spec=spec)          # expected log-probs
            target = args[1]
            weight = args[2]
            reduction = _reduction_from_int(int(args[3]))
            ignore_index = int(args[4])

            if weight is not None:
                raise NotImplementedError("nll_loss_forward: weight is not implemented.")
            if not isinstance(target, torch.Tensor):
                raise TypeError("nll_loss_forward: target must be a torch.Tensor.")
            if target.dtype not in (torch.int64, torch.long):
                raise TypeError("nll_loss_forward: target must be int64 class indices.")

            class_dim = 0 if x.dim() == 1 else 1
            idx = target.unsqueeze(class_dim)

            loss = PolyTensor.from_coeffs(torch.stack(
                [-torch.gather(x.coeffs[k], dim=class_dim, index=idx).squeeze(class_dim) for k in range(x.degree + 1)],
                dim=0,
            ))

            if ignore_index >= 0:
                mask = (target != ignore_index)
                loss = PolyTensor.from_coeffs(torch.stack(
                    [torch.where(mask, loss.coeffs[k], torch.zeros_like(loss.coeffs[k])) for k in range(x.degree + 1)],
                    dim=0,
                ))
                total_weight = mask.sum().to(device=spec.device, dtype=spec.dtype)
            else:
                total_weight = torch.tensor(target.numel(), device=spec.device, dtype=spec.dtype)

            if reduction == "none":
                return loss, total_weight
            if reduction == "sum":
                out = PolyTensor.from_coeffs(torch.stack([loss.coeffs[k].sum() for k in range(x.degree + 1)], dim=0))
                return out, total_weight
            if reduction == "mean":
                denom = total_weight.clamp(min=1)
                out = PolyTensor.from_coeffs(torch.stack([loss.coeffs[k].sum() / denom for k in range(x.degree + 1)], dim=0))
                return out, total_weight

            raise RuntimeError("unreachable")

        # ---- masked_fill (used in cross_entropy label_smoothing path) ----
        if fstr.startswith("aten.masked_fill_.Scalar") or fstr.startswith("aten.masked_fill.Scalar"):
            x = cls._promote(args[0], spec=spec)
            mask = args[1]
            value = args[2]

            out_coeffs = x.coeffs.clone()
            fill0 = torch.as_tensor(value, device=out_coeffs.device, dtype=out_coeffs.dtype)
            out_coeffs[0] = torch.where(mask, fill0, out_coeffs[0])
            for k in range(1, x.degree + 1):
                out_coeffs[k] = torch.where(mask, torch.zeros_like(out_coeffs[k]), out_coeffs[k])

            if fstr.startswith("aten.masked_fill_.Scalar"):
                x.coeffs.copy_(out_coeffs)
                return x
            return PolyTensor.from_coeffs(out_coeffs)

        # ---- linear algebra: matmul / mm / addmm / linear ----
        if fstr.startswith("aten.matmul.") or fstr.startswith("aten.mm."):
            a = cls._promote(args[0], spec=spec)
            b = cls._promote(args[1], spec=spec)

            def bilinear(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return u.matmul(v)

            return PolyTensor.from_coeffs(poly_arith.cauchy_bilinear(a.coeffs, b.coeffs, bilinear))

        if fstr.startswith("aten.addmm."):
            inp = cls._promote(args[0], spec=spec)
            mat1 = cls._promote(args[1], spec=spec)
            mat2 = cls._promote(args[2], spec=spec)
            beta = kwargs.get("beta", 1)
            alpha = kwargs.get("alpha", 1)

            def bilinear(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return u.matmul(v)

            prod = poly_arith.scale(poly_arith.cauchy_bilinear(mat1.coeffs, mat2.coeffs, bilinear), alpha)
            inp_scaled = poly_arith.scale(inp.coeffs, beta)
            return PolyTensor.from_coeffs(poly_arith.add(inp_scaled, prod))

        if fstr.startswith("aten.linear."):
            x = cls._promote(args[0], spec=spec)
            w = cls._promote(args[1], spec=spec)
            bias = args[2] if len(args) >= 3 else kwargs.get("bias", None)
            b_poly = None if bias is None else cls._promote(bias, spec=spec)

            def bilinear(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return F.linear(u, v, bias=None)

            out = PolyTensor.from_coeffs(poly_arith.cauchy_bilinear(x.coeffs, w.coeffs, bilinear))
            if b_poly is not None:
                out = PolyTensor.from_coeffs(poly_arith.add(out.coeffs, b_poly.coeffs))
            return out

        # ---- convolution ----
        if fstr.startswith("aten.convolution."):
            x = cls._promote(args[0], spec=spec)
            w = cls._promote(args[1], spec=spec)
            bias = args[2]
            stride = args[3]
            padding = args[4]
            dilation = args[5]
            transposed = bool(args[6])
            _output_padding = args[7]
            groups = int(args[8])

            if transposed:
                raise NotImplementedError("Transposed convolution is not implemented (not needed for airbench).")

            b_poly = None if bias is None else cls._promote(bias, spec=spec)

            def bilinear(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return F.conv2d(u, v, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

            out = PolyTensor.from_coeffs(poly_arith.cauchy_bilinear(x.coeffs, w.coeffs, bilinear))
            if b_poly is not None:
                out = PolyTensor.from_coeffs(poly_arith.add(out.coeffs, b_poly.coeffs))
            return out

        if fstr.startswith("aten.conv2d."):
            x = cls._promote(args[0], spec=spec)
            w = cls._promote(args[1], spec=spec)
            bias = args[2]
            stride = args[3]
            padding = args[4]
            dilation = args[5]
            groups = int(args[6])

            b_poly = None if bias is None else cls._promote(bias, spec=spec)

            def bilinear(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return F.conv2d(u, v, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

            out = PolyTensor.from_coeffs(poly_arith.cauchy_bilinear(x.coeffs, w.coeffs, bilinear))
            if b_poly is not None:
                out = PolyTensor.from_coeffs(poly_arith.add(out.coeffs, b_poly.coeffs))
            return out

        # ---- batch norm ----
        if fstr.startswith("aten.batch_norm."):
            x = cls._promote(args[0], spec=spec)
            weight = args[1]
            bias = args[2]
            running_mean = args[3]
            running_var = args[4]
            training = bool(args[5])
            momentum = args[6]
            eps = float(args[7])
            # args[8] is cudnn_enabled; ignored

            w_poly = None if weight is None else cls._promote(weight, spec=spec)
            b_poly = None if bias is None else cls._promote(bias, spec=spec)

            if x.dim() != 4:
                raise NotImplementedError("Only 4D NCHW BatchNorm2d is supported for airbench.")

            reduce_dims = (0, 2, 3)

            mean_coeffs = [x.coeffs[k].mean(dim=reduce_dims, keepdim=True) for k in range(x.degree + 1)]
            mean = PolyTensor.from_coeffs(torch.stack(mean_coeffs, dim=0))

            centered = PolyTensor.from_coeffs(poly_arith.sub(x.coeffs, mean.coeffs))
            sq = PolyTensor.from_coeffs(poly_arith.mul(centered.coeffs, centered.coeffs))

            var_coeffs = [sq.coeffs[k].mean(dim=reduce_dims, keepdim=True) for k in range(x.degree + 1)]
            var = PolyTensor.from_coeffs(torch.stack(var_coeffs, dim=0))

            eps_poly = cls._promote(torch.tensor(eps, device=spec.device, dtype=spec.dtype), spec=spec)
            var_eps = PolyTensor.from_coeffs(poly_arith.add(var.coeffs, eps_poly.coeffs))

            inv_std = cls._elementwise_pow(var_eps, -0.5)
            y = PolyTensor.from_coeffs(poly_arith.mul(centered.coeffs, inv_std.coeffs))

            if w_poly is not None:
                wB = PolyTensor.from_coeffs(torch.stack([w_poly.coeffs[k].view(1, -1, 1, 1) for k in range(x.degree + 1)], dim=0))
                y = PolyTensor.from_coeffs(poly_arith.mul(y.coeffs, wB.coeffs))
            if b_poly is not None:
                bB = PolyTensor.from_coeffs(torch.stack([b_poly.coeffs[k].view(1, -1, 1, 1) for k in range(x.degree + 1)], dim=0))
                y = PolyTensor.from_coeffs(poly_arith.add(y.coeffs, bB.coeffs))

            if training and isinstance(running_mean, torch.Tensor) and isinstance(running_var, torch.Tensor):
                mom = float(momentum) if momentum is not None else 0.0
                with torch.no_grad():
                    batch_mean = mean.primal().view(-1)
                    batch_var_unbiased = x.primal().var(dim=reduce_dims, unbiased=True).view(-1)
                    running_mean.mul_(1.0 - mom).add_(batch_mean, alpha=mom)
                    running_var.mul_(1.0 - mom).add_(batch_var_unbiased, alpha=mom)

            return y

        # ---- cross entropy ----
        if "cross_entropy_loss" in fstr and fstr.startswith("aten."):
            x = cls._promote(args[0], spec=spec)
            target = args[1]
            weight = args[2]
            reduction = _reduction_from_int(int(args[3]))
            ignore_index = int(args[4])
            label_smoothing = float(args[5])

            if weight is not None:
                raise NotImplementedError("cross_entropy_loss: weight is not implemented (airbench uses weight=None).")
            if not isinstance(target, torch.Tensor):
                raise TypeError("cross_entropy_loss: target must be a torch.Tensor.")
            if target.dtype not in (torch.int64, torch.long):
                raise TypeError("cross_entropy_loss: target must be int64 class indices for this implementation.")

            class_dim = 0 if x.dim() == 1 else 1

            lse = cls._logsumexp_coeffs(x, dim=(class_dim,), keepdim=True)
            log_probs = PolyTensor.from_coeffs(poly_arith.sub(x.coeffs, lse))

            idx = target.unsqueeze(class_dim)
            loss_coeffs = []
            for k in range(x.degree + 1):
                picked = torch.gather(log_probs.coeffs[k], dim=class_dim, index=idx).squeeze(class_dim)
                loss_coeffs.append(-picked)
            loss = PolyTensor.from_coeffs(torch.stack(loss_coeffs, dim=0))

            if label_smoothing != 0.0:
                smooth = PolyTensor.from_coeffs(torch.stack([-log_probs.coeffs[k].mean(dim=class_dim) for k in range(x.degree + 1)], dim=0))
                loss = PolyTensor.from_coeffs(
                    poly_arith.add(
                        poly_arith.scale(loss.coeffs, 1.0 - label_smoothing),
                        poly_arith.scale(smooth.coeffs, label_smoothing),
                    )
                )

            if ignore_index >= 0:
                mask = (target != ignore_index)
                loss = PolyTensor.from_coeffs(torch.stack(
                    [torch.where(mask, loss.coeffs[k], torch.zeros_like(loss.coeffs[k])) for k in range(x.degree + 1)],
                    dim=0,
                ))

            if reduction == "none":
                return loss
            if reduction == "sum":
                return PolyTensor.from_coeffs(torch.stack([loss.coeffs[k].sum() for k in range(x.degree + 1)], dim=0))
            if reduction == "mean":
                if ignore_index >= 0:
                    denom = (target != ignore_index).sum().clamp(min=1).to(dtype=spec.dtype)
                else:
                    denom = torch.tensor(target.numel(), device=spec.device, dtype=spec.dtype)
                return PolyTensor.from_coeffs(torch.stack([loss.coeffs[k].sum() / denom for k in range(x.degree + 1)], dim=0))

            raise RuntimeError("unreachable")

        raise NotImplementedError(f"PolyTensor does not support op: {func}")


class LSEPool2d(torch.nn.Module):
    """
    Log-sum-exp pooling as a smooth replacement for MaxPool2d.

    For each channel independently:
        y = log( sum_{window} exp(x) )

    Implemented via depthwise conv on exp(x) with a ones kernel.
    """
    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("LSEPool2d expects NCHW input")
        C = x.shape[1]
        ones = torch.ones((C, 1, self.kernel_size[0], self.kernel_size[1]), device=x.device, dtype=x.dtype)
        y = F.conv2d(torch.exp(x), ones, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=C)
        return torch.log(y)
