"""
PolyTensor: a minimal polynomial-ring Tensor subclass.

Key design points (MVP):
- Wrapper subclass: the outer PolyTensor has *no storage* and presents the same
  shape/dtype/device as the primal tensor; real data lives in self._coeffs.
  This matches the general wrapper subclass pattern described by PyTorch devs. 

- self._coeffs is a plain torch.Tensor with shape (deg+1, *shape) and
  requires_grad=False. Gradients are propagated by installing custom autograd.Function
  nodes for add/sub/mul/neg that implement *polynomial ring* reverse-mode.

- __torch_dispatch__ intercepts a tiny whitelist of ATen ops, creating those
  autograd.Function nodes for out-of-place ops and doing direct coefficient mutation
  for in-place ops.

Limitations:
- Only elementwise +, -, unary -, * and in-place variants; detach supported.
- Any other op raises NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch

import poly_arith

aten = torch.ops.aten


@dataclass(frozen=True)
class _Spec:
    deg: int
    device: torch.device
    dtype: torch.dtype


def _is_plain_tensor_requires_grad(x: Any) -> bool:
    return isinstance(x, torch.Tensor) and (not isinstance(x, PolyTensor)) and bool(x.requires_grad)


class PolyTensor(torch.Tensor):
    # Disable __torch_function__ to ensure __torch_dispatch__ is the primary override.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[attr-defined]

    # ----- Construction -----

    @staticmethod
    def _wrap(coeffs: torch.Tensor, *, requires_grad: bool) -> "PolyTensor":
        poly_arith._check_coeffs(coeffs)
        # Ensure there is no coefficient-level autograd graph.
        coeffs = coeffs.detach()
        coeffs.requires_grad_(False)

        shape = tuple(coeffs.shape[1:])
        out = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            PolyTensor,
            shape,
            dtype=coeffs.dtype,
            device=coeffs.device,
            requires_grad=requires_grad,
        )
        out._coeffs = coeffs
        return out

    @classmethod
    def from_coeffs(cls, coeffs: torch.Tensor, *, requires_grad: bool = False) -> "PolyTensor":
        """
        coeffs must have shape (deg+1, *shape).
        """
        return cls._wrap(coeffs, requires_grad=requires_grad)

    @classmethod
    def from_primal(
        cls,
        primal: torch.Tensor,
        degree: int,
        tangent: Optional[torch.Tensor] = None,
        *,
        requires_grad: bool = True,
    ) -> "PolyTensor":
        """
        Construct polynomial with:
          coeff[0] = primal
          coeff[1] = tangent (optional; default zeros)
          coeff[k>1] = 0

        If you want higher-order initial coefficients, build a coeff tensor and call from_coeffs.
        """
        if not isinstance(primal, torch.Tensor):
            raise TypeError("primal must be a torch.Tensor")
        if degree < 0:
            raise ValueError("degree must be >= 0")

        coeffs = torch.zeros((degree + 1,) + tuple(primal.shape), device=primal.device, dtype=primal.dtype)
        coeffs[0] = primal
        if degree >= 1:
            if tangent is None:
                coeffs[1] = torch.zeros_like(primal)
            else:
                if tangent.shape != primal.shape:
                    raise ValueError(f"tangent must match primal.shape; got {tangent.shape} vs {primal.shape}")
                if tangent.device != primal.device:
                    raise RuntimeError("tangent must be on same device as primal for this MVP")
                coeffs[1] = tangent.to(dtype=primal.dtype)

        return cls._wrap(coeffs, requires_grad=requires_grad)

    # ----- Introspection -----

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

    # ----- Convenience methods that avoid dispatching unsupported ops -----

    def to(self, *args: Any, **kwargs: Any) -> "PolyTensor":
        """
        Move/cast coefficients and rewrap.
        """
        coeffs = self._coeffs.to(*args, **kwargs)
        return PolyTensor.from_coeffs(coeffs, requires_grad=self.requires_grad)

    def detach(self) -> "PolyTensor":
        return PolyTensor.from_coeffs(self._coeffs.detach(), requires_grad=False)

    def requires_grad_(self, requires_grad: bool = True) -> "PolyTensor":
        torch.Tensor.requires_grad_(self, requires_grad)  # type: ignore[misc]
        return self

    def backward(
        self,
        gradient: Optional["PolyTensor | torch.Tensor"] = None,
        retain_graph: Optional[bool] = None,
        create_graph: bool = False,
        inputs: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> None:
        """
        Backward for PolyTensor.

        If gradient is None, seeds with a constant-1 polynomial matching the degree/device/dtype.
        """
        if self.numel() != 1 and gradient is None:
            raise RuntimeError("PolyTensor.backward() with gradient=None requires a scalar (numel==1) output.")

        if gradient is None:
            g0 = torch.ones_like(self.primal())
            gradient = PolyTensor.from_primal(g0, self.degree, tangent=None, requires_grad=False)
        else:
            if isinstance(gradient, PolyTensor):
                if gradient.degree != self.degree:
                    raise ValueError(f"Gradient degree {gradient.degree} does not match output degree {self.degree}")
            elif isinstance(gradient, torch.Tensor):
                gradient = PolyTensor.from_primal(gradient, self.degree, tangent=None, requires_grad=False)
            else:
                raise TypeError("gradient must be a PolyTensor, torch.Tensor, or None")

        torch.autograd.backward(self, gradient, retain_graph=retain_graph, create_graph=create_graph, inputs=inputs)

    def __repr__(self) -> str:
        return (
            f"PolyTensor(degree={self.degree}, shape={tuple(self.shape)}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    # ----- Dispatch helpers -----

    @staticmethod
    def _spec_from_args(args: Tuple[Any, ...]) -> _Spec:
        for a in args:
            if isinstance(a, PolyTensor):
                return _Spec(deg=a.degree, device=a.device, dtype=a.dtype)
        raise RuntimeError("PolyTensor dispatch called without any PolyTensor arguments (unexpected).")

    @staticmethod
    def _promote_to_poly(x: Any, *, spec: _Spec) -> "PolyTensor":
        """
        Promote scalar/torch.Tensor to a constant PolyTensor of matching degree/device/dtype.
        """
        if isinstance(x, PolyTensor):
            if x.degree != spec.deg:
                raise ValueError(f"Degree mismatch: {x.degree} vs {spec.deg}")
            if x.device != spec.device:
                raise RuntimeError(f"Device mismatch: {x.device} vs {spec.device}")
            # Allow dtype promotion at op time; keep as-is here.
            return x

        if _is_plain_tensor_requires_grad(x):
            raise RuntimeError(
                "PolyTensor MVP does not support gradients w.r.t. plain torch.Tensors. "
                "Wrap that tensor as PolyTensor too, or set requires_grad=False."
            )

        if isinstance(x, torch.Tensor):
            if x.device != spec.device:
                raise RuntimeError(f"Device mismatch: {x.device} vs {spec.device}")
            coeffs = poly_arith.make_constant_coeffs(x, deg=spec.deg, device=spec.device, dtype=spec.dtype)
            return PolyTensor.from_coeffs(coeffs, requires_grad=False)

        # Python scalar
        coeffs = poly_arith.make_constant_coeffs(x, deg=spec.deg, device=spec.device, dtype=spec.dtype)
        return PolyTensor.from_coeffs(coeffs, requires_grad=False)

    @staticmethod
    def _check_inplace_allowed(a: "PolyTensor") -> None:
        # Requested MVP rule: reject in-place ops on leaf requiring grad (unless user disables grad mode).
        if a.is_leaf and a.requires_grad and torch.is_grad_enabled():
            raise RuntimeError(
                "In-place op on a leaf PolyTensor that requires grad is not supported in this MVP.\n"
                "Do parameter updates under `with torch.no_grad():`, or operate on `a.detach()`."
            )

    # ----- __torch_dispatch__ -----

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        spec = cls._spec_from_args(args)

        # ---- detach ----
        if func is aten.detach.default:
            (a,) = args
            if not isinstance(a, PolyTensor):
                return func(*args, **kwargs)
            return a.detach()

        if hasattr(aten, "detach_") and func is aten.detach_.default:
            (a,) = args
            if not isinstance(a, PolyTensor):
                return func(*args, **kwargs)
            cls._check_inplace_allowed(a)
            a._coeffs = a._coeffs.detach()
            a.requires_grad_(False)
            return a

        # ---- add / sub / mul / neg (out-of-place go through custom autograd.Function) ----
        if func in (aten.add.Tensor, aten.add.Scalar):
            a, b = args[0], args[1]
            alpha = kwargs.get("alpha", 1)
            a_p = cls._promote_to_poly(a, spec=spec)
            b_p = cls._promote_to_poly(b, spec=spec)
            return _PolyAdd.apply(a_p, b_p, alpha)

        if func in (aten.sub.Tensor, aten.sub.Scalar):
            a, b = args[0], args[1]
            alpha = kwargs.get("alpha", 1)
            a_p = cls._promote_to_poly(a, spec=spec)
            b_p = cls._promote_to_poly(b, spec=spec)
            return _PolySub.apply(a_p, b_p, alpha)

        if func in (aten.mul.Tensor, aten.mul.Scalar):
            a, b = args[0], args[1]
            a_p = cls._promote_to_poly(a, spec=spec)
            b_p = cls._promote_to_poly(b, spec=spec)
            return _PolyMul.apply(a_p, b_p)

        if func is aten.neg.default:
            (a,) = args
            a_p = cls._promote_to_poly(a, spec=spec)
            return _PolyNeg.apply(a_p)

        # ---- in-place variants (mutate coefficient storage directly) ----
        if hasattr(aten, "add_") and func in (aten.add_.Tensor, aten.add_.Scalar):
            a, b = args[0], args[1]
            alpha = kwargs.get("alpha", 1)
            if not isinstance(a, PolyTensor):
                raise TypeError("add_ first argument must be a PolyTensor")
            cls._check_inplace_allowed(a)
            b_p = cls._promote_to_poly(b, spec=spec)

            b_coeffs = b_p._coeffs.to(dtype=a.dtype)  # in-place ops follow lhs dtype
            if alpha != 1:
                b_coeffs = poly_arith.scale(b_coeffs, alpha)
            poly_arith.add_(a._coeffs, b_coeffs)
            return a

        if hasattr(aten, "sub_") and func in (aten.sub_.Tensor, aten.sub_.Scalar):
            a, b = args[0], args[1]
            alpha = kwargs.get("alpha", 1)
            if not isinstance(a, PolyTensor):
                raise TypeError("sub_ first argument must be a PolyTensor")
            cls._check_inplace_allowed(a)
            b_p = cls._promote_to_poly(b, spec=spec)

            b_coeffs = b_p._coeffs.to(dtype=a.dtype)
            if alpha != 1:
                b_coeffs = poly_arith.scale(b_coeffs, alpha)
            poly_arith.sub_(a._coeffs, b_coeffs)
            return a

        if hasattr(aten, "mul_") and func in (aten.mul_.Tensor, aten.mul_.Scalar):
            a, b = args[0], args[1]
            if not isinstance(a, PolyTensor):
                raise TypeError("mul_ first argument must be a PolyTensor")
            cls._check_inplace_allowed(a)
            b_p = cls._promote_to_poly(b, spec=spec)

            b_coeffs = b_p._coeffs.to(dtype=a.dtype)
            poly_arith.mul_(a._coeffs, b_coeffs)
            return a

        # Tensor.neg_() exists; route if it shows up as an ATen op.
        if hasattr(aten, "neg_") and func is aten.neg_.default:
            (a,) = args
            if not isinstance(a, PolyTensor):
                raise TypeError("neg_ argument must be a PolyTensor")
            cls._check_inplace_allowed(a)
            poly_arith.neg_(a._coeffs)
            return a

        raise NotImplementedError(
            f"PolyTensor MVP supports only add/sub/mul/neg (and in-place variants) + detach. Got: {func}"
        )


# ------------------ custom autograd nodes (ring reverse-mode) ------------------

class _PolyAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: PolyTensor, b: PolyTensor, alpha: int | float) -> PolyTensor:
        ctx.alpha = alpha
        ctx.a_shape = tuple(a.shape)
        ctx.b_shape = tuple(b.shape)

        # Promote dtypes like PyTorch would for out-of-place.
        out_dtype = torch.result_type(a.coeffs, b.coeffs)
        a_c = a.coeffs.to(dtype=out_dtype)
        b_c = b.coeffs.to(dtype=out_dtype)
        if alpha != 1:
            b_c = poly_arith.scale(b_c, alpha)

        out_c = poly_arith.add(a_c, b_c)
        requires_grad = (a.requires_grad or b.requires_grad) and torch.is_grad_enabled()
        return PolyTensor.from_coeffs(out_c, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx, grad_out: PolyTensor):
        alpha = ctx.alpha
        a_shape = ctx.a_shape
        b_shape = ctx.b_shape

        g = grad_out.coeffs

        ga = poly_arith.sum_to_shape_coeffs(g, a_shape)
        gb = poly_arith.sum_to_shape_coeffs(g, b_shape)
        if alpha != 1:
            gb = poly_arith.scale(gb, alpha)

        # Return PolyTensor grads; alpha has no gradient.
        return PolyTensor.from_coeffs(ga, requires_grad=False), PolyTensor.from_coeffs(gb, requires_grad=False), None


class _PolySub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: PolyTensor, b: PolyTensor, alpha: int | float) -> PolyTensor:
        ctx.alpha = alpha
        ctx.a_shape = tuple(a.shape)
        ctx.b_shape = tuple(b.shape)

        out_dtype = torch.result_type(a.coeffs, b.coeffs)
        a_c = a.coeffs.to(dtype=out_dtype)
        b_c = b.coeffs.to(dtype=out_dtype)
        if alpha != 1:
            b_c = poly_arith.scale(b_c, alpha)

        out_c = poly_arith.sub(a_c, b_c)
        requires_grad = (a.requires_grad or b.requires_grad) and torch.is_grad_enabled()
        return PolyTensor.from_coeffs(out_c, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx, grad_out: PolyTensor):
        alpha = ctx.alpha
        a_shape = ctx.a_shape
        b_shape = ctx.b_shape

        g = grad_out.coeffs
        ga = poly_arith.sum_to_shape_coeffs(g, a_shape)
        gb = poly_arith.sum_to_shape_coeffs(g, b_shape)

        # d(a - alpha*b)/db = -alpha
        gb = poly_arith.scale(gb, -alpha)

        return PolyTensor.from_coeffs(ga, requires_grad=False), PolyTensor.from_coeffs(gb, requires_grad=False), None


class _PolyNeg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: PolyTensor) -> PolyTensor:
        ctx.a_shape = tuple(a.shape)
        out_c = poly_arith.neg(a.coeffs)
        requires_grad = a.requires_grad and torch.is_grad_enabled()
        return PolyTensor.from_coeffs(out_c, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx, grad_out: PolyTensor):
        a_shape = ctx.a_shape
        g = poly_arith.neg(grad_out.coeffs)
        ga = poly_arith.sum_to_shape_coeffs(g, a_shape)
        return PolyTensor.from_coeffs(ga, requires_grad=False)


class _PolyMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: PolyTensor, b: PolyTensor) -> PolyTensor:
        ctx.a_shape = tuple(a.shape)
        ctx.b_shape = tuple(b.shape)

        # For ring backward we need the operands' coeffs.
        # Save as plain tensors.
        ctx.save_for_backward(a.coeffs, b.coeffs)

        out_dtype = torch.result_type(a.coeffs, b.coeffs)
        a_c = a.coeffs.to(dtype=out_dtype)
        b_c = b.coeffs.to(dtype=out_dtype)

        out_c = poly_arith.mul(a_c, b_c)
        requires_grad = (a.requires_grad or b.requires_grad) and torch.is_grad_enabled()
        return PolyTensor.from_coeffs(out_c, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx, grad_out: PolyTensor):
        a_shape = ctx.a_shape
        b_shape = ctx.b_shape
        a_c, b_c = ctx.saved_tensors
        g_c = grad_out.coeffs

        # Ring reverse-mode:
        # d(a*b)/da = b, so grad_a = g * b  (ring multiplication)
        # d(a*b)/db = a, so grad_b = g * a
        ga = poly_arith.mul(g_c, b_c)
        gb = poly_arith.mul(g_c, a_c)

        # Reduce to operand shapes due to broadcasting.
        ga = poly_arith.sum_to_shape_coeffs(ga, a_shape)
        gb = poly_arith.sum_to_shape_coeffs(gb, b_shape)

        return PolyTensor.from_coeffs(ga, requires_grad=False), PolyTensor.from_coeffs(gb, requires_grad=False)
