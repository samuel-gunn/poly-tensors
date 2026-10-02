"""Arithmetic and storage rules that retain extended coefficient range."""

import torch

from ._range import attach_scaled, get_scaled
from ._scaled import ScaledTensor as S


NOT_HANDLED = object()


def dispatch_range(func, args, kwargs, *, polys, degree, wrap, correct, preserve):
    aten = torch.ops.aten
    count = degree + 1
    like = polys[0].value

    def is_poly(value):
        return hasattr(value, "coeffs")

    def lift(value):
        if is_poly(value):
            return get_scaled(value) or tuple(S.from_tensor(c) for c in value.coeffs)
        if not isinstance(value, torch.Tensor):
            dtype = torch.result_type(like, value)
            value = torch.as_tensor(value, dtype=dtype, device=like.device)
        if not (value.is_floating_point() or value.is_complex()):
            value = value.to(dtype=like.dtype)
        return (S.from_tensor(value), *(S.from_tensor(torch.zeros_like(value)) for _ in range(degree)))

    def dtypes(values):
        result = []
        for k in range(count):
            representative = None
            for value in values:
                coefficient = value.coeffs[k] if is_poly(value) else value
                if coefficient is None:
                    continue
                if representative is None:
                    if isinstance(coefficient, torch.Tensor):
                        representative = coefficient
                        continue
                    representative = like
                dtype = torch.result_type(representative, coefficient)
                scalar = representative.ndim == 0 and (not isinstance(coefficient, torch.Tensor) or coefficient.ndim == 0)
                representative = torch.empty(() if scalar else (1,), dtype=dtype)
            result.append(like.dtype if representative is None else representative.dtype)
        return result

    def output(coefficients, values, *, dtype=None):
        coefficients = tuple(coefficients)
        targets = dtypes(values) if dtype is None else [dtype] * count
        if dtype is not None and not dtype.is_complex:
            complex_dtype = {torch.float16: torch.complex32, torch.float32: torch.complex64,
                             torch.float64: torch.complex128}.get(dtype)
            if any(c.mantissa.is_complex() for c in coefficients):
                if complex_dtype is None:
                    raise TypeError(f"{dtype} cannot represent complex PolyTensor directions")
                targets = [complex_dtype if c.mantissa.is_complex() else dtype for c in coefficients]
        return attach_scaled(wrap(c.to_tensor(target) for c, target in zip(coefficients, targets)), coefficients)

    def convolution(a, b, *, matrix=False):
        from ._scaled_matmul import scaled_polynomial_product

        return scaled_polynomial_product(a, b, matrix=matrix)

    def quotient(a, b):
        result = []
        for k in range(count):
            remainder = a[k]
            for i in range(1, k + 1):
                remainder = remainder - b[i] * result[k - i]
            result.append(remainder / b[0])
        return result

    def write(target, coefficients):
        coefficients = tuple(coefficients)
        if any(c.mantissa.is_complex() and not dest.is_complex()
               for c, dest in zip(coefficients, target.coeffs)):
            raise RuntimeError("cannot copy complex PolyTensor coefficients into real coefficient storage")
        for dest, value in zip(target.coeffs, coefficients):
            dest.copy_(value.to_tensor(dest.dtype))
        return attach_scaled(target, coefficients)

    add = (aten.add.Tensor, aten.add.Scalar, aten.add_.Tensor, aten.add_.Scalar)
    sub = (aten.sub.Tensor, aten.sub.Scalar, aten.sub_.Tensor, aten.sub_.Scalar)
    mul = (aten.mul.Tensor, aten.mul.Scalar, aten.mul_.Tensor, aten.mul_.Scalar)
    div = (aten.div.Tensor, aten.div.Scalar, aten.div_.Tensor, aten.div_.Scalar)
    if func in (*add, *sub, *mul, *div):
        a, b = args[:2]
        A, B = lift(a), lift(b)
        alpha = kwargs.get("alpha", 1)
        if func in add:
            result = tuple(x + alpha * y for x, y in zip(A, B))
        elif func in sub:
            result = tuple(x - alpha * y for x, y in zip(A, B))
        elif func in mul:
            result = convolution(A, B)
        else:
            if kwargs.get("rounding_mode") is not None:
                raise NotImplementedError("PolyTensor does not implement rounded division")
            result = quotient(A, B)
        return write(a, result) if func._schema.is_mutable else output(result, (a, b))

    if func is aten.rsub.Scalar:
        x, other = args[:2]
        alpha = kwargs.get("alpha", 1)
        return output((b - alpha * a for a, b in zip(lift(x), lift(other))), (x, other))
    if func is aten.neg.default:
        return output((-c for c in lift(args[0])), args[:1])
    if func is aten.reciprocal.default:
        return output(quotient(lift(1), lift(args[0])), args[:1])

    if func in (aten.mm.default, aten.matmul.default, aten.bmm.default, aten.mv.default, aten.dot.default):
        result = convolution(lift(args[0]), lift(args[1]), matrix=True)
        return correct(output(result, args[:2]))
    if func is aten.addmm.default:
        bias, a, b = args[:3]
        beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
        result = tuple(alpha * c for c in convolution(lift(a), lift(b), matrix=True))
        if beta != 0:
            result = tuple(c + beta * bias_c for c, bias_c in zip(result, lift(bias)))
        return output(result, args[:3])
    if func is aten.linear.default:
        x, weight = args[:2]
        result = convolution(lift(x), tuple(c.transpose(-2, -1) for c in lift(weight)), matrix=True)
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        if bias is not None:
            result = tuple(c + b for c, b in zip(result, lift(bias)))
        return output(result, (x, weight, bias))

    if func in (aten.sum.default, aten.sum.dim_IntList, aten.mean.default, aten.mean.dim):
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        # An empty dimension list reduces all dimensions in PyTorch.
        dim = tuple(dim) if isinstance(dim, list) else dim
        if dim == ():
            dim = None
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        result = tuple(c.sum(dim, keepdim) for c in lift(args[0]))
        if func in (aten.mean.default, aten.mean.dim):
            shape = args[0].shape
            axes = range(len(shape)) if dim is None else ((dim,) if isinstance(dim, int) else dim)
            divisor = 1
            for axis in axes:
                divisor *= shape[axis] if shape else 1
            result = tuple(c / divisor for c in result)
        return output(result, args[:1], dtype=kwargs.get("dtype"))

    if func in (aten.sum_to_size.default, aten._grad_sum_to_size.default):
        shape = args[1]
        if shape is None:
            return args[0]
        def reduce(c):
            current = c.mantissa.shape
            leading = len(current) - len(shape)
            axes = tuple(range(leading)) + tuple(i + leading for i, size in enumerate(shape)
                                                if size == 1 and current[i + leading] != 1)
            if axes:
                c = c.sum(axes, keepdim=True)
            return c.map_tensor(lambda t: t.reshape(shape))
        return output((reduce(c) for c in lift(args[0])), args[:1])

    if func in (aten._conj.default, aten._conj_physical.default, aten.conj_physical.default):
        result = wrap(func(c) for c in args[0].coeffs)
        return correct(attach_scaled(result, (c.conj() for c in lift(args[0]))))

    if func in (aten.real.default, aten.imag.default):
        result = wrap(func(c) for c in args[0].coeffs)
        part = "real" if func is aten.real.default else "imag"
        return correct(attach_scaled(result, (getattr(c, part) for c in lift(args[0]))))

    if func in (aten.where.self, aten.where.ScalarOther, aten.where.ScalarSelf, aten.where.Scalar):
        condition, a, b = args[:3]
        if is_poly(condition):
            condition = condition.value
        return output((x.masked_fill(~condition, y) for x, y in zip(lift(a), lift(b))), (a, b))

    if func in (aten.detach.default, aten.alias.default, aten.clone.default):
        source = args[0]
        result = wrap(func(c, **kwargs) for c in source.coeffs)
        scaled = get_scaled(source)
        if scaled is not None:
            result = attach_scaled(result, (c.map_tensor(lambda t: func(t, **kwargs)) for c in scaled))
        return correct(preserve(source, result))

    if func is aten.detach_.default:
        source = args[0]
        scaled = get_scaled(source)
        source._set_coeffs(c.detach() for c in source.coeffs)
        if scaled is not None:
            attach_scaled(source, (c.map_tensor(lambda t: t.detach()) for c in scaled))
        return source

    shape_ops = (
        aten.t.default, aten.transpose.int, aten.permute.default, aten.view.default,
        aten.reshape.default, aten._unsafe_view.default, aten.flatten.using_ints,
        aten.slice.Tensor, aten.select.int, aten.unsqueeze.default, aten.squeeze.dim,
        aten.squeeze.default, aten.expand.default, aten.index_select.default,
        aten.gather.default, aten.index.Tensor,
    )
    if func in shape_ops:
        source = args[0]
        def transform(t):
            return func(t, *args[1:], **kwargs)
        # Public views keep their native aliasing. Hidden scales have their
        # own layouts and are transformed by logical shape, not storage offset.
        result = wrap(transform(c) for c in source.coeffs)
        scaled = get_scaled(source)
        if scaled is not None:
            result = attach_scaled(result, (c.map_tensor(transform) for c in scaled))
        return correct(result)

    if func in (aten.cat.default, aten.stack.default):
        sources = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        operation = torch.cat if func is aten.cat.default else torch.stack
        def combine(items):
            if any(item.mantissa.is_complex() for item in items):
                return S._from_parts(combine([item.real for item in items]),
                                     combine([item.imag for item in items]))
            return S(operation([item.mantissa for item in items], dim=dim),
                     operation([item.exponent.expand(item.mantissa.shape) for item in items], dim=dim))
        groups = [lift(source) for source in sources]
        return output((combine([group[k] for group in groups]) for k in range(count)), sources)

    if func is aten.copy_.default:
        return write(args[0], lift(args[1]))
    if func is aten.zero_.default:
        return write(args[0], lift(0))

    if func in (aten.masked_fill.Scalar, aten.masked_fill.Tensor,
                aten.masked_fill_.Scalar, aten.masked_fill_.Tensor):
        source, mask, value = args[:3]
        if is_poly(mask):
            mask = mask.value
        result = tuple(c.masked_fill(mask, fill) for c, fill in zip(lift(source), lift(value)))
        return write(source, result) if func._schema.is_mutable else output(result, (source, value))

    return NOT_HANDLED
