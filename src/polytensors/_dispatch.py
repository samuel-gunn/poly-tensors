"""PyTorch operator rules for PolyTensor."""

import math

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from ._series import SeriesOps

aten = torch.ops.aten


def dispatch(func, types, args=(), kwargs=None, *, coefficient_autograd=False):
    from ._tensor import PolyTensor

    kwargs = {} if kwargs is None else kwargs

    polys = []
    poly_wrappers = []

    def is_poly_wrapper(x):
        return type(x) is PolyTensor

    def is_initialized_poly(x):
        return is_poly_wrapper(x) and hasattr(x, "coeffs")

    def collect(x):
        if is_initialized_poly(x):
            polys.append(x)
            poly_wrappers.append(x)
        elif is_poly_wrapper(x):
            poly_wrappers.append(x)
        elif isinstance(x, (tuple, list)):
            for y in x:
                collect(y)
        elif isinstance(x, dict):
            for y in x.values():
                collect(y)

    collect(args)
    collect(kwargs)

    if len(polys) != len(poly_wrappers):
        raise RuntimeError(
            "PolyTensor wrapper reached dispatch without coefficient storage. "
            "Keep every PolyTensor wrapper alive through forward and backward; "
            "silently replacing this value with zero would corrupt derivatives."
        )

    if not polys:
        return func(*args, **kwargs)

    D = polys[0].degree
    if any(poly.degree != D for poly in polys[1:]):
        raise ValueError("PolyTensor operands must have the same degree")
    series = SeriesOps(D)
    like = polys[0].value

    def lift(x):
        if is_initialized_poly(x):
            return x.coeffs
        if isinstance(x, torch.Tensor):
            return (x,) + tuple(torch.zeros_like(x) for _ in range(D))
        return (x,) + tuple(torch.zeros_like(like) for _ in range(D))

    def clone_coeffs(cs):
        return tuple(c.clone() if isinstance(c, torch.Tensor) else c for c in cs)

    def wrap(cs, *, clone=False):
        if clone:
            cs = clone_coeffs(cs)
        else:
            cs = tuple(cs)
        # In coefficient-autograd mode, differentiation belongs exclusively
        # to the ordinary coefficient tensors.  Making the storage-less
        # wrapper differentiable would re-enter wrapper-subclass autograd and
        # recreate its lifetime and aliasing constraints.
        requires_grad = False if coefficient_autograd else any(
            isinstance(c, torch.Tensor) and c.requires_grad for c in cs
        )
        return PolyTensor(cs, requires_grad=requires_grad)

    def correct(out):
        if coefficient_autograd:
            # Every coefficient already owns its ordinary view/alias graph.
            # Wrapper alias correction performs a differentiable ``set_`` on
            # a storage-less leaf in this mode and is both unnecessary and
            # rejected by autograd.
            return out
        return return_and_correct_aliasing(func, args, kwargs, out)

    def is_poly(x):
        return is_initialized_poly(x)

    def has_poly(x):
        if is_initialized_poly(x):
            return True
        if isinstance(x, (tuple, list)):
            return any(has_poly(y) for y in x)
        if isinstance(x, dict):
            return any(has_poly(y) for y in x.values())
        return False

    def plain(x):
        if is_poly_wrapper(x):
            return x.value
        if isinstance(x, tuple):
            return tuple(plain(y) for y in x)
        if isinstance(x, list):
            return [plain(y) for y in x]
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        return x

    def coefficient_dtype(coefficient, dtype):
        if not coefficient.is_complex() or dtype is None or dtype.is_complex:
            return dtype
        complex_dtype = {
            torch.float16: torch.complex32,
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
        }.get(dtype)
        if complex_dtype is None:
            raise TypeError(f"{dtype} cannot represent complex PolyTensor directions")
        return complex_dtype

    def coefficient_options(coefficient, options):
        options = dict(options)
        if "dtype" in options:
            options["dtype"] = coefficient_dtype(coefficient, options["dtype"])
        return options

    def check_coefficient_storage(destinations, sources):
        if any(src.is_complex() and not dst.is_complex()
               for dst, src in zip(destinations, sources)):
            raise RuntimeError(
                "cannot copy complex PolyTensor coefficients into real coefficient storage; "
                "initialize complex directions on the destination first"
            )

    def copy_coefficients(destinations, sources, *, non_blocking=False):
        sources = tuple(sources)
        check_coefficient_storage(destinations, sources)
        for dst, src in zip(destinations, sources):
            dst.copy_(src, non_blocking=non_blocking)

    schema = getattr(func, "_schema", None)
    name = getattr(schema, "name", "")
    mutates_first_arg = bool(getattr(schema, "is_mutable", False)) or name.rsplit("::", 1)[-1].endswith("_")
    if args and mutates_first_arg and not is_poly(args[0]) and has_poly((args[1:], kwargs)):
        target = args[0]
        source_types = ", ".join(type(arg).__name__ for arg in args[1:4])
        raise RuntimeError(
            "cannot update a regular Tensor in-place with a PolyTensor; "
            f"func={func}, target_shape={getattr(target, 'shape', None)}, "
            f"target_dtype={getattr(target, 'dtype', None)}, "
            f"source_types=[{source_types}]"
        )

    def add_terms(xs):
        s = xs[0]
        for x in xs[1:]:
            s = s + x
        return s

    def unstack_coeffs(y):
        return tuple(y.unbind(0))

    def matmul_poly_tensor(A, b):
        if A[0].dim() < 2 or b.dim() < 2:
            return None
        if any(c.dtype != b.dtype for c in A):
            return None
        A_stack = torch.stack(A)
        extra_batch_dims = max(0, b.dim() - A[0].dim())
        A_stack = A_stack.reshape(
            (D + 1,) + (1,) * extra_batch_dims + A_stack.shape[1:]
        )
        out = list(unstack_coeffs(torch.matmul(A_stack, b.unsqueeze(0))))
        out[0] = torch.matmul(A[0], b)
        return tuple(out)

    def matmul_tensor_poly(a, B):
        if a.dim() < 2 or B[0].dim() < 2:
            return None
        if any(c.dtype != a.dtype for c in B):
            return None
        B_stack = torch.stack(B)
        extra_batch_dims = max(0, a.dim() - B[0].dim())
        B_stack = B_stack.reshape(
            (D + 1,) + (1,) * extra_batch_dims + B_stack.shape[1:]
        )
        out = list(unstack_coeffs(torch.matmul(a.unsqueeze(0), B_stack)))
        out[0] = torch.matmul(a, B[0])
        return tuple(out)

    def matmul(a, b):
        a_is_poly = is_poly(a)
        b_is_poly = is_poly(b)
        if a_is_poly and not b_is_poly:
            out = matmul_poly_tensor(a.coeffs, b)
            if out is not None:
                return out
        if b_is_poly and not a_is_poly:
            out = matmul_tensor_poly(a, b.coeffs)
            if out is not None:
                return out
        return bilinear(torch.matmul, a, b)

    def bmm_poly_tensor(A, b):
        if A[0].dim() != 3 or b.dim() != 3:
            return None
        if any(c.dtype != b.dtype for c in A):
            return None
        batch, left, inner = A[0].shape
        right = b.shape[-1]
        A_stack = torch.stack(A).reshape((D + 1) * batch, left, inner)
        b_stack = (
            b.unsqueeze(0)
            .expand(D + 1, *b.shape)
            .reshape((D + 1) * batch, inner, right)
        )
        y = torch.bmm(A_stack, b_stack).reshape(D + 1, batch, left, right)
        out = list(unstack_coeffs(y))
        out[0] = torch.bmm(A[0], b)
        return tuple(out)

    def bmm_tensor_poly(a, B):
        if a.dim() != 3 or B[0].dim() != 3:
            return None
        if any(c.dtype != a.dtype for c in B):
            return None
        batch, left, inner = a.shape
        right = B[0].shape[-1]
        a_stack = (
            a.unsqueeze(0)
            .expand(D + 1, *a.shape)
            .reshape((D + 1) * batch, left, inner)
        )
        B_stack = torch.stack(B).reshape((D + 1) * batch, inner, right)
        y = torch.bmm(a_stack, B_stack).reshape(D + 1, batch, left, right)
        out = list(unstack_coeffs(y))
        out[0] = torch.bmm(a, B[0])
        return tuple(out)

    def bmm(a, b):
        a_is_poly = is_poly(a)
        b_is_poly = is_poly(b)
        if a_is_poly and not b_is_poly:
            out = bmm_poly_tensor(a.coeffs, b)
            if out is not None:
                return out
        if b_is_poly and not a_is_poly:
            out = bmm_tensor_poly(a, b.coeffs)
            if out is not None:
                return out
        return bilinear(torch.bmm, a, b)

    def bilinear(op, a, b):
        def apply(left, right):
            # Real base values may carry complex directions.  Matrix kernels
            # require matching dtypes, unlike elementwise multiplication.
            dtype = torch.promote_types(left.dtype, right.dtype)
            return op(left.to(dtype=dtype), right.to(dtype=dtype))

        a_is_poly = is_poly(a)
        b_is_poly = is_poly(b)
        if a_is_poly and not b_is_poly:
            A = a.coeffs
            return [apply(A[k], b) for k in range(D + 1)]
        if b_is_poly and not a_is_poly:
            B = b.coeffs
            return [apply(a, B[k]) for k in range(D + 1)]
        if not a_is_poly and not b_is_poly:
            y0 = apply(a, b)
            return [y0] + [torch.zeros_like(y0) for _ in range(D)]

        A, B = lift(a), lift(b)

        out = []
        for k in range(D + 1):
            s = apply(A[0], B[k])
            for i in range(1, k + 1):
                s = s + apply(A[i], B[k - i])
            out.append(s)
        return out
    def poly_cross_entropy_loss(X, target, weight, reduction, ignore_index):
        if is_poly(target) or is_poly(weight):
            raise NotImplementedError(
                "PolyTensor cross entropy requires ordinary targets and class weights"
            )
        if isinstance(reduction, str):
            reduction_name = reduction
            reduction = {"none": 0, "mean": 1, "sum": 2}[reduction]
        else:
            reduction_name = {0: "none", 1: "mean", 2: "sum"}[reduction]
        if target.shape == X[0].shape:
            raise NotImplementedError("PolyTensor cross entropy does not implement soft labels")
        if X[0].dim() < 2:
            raise NotImplementedError("PolyTensor cross entropy expects a class dimension")

        target = plain(target)
        weight = plain(weight)
        valid = target != ignore_index
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        gather_index = safe_target.unsqueeze(1)

        # Cross entropy's native backward is expressed through the native
        # log-softmax output.  Use that same rooted jet rather than a
        # separately evaluated logsumexp-minus-target identity, whose
        # float32 gradient can differ by several ulps.
        log_probabilities = series.poly_log_softmax(X, 1)
        target_log_probabilities = [
            x.gather(1, gather_index).squeeze(1)
            for x in log_probabilities
        ]
        losses = [
            torch.where(
                valid,
                -target_log_probabilities[k],
                torch.zeros_like(target_log_probabilities[k]),
            )
            for k in range(D + 1)
        ]

        if weight is not None:
            sample_weight = weight.gather(
                0,
                safe_target.reshape(-1),
            ).reshape_as(safe_target)
            sample_weight = torch.where(valid, sample_weight, torch.zeros_like(sample_weight))
            losses = [loss * sample_weight for loss in losses]
            total_weight = sample_weight.sum()
        else:
            total_weight = valid.sum().to(device=losses[0].device, dtype=losses[0].dtype)

        if reduction == 0:
            out = losses
        if reduction == 1:
            out = [loss.sum() / total_weight for loss in losses]
        elif reduction == 2:
            out = [loss.sum() for loss in losses]
        elif reduction != 0:
            raise ValueError(f"unknown reduction {reduction!r}")

        native_loss = torch.nn.functional.cross_entropy(
            X[0],
            target,
            weight=weight,
            reduction=reduction_name,
            ignore_index=ignore_index,
        )
        return [native_loss, *out[1:]]

    if func in (aten.add.Tensor, aten.add.Scalar):
        a, b = args[:2]
        alpha = kwargs.get("alpha", 1)
        if is_poly(a) and not is_poly(b):
            return wrap((a.coeffs[0] + alpha * b, *clone_coeffs(a.coeffs[1:])))
        if not is_poly(a) and is_poly(b):
            if alpha == 1:
                return wrap((a + b.coeffs[0], *clone_coeffs(b.coeffs[1:])))
            return wrap((a + alpha * b.coeffs[0], *(alpha * c for c in b.coeffs[1:])))
        A, B = lift(a), lift(b)
        return wrap(A[k] + alpha * B[k] for k in range(D + 1))

    if func in (aten.sub.Tensor, aten.sub.Scalar):
        a, b = args[:2]
        alpha = kwargs.get("alpha", 1)
        if is_poly(a) and not is_poly(b):
            return wrap((a.coeffs[0] - alpha * b, *clone_coeffs(a.coeffs[1:])))
        if not is_poly(a) and is_poly(b):
            return wrap((a - alpha * b.coeffs[0], *(-alpha * c for c in b.coeffs[1:])))
        A, B = lift(a), lift(b)
        return wrap(A[k] - alpha * B[k] for k in range(D + 1))

    if func is aten.rsub.Scalar:
        x, other = args[:2]
        alpha = kwargs.get("alpha", 1)
        return wrap((other - alpha * x.coeffs[0], *(-alpha * c for c in x.coeffs[1:])))

    if func in (aten.mul.Tensor, aten.mul.Scalar):
        a, b = args[:2]
        if is_poly(a) and not is_poly(b):
            return wrap(c * b for c in a.coeffs)
        if not is_poly(a) and is_poly(b):
            return wrap(a * c for c in b.coeffs)
        return wrap(series.conv(lift(a), lift(b)))

    if func in (aten.div.Tensor, aten.div.Scalar):
        a, b = args[:2]
        rounding_mode = kwargs.get("rounding_mode")
        if rounding_mode is not None:
            raise NotImplementedError("PolyTensor does not implement rounded division")
        if is_poly(a) and not is_poly(b):
            return wrap(c / b for c in a.coeffs)
        return wrap(series.conv(lift(a), series.poly_reciprocal(lift(b))))

    if func is aten.reciprocal.default:
        return wrap(series.poly_reciprocal(lift(args[0])))

    if func is aten.mm.default:
        return wrap(matmul(args[0], args[1]))

    if func in (aten.dot.default, aten.mv.default):
        return wrap(bilinear(func, args[0], args[1]))

    if func is aten.matmul.default:
        return correct(wrap(matmul(args[0], args[1])))

    if func is aten.bmm.default:
        return correct(wrap(bmm(args[0], args[1])))

    if func is aten.scaled_dot_product_attention.default:
        query, key, value = args[:3]
        attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
        is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
        scale = kwargs.get("scale")
        enable_gqa = kwargs.get("enable_gqa", False)
        if enable_gqa:
            raise NotImplementedError("PolyTensor does not implement GQA head repetition")

        scale = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if is_causal:
            causal_mask = torch.ones(
                scores.shape[-2:],
                dtype=torch.bool,
                device=scores.device,
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn_weight = torch.softmax(scores, dim=-1)
        if dropout_p != 0.0:
            attn_weight = torch.dropout(attn_weight, dropout_p, True)
        return torch.matmul(attn_weight, value)

    if func is aten._scaled_dot_product_flash_attention_for_cpu.default:
        query, key, value = args[:3]
        dropout_p = args[3] if len(args) > 3 else kwargs.get("dropout_p", 0.0)
        is_causal = args[4] if len(args) > 4 else kwargs.get("is_causal", False)
        attn_mask = kwargs.get("attn_mask")
        scale = kwargs.get("scale")

        scale = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if is_causal:
            causal_mask = torch.ones(
                scores.shape[-2:],
                dtype=torch.bool,
                device=scores.device,
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        logsumexp = torch.logsumexp(scores, dim=-1)
        attn_weight = torch.softmax(scores, dim=-1)
        if dropout_p != 0.0:
            attn_weight = torch.dropout(attn_weight, dropout_p, True)
        return torch.matmul(attn_weight, value), logsumexp

    if func is aten.addmm.default:
        x, a, b = args[:3]
        beta = kwargs.get("beta", 1)
        alpha = kwargs.get("alpha", 1)
        M = matmul(a, b)
        if beta == 0:
            return wrap(alpha * m for m in M)
        if is_poly(x):
            X = x.coeffs
            return wrap(beta * X[k] + alpha * M[k] for k in range(D + 1))
        out = [alpha * m for m in M]
        out[0] = beta * x + out[0]
        return wrap(out)

    if func is aten.linear.default:
        x, weight = args[:2]
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        if is_poly(weight):
            if is_poly(x):
                Y = bilinear(
                    lambda a, b: torch.matmul(a, b.transpose(-2, -1)),
                    x,
                    weight,
                )
            else:
                weight_t = tuple(w.transpose(-2, -1) for w in weight.coeffs)
                Y = matmul_tensor_poly(x, weight_t)
                if Y is None:
                    Y = bilinear(
                        lambda a, b: torch.matmul(a, b.transpose(-2, -1)),
                        x,
                        weight,
                    )
        elif is_poly(x):
            Y = matmul_poly_tensor(
                x.coeffs,
                weight.transpose(-2, -1),
            )
            if Y is None:
                Y = bilinear(
                    lambda a, b: torch.matmul(a, b.transpose(-2, -1)),
                    x,
                    weight,
                )
        else:
            Y = bilinear(
                lambda a, b: torch.matmul(a, b.transpose(-2, -1)),
                x,
                weight,
            )
        if bias is not None:
            if is_poly(bias):
                B = bias.coeffs
                Y = [Y[k] + B[k] for k in range(D + 1)]
            else:
                Y = list(Y)
                Y[0] = Y[0] + bias
        return wrap(Y)

    if func in (aten.convolution.default, aten.conv2d.default):
        x, weight, bias = args[:3]
        rest = args[3:]
        Y = bilinear(lambda a, b: func(a, b, None, *rest, **kwargs), x, weight)
        if bias is not None:
            B = lift(bias)
            Y = [
                Y[k] + B[k].reshape(1, -1, *([1] * (Y[k].dim() - 2)))
                for k in range(D + 1)
            ]
        return wrap(Y)

    if func is aten.neg.default:
        A = lift(args[0])
        return wrap(-A[k] for k in range(D + 1))

    if func is aten.pow.Tensor_Scalar:
        x, n = args[:2]
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        if not isinstance(n, int) or n < 0:
            raise NotImplementedError("only nonnegative integer powers")

        y = PolyTensor.constant(torch.ones_like(lift(x)[0]), D)
        for _ in range(n):
            y = y * x
        return y

    if func in (aten.add_.Tensor, aten.add_.Scalar):
        a, b = args[:2]
        alpha = kwargs.get("alpha", 1)
        if not is_poly(b):
            a.coeffs[0].add_(b, alpha=alpha)
            return a
        A, B = lift(a), lift(b)
        check_coefficient_storage(A, B)
        for k in range(D + 1):
            A[k].add_(B[k], alpha=alpha)
        return a

    if func in (aten.sub_.Tensor, aten.sub_.Scalar):
        a, b = args[:2]
        alpha = kwargs.get("alpha", 1)
        if not is_poly(b):
            a.coeffs[0].sub_(b, alpha=alpha)
            return a
        A, B = lift(a), lift(b)
        check_coefficient_storage(A, B)
        for k in range(D + 1):
            A[k].sub_(B[k], alpha=alpha)
        return a

    if func in (aten.mul_.Tensor, aten.mul_.Scalar):
        a, b = args[:2]
        if not is_poly(b):
            for coeff in a.coeffs:
                coeff.mul_(b)
            return a
        out = series.conv(lift(a), lift(b))
        copy_coefficients(a.coeffs, out)
        return a

    if func in (aten.div_.Tensor, aten.div_.Scalar):
        a, b = args[:2]
        rounding_mode = kwargs.get("rounding_mode")
        if rounding_mode is not None:
            raise NotImplementedError("PolyTensor does not implement rounded in-place division")
        if not is_poly(b):
            for coeff in a.coeffs:
                coeff.div_(b)
            return a
        out = series.conv(lift(a), series.poly_reciprocal(lift(b)))
        copy_coefficients(a.coeffs, out)
        return a

    if func in (aten.masked_fill.Scalar, aten.masked_fill.Tensor):
        x, mask, value = args[:3]
        X = lift(x)
        if is_poly(value):
            V = lift(value)
            return wrap(torch.where(plain(mask), V[k], X[k]) for k in range(D + 1))
        return wrap(
            (
                X[0].masked_fill(plain(mask), value),
                *(c.masked_fill(plain(mask), 0) for c in X[1:]),
            )
        )

    if func in (aten.masked_fill_.Scalar, aten.masked_fill_.Tensor):
        x, mask, value = args[:3]
        if is_poly(value):
            X, V = lift(x), lift(value)
            copy_coefficients(X, (torch.where(plain(mask), V[k], X[k]) for k in range(D + 1)))
            return x
        x.coeffs[0].masked_fill_(plain(mask), value)
        for coeff in x.coeffs[1:]:
            coeff.masked_fill_(plain(mask), 0)
        return x

    if func in (
        aten.where.self,
        aten.where.ScalarOther,
        aten.where.ScalarSelf,
        aten.where.Scalar,
    ):
        condition = plain(args[0])
        a = args[1]
        b = args[2]
        A, B = lift(a), lift(b)
        return wrap(torch.where(condition, A[k], B[k]) for k in range(D + 1))

    if func is aten.exp.default:
        return wrap(series.poly_exp(lift(args[0])))

    if func is aten.sin.default:
        return wrap(series.poly_sin(lift(args[0])))

    if func is aten.cos.default:
        return wrap(series.poly_cos(lift(args[0])))

    if func is aten.log.default:
        return wrap(series.poly_log(lift(args[0])))

    if func is aten.sqrt.default:
        return wrap(series.poly_sqrt(lift(args[0])))

    if func is aten.rsqrt.default:
        return wrap(series.poly_rsqrt(lift(args[0])))

    if func is aten.logsumexp.default:
        x = args[0]
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        return wrap(series.poly_logsumexp(lift(x), dim, keepdim))

    if func is aten.sigmoid.default:
        return wrap(series.poly_sigmoid(lift(args[0])))

    if func is aten.tanh.default:
        return wrap(series.poly_tanh(lift(args[0]))[0])

    if func is aten.silu.default:
        X = lift(args[0])
        S = series.poly_sigmoid(X)
        out = [torch.nn.functional.silu(X[0])]
        for k in range(1, D + 1):
            coefficient = X[0] * S[k]
            for i in range(1, k + 1):
                coefficient = coefficient + X[i] * S[k - i]
            out.append(coefficient)
        return wrap(out)

    if func is aten.gelu.default:
        approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
        out = series.poly_gelu(lift(args[0]), approximate)
        out[0] = torch.nn.functional.gelu(lift(args[0])[0], approximate=approximate)
        return wrap(out)

    if func is aten._log_softmax.default:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        half_to_float = args[2] if len(args) > 2 else kwargs.get("half_to_float", False)
        coefficients = lift(args[0])
        if half_to_float:
            coefficients = [c.float() for c in coefficients]
        return wrap(series.poly_log_softmax(coefficients, dim))

    if func is aten.log_softmax.int:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
        if dtype is not None:
            out = series.poly_log_softmax([c.to(dtype=dtype) for c in lift(args[0])], dim)
        else:
            out = series.poly_log_softmax(lift(args[0]), dim)
        return wrap(out)

    if func is aten._softmax.default:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        half_to_float = args[2] if len(args) > 2 else kwargs.get("half_to_float", False)
        if half_to_float:
            return wrap(series.poly_softmax([c.float() for c in lift(args[0])], dim))
        return wrap(series.poly_softmax(lift(args[0]), dim))

    if func is aten.softmax.int:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
        if dtype is not None:
            out = series.poly_softmax([c.to(dtype=dtype) for c in lift(args[0])], dim)
        else:
            out = series.poly_softmax(lift(args[0]), dim)
        return wrap(out)

    if func is aten.embedding.default:
        weight = args[0]
        return wrap(func(w, *plain(args[1:]), **plain(kwargs)) for w in lift(weight))

    if func is aten.avg_pool2d.default:
        return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

    if func in (aten.mean.default, aten.mean.dim):
        return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

    if func in (aten.sum.default, aten.sum.dim_IntList):
        return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

    if func in (aten.sum_to_size.default, aten._grad_sum_to_size.default):
        return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

    if func is aten.nll_loss_forward.default:
        if has_poly((args[1:], kwargs)):
            raise NotImplementedError(
                "PolyTensor NLL loss requires ordinary targets and class weights"
            )
        outs = [func(c, *plain(args[1:]), **plain(kwargs)) for c in lift(args[0])]
        return wrap(out[0] for out in outs), outs[0][1]

    if func is aten.cross_entropy_loss.default:
        x, target = args[:2]
        weight = args[2] if len(args) > 2 else kwargs.get("weight")
        reduction = args[3] if len(args) > 3 else kwargs.get("reduction", 1)
        ignore_index = args[4] if len(args) > 4 else kwargs.get("ignore_index", -100)
        label_smoothing = args[5] if len(args) > 5 else kwargs.get("label_smoothing", 0.0)
        if label_smoothing != 0.0:
            raise NotImplementedError("PolyTensor does not implement label-smoothed cross entropy")

        return wrap(poly_cross_entropy_loss(lift(x), target, weight, reduction, ignore_index))

    if func is aten.nll_loss_backward.default:
        if has_poly((args[2:], kwargs)):
            raise NotImplementedError(
                "PolyTensor NLL loss requires ordinary targets, class weights, and total weight"
            )
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten._log_softmax_backward_data.default:
        grad_output, output, dim = args[:3]
        input_dtype = args[3] if len(args) > 3 else kwargs.get("input_dtype")
        G = lift(grad_output)
        O = lift(output)

        S = [g.sum(dim=dim, keepdim=True) for g in G]
        E = series.poly_exp(O)

        out = []
        for k in range(D + 1):
            s = E[0] * S[k]
            for i in range(1, k + 1):
                s = s + E[i] * S[k - i]
            y = G[k] - s
            out.append(y.to(dtype=input_dtype) if input_dtype is not None else y)

        return wrap(out)

    if func is aten._softmax_backward_data.default:
        grad_output, output, dim = args[:3]
        input_dtype = args[3] if len(args) > 3 else kwargs.get("input_dtype")
        G = lift(grad_output)
        O = lift(output)
        GO = series.conv(G, O)
        S = [go.sum(dim=dim, keepdim=True) for go in GO]
        out = series.conv(O, [G[k] - S[k] for k in range(D + 1)])
        return wrap(y.to(dtype=input_dtype) if input_dtype is not None else y for y in out)

    if func is aten.sigmoid_backward.default:
        G = lift(args[0])
        O = lift(args[1])
        one_minus_o = [1 - O[0]] + [-O[k] for k in range(1, D + 1)]
        return wrap(series.conv(G, series.conv(O, one_minus_o)))

    if func is aten.tanh_backward.default:
        G = lift(args[0])
        O = lift(args[1])
        O2 = series.conv(O, O)
        one_minus_o2 = [1 - O2[0]] + [-O2[k] for k in range(1, D + 1)]
        return wrap(series.conv(G, one_minus_o2))

    if func is aten.silu_backward.default:
        G = lift(args[0])
        X = lift(args[1])
        S = series.poly_sigmoid(X)
        one_minus_s = [1 - S[0]] + [-S[k] for k in range(1, D + 1)]
        x_s_one_minus_s = series.conv(X, series.conv(S, one_minus_s))
        derivative = [S[k] + x_s_one_minus_s[k] for k in range(D + 1)]
        return wrap(series.conv(G, derivative))

    if func is aten.gelu_backward.default:
        grad_output, x = args[:2]
        approximate = args[2] if len(args) > 2 else kwargs.get("approximate", "none")
        return wrap(series.conv(lift(grad_output), series.poly_gelu_grad(lift(x), approximate)))

    if func is aten.isnan.default:
        out = torch.isnan(lift(args[0])[0])
        for c in lift(args[0])[1:]:
            out = out | torch.isnan(c)
        return out

    if func is aten.native_dropout.default:
        x = args[0]
        p = args[1] if len(args) > 1 else kwargs["p"]
        train = args[2] if len(args) > 2 else kwargs["train"]
        X = lift(x)
        y0, mask = func(X[0], p, train)
        if not train or p == 0:
            return correct((wrap((y0, *X[1:])), mask))
        if p == 1:
            return wrap(torch.zeros_like(c) for c in X), mask
        scale = 1.0 / (1.0 - p)
        mask_values = mask.to(dtype=X[0].dtype)
        return wrap((y0, *(c * mask_values * scale for c in X[1:]))), mask

    if func is aten.dropout.default:
        x = args[0]
        p = args[1] if len(args) > 1 else kwargs["p"]
        train = args[2] if len(args) > 2 else kwargs["train"]
        X = lift(x)
        if not train or p == 0:
            y = func(X[0], p, train)
            return correct(wrap((y, *X[1:])))
        if p == 1:
            return wrap(torch.zeros_like(c) for c in X)
        y, mask = aten.native_dropout.default(X[0], p, train)
        mask = mask.to(dtype=X[0].dtype)
        scale = 1.0 / (1.0 - p)
        return wrap((y, *(c * mask * scale for c in X[1:])))

    if func is aten.native_dropout_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.avg_pool2d_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.embedding_dense_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.slice_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.select_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.convolution_backward.default:
        grad_output, x, weight = args[:3]
        rest = plain(args[3:])
        call_kwargs = plain(kwargs)
        if rest:
            output_mask = rest[-1]
            call_args = rest[:-1]
        else:
            output_mask = call_kwargs.pop("output_mask")
            call_args = ()
        G = lift(grad_output)
        X = lift(x)
        W = lift(weight)
        outputs = []

        if output_mask[0]:
            outputs.append(
                wrap(
                    add_terms([
                        func(
                            G[i],
                            X[0],
                            W[k - i],
                            *call_args,
                            (True, False, False),
                            **call_kwargs,
                        )[0]
                        for i in range(k + 1)
                    ])
                    for k in range(D + 1)
                )
            )
        else:
            outputs.append(None)

        if output_mask[1]:
            outputs.append(
                wrap(
                    add_terms([
                        func(
                            G[i],
                            X[k - i],
                            W[0],
                            *call_args,
                            (False, True, False),
                            **call_kwargs,
                        )[1]
                        for i in range(k + 1)
                    ])
                    for k in range(D + 1)
                )
            )
        else:
            outputs.append(None)

        if output_mask[2]:
            outputs.append(
                wrap(
                    func(
                        G[k],
                        X[0],
                        W[0],
                        *call_args,
                        (False, False, True),
                        **call_kwargs,
                    )[2]
                    for k in range(D + 1)
                )
            )
        else:
            outputs.append(None)

        return tuple(outputs)

    # Minimal autograd / optimizer plumbing.
    if func is aten.ones_like.default:
        x0 = lift(args[0])[0]
        out_zero = func(x0, *args[1:], **kwargs)
        return wrap((out_zero, *(torch.zeros_like(out_zero) for _ in range(D))))

    if func is aten.zeros_like.default:
        return wrap(torch.zeros_like(c, *args[1:], **coefficient_options(c, kwargs)) for c in lift(args[0]))

    if func is aten.empty_like.default:
        return wrap(torch.empty_like(c, *args[1:], **coefficient_options(c, kwargs)) for c in lift(args[0]))

    if func is aten.full_like.default:
        call_kwargs = dict(kwargs)
        fill_value = args[1] if len(args) > 1 else call_kwargs.pop("fill_value")
        out_zero = torch.full_like(lift(args[0])[0], fill_value, *args[2:], **call_kwargs)
        return wrap((out_zero, *(torch.zeros_like(out_zero) for _ in range(D))))

    if func is aten.new_empty_strided.default:
        return wrap(func(c, *args[1:], **coefficient_options(c, kwargs)) for c in lift(args[0]))

    if func is aten.detach.default:
        return PolyTensor(tuple(c.detach() for c in lift(args[0])))

    if func is aten.detach_.default:
        x = args[0]
        x._set_coeffs(c.detach() for c in lift(x))
        return x

    if func is aten.alias.default:
        return correct(PolyTensor(tuple(func(c) for c in lift(args[0]))))

    if func in (
        aten._conj.default,
        aten._conj_physical.default,
        aten.conj_physical.default,
        aten.real.default,
        aten.imag.default,
        aten.view_as_real.default,
        aten.view_as_complex.default,
    ):
        return correct(wrap(func(c, *args[1:], **kwargs) for c in lift(args[0])))

    if func is aten.clone.default:
        x = args[0]
        return PolyTensor(tuple(c.clone(**kwargs) for c in lift(x)), requires_grad=x.requires_grad)

    if func in (
        aten._to_copy.default,
        aten.to.dtype,
        aten.to.device,
        aten.to.other,
        aten.to.dtype_layout,
    ):
        def convert(c):
            other_args = list(args[1:])
            options = coefficient_options(c, kwargs)
            if func in (aten.to.dtype, aten.to.dtype_layout) and other_args:
                other_args[0] = coefficient_dtype(c, other_args[0])
            elif func is aten.to.device and len(other_args) > 1:
                other_args[1] = coefficient_dtype(c, other_args[1])
            elif func is aten.to.other:
                other = other_args[0] if other_args else options.pop("other")
                non_blocking = other_args[1] if len(other_args) > 1 else options.get("non_blocking", False)
                copy = other_args[2] if len(other_args) > 2 else options.get("copy", False)
                return c.to(
                    device=other.device,
                    dtype=coefficient_dtype(c, other.dtype),
                    non_blocking=non_blocking,
                    copy=copy,
                    memory_format=options.get("memory_format", torch.preserve_format),
                )
            return func(c, *other_args, **options)

        return wrap(convert(c) for c in lift(args[0]))

    if func is aten.zero_.default:
        x = args[0]
        for coeff in lift(x):
            coeff.zero_()
        return x

    if func in (aten.bernoulli_.float, aten.bernoulli_.Tensor):
        x = args[0]
        x.coeffs[0].bernoulli_(*plain(args[1:]), **plain(kwargs))
        for coeff in x.coeffs[1:]:
            coeff.zero_()
        return x

    if func is aten.copy_.default:
        x, src = args[:2]
        X, S = lift(x), lift(src)
        non_blocking = args[2] if len(args) > 2 else kwargs.get("non_blocking", False)
        copy_coefficients(X, S, non_blocking=non_blocking)
        return x

    if func in (
        aten.t.default,
        aten.transpose.int,
        aten.permute.default,
        aten.view.default,
        aten.reshape.default,
        aten._unsafe_view.default,
        aten.flatten.using_ints,
        aten.as_strided.default,
        aten.slice.Tensor,
        aten.select.int,
        aten.unsqueeze.default,
        aten.squeeze.dim,
        aten.squeeze.default,
    ):
        return correct(wrap(func(c, *args[1:], **kwargs) for c in lift(args[0])))

    if func is aten.expand.default:
        x, size = args[:2]
        return correct(wrap(c.expand(size) for c in lift(x)))

    if func is aten.cat.default:
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        return wrap(
            torch.cat([lift(tensor)[k] for tensor in tensors], dim=dim)
            for k in range(D + 1)
        )

    if func is aten.stack.default:
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        return wrap(
            torch.stack([lift(tensor)[k] for tensor in tensors], dim=dim)
            for k in range(D + 1)
        )

    if func is aten.index_select.default:
        return wrap(func(c, *plain(args[1:]), **plain(kwargs)) for c in lift(args[0]))

    if func is aten.gather.default:
        return wrap(func(c, *plain(args[1:]), **plain(kwargs)) for c in lift(args[0]))

    if func is aten.index.Tensor:
        return wrap(func(c, *plain(args[1:]), **plain(kwargs)) for c in lift(args[0]))

    raise NotImplementedError(f"PolyTensor does not implement {func}")
