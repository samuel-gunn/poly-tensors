import math

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

aten = torch.ops.aten


class PolyTensor(torch.Tensor):
    @staticmethod
    def _coerce_coeffs(coeffs, degree=None):
        if isinstance(coeffs, PolyTensor):
            coeffs = coeffs.coeffs
        coeffs = tuple(coeffs)
        base = coeffs[0] if isinstance(coeffs[0], torch.Tensor) else torch.as_tensor(coeffs[0])
        coeffs = tuple(
            c if isinstance(c, torch.Tensor)
            else torch.as_tensor(c, device=base.device, dtype=base.dtype)
            for c in coeffs
        )
        if degree is not None:
            if degree < len(coeffs) - 1:
                raise ValueError("degree must be at least the degree of coeffs")
            coeffs = coeffs + tuple(torch.zeros_like(base) for _ in range(degree + 1 - len(coeffs)))
        shape = torch.broadcast_shapes(*(c.shape for c in coeffs))
        return tuple(c.expand(shape) for c in coeffs)

    def _set_coeffs(self, coeffs):
        self.coeffs = self._coerce_coeffs(coeffs)
        self.degree = len(self.coeffs) - 1

    @staticmethod
    def __new__(cls, coeffs, *, degree=None, requires_grad=False):
        coeffs = cls._coerce_coeffs(coeffs, degree)
        base = coeffs[0]

        out = torch.Tensor._make_wrapper_subclass(
            cls,
            base.shape,
            strides=base.stride(),
            storage_offset=base.storage_offset(),
            dtype=base.dtype,
            layout=base.layout,
            device=base.device,
            requires_grad=requires_grad,
        )
        out._set_coeffs(coeffs)
        return out

    def __init__(self, coeffs, *, degree=None, requires_grad=False):
        pass

    @property
    def value(self):
        return self.coeffs[0]

    @property
    def tangent(self):
        return self.coeffs[1]

    @classmethod
    def constant(cls, x, degree):
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        return cls((x,) + tuple(torch.zeros_like(x) for _ in range(degree)))

    def backward(self, gradient=None, **kwargs):
        if gradient is None:
            gradient = PolyTensor.constant(torch.ones_like(self.value), self.degree)
        elif not isinstance(gradient, PolyTensor):
            gradient = PolyTensor.constant(gradient, self.degree)

        torch.autograd.backward(self, gradient, **kwargs)

    def __repr__(self):
        if not hasattr(self, "coeffs"):
            return f"PolyTensor(<uninitialized>, requires_grad={self.requires_grad})"
        return f"PolyTensor({self.coeffs}, requires_grad={self.requires_grad})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        polys = []

        def collect(x):
            if isinstance(x, PolyTensor):
                polys.append(x)
            elif isinstance(x, (tuple, list)):
                for y in x:
                    collect(y)
            elif isinstance(x, dict):
                for y in x.values():
                    collect(y)

        collect(args)
        collect(kwargs)

        if not polys:
            return func(*args, **kwargs)

        D = polys[0].degree
        like = polys[0].value

        def lift(x):
            if isinstance(x, PolyTensor):
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
            return PolyTensor(cs, requires_grad=False)

        def correct(out):
            return return_and_correct_aliasing(func, args, kwargs, out)

        def is_poly(x):
            return isinstance(x, PolyTensor)

        def has_poly(x):
            if isinstance(x, PolyTensor):
                return True
            if isinstance(x, (tuple, list)):
                return any(has_poly(y) for y in x)
            if isinstance(x, dict):
                return any(has_poly(y) for y in x.values())
            return False

        def plain(x):
            if isinstance(x, PolyTensor):
                if not hasattr(x, "coeffs"):
                    return torch.empty(x.shape, dtype=x.dtype, device=x.device)
                return x.value
            if isinstance(x, tuple):
                return tuple(plain(y) for y in x)
            if isinstance(x, list):
                return [plain(y) for y in x]
            if isinstance(x, dict):
                return {k: plain(v) for k, v in x.items()}
            return x

        schema = getattr(func, "_schema", None)
        name = getattr(schema, "name", "")
        mutates_first_arg = bool(getattr(schema, "is_mutable", False)) or name.rsplit("::", 1)[-1].endswith("_")
        if args and mutates_first_arg and not is_poly(args[0]) and has_poly((args[1:], kwargs)):
            raise RuntimeError("cannot update a regular Tensor in-place with a PolyTensor")

        def conv(a, b):
            out = []
            for k in range(D + 1):
                s = a[0] * b[k]
                for i in range(1, k + 1):
                    s = s + a[i] * b[k - i]
                out.append(s)
            return out

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
            return unstack_coeffs(torch.matmul(torch.stack(A), b))

        def matmul_tensor_poly(a, B):
            if a.dim() < 2 or B[0].dim() < 2:
                return None
            B_stack = torch.stack(B)
            extra_batch_dims = max(0, a.dim() - B[0].dim())
            B_stack = B_stack.reshape(
                (D + 1,) + (1,) * extra_batch_dims + B_stack.shape[1:]
            )
            return unstack_coeffs(torch.matmul(a.unsqueeze(0), B_stack))

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
            batch, left, inner = A[0].shape
            right = b.shape[-1]
            A_stack = torch.stack(A).reshape((D + 1) * batch, left, inner)
            b_stack = (
                b.unsqueeze(0)
                .expand(D + 1, *b.shape)
                .reshape((D + 1) * batch, inner, right)
            )
            y = torch.bmm(A_stack, b_stack).reshape(D + 1, batch, left, right)
            return unstack_coeffs(y)

        def bmm_tensor_poly(a, B):
            if a.dim() != 3 or B[0].dim() != 3:
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
            return unstack_coeffs(y)

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
            a_is_poly = is_poly(a)
            b_is_poly = is_poly(b)
            if a_is_poly and not b_is_poly:
                A = a.coeffs
                return [op(A[k], b) for k in range(D + 1)]
            if b_is_poly and not a_is_poly:
                B = b.coeffs
                return [op(a, B[k]) for k in range(D + 1)]
            if not a_is_poly and not b_is_poly:
                y0 = op(a, b)
                return [y0] + [torch.zeros_like(y0) for _ in range(D)]

            A, B = lift(a), lift(b)

            out = []
            for k in range(D + 1):
                s = op(A[0], B[k])
                for i in range(1, k + 1):
                    s = s + op(A[i], B[k - i])
                out.append(s)
            return out

        def poly_exp(X):
            Y = [None] * (D + 1)
            Y[0] = torch.exp(X[0])

            for k in range(1, D + 1):
                s = X[1] * Y[k - 1]
                for i in range(2, k + 1):
                    s = s + i * X[i] * Y[k - i]
                Y[k] = s / k

            return Y

        def poly_log(X):
            Y = [None] * (D + 1)
            Y[0] = torch.log(X[0])

            for k in range(1, D + 1):
                s = k * X[k]
                for i in range(1, k):
                    s = s - i * Y[i] * X[k - i]
                Y[k] = s / (k * X[0])

            return Y

        def poly_reciprocal(X):
            Y = [None] * (D + 1)
            Y[0] = 1 / X[0]

            for k in range(1, D + 1):
                s = X[1] * Y[k - 1]
                for i in range(2, k + 1):
                    s = s + X[i] * Y[k - i]
                Y[k] = -s / X[0]

            return Y

        def poly_sqrt(X):
            Y = [None] * (D + 1)
            Y[0] = torch.sqrt(X[0])

            for k in range(1, D + 1):
                s = torch.zeros_like(X[0])
                for i in range(1, k):
                    s = s + Y[i] * Y[k - i]
                Y[k] = (X[k] - s) / (2 * Y[0])

            return Y

        def poly_rsqrt(X):
            return poly_reciprocal(poly_sqrt(X))

        def poly_log_softmax(X, dim):
            M = X[0].max(dim=dim, keepdim=True).values
            Z = [X[0] - M] + list(X[1:])
            E = poly_exp(Z)
            S = [e.sum(dim=dim, keepdim=True) for e in E]
            L = poly_log(S)
            L[0] = L[0] + M
            return [X[k] - L[k] for k in range(D + 1)]

        def poly_softmax(X, dim):
            M = X[0].max(dim=dim, keepdim=True).values
            Z = [X[0] - M] + list(X[1:])
            E = poly_exp(Z)
            S = [e.sum(dim=dim, keepdim=True) for e in E]
            return conv(E, poly_reciprocal(S))

        def poly_logsumexp(X, dim, keepdim=False):
            if isinstance(dim, (tuple, list)):
                if len(dim) != 1:
                    raise NotImplementedError("PolyTensor logsumexp supports one dimension")
                dim = dim[0]
            M = X[0].max(dim=dim, keepdim=True).values
            Z = [X[0] - M] + list(X[1:])
            E = poly_exp(Z)
            S = [e.sum(dim=dim, keepdim=True) for e in E]
            L = poly_log(S)
            L[0] = L[0] + M
            if not keepdim:
                L = [l.squeeze(dim) for l in L]
            return L

        def poly_cross_entropy_loss(X, target, weight, reduction, ignore_index):
            if isinstance(reduction, str):
                reduction = {"none": 0, "mean": 1, "sum": 2}[reduction]
            if target.shape == X[0].shape:
                raise NotImplementedError("PolyTensor cross entropy does not implement soft labels")
            if X[0].dim() < 2:
                raise NotImplementedError("PolyTensor cross entropy expects a class dimension")

            target = plain(target)
            weight = plain(weight)
            valid = target != ignore_index
            safe_target = torch.where(valid, target, torch.zeros_like(target))
            gather_index = safe_target.unsqueeze(1)

            log_normalizer = poly_logsumexp(X, 1)
            target_logits = [
                x.gather(1, gather_index).squeeze(1)
                for x in X
            ]
            losses = [
                torch.where(
                    valid,
                    log_normalizer[k] - target_logits[k],
                    torch.zeros_like(log_normalizer[k]),
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
                return losses
            if reduction == 1:
                return [loss.sum() / total_weight for loss in losses]
            if reduction == 2:
                return [loss.sum() for loss in losses]
            raise ValueError(f"unknown reduction {reduction!r}")

        def poly_one(like):
            return [torch.ones_like(like)] + [torch.zeros_like(like) for _ in range(D)]

        def poly_sigmoid(X):
            T, _ = poly_tanh([0.5 * x for x in X])
            return [0.5 * (1 + T[0])] + [0.5 * t for t in T[1:]]

        def poly_tanh(X):
            Y = [None] * (D + 1)
            Q = [None] * (D + 1)
            Y[0] = torch.tanh(X[0])
            Q[0] = 1 - Y[0] * Y[0]

            for k in range(1, D + 1):
                s = X[1] * Q[k - 1]
                for i in range(2, k + 1):
                    s = s + i * X[i] * Q[k - i]
                Y[k] = s / k

                y2 = Y[0] * Y[k]
                for i in range(1, k + 1):
                    y2 = y2 + Y[i] * Y[k - i]
                Q[k] = -y2

            return Y, Q

        def poly_normal_cdf_and_pdf(X):
            X2 = conv(X, X)
            phi = poly_exp([-0.5 * x for x in X2])
            phi = [x / math.sqrt(2 * math.pi) for x in phi]

            cdf = [None] * (D + 1)
            cdf[0] = 0.5 * (1 + torch.erf(X[0] / math.sqrt(2)))

            for k in range(1, D + 1):
                s = X[1] * phi[k - 1]
                for i in range(2, k + 1):
                    s = s + i * X[i] * phi[k - i]
                cdf[k] = s / k

            return cdf, phi

        def poly_gelu(X, approximate):
            if approximate == "none":
                cdf, _ = poly_normal_cdf_and_pdf(X)
                return conv(X, cdf)

            if approximate == "tanh":
                one = poly_one(X[0])
                X2 = conv(X, X)
                X3 = conv(X2, X)
                scale = math.sqrt(2 / math.pi)
                U = [scale * (X[k] + 0.044715 * X3[k]) for k in range(D + 1)]
                T, _ = poly_tanh(U)
                return [0.5 * y for y in conv(X, [one[k] + T[k] for k in range(D + 1)])]

            raise NotImplementedError(f"PolyTensor does not implement GELU approximate={approximate!r}")

        def poly_gelu_grad(X, approximate):
            if approximate == "none":
                cdf, phi = poly_normal_cdf_and_pdf(X)
                x_phi = conv(X, phi)
                return [cdf[k] + x_phi[k] for k in range(D + 1)]

            if approximate == "tanh":
                one = poly_one(X[0])
                X2 = conv(X, X)
                X3 = conv(X2, X)
                scale = math.sqrt(2 / math.pi)
                U = [scale * (X[k] + 0.044715 * X3[k]) for k in range(D + 1)]
                T, Q = poly_tanh(U)
                Udx = [scale * (one[k] + 3 * 0.044715 * X2[k]) for k in range(D + 1)]
                x_qudx = conv(X, conv(Q, Udx))
                return [0.5 * (one[k] + T[k] + x_qudx[k]) for k in range(D + 1)]

            raise NotImplementedError(f"PolyTensor does not implement GELU approximate={approximate!r}")

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
            return wrap(conv(lift(a), lift(b)))

        if func in (aten.div.Tensor, aten.div.Scalar):
            a, b = args[:2]
            rounding_mode = kwargs.get("rounding_mode")
            if rounding_mode is not None:
                raise NotImplementedError("PolyTensor does not implement rounded division")
            if is_poly(a) and not is_poly(b):
                return wrap(c / b for c in a.coeffs)
            return wrap(conv(lift(a), poly_reciprocal(lift(b))))

        if func is aten.reciprocal.default:
            return wrap(poly_reciprocal(lift(args[0])))

        if func is aten.mm.default:
            return wrap(matmul(args[0], args[1]))

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
                for coeff in a.coeffs:
                    coeff.add_(b, alpha=alpha)
                return a
            A, B = lift(a), lift(b)
            for k in range(D + 1):
                A[k].add_(B[k], alpha=alpha)
            return a

        if func in (aten.sub_.Tensor, aten.sub_.Scalar):
            a, b = args[:2]
            alpha = kwargs.get("alpha", 1)
            if not is_poly(b):
                for coeff in a.coeffs:
                    coeff.sub_(b, alpha=alpha)
                return a
            A, B = lift(a), lift(b)
            for k in range(D + 1):
                A[k].sub_(B[k], alpha=alpha)
            return a

        if func in (aten.mul_.Tensor, aten.mul_.Scalar):
            a, b = args[:2]
            if not is_poly(b):
                for coeff in a.coeffs:
                    coeff.mul_(b)
                return a
            out = conv(lift(a), lift(b))
            for dst, src in zip(a.coeffs, out):
                dst.copy_(src)
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
            out = conv(lift(a), poly_reciprocal(lift(b)))
            for dst, src in zip(a.coeffs, out):
                dst.copy_(src)
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
                for k in range(D + 1):
                    X[k].copy_(torch.where(plain(mask), V[k], X[k]))
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
            return wrap(poly_exp(lift(args[0])))

        if func is aten.log.default:
            return wrap(poly_log(lift(args[0])))

        if func is aten.sqrt.default:
            return wrap(poly_sqrt(lift(args[0])))

        if func is aten.rsqrt.default:
            return wrap(poly_rsqrt(lift(args[0])))

        if func is aten.logsumexp.default:
            x = args[0]
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            return wrap(poly_logsumexp(lift(x), dim, keepdim))

        if func is aten.sigmoid.default:
            return wrap(poly_sigmoid(lift(args[0])))

        if func is aten.tanh.default:
            return wrap(poly_tanh(lift(args[0]))[0])

        if func is aten.silu.default:
            X = lift(args[0])
            return wrap(conv(X, poly_sigmoid(X)))

        if func is aten.gelu.default:
            approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
            return wrap(poly_gelu(lift(args[0]), approximate))

        if func is aten._log_softmax.default:
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            return wrap(poly_log_softmax(lift(args[0]), dim))

        if func is aten.log_softmax.int:
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
            out = poly_log_softmax(lift(args[0]), dim)
            if dtype is not None:
                out = [o.to(dtype=dtype) for o in out]
            return wrap(out)

        if func is aten._softmax.default:
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            half_to_float = args[2] if len(args) > 2 else kwargs.get("half_to_float", False)
            if half_to_float:
                raise NotImplementedError("PolyTensor does not implement half_to_float softmax")
            return wrap(poly_softmax(lift(args[0]), dim))

        if func is aten.softmax.int:
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
            out = poly_softmax(lift(args[0]), dim)
            if dtype is not None:
                out = [o.to(dtype=dtype) for o in out]
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
            grad_output = args[0]
            return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

        if func is aten._log_softmax_backward_data.default:
            grad_output, output, dim = args[:3]
            input_dtype = args[3] if len(args) > 3 else kwargs.get("input_dtype")
            G = lift(grad_output)
            O = lift(output)

            S = [g.sum(dim=dim, keepdim=True) for g in G]
            E = poly_exp(O)

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
            GO = conv(G, O)
            S = [go.sum(dim=dim, keepdim=True) for go in GO]
            out = conv(O, [G[k] - S[k] for k in range(D + 1)])
            return wrap(y.to(dtype=input_dtype) if input_dtype is not None else y for y in out)

        if func is aten.sigmoid_backward.default:
            G = lift(args[0])
            O = lift(args[1])
            one_minus_o = [1 - O[0]] + [-O[k] for k in range(1, D + 1)]
            return wrap(conv(G, conv(O, one_minus_o)))

        if func is aten.tanh_backward.default:
            G = lift(args[0])
            O = lift(args[1])
            O2 = conv(O, O)
            one_minus_o2 = [1 - O2[0]] + [-O2[k] for k in range(1, D + 1)]
            return wrap(conv(G, one_minus_o2))

        if func is aten.silu_backward.default:
            G = lift(args[0])
            X = lift(args[1])
            S = poly_sigmoid(X)
            one_minus_s = [1 - S[0]] + [-S[k] for k in range(1, D + 1)]
            x_s_one_minus_s = conv(X, conv(S, one_minus_s))
            derivative = [S[k] + x_s_one_minus_s[k] for k in range(D + 1)]
            return wrap(conv(G, derivative))

        if func is aten.gelu_backward.default:
            grad_output, x = args[:2]
            approximate = args[2] if len(args) > 2 else kwargs.get("approximate", "none")
            return wrap(conv(lift(grad_output), poly_gelu_grad(lift(x), approximate)))

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
            return PolyTensor((torch.ones_like(x0),) + tuple(torch.zeros_like(x0) for _ in range(D)))

        if func is aten.zeros_like.default:
            return wrap(torch.zeros_like(c, *args[1:], **kwargs) for c in lift(args[0]))

        if func is aten.empty_like.default:
            return wrap(torch.empty_like(c, *args[1:], **kwargs) for c in lift(args[0]))

        if func is aten.full_like.default:
            call_kwargs = dict(kwargs)
            fill_value = args[1] if len(args) > 1 else call_kwargs.pop("fill_value")
            return wrap(
                torch.full_like(c, fill_value, *args[2:], **call_kwargs)
                for c in lift(args[0])
            )

        if func is aten.new_empty_strided.default:
            return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

        if func is aten.detach.default:
            return PolyTensor(tuple(c.detach() for c in lift(args[0])))

        if func is aten.detach_.default:
            x = args[0]
            x._set_coeffs(c.detach() for c in lift(x))
            return x

        if func is aten.alias.default:
            return correct(PolyTensor(tuple(func(c) for c in lift(args[0]))))

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
            return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

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
            x._set_coeffs(X[k].copy_(S[k]) for k in range(D + 1))
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
