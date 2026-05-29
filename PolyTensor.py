import math

import torch
import torch.nn.functional as F
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

    def flatten(self, start_dim=0, end_dim=-1):
        return PolyTensor(
            tuple(c.flatten(start_dim, end_dim).clone() for c in self.coeffs),
            requires_grad=self.requires_grad,
        )

    def transpose(self, dim0, dim1):
        return PolyTensor(
            tuple(c.transpose(dim0, dim1).clone() for c in self.coeffs),
            requires_grad=self.requires_grad,
        )

    def t(self):
        return PolyTensor(tuple(c.t().clone() for c in self.coeffs), requires_grad=self.requires_grad)

    def view(self, *shape):
        return PolyTensor(tuple(c.view(*shape).clone() for c in self.coeffs), requires_grad=self.requires_grad)

    def reshape(self, *shape):
        return PolyTensor(tuple(c.reshape(*shape).clone() for c in self.coeffs), requires_grad=self.requires_grad)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = getattr(func, "__name__", "")

        if func is F.linear or name == "linear":
            return _poly_linear(*args, **kwargs)

        if func is F.gelu or name == "gelu":
            approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
            return _poly_gelu(args[0], approximate)

        if name == "flatten":
            return PolyTensor.flatten(args[0], *args[1:], **kwargs)

        if name == "transpose":
            return PolyTensor.transpose(args[0], *args[1:], **kwargs)

        if name == "t":
            return PolyTensor.t(args[0])

        if name == "view":
            return PolyTensor.view(args[0], *args[1:], **kwargs)

        if name == "reshape":
            return PolyTensor.reshape(args[0], *args[1:], **kwargs)

        if func is torch.matmul and args[0].dim() == 2 and args[1].dim() == 2:
            return _poly_mm(args[0], args[1])

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

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

        def wrap(cs):
            return PolyTensor(
                tuple(c.clone() if isinstance(c, torch.Tensor) else c for c in cs),
                requires_grad=any(p.requires_grad for p in polys),
            )

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

        def bilinear(op, a, b):
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

        def poly_log_softmax(X, dim):
            M = X[0].max(dim=dim, keepdim=True).values
            Z = [X[0] - M] + list(X[1:])
            E = poly_exp(Z)
            S = [e.sum(dim=dim, keepdim=True) for e in E]
            L = poly_log(S)
            L[0] = L[0] + M
            return [X[k] - L[k] for k in range(D + 1)]

        def poly_one(like):
            return [torch.ones_like(like)] + [torch.zeros_like(like) for _ in range(D)]

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
                return wrap((a.coeffs[0] + alpha * b, *a.coeffs[1:]))
            if not is_poly(a) and is_poly(b):
                if alpha == 1:
                    return wrap((a + b.coeffs[0], *b.coeffs[1:]))
                return wrap((a + alpha * b.coeffs[0], *(alpha * c for c in b.coeffs[1:])))
            A, B = lift(a), lift(b)
            return wrap(A[k] + alpha * B[k] for k in range(D + 1))

        if func in (aten.sub.Tensor, aten.sub.Scalar):
            a, b = args[:2]
            alpha = kwargs.get("alpha", 1)
            if is_poly(a) and not is_poly(b):
                return wrap((a.coeffs[0] - alpha * b, *a.coeffs[1:]))
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

        if func is aten.mm.default:
            return _poly_mm(args[0], args[1])

        if func is aten.matmul.default:
            if args[0].dim() == 2 and args[1].dim() == 2:
                return _poly_mm(args[0], args[1])
            return correct(wrap(bilinear(torch.matmul, args[0], args[1])))

        if func is aten.bmm.default:
            return correct(wrap(bilinear(torch.bmm, args[0], args[1])))

        if func is aten.addmm.default:
            x, a, b = args[:3]
            beta = kwargs.get("beta", 1)
            alpha = kwargs.get("alpha", 1)
            X = lift(x)
            M = bilinear(torch.mm, a, b)
            return wrap(beta * X[k] + alpha * M[k] for k in range(D + 1))

        if func is aten.linear.default:
            x, weight = args[:2]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            Y = bilinear(lambda a, b: torch.matmul(a, b.transpose(-2, -1).clone()), x, weight)
            if bias is not None:
                B = lift(bias)
                Y = [Y[k] + B[k] for k in range(D + 1)]
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
                a._set_coeffs((a.coeffs[0] + alpha * b, *a.coeffs[1:]))
                return a
            A, B = lift(a), lift(b)
            a._set_coeffs(A[k] + alpha * B[k] for k in range(D + 1))
            return a

        if func in (aten.sub_.Tensor, aten.sub_.Scalar):
            a, b = args[:2]
            alpha = kwargs.get("alpha", 1)
            if not is_poly(b):
                a._set_coeffs((a.coeffs[0] - alpha * b, *a.coeffs[1:]))
                return a
            A, B = lift(a), lift(b)
            a._set_coeffs(A[k] - alpha * B[k] for k in range(D + 1))
            return a

        if func in (aten.mul_.Tensor, aten.mul_.Scalar):
            a, b = args[:2]
            if not is_poly(b):
                a._set_coeffs(c * b for c in a.coeffs)
                return a
            a._set_coeffs(conv(lift(a), lift(b)))
            return a

        if func is aten.exp.default:
            return wrap(poly_exp(lift(args[0])))

        if func is aten.log.default:
            return wrap(poly_log(lift(args[0])))

        if func is aten.gelu.default:
            approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
            return wrap(poly_gelu(lift(args[0]), approximate))

        if func is aten._log_softmax.default:
            dim = args[1] if len(args) > 1 else kwargs["dim"]
            return wrap(poly_log_softmax(lift(args[0]), dim))

        if func is aten.avg_pool2d.default:
            return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

        if func in (aten.sum.default, aten.sum.dim_IntList):
            return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

        if func in (aten.sum_to_size.default, aten._grad_sum_to_size.default):
            return wrap(func(c, *args[1:], **kwargs).clone() for c in lift(args[0]))

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

            L = poly_log_softmax(lift(x), 1)
            return wrap(
                aten.nll_loss_forward.default(c, plain(target), plain(weight), reduction, ignore_index)[0]
                for c in L
            )

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

        if func is aten.gelu_backward.default:
            grad_output, x = args[:2]
            approximate = args[2] if len(args) > 2 else kwargs.get("approximate", "none")
            return wrap(conv(lift(grad_output), poly_gelu_grad(lift(x), approximate)))

        if func is aten.isnan.default:
            out = torch.isnan(lift(args[0])[0])
            for c in lift(args[0])[1:]:
                out = out | torch.isnan(c)
            return out

        if func is aten.avg_pool2d_backward.default:
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

        if func is aten.new_empty_strided.default:
            return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

        if func is aten.detach.default:
            return PolyTensor(tuple(c.detach() for c in lift(args[0])))

        if func is aten.detach_.default:
            x = args[0]
            x._set_coeffs(c.detach() for c in lift(x))
            return x

        if func is aten.clone.default:
            x = args[0]
            return PolyTensor(tuple(c.clone(**kwargs) for c in lift(x)), requires_grad=x.requires_grad)

        if func is aten.zero_.default:
            x = args[0]
            x._set_coeffs(torch.zeros_like(c) for c in lift(x))
            return x

        if func is aten.copy_.default:
            x, src = args[:2]
            X, S = lift(x), lift(src)
            x._set_coeffs(X[k].copy_(S[k]) for k in range(D + 1))
            return x

        if func in (
            aten.t.default,
            aten.transpose.int,
            aten.view.default,
            aten.reshape.default,
            aten._unsafe_view.default,
            aten.flatten.using_ints,
            aten.as_strided.default,
        ):
            return wrap(func(c, *args[1:], **kwargs).clone() for c in lift(args[0]))

        if func is aten.expand.default:
            x, size = args[:2]
            return wrap(c.expand(size).clone() for c in lift(x))

        raise NotImplementedError(f"PolyTensor does not implement {func}")


def _lift_coeffs_for_degree(x, degree, like):
    if isinstance(x, PolyTensor):
        return x.coeffs
    if isinstance(x, torch.Tensor):
        return (x,) + tuple(torch.zeros_like(x) for _ in range(degree))
    return (x,) + tuple(torch.zeros_like(like) for _ in range(degree))


def _sum_terms(xs):
    s = xs[0]
    for x in xs[1:]:
        s = s + x
    return s


def _coeff_conv(A, B):
    degree = len(A) - 1
    out = []
    for k in range(degree + 1):
        out.append(_sum_terms([A[i] * B[k - i] for i in range(k + 1)]))
    return out


def _coeff_exp(X):
    degree = len(X) - 1
    Y = [None] * (degree + 1)
    Y[0] = torch.exp(X[0])
    for k in range(1, degree + 1):
        Y[k] = _sum_terms([i * X[i] * Y[k - i] for i in range(1, k + 1)]) / k
    return Y


def _coeff_tanh(X):
    degree = len(X) - 1
    Y = [None] * (degree + 1)
    Q = [None] * (degree + 1)
    Y[0] = torch.tanh(X[0])
    Q[0] = 1 - Y[0] * Y[0]
    for k in range(1, degree + 1):
        Y[k] = _sum_terms([i * X[i] * Q[k - i] for i in range(1, k + 1)]) / k
        y2 = _sum_terms([Y[i] * Y[k - i] for i in range(k + 1)])
        Q[k] = -y2
    return Y, Q


def _coeff_normal_cdf_and_pdf(X):
    degree = len(X) - 1
    X2 = _coeff_conv(X, X)
    phi = [x / math.sqrt(2 * math.pi) for x in _coeff_exp([-0.5 * x for x in X2])]
    cdf = [None] * (degree + 1)
    cdf[0] = 0.5 * (1 + torch.erf(X[0] / math.sqrt(2)))
    for k in range(1, degree + 1):
        cdf[k] = _sum_terms([i * X[i] * phi[k - i] for i in range(1, k + 1)]) / k
    return cdf, phi


def _coeff_gelu(X, approximate):
    degree = len(X) - 1
    if approximate == "none":
        cdf, _ = _coeff_normal_cdf_and_pdf(X)
        return _coeff_conv(X, cdf)
    if approximate == "tanh":
        one = [torch.ones_like(X[0])] + [torch.zeros_like(X[0]) for _ in range(degree)]
        X2 = _coeff_conv(X, X)
        X3 = _coeff_conv(X2, X)
        scale = math.sqrt(2 / math.pi)
        U = [scale * (X[k] + 0.044715 * X3[k]) for k in range(degree + 1)]
        T, _ = _coeff_tanh(U)
        return [0.5 * y for y in _coeff_conv(X, [one[k] + T[k] for k in range(degree + 1)])]
    raise NotImplementedError(f"PolyTensor does not implement GELU approximate={approximate!r}")


def _coeff_gelu_grad(X, approximate):
    degree = len(X) - 1
    if approximate == "none":
        cdf, phi = _coeff_normal_cdf_and_pdf(X)
        x_phi = _coeff_conv(X, phi)
        return [cdf[k] + x_phi[k] for k in range(degree + 1)]
    if approximate == "tanh":
        one = [torch.ones_like(X[0])] + [torch.zeros_like(X[0]) for _ in range(degree)]
        X2 = _coeff_conv(X, X)
        X3 = _coeff_conv(X2, X)
        scale = math.sqrt(2 / math.pi)
        U = [scale * (X[k] + 0.044715 * X3[k]) for k in range(degree + 1)]
        T, Q = _coeff_tanh(U)
        Udx = [scale * (one[k] + 3 * 0.044715 * X2[k]) for k in range(degree + 1)]
        x_qudx = _coeff_conv(X, _coeff_conv(Q, Udx))
        return [0.5 * (one[k] + T[k] + x_qudx[k]) for k in range(degree + 1)]
    raise NotImplementedError(f"PolyTensor does not implement GELU approximate={approximate!r}")


class _PolyGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, approximate):
        if not isinstance(x, PolyTensor):
            ctx.approximate = None
            return torch.nn.functional.gelu(x, approximate=approximate)

        ctx.approximate = approximate
        ctx.degree = x.degree
        ctx.save_for_backward(*x.coeffs)
        return PolyTensor(tuple(_coeff_gelu(x.coeffs, approximate)))

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.approximate is None:
            raise RuntimeError("_PolyGelu backward expected a PolyTensor input")

        X = ctx.saved_tensors
        G = _lift_coeffs_for_degree(
            grad_output,
            ctx.degree,
            grad_output.value if isinstance(grad_output, PolyTensor) else grad_output,
        )
        return PolyTensor(tuple(_coeff_conv(G, _coeff_gelu_grad(X, ctx.approximate)))), None


def _poly_gelu(x, approximate="none"):
    return _PolyGelu.apply(x, approximate)


class _PolyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        x_is_poly = isinstance(x, PolyTensor)
        weight_is_poly = isinstance(weight, PolyTensor)
        bias_is_poly = isinstance(bias, PolyTensor)
        polys = [p for p in (x, weight, bias) if isinstance(p, PolyTensor)]
        degree = polys[0].degree
        like = polys[0].value

        X = _lift_coeffs_for_degree(x, degree, like)
        W = _lift_coeffs_for_degree(weight, degree, like)
        B = None if bias is None else _lift_coeffs_for_degree(bias, degree, like)

        ctx.x_is_poly = x_is_poly
        ctx.weight_is_poly = weight_is_poly
        ctx.bias_is_poly = bias_is_poly
        ctx.has_bias = bias is not None
        ctx.degree = degree
        ctx.save_for_backward(*(X + W + (() if B is None else B)))

        out = []
        for k in range(degree + 1):
            y = _sum_terms([X[i].mm(W[k - i].transpose(-2, -1)) for i in range(k + 1)])
            if B is not None:
                y = y + B[k]
            out.append(y)
        return PolyTensor(tuple(out))

    @staticmethod
    def backward(ctx, grad_output):
        degree = ctx.degree
        saved = ctx.saved_tensors
        X = saved[: degree + 1]
        W = saved[degree + 1 : 2 * (degree + 1)]
        G = _lift_coeffs_for_degree(
            grad_output,
            degree,
            grad_output.value if isinstance(grad_output, PolyTensor) else grad_output,
        )

        grad_x = []
        grad_weight = []
        grad_bias = []
        for k in range(degree + 1):
            grad_x.append(_sum_terms([G[i].mm(W[k - i]) for i in range(k + 1)]))
            grad_weight.append(_sum_terms([G[i].transpose(-2, -1).mm(X[k - i]) for i in range(k + 1)]))
            grad_bias.append(G[k].sum(dim=tuple(range(G[k].dim() - 1))))

        if ctx.x_is_poly:
            grad_x = PolyTensor(tuple(grad_x))
        else:
            grad_x = grad_x[0]

        if ctx.weight_is_poly:
            grad_weight = PolyTensor(tuple(grad_weight))
        else:
            grad_weight = grad_weight[0]

        if ctx.has_bias:
            if ctx.bias_is_poly:
                grad_bias = PolyTensor(tuple(grad_bias))
            else:
                grad_bias = grad_bias[0]
        else:
            grad_bias = None

        return grad_x, grad_weight, grad_bias


def _poly_linear(x, weight, bias=None):
    if not any(isinstance(p, PolyTensor) for p in (x, weight, bias)):
        return F.linear(x, weight, bias)
    return _PolyLinear.apply(x, weight, bias)


class _PolyMm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a_is_poly = isinstance(a, PolyTensor)
        b_is_poly = isinstance(b, PolyTensor)
        degree = a.degree if a_is_poly else b.degree
        like = a.value if a_is_poly else b.value
        A = _lift_coeffs_for_degree(a, degree, like)
        B = _lift_coeffs_for_degree(b, degree, like)

        ctx.a_is_poly = a_is_poly
        ctx.b_is_poly = b_is_poly
        ctx.degree = degree
        ctx.save_for_backward(*(A + B))

        out = []
        for k in range(degree + 1):
            out.append(_sum_terms([A[i].mm(B[k - i]) for i in range(k + 1)]))
        return PolyTensor(tuple(out))

    @staticmethod
    def backward(ctx, grad_output):
        degree = ctx.degree
        saved = ctx.saved_tensors
        A = saved[: degree + 1]
        B = saved[degree + 1 :]
        G = _lift_coeffs_for_degree(grad_output, degree, grad_output.value if isinstance(grad_output, PolyTensor) else grad_output)

        grad_a = []
        grad_b = []
        for k in range(degree + 1):
            grad_a.append(_sum_terms([G[i].mm(B[k - i].transpose(-2, -1)) for i in range(k + 1)]))
            grad_b.append(_sum_terms([A[i].transpose(-2, -1).mm(G[k - i]) for i in range(k + 1)]))

        if ctx.a_is_poly:
            grad_a = PolyTensor(tuple(grad_a))
        else:
            grad_a = grad_a[0]

        if ctx.b_is_poly:
            grad_b = PolyTensor(tuple(grad_b))
        else:
            grad_b = grad_b[0]

        return grad_a, grad_b


def _poly_mm(a, b):
    return _PolyMm.apply(a, b)
