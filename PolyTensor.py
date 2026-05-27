import torch

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
            return PolyTensor(tuple(cs), requires_grad=any(p.requires_grad for p in polys))

        def is_poly(x):
            return isinstance(x, PolyTensor)

        def conv(a, b):
            out = []
            for k in range(D + 1):
                s = a[0] * b[k]
                for i in range(1, k + 1):
                    s = s + a[i] * b[k - i]
                out.append(s)
            return out

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
            x = args[0]
            X = lift(x)

            Y = [None] * (D + 1)
            Y[0] = torch.exp(X[0])

            for k in range(1, D + 1):
                s = X[1] * Y[k - 1]
                for i in range(2, k + 1):
                    s = s + i * X[i] * Y[k - i]
                Y[k] = s / k

            return wrap(Y)

        if func is aten.sum.default:
            return wrap(c.sum(**kwargs) for c in lift(args[0]))

        # Minimal autograd / optimizer plumbing.
        if func is aten.ones_like.default:
            x0 = lift(args[0])[0]
            return PolyTensor((torch.ones_like(x0),) + tuple(torch.zeros_like(x0) for _ in range(D)))

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

        if func is aten.expand.default:
            x, size = args[:2]
            return wrap(c.expand(size) for c in lift(x))

        raise NotImplementedError(f"PolyTensor does not implement {func}")
