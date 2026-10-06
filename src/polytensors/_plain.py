"""Plain-range coefficient arithmetic.

``PlainTensor`` implements the subset of the ``ScaledTensor`` interface used by
the arithmetic and shape rules, on ordinary tensors with no private exponent.
Inside ``PolyTensor.plain_range()`` the rules use it instead of
``ScaledTensor``: results are those of ordinary floating-point arithmetic in
the coefficient dtype, without the extended exponent range, and much faster.
"""

import contextvars

import torch


_plain_range = contextvars.ContextVar("polytensor_plain_range", default=False)


def plain_range_enabled():
    return _plain_range.get()


class PlainTensor:
    """An ordinary tensor with the ScaledTensor arithmetic interface."""

    __slots__ = ("mantissa",)
    exponent = None
    _parts = None

    def __init__(self, mantissa, exponent=None):
        if exponent is not None and not (isinstance(exponent, (int, float)) and exponent == 0):
            raise ValueError("PlainTensor has no exponent")
        self.mantissa = mantissa

    @classmethod
    def from_tensor(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.get_default_dtype())
        return cls(value)

    def _coerce(self, other):
        if isinstance(other, PlainTensor):
            return other.mantissa
        return other

    @property
    def real(self):
        return PlainTensor(self.mantissa.real if self.mantissa.is_complex() else self.mantissa)

    @property
    def imag(self):
        return PlainTensor(self.mantissa.imag if self.mantissa.is_complex() else torch.zeros_like(self.mantissa))

    def conj(self):
        return PlainTensor(self.mantissa.conj())

    def map_tensor(self, function):
        return PlainTensor(function(self.mantissa))

    def transpose(self, dim0, dim1):
        return PlainTensor(self.mantissa.transpose(dim0, dim1))

    def unsqueeze(self, dim):
        return PlainTensor(self.mantissa.unsqueeze(dim))

    def squeeze(self, dim):
        return PlainTensor(self.mantissa.squeeze(dim))

    def masked_fill(self, mask, value=0):
        other = self._coerce(value)
        if not isinstance(other, torch.Tensor):
            other = torch.as_tensor(other, dtype=self.mantissa.dtype, device=self.mantissa.device)
        dtype = torch.promote_types(self.mantissa.dtype, other.dtype)
        return PlainTensor(torch.where(mask, other.to(dtype), self.mantissa.to(dtype)))

    def matmul(self, other):
        a, b = self.mantissa, self._coerce(other)
        dtype = torch.promote_types(a.dtype, b.dtype)
        return PlainTensor(torch.matmul(a.to(dtype), b.to(dtype)))

    __matmul__ = matmul

    def __neg__(self):
        return PlainTensor(-self.mantissa)

    def __add__(self, other):
        return PlainTensor(self.mantissa + self._coerce(other))

    __radd__ = __add__

    def __sub__(self, other):
        return PlainTensor(self.mantissa - self._coerce(other))

    def __rsub__(self, other):
        return PlainTensor(self._coerce(other) - self.mantissa)

    def __mul__(self, other):
        return PlainTensor(self.mantissa * self._coerce(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return PlainTensor(self.mantissa / self._coerce(other))

    def __rtruediv__(self, other):
        return PlainTensor(self._coerce(other) / self.mantissa)

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return PlainTensor(self.mantissa.sum())
        return PlainTensor(self.mantissa.sum(dim=dim, keepdim=keepdim))

    def to_tensor(self, dtype=None):
        return self.mantissa if dtype is None else self.mantissa.to(dtype=dtype)


def plain_polynomial_product(left, right, *, matrix=False):
    """Coefficients of the truncated product of two coefficient sequences."""
    count = len(left)
    result = []
    for k in range(count):
        total = None
        for i in range(k + 1):
            term = left[i].matmul(right[k - i]) if matrix else left[i] * right[k - i]
            total = term if total is None else total + term
        result.append(total)
    return tuple(result)


def plain_concatenate(items, operation, dim):
    mantissas = [item.mantissa for item in items]
    dtype = mantissas[0].dtype
    for m in mantissas[1:]:
        dtype = torch.promote_types(dtype, m.dtype)
    return PlainTensor(operation([m.to(dtype) for m in mantissas], dim=dim))


# ---------------------------------------------------------------------------
# Plain-range nonlinear rules. Each function takes and returns coefficient
# sequences of ordinary tensors (coefficient 0 real for a real base).

def _conv(a, b, degree):
    return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(degree + 1)]


def plain_exp(X):
    """exp along a series: Y' = X' Y gives Y_k = sum_i (i/k) X_i Y_{k-i}."""
    degree = len(X) - 1
    Y = [torch.exp(X[0])]
    for k in range(1, degree + 1):
        Y.append(sum((i / k) * X[i] * Y[k - i] for i in range(1, k + 1)))
    return Y


def plain_log(X):
    degree = len(X) - 1
    Y = [torch.log(X[0])]
    normalized = [None] + [x / X[0] for x in X[1:]]
    for k in range(1, degree + 1):
        s = normalized[k]
        for i in range(1, k):
            s = s - (i / k) * Y[i] * normalized[k - i]
        Y.append(s)
    return Y


def _compose(derivatives, X):
    """f(X(t)) given f^{(j)}(X_0) for j = 0..degree (Faa di Bruno via powers of h)."""
    degree = len(X) - 1
    h = [torch.zeros_like(X[1]) if degree else None] + list(X[1:])
    out = [derivatives[0]] + [0 for _ in range(degree)]
    power = h
    factorial = 1.0
    for j in range(1, degree + 1):
        factorial *= j
        coefficient = derivatives[j] / factorial
        for k in range(j, degree + 1):
            out[k] = out[k] + coefficient * power[k]
        if j < degree:
            power = _conv(power, h, degree)
    return [o if isinstance(o, torch.Tensor) else torch.zeros_like(X[1]) for o in out]


def _hermite(x, count):
    """Probabilists' Hermite polynomials He_0..He_{count-1} at x."""
    values = [torch.ones_like(x), x]
    for n in range(1, count - 1):
        values.append(x * values[n] - n * values[n - 1])
    return values[:count]


def _gelu_derivatives(x, count, offset=0):
    """f^{(n)} for f(x) = x Phi(x), n = offset .. offset+count-1."""
    import math
    phi = torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    he = _hermite(x, offset + count + 1)

    def Phi_derivative(m):           # Phi^{(m)}, m >= 0
        if m == 0:
            return Phi
        sign = -1.0 if (m - 1) % 2 else 1.0
        return sign * he[m - 1] * phi

    out = []
    for n in range(offset, offset + count):
        if n == 0:
            out.append(x * Phi)
        else:
            out.append(x * Phi_derivative(n) + n * Phi_derivative(n - 1))
    return out


def _output_polynomial_derivatives(y, count, offset, derivative_of_output):
    """Derivatives of a function whose derivative is a polynomial g(y) of its output y.

    P_0(y) = y and P_{n+1}(y) = P_n'(y) g(y); returns P_n(y) for n = offset .. offset+count-1.
    ``derivative_of_output`` gives the coefficients of g in increasing powers.
    """
    polynomial = [0.0, 1.0]
    polynomials = [polynomial]
    for _ in range(offset + count - 1):
        derivative = [i * c for i, c in enumerate(polynomial)][1:] or [0.0]
        product = [0.0] * (len(derivative) + len(derivative_of_output) - 1)
        for i, a in enumerate(derivative):
            for j, b in enumerate(derivative_of_output):
                product[i + j] += a * b
        polynomial = product
        polynomials.append(polynomial)
    out = []
    for p in polynomials[offset:offset + count]:
        value = torch.zeros_like(y)
        for c in reversed(p):
            value = value * y + c
        out.append(value)
    return out


def activation_derivatives(kind, x, count, offset=0):
    if kind == "gelu":
        return _gelu_derivatives(x, count, offset)
    if kind == "tanh":
        return _output_polynomial_derivatives(torch.tanh(x), count, offset, [1.0, 0.0, -1.0])
    if kind == "sigmoid":
        return _output_polynomial_derivatives(torch.sigmoid(x), count, offset, [0.0, 1.0, -1.0])
    raise NotImplementedError(kind)


def plain_activation(X, kind, derivative_order=0):
    degree = len(X) - 1
    derivatives = activation_derivatives(kind, X[0], degree + 1, offset=derivative_order)
    return _compose(derivatives, X)


def plain_softmax_family(X, dim):
    """Returns (softmax, log_softmax, logsumexp keepdim) series along ``dim``."""
    lse0 = torch.logsumexp(X[0], dim=dim, keepdim=True)
    centered = [X[0] - lse0] + list(X[1:])
    E = plain_exp(centered)
    Z = [e.sum(dim=dim, keepdim=True) for e in E]
    log_z = plain_log(Z)
    log_softmax = [torch.log_softmax(X[0], dim=dim)] + [c - l for c, l in zip(centered[1:], log_z[1:])]
    softmax = plain_exp(log_softmax)
    softmax[0] = torch.softmax(X[0], dim=dim)
    lse = [lse0] + list(log_z[1:])
    return softmax, log_softmax, lse


def plain_rule(func, args, kwargs, lift, wrap):
    """Plain-range rules for nonlinear operators; NotImplemented if not handled."""
    aten = torch.ops.aten
    if func is aten.exp.default:
        return wrap(plain_exp(lift(args[0])))
    if func is aten.log.default:
        return wrap(plain_log(lift(args[0])))
    if func in (aten.tanh.default, aten.sigmoid.default):
        return wrap(plain_activation(lift(args[0]), "tanh" if func is aten.tanh.default else "sigmoid"))
    if func is aten.gelu.default:
        approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
        if approximate != "none":
            return NotImplemented
        return wrap(plain_activation(lift(args[0]), "gelu"))
    if func is aten.gelu_backward.default:
        approximate = args[2] if len(args) > 2 else kwargs.get("approximate", "none")
        if approximate != "none":
            return NotImplemented
        G, X = lift(args[0]), lift(args[1])
        return wrap(_conv(G, plain_activation(X, "gelu", derivative_order=1), len(X) - 1))
    if func is aten.tanh_backward.default:
        G, Y = lift(args[0]), lift(args[1])
        degree = len(Y) - 1
        one_minus = [1 - y if k == 0 else -y for k, y in enumerate(_conv(Y, Y, degree))]
        return wrap(_conv(G, one_minus, degree))
    if func is aten.sigmoid_backward.default:
        G, Y = lift(args[0]), lift(args[1])
        degree = len(Y) - 1
        one_minus = [1 - y if k == 0 else -y for k, y in enumerate(Y)]
        return wrap(_conv(G, _conv(Y, one_minus, degree), degree))
    if func in (aten._softmax.default, aten._log_softmax.default):
        dim = args[1]
        half_to_float = args[2] if len(args) > 2 else False
        if half_to_float:
            return NotImplemented
        softmax, log_softmax, _ = plain_softmax_family(lift(args[0]), dim)
        return wrap(softmax if func is aten._softmax.default else log_softmax)
    if func is aten.logsumexp.default:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if isinstance(dim, (tuple, list)):
            if len(dim) != 1:
                return NotImplemented
            dim = dim[0]
        _, _, lse = plain_softmax_family(lift(args[0]), dim)
        return wrap(lse if keepdim else [c.squeeze(dim) for c in lse])
    if func is aten._softmax_backward_data.default:
        G, Y = lift(args[0]), lift(args[1])
        dim = args[2]
        degree = len(Y) - 1
        inner = [s.sum(dim=dim, keepdim=True) for s in _conv(G, Y, degree)]
        return wrap(_conv(Y, [g - i for g, i in zip(G, inner)], degree))
    if func is aten._log_softmax_backward_data.default:
        G, Y = lift(args[0]), lift(args[1])
        dim = args[2]
        degree = len(Y) - 1
        probabilities = plain_exp(Y)
        total = [g.sum(dim=dim, keepdim=True) for g in G]
        return wrap([g - p for g, p in zip(G, _conv(probabilities, total, degree))])
    return NotImplemented
