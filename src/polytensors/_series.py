"""Truncated power-series arithmetic on ordinary PyTorch tensors."""

import math

import torch

from ._scaled import ScaledTensor


class SeriesOps:
    def __init__(self, degree):
        self.degree = degree

    def conv(self, a, b):
        out = []
        for k in range(self.degree + 1):
            s = a[0] * b[k]
            for i in range(1, k + 1):
                s = s + a[i] * b[k - i]
            out.append(s)
        return out

    def poly_exp(self, X):
        if torch.is_grad_enabled() and any(x.requires_grad for x in X):
            from ._exponential import exponential_series

            return exponential_series(X)
        Y = self._poly_exp_scaled(X)
        return [torch.exp(X[0]), *(y.to_tensor(dtype=X[-1].dtype) for y in Y[1:])]

    def _poly_exp_scaled(self, X):
        # Retain the exponential's exponent while generating coefficients.
        # exp(x0) may underflow even though exp(x0) * direction**k / k!
        # is representable, so a rounded coefficient zero cannot seed this
        # recurrence.
        Y = [ScaledTensor.from_log(X[0])]
        for k in range(1, self.degree + 1):
            s = (ScaledTensor.from_tensor(X[1]) / k) * Y[k - 1]
            for i in range(2, k + 1):
                s = s + (ScaledTensor.from_tensor(X[i]) * (i / k)) * Y[k - i]
            Y.append(s)
        return Y

    def poly_log(self, X):
        Y = [None] * (self.degree + 1)
        Y[0] = torch.log(X[0])
        normalized = [None] + [x / X[0] for x in X[1:]]

        for k in range(1, self.degree + 1):
            s = normalized[k]
            for i in range(1, k):
                s = s - (i / k) * Y[i] * normalized[k - i]
            Y[k] = s

        return Y

    def poly_reciprocal(self, X):
        Y = [None] * (self.degree + 1)
        Y[0] = torch.reciprocal(X[0])
        normalized = [None] + [x / X[0] for x in X[1:]]

        # X * Y = 1. Normalize before multiplying: forming Y[0]**2 can
        # overflow even for a constant input with a finite reciprocal.
        for k in range(1, self.degree + 1):
            s = normalized[1] * Y[k - 1]
            for i in range(2, k + 1):
                s = s + normalized[i] * Y[k - i]
            Y[k] = -s

        return Y

    def poly_sqrt(self, X):
        value = torch.sqrt(X[0])
        Y = [ScaledTensor.from_tensor(value)]
        denominator = 2 * Y[0]
        for k in range(1, self.degree + 1):
            residual = ScaledTensor.from_tensor(X[k])
            for i in range(1, k):
                residual = residual - Y[i] * Y[k - i]
            Y.append(residual / denominator)
        return [value, *(y.to_tensor(dtype=X[-1].dtype) for y in Y[1:])]

    def poly_rsqrt(self, X):
        Y = [None] * (self.degree + 1)
        Y[0] = torch.rsqrt(X[0])
        normalized = [None] + [x / X[0] for x in X[1:]]

        # 2 X Y' + X' Y = 0 avoids cubing a potentially large Y[0].
        # Coefficient zero still uses the native PyTorch rsqrt kernel.
        for k in range(1, self.degree + 1):
            s = ((2 * k - 1) / (2 * k)) * normalized[1] * Y[k - 1]
            for i in range(2, k + 1):
                s = s + ((2 * k - i) / (2 * k)) * normalized[i] * Y[k - i]
            Y[k] = -s

        return Y

    def poly_sin_cos(self, X):
        """Use sin(X)' = cos(X) X' and cos(X)' = -sin(X) X'."""
        sine = [torch.sin(X[0])]
        cosine = [torch.cos(X[0])]
        for k in range(1, self.degree + 1):
            s = (X[1] / k) * cosine[k - 1]
            c = -(X[1] / k) * sine[k - 1]
            for i in range(2, k + 1):
                derivative = (i / k) * X[i]
                s = s + derivative * cosine[k - i]
                c = c - derivative * sine[k - i]
            sine.append(s)
            cosine.append(c)
        return sine, cosine

    def poly_sin(self, X):
        return self.poly_sin_cos(X)[0]

    def poly_cos(self, X):
        return self.poly_sin_cos(X)[1]

    def poly_softmax_and_logsumexp(self,
        X,
        dim,
        *,
        probability_zero,
        logsumexp_zero,
        log_probability_zero=None,
        _return_scaled=False,
    ):
        """Generate normalized-exponential and normalizer coefficients.

        Remove a common shift from every input coefficient. Choose the
        coefficient at the largest base logit as that shift: tiny tail
        probabilities then multiply direction *differences* without being
        lost when subtracting two large means. Scaled arithmetic preserves
        their products even when the native probabilities round to zero.

        Coefficient recurrences divide by the order before multiplying.
        ``log_probability_zero`` requests log-softmax as the second result;
        its higher coefficients use centered quantities directly.
        """
        if not _return_scaled and torch.is_grad_enabled() and any(x.requires_grad for x in X):
            from ._normalization import normalization_series

            return normalization_series(X, dim, log_probability_zero is not None)
        if _return_scaled and torch.is_grad_enabled() and X[0].requires_grad:
            # A later reverse derivative must also preserve the complement
            # of a base probability rounded to one. Native log-softmax's
            # backward only sees its rounded output, so reuse our analytic
            # rule for this coefficient-zero normalization as well.
            from ._normalization import normalization_series

            log_probability = normalization_series((X[0],), dim, True)[1][0]
        else:
            log_probability = torch.log_softmax(X[0], dim=dim)
        P = [ScaledTensor.from_log(log_probability)]
        source = [ScaledTensor.from_tensor(x) for x in X]
        centered = [None]
        normalizers = [None]
        L = [ScaledTensor.from_tensor(logsumexp_zero)]

        index = None if X[0].numel() == 0 else X[0].detach().argmax(dim=dim, keepdim=True)
        for coefficient, scaled in zip(X[1:], source[1:]):
            # The shift is algebraically arbitrary; its detached selection
            # need not become part of the coefficient autograd graph.
            offset = (
                coefficient.detach().sum(dim=dim, keepdim=True)
                if index is None else coefficient.detach().gather(dim, index)
            )
            centered.append(scaled - offset)

        # From L' = sum(P * X') and P' = P * (X' - L'). Store
        # coefficients rather than unscaled derivatives, so neither the
        # input nor a large intermediate is ever multiplied by the order.
        for k in range(1, self.degree + 1):
            normalizer = (P[0] * centered[k]).sum(dim=dim, keepdim=True)
            # Compute the uncentered P[0] term directly: adding the offset
            # back after averaging can cancel a small valid derivative.
            uncentered = (P[0] * source[k]).sum(dim=dim, keepdim=True)
            for i in range(1, k):
                term = (
                    P[i] * centered[k - i] * ((k - i) / k)
                ).sum(dim=dim, keepdim=True)
                normalizer = normalizer + term
                uncentered = uncentered + term
            normalizers.append(normalizer)
            L.append(uncentered)

            coefficient = P[0] * (centered[k] - normalizers[k])
            for i in range(1, k):
                coefficient = coefficient + (
                    P[i] * (centered[k - i] - normalizers[k - i]) * ((k - i) / k)
                )
            P.append(coefficient)

        if _return_scaled:
            if log_probability_zero is not None:
                L = [ScaledTensor.from_tensor(log_probability_zero)] + [
                    centered[k] - normalizers[k] for k in range(1, self.degree + 1)
                ]
            return P, L
        probabilities = [
            probability_zero,
            *(p.to_tensor(dtype=X[-1].dtype) for p in P[1:]),
        ]
        if log_probability_zero is not None:
            return probabilities, [
                log_probability_zero,
                *((centered[k] - normalizers[k]).to_tensor(dtype=X[-1].dtype)
                  for k in range(1, self.degree + 1)),
            ]
        return probabilities, [logsumexp_zero, *(l.to_tensor(dtype=X[-1].dtype) for l in L[1:])]

    def poly_log_softmax(self, X, dim):
        out_zero = torch.log_softmax(X[0], dim=dim)
        _, out = self.poly_softmax_and_logsumexp(
            X,
            dim,
            probability_zero=torch.exp(out_zero),
            logsumexp_zero=torch.logsumexp(X[0], dim=dim, keepdim=True),
            log_probability_zero=out_zero,
        )
        return out

    def poly_softmax(self, X, dim):
        P, _ = self.poly_softmax_and_logsumexp(
            X,
            dim,
            probability_zero=torch.softmax(X[0], dim=dim),
            logsumexp_zero=torch.logsumexp(X[0], dim=dim, keepdim=True),
        )
        return P

    def poly_logsumexp(self, X, dim, keepdim=False):
        if isinstance(dim, (tuple, list)):
            if len(dim) != 1:
                raise NotImplementedError("PolyTensor logsumexp supports one dimension")
            dim = dim[0]
        logsumexp_zero = torch.logsumexp(X[0], dim=dim, keepdim=True)
        probability_zero = torch.softmax(X[0], dim=dim)
        _, L = self.poly_softmax_and_logsumexp(
            X,
            dim,
            probability_zero=probability_zero,
            logsumexp_zero=logsumexp_zero,
        )
        if not keepdim:
            L = [l.squeeze(dim) for l in L]
        return L

    def poly_one(self, like):
        return [torch.ones_like(like)] + [torch.zeros_like(like) for _ in range(self.degree)]

    def poly_sigmoid(self, X):
        from ._activations import activation_series
        return activation_series(X, "sigmoid")

    def poly_sigmoid_grad(self, X):
        from ._activations import activation_series
        return activation_series(X, "sigmoid", derivative_order=1)

    def poly_tanh(self, X):
        from ._activations import activation_series
        return (activation_series(X, "tanh"),
                activation_series(X, "tanh", derivative_order=1))

    def poly_silu(self, X):
        from ._activations import activation_series
        return activation_series(X, "silu")

    def poly_silu_grad(self, X):
        from ._activations import activation_series
        return activation_series(X, "silu", derivative_order=1)

    def poly_gelu(self, X, approximate):
        from ._activations import activation_series
        if approximate not in ("none", "tanh"):
            raise NotImplementedError(f"PolyTensor does not implement GELU approximate={approximate!r}")
        return activation_series(X, "gelu" if approximate == "none" else "gelu_tanh")

    def poly_gelu_grad(self, X, approximate):
        from ._activations import activation_series
        if approximate not in ("none", "tanh"):
            raise NotImplementedError(f"PolyTensor does not implement GELU approximate={approximate!r}")
        return activation_series(X, "gelu" if approximate == "none" else "gelu_tanh", derivative_order=1)
