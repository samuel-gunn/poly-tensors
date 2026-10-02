"""Truncated power-series arithmetic on ordinary PyTorch tensors."""

import math

import torch


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
        Y = [None] * (self.degree + 1)
        Y[0] = torch.exp(X[0])

        for k in range(1, self.degree + 1):
            s = (X[1] / k) * Y[k - 1]
            for i in range(2, k + 1):
                s = s + ((i / k) * X[i]) * Y[k - i]
            Y[k] = s

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
        Y = [None] * (self.degree + 1)
        Y[0] = torch.sqrt(X[0])

        for k in range(1, self.degree + 1):
            s = torch.zeros_like(X[0])
            for i in range(1, k):
                s = s + Y[i] * Y[k - i]
            Y[k] = (X[k] - s) / (2 * Y[0])

        return Y

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
    ):
        """Generate normalized-exponential and normalizer coefficients.

        Higher coefficients are generated from

            p' = p * (x' - sum(p * x')),
            logsumexp(x)' = sum(p * x'),

        rooted at the supplied coefficient-zero values. Softmax and
        logsumexp use the native softmax probabilities; log-softmax uses
        exp of its native output. All avoid subtracting a large common
        offset from a rounded logsumexp value.
        """

        P = [None] * (self.degree + 1)
        L = [None] * (self.degree + 1)
        centered_derivative = [None] * self.degree

        P[0] = probability_zero
        L[0] = logsumexp_zero

        # Coefficient m of sum(P(t) * X'(t)).  At iteration m all
        # required P[0:m+1] coefficients are already available.
        for m in range(self.degree):
            logsumexp_derivative = torch.zeros_like(L[0])
            for i in range(m + 1):
                logsumexp_derivative = logsumexp_derivative + (
                    P[i] * ((m - i + 1) * X[m - i + 1])
                ).sum(dim=dim, keepdim=True)

            L[m + 1] = logsumexp_derivative / (m + 1)
            centered_derivative[m] = (
                (m + 1) * X[m + 1] - logsumexp_derivative
            )

            softmax_derivative = torch.zeros_like(P[0])
            for i in range(m + 1):
                softmax_derivative = softmax_derivative + (
                    P[i] * centered_derivative[m - i]
                )
            P[m + 1] = softmax_derivative / (m + 1)

        return P, L

    def poly_log_softmax(self, X, dim):
        out_zero = torch.log_softmax(X[0], dim=dim)
        probability_zero = torch.exp(out_zero)
        _, L = self.poly_softmax_and_logsumexp(
            X,
            dim,
            probability_zero=probability_zero,
            logsumexp_zero=torch.logsumexp(X[0], dim=dim, keepdim=True),
        )
        return [
            out_zero,
            *(X[k] - L[k] for k in range(1, self.degree + 1)),
        ]

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
        # Generate coefficients from s' = s(1-s)x', rooted at the exact
        # native sigmoid value used by PyTorch's SiLU backward.
        Y = [None] * (self.degree + 1)
        Q = [None] * self.degree
        Y[0] = torch.sigmoid(X[0])

        for m in range(self.degree):
            square_coefficient = Y[0] * Y[m]
            for i in range(1, m + 1):
                square_coefficient = square_coefficient + Y[i] * Y[m - i]
            Q[m] = Y[m] - square_coefficient

            derivative_coefficient = Q[0] * ((m + 1) * X[m + 1])
            for i in range(1, m + 1):
                derivative_coefficient = derivative_coefficient + (
                    Q[i] * ((m - i + 1) * X[m - i + 1])
                )
            Y[m + 1] = derivative_coefficient / (m + 1)

        return Y

    def poly_tanh(self, X):
        Y = [None] * (self.degree + 1)
        Q = [None] * (self.degree + 1)
        Y[0] = torch.tanh(X[0])
        Q[0] = 1 - Y[0] * Y[0]

        for k in range(1, self.degree + 1):
            s = X[1] * Q[k - 1]
            for i in range(2, k + 1):
                s = s + i * X[i] * Q[k - i]
            Y[k] = s / k

            y2 = Y[0] * Y[k]
            for i in range(1, k + 1):
                y2 = y2 + Y[i] * Y[k - i]
            Q[k] = -y2

        return Y, Q

    def poly_normal_cdf_and_pdf(self, X):
        X2 = self.conv(X, X)
        phi = self.poly_exp([-0.5 * x for x in X2])
        phi = [x / math.sqrt(2 * math.pi) for x in phi]

        cdf = [None] * (self.degree + 1)
        cdf[0] = 0.5 * (1 + torch.erf(X[0] / math.sqrt(2)))

        for k in range(1, self.degree + 1):
            s = X[1] * phi[k - 1]
            for i in range(2, k + 1):
                s = s + i * X[i] * phi[k - i]
            cdf[k] = s / k

        return cdf, phi

    def poly_gelu(self, X, approximate):
        if approximate == "none":
            cdf, _ = self.poly_normal_cdf_and_pdf(X)
            return self.conv(X, cdf)

        if approximate == "tanh":
            one = self.poly_one(X[0])
            X2 = self.conv(X, X)
            X3 = self.conv(X2, X)
            scale = math.sqrt(2 / math.pi)
            U = [scale * (X[k] + 0.044715 * X3[k]) for k in range(self.degree + 1)]
            T, _ = self.poly_tanh(U)
            return [0.5 * y for y in self.conv(X, [one[k] + T[k] for k in range(self.degree + 1)])]

        raise NotImplementedError(f"PolyTensor does not implement GELU approximate={approximate!r}")

    def poly_gelu_grad(self, X, approximate):
        if approximate == "none":
            cdf, phi = self.poly_normal_cdf_and_pdf(X)
            x_phi = self.conv(X, phi)
            return [cdf[k] + x_phi[k] for k in range(self.degree + 1)]

        if approximate == "tanh":
            one = self.poly_one(X[0])
            X2 = self.conv(X, X)
            X3 = self.conv(X2, X)
            scale = math.sqrt(2 / math.pi)
            U = [scale * (X[k] + 0.044715 * X3[k]) for k in range(self.degree + 1)]
            T, Q = self.poly_tanh(U)
            Udx = [scale * (one[k] + 3 * 0.044715 * X2[k]) for k in range(self.degree + 1)]
            x_qudx = self.conv(X, self.conv(Q, Udx))
            return [0.5 * (one[k] + T[k] + x_qudx[k]) for k in range(self.degree + 1)]

        raise NotImplementedError(f"PolyTensor does not implement GELU approximate={approximate!r}")
