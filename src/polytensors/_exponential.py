"""Autograd for exponential coefficients before rounding small derivatives."""

import torch

from ._scaled import ScaledTensor


class _ExponentialSeries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *coefficients):
        from ._series import SeriesOps

        ctx.save_for_backward(*coefficients)
        ctx.set_materialize_grads(False)
        return tuple(SeriesOps(len(coefficients) - 1).poly_exp(coefficients))

    @staticmethod
    def backward(ctx, *gradients):
        from ._series import SeriesOps

        coefficients = ctx.saved_tensors
        # d exp(X(t))_k / d X_j = exp(X(t))_(k-j). Multiply the
        # upstream gradient before converting back to a normal tensor.
        derivatives = SeriesOps(len(coefficients) - 1)._poly_exp_scaled(coefficients)
        result = []
        for j, coefficient in enumerate(coefficients):
            gradient = None
            for k in range(j, len(coefficients)):
                if gradients[k] is None:
                    continue
                contribution = derivatives[k - j].conj() * ScaledTensor.from_tensor(gradients[k])
                gradient = contribution if gradient is None else gradient + contribution
            if gradient is not None:
                if not coefficient.is_complex():
                    gradient = gradient.real
                gradient = gradient.to_tensor(dtype=coefficient.dtype)
            result.append(gradient)
        return tuple(result)


def exponential_series(coefficients):
    return list(_ExponentialSeries.apply(*coefficients))
