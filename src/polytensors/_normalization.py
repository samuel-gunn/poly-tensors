"""Coefficient-autograd rules for normalization with scaled intermediates."""

import torch

from ._scaled import ScaledTensor


def _evaluate(coefficients, dim, log_probabilities, *, scaled=False):
    from ._series import SeriesOps

    base = coefficients[0]
    log_probability = torch.log_softmax(base, dim=dim)
    return SeriesOps(len(coefficients) - 1).poly_softmax_and_logsumexp(
        coefficients,
        dim,
        probability_zero=torch.exp(log_probability) if log_probabilities else torch.softmax(base, dim=dim),
        logsumexp_zero=torch.logsumexp(base, dim=dim, keepdim=True),
        log_probability_zero=log_probability if log_probabilities else None,
        _return_scaled=scaled,
    )


class _NormalizationSeries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, log_probabilities, *coefficients):
        ctx.dim = dim
        ctx.log_probabilities = log_probabilities
        ctx.save_for_backward(*coefficients)
        ctx.set_materialize_grads(False)
        probability, normalizer = _evaluate(coefficients, dim, log_probabilities)
        return tuple(probability + normalizer)

    @staticmethod
    def backward(ctx, *gradients):
        return (None, None, *_coefficient_vjp(ctx.saved_tensors, gradients, ctx.dim, ctx.log_probabilities))


def _coefficient_vjp(coefficients, gradients, dim, log_probabilities):
    """Analytic coefficient adjoint; upstream coefficients may remain scaled."""
    count = len(coefficients)
    probabilities, _ = _evaluate(coefficients, dim, log_probabilities, scaled=True)
    probabilities = [p.conj() for p in probabilities]
    gradients_to_inputs = [None] * count
    base = coefficients[0]
    index = None if base.numel() == 0 else base.detach().argmax(dim=dim, keepdim=True)

    def accumulate(order, contribution):
        previous = gradients_to_inputs[order]
        gradients_to_inputs[order] = contribution if previous is None else previous + contribution

    # J_softmax(t)^T g = P(t) * (g - sum(P(t) * g)). Subtract
    # g at the largest-probability entry before computing the mean;
    # this also preserves the complement of a probability rounded to 1.
    for output_order, gradient in enumerate(gradients[:count]):
        if gradient is None:
            continue
        gradient = ScaledTensor.from_tensor(gradient)
        offset = (
            gradient.map_tensor(lambda t: t.detach()).sum(dim=dim, keepdim=True)
            if index is None else gradient.map_tensor(lambda t: t.detach().gather(dim, index))
        )
        centered = gradient - offset
        means = [(p * centered).sum(dim=dim, keepdim=True) for p in probabilities[:output_order + 1]]
        for input_order in range(output_order + 1):
            order = output_order - input_order
            contribution = probabilities[order] * centered
            for i in range(order + 1):
                contribution = contribution - probabilities[i] * means[order - i]
            accumulate(input_order, contribution)

    for output_order, gradient in enumerate(gradients[count:]):
        if gradient is None:
            continue
        scaled_gradient = ScaledTensor.from_tensor(gradient)
        if log_probabilities:
            total = scaled_gradient.sum(dim=dim, keepdim=True)
            for input_order in range(output_order + 1):
                order = output_order - input_order
                contribution = -(probabilities[order] * total)
                if order == 0:
                    contribution = contribution + scaled_gradient
                    if index is not None:
                        # At the largest probability, evaluate its
                        # complement from the remaining probabilities.
                        selected = torch.zeros_like(base).scatter(dim, index, 1)
                        remaining = 1 - selected
                        tail = (probabilities[0] * remaining).sum(dim=dim, keepdim=True)
                        others = (scaled_gradient * remaining).sum(dim=dim, keepdim=True)
                        stable = scaled_gradient * tail - probabilities[0] * others
                        contribution = contribution * remaining + stable * selected
                accumulate(input_order, contribution)
        else:
            for input_order in range(output_order + 1):
                accumulate(input_order, probabilities[output_order - input_order] * scaled_gradient)

    result = []
    for coefficient, gradient in zip(coefficients, gradients_to_inputs):
        if gradient is None:
            result.append(None)
        else:
            if not coefficient.is_complex():
                gradient = gradient.real
            result.append(gradient.to_tensor(dtype=coefficient.dtype))
    return result


def normalization_series(coefficients, dim, log_probabilities=False):
    outputs = _NormalizationSeries.apply(dim, log_probabilities, *coefficients)
    count = len(coefficients)
    return list(outputs[:count]), list(outputs[count:])


class _ScaledNormalizationSeries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layout, dim, log_probabilities, *coefficients):
        ctx.dim = dim
        ctx.log_probabilities = log_probabilities
        ctx.save_for_backward(*coefficients)
        ctx.set_materialize_grads(False)
        probabilities, normalizers = _evaluate(coefficients, dim, log_probabilities, scaled=True)
        mantissas = []
        for coefficient in (*probabilities, *normalizers):
            parts = (coefficient.real, coefficient.imag) if coefficient.mantissa.is_complex() else (coefficient,)
            layout.append(tuple(part.exponent for part in parts))
            mantissas.extend(part.mantissa for part in parts)
        ctx.layout = layout
        return tuple(mantissas)

    @staticmethod
    def backward(ctx, *mantissa_gradients):
        gradients = []
        position = 0
        for powers in ctx.layout:
            parts = mantissa_gradients[position:position + len(powers)]
            position += len(powers)
            if all(part is None for part in parts):
                gradients.append(None)
                continue
            template = next(part for part in parts if part is not None)
            scaled = tuple(
                ScaledTensor(torch.zeros_like(template) if part is None else part, -power)
                for part, power in zip(parts, powers)
            )
            gradients.append(scaled[0] if len(scaled) == 1 else ScaledTensor._from_parts(*scaled))
        return (None, None, None, *_coefficient_vjp(ctx.saved_tensors, gradients, ctx.dim, ctx.log_probabilities))


def normalization_scaled_series(coefficients, dim, log_probabilities=False):
    """Retain range across operations and an analytic coefficient-autograd rule.

    The autograd boundary exposes normalized mantissas, so even an upstream
    gradient larger than the tensor dtype can reach the analytic Jacobian as a
    scaled value. Real and imaginary parts have independent exponent ranges.
    """
    if not torch.is_grad_enabled() or not any(c.requires_grad for c in coefficients):
        return _evaluate(coefficients, dim, log_probabilities, scaled=True)
    layout = []
    mantissas = _ScaledNormalizationSeries.apply(layout, dim, log_probabilities, *coefficients)
    result = []
    position = 0
    for powers in layout:
        parts = tuple(ScaledTensor(mantissa, power) for mantissa, power in zip(mantissas[position:], powers))
        position += len(powers)
        result.append(parts[0] if len(parts) == 1 else ScaledTensor._from_parts(*parts))
    count = len(coefficients)
    return result[:count], result[count:]


def normalization_backward(gradients, coefficients, dim, *, log=False, _return_scaled=False):
    """Evaluate a softmax/log-softmax gradient jet before rounding its terms.

    The saved *inputs* are essential: an output probability rounded to zero
    cannot recover a representable product with a large upstream gradient.
    """
    probabilities, _ = _evaluate(coefficients, dim, log, scaled=True)
    base = coefficients[0]
    index = None if base.numel() == 0 else base.detach().argmax(dim=dim, keepdim=True)
    count = len(coefficients)
    G = [ScaledTensor.from_tensor(g) for g in gradients]
    output = []
    if log:
        totals = [g.sum(dim=dim, keepdim=True) for g in G]
        selected = None if index is None else torch.zeros_like(base).scatter(dim, index, 1)
        remaining = None if selected is None else 1 - selected
        tail = None if selected is None else (probabilities[0] * remaining).sum(dim=dim, keepdim=True)
        for k in range(count):
            value = G[k] - probabilities[0] * totals[k]
            if selected is not None:
                others = (G[k] * remaining).sum(dim=dim, keepdim=True)
                stable = G[k] * tail - probabilities[0] * others
                value = value * remaining + stable * selected
            for i in range(1, k + 1):
                value = value - probabilities[i] * totals[k - i]
            output.append(value)
    else:
        centered = []
        for scaled in G:
            offset = (scaled.map_tensor(lambda t: t.detach()).sum(dim=dim, keepdim=True)
                      if index is None else scaled.map_tensor(lambda t: t.detach().gather(dim, index)))
            centered.append(scaled - offset)
        means = []
        for k in range(count):
            mean = (probabilities[0] * centered[k]).sum(dim=dim, keepdim=True)
            for i in range(1, k + 1):
                mean = mean + (probabilities[i] * centered[k - i]).sum(dim=dim, keepdim=True)
            means.append(mean)
            value = probabilities[0] * (centered[k] - means[k])
            for i in range(1, k + 1):
                value = value + probabilities[i] * (centered[k - i] - means[k - i])
            output.append(value)
    if _return_scaled:
        return output
    return [
        value.to_tensor(dtype=torch.promote_types(coefficients[k].dtype, G[k].mantissa.dtype))
        for k, value in enumerate(output)
    ]
