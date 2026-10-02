"""Polynomial attention with a finite normalization path for masked rows."""

import math

import torch

from ._scaled import ScaledTensor
from ._series import SeriesOps


def _working_precision(coefficients):
    """Match attention's float32 accumulation for half/bfloat16 inputs."""
    if coefficients is None:
        return None
    return tuple(
        c if isinstance(c, ScaledTensor)
        else c.to(dtype=torch.float32) if c.dtype in (torch.float16, torch.bfloat16)
        else c.to(dtype=torch.complex64) if c.dtype == torch.complex32
        else c
        for c in coefficients
    )


def _output_precision(coefficients, dtype):
    def convert(c):
        target = dtype
        if c.is_complex() and not dtype.is_complex:
            target = {
                torch.float16: torch.complex32,
                torch.bfloat16: torch.complex64,
                torch.float32: torch.complex64,
                torch.float64: torch.complex128,
            }[dtype]
        return c.to(dtype=target)

    return tuple(convert(c) for c in coefficients)


def _matmul(left, right):
    def multiply(a, b):
        if isinstance(a, ScaledTensor) or isinstance(b, ScaledTensor):
            return ScaledTensor.from_tensor(a).matmul(b)
        dtype = torch.promote_types(a.dtype, b.dtype)
        return a.to(dtype=dtype) @ b.to(dtype=dtype)

    return tuple(
        sum((multiply(left[i], right[k - i]) for i in range(k + 1)))
        for k in range(len(left))
    )


def _transpose(coefficients):
    return tuple(c.transpose(-2, -1) for c in coefficients)


def _safe_scores(coefficients, dim=-1):
    excluded = torch.isneginf(coefficients[0])
    empty = excluded.all(dim=dim, keepdim=True)
    safe = (
        coefficients[0].masked_fill(empty, 0),
        *(c.masked_fill(excluded, 0) for c in coefficients[1:]),
    )
    return safe, excluded, empty


def safe_softmax(coefficients, dim=-1, *, scaled=False):
    """Return probabilities and log-normalizers, with empty rows set to zero.

    A -inf score denotes a structurally excluded key. Replace fully excluded
    rows before normalizing so both coefficient and reverse-mode derivatives
    avoid the undefined all--inf softmax. NaNs and +inf remain visible.
    """
    safe, excluded, empty = _safe_scores(coefficients, dim)
    series = SeriesOps(len(coefficients) - 1)
    probabilities, normalizers = series.poly_softmax_and_logsumexp(
        safe,
        dim,
        probability_zero=torch.softmax(safe[0], dim=dim),
        logsumexp_zero=torch.logsumexp(safe[0], dim=dim, keepdim=True),
        _return_scaled=scaled,
    )
    return (
        tuple(c.masked_fill(excluded, 0) for c in probabilities),
        tuple(c.masked_fill(empty, 0).squeeze(dim) for c in normalizers),
    )


def _attention_weights(query, key, attn_mask, is_causal, scale):
    scale = 1 / math.sqrt(query[0].shape[-1]) if scale is None else scale
    # Scaling before the dot product avoids overflowing an unscaled score
    # whose final scaled value is representable. Split the factor symmetrically
    # so neither input absorbs the entire scale.
    factor = math.sqrt(abs(scale))
    scaled_query = tuple(ScaledTensor.from_tensor(c) * factor for c in query)
    scaled_key = tuple(ScaledTensor.from_tensor(c) * math.copysign(factor, scale) for c in key)
    scores = tuple(c.to_tensor() for c in _matmul(scaled_query, _transpose(scaled_key)))
    excluded = torch.zeros(scores[0].shape[-2:], dtype=torch.bool, device=scores[0].device)
    if is_causal:
        excluded = ~torch.ones_like(excluded).tril()
    if attn_mask is not None:
        if attn_mask[0].dtype == torch.bool:
            excluded = excluded | ~attn_mask[0]
        else:
            # Treat additive -inf entries like a Boolean mask, including their
            # higher coefficients, rather than adding infinities to scores.
            mask_excluded = torch.isneginf(attn_mask[0])
            excluded = excluded | mask_excluded
            scores = tuple(
                score + mask.masked_fill(mask_excluded, 0)
                for score, mask in zip(scores, attn_mask)
            )
    scores = (
        scores[0].masked_fill(excluded, float("-inf")),
        *(c.masked_fill(excluded, 0) for c in scores[1:]),
    )
    probabilities, normalizers = safe_softmax(scores, scaled=True)
    return probabilities, normalizers, scaled_query, scaled_key, factor, scale, scores


def _evaluate(query, key, value, attn_mask, dropout_p, is_causal, scale):
    probabilities, normalizers, *_ = _attention_weights(query, key, attn_mask, is_causal, scale)
    multiplier = None
    if dropout_p:
        # One dropout mask belongs to the whole polynomial, not to each order.
        template = probabilities[0].mantissa
        base, mask = torch.ops.aten.native_dropout.default(torch.ones_like(template), dropout_p, True)
        multiplier = mask.to(dtype=base.dtype) / (1 - dropout_p) if dropout_p < 1 else mask.to(dtype=base.dtype)
        probabilities = tuple(c * multiplier for c in probabilities)
    return _matmul(probabilities, value), normalizers, multiplier


def _sum_to_size(value, shape):
    current = value.mantissa.shape
    leading = len(current) - len(shape)
    dimensions = tuple(range(leading)) + tuple(
        i + leading for i, size in enumerate(shape) if size == 1 and current[i + leading] != 1
    )
    if dimensions:
        value = value.sum(dim=dimensions, keepdim=True)
    return value.map_tensor(lambda tensor: tensor.reshape(shape))


def _backward_scaled(grad_output, grad_normalizer, query, key, value, attn_mask, is_causal, scale, multiplier=None):
    probabilities, _, scaled_query, scaled_key, factor, scale, scores = _attention_weights(
        query, key, attn_mask, is_causal, scale
    )
    safe_scores, excluded, _ = _safe_scores(scores)
    count = len(query)
    dropped = probabilities if multiplier is None else tuple(p * multiplier for p in probabilities)
    grad_value = _matmul(_transpose(dropped), grad_output)
    # Both operands may be large even though multiplication by a tiny
    # probability makes the score derivative representable.
    grad_probability = _matmul(
        tuple(ScaledTensor.from_tensor(g) for g in grad_output), _transpose(value)
    )
    if multiplier is not None:
        grad_probability = tuple(g * multiplier for g in grad_probability)
    index = safe_scores[0].detach().argmax(dim=-1, keepdim=True) if safe_scores[0].numel() else None
    centered = []
    for gradient in grad_probability:
        gradient = gradient.masked_fill(excluded, 0)
        offset = gradient.sum(dim=-1, keepdim=True) if index is None else gradient.map_tensor(
            lambda tensor: tensor.gather(-1, index)
        )
        centered.append(gradient - offset)
    means = []
    grad_score = []
    for k in range(count):
        mean = sum((probabilities[i] * centered[k - i]).sum(dim=-1, keepdim=True) for i in range(k + 1))
        means.append(mean)
        coefficient = sum(probabilities[i] * (centered[k - i] - means[k - i]) for i in range(k + 1))
        if grad_normalizer is not None:
            coefficient = coefficient + sum(
                probabilities[i] * ScaledTensor.from_tensor(grad_normalizer[k - i]).map_tensor(lambda t: t.unsqueeze(-1))
                for i in range(k + 1)
            )
        grad_score.append(coefficient.masked_fill(excluded, 0))
    grad_query = tuple(c * factor for c in _matmul(grad_score, scaled_key))
    grad_key = tuple(c * math.copysign(factor, scale) for c in _matmul(_transpose(grad_score), scaled_query))
    gradients = (grad_query, grad_key, grad_value)
    if attn_mask is not None and attn_mask[0].dtype != torch.bool:
        gradients += (tuple(grad_score),)
    return gradients


class _AttentionSeries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layout, count, has_mask, dropout_p, is_causal, scale, *coefficients):
        ctx.count, ctx.has_mask = count, has_mask
        ctx.is_causal, ctx.scale = is_causal, scale
        ctx.save_for_backward(*coefficients)
        ctx.set_materialize_grads(False)
        query, key, value = (coefficients[i * count:(i + 1) * count] for i in range(3))
        attn_mask = coefficients[3 * count:] if has_mask else None
        output, normalizer, ctx.multiplier = _evaluate(query, key, value, attn_mask, dropout_p, is_causal, scale)
        # Return normalized mantissas through the custom autograd boundary.
        # Detached scales stay outside it, so a later multiplication can use
        # a coefficient whose public tensor has already underflowed.
        mantissas = []
        for coefficient in (*output, *normalizer):
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
            scaled = tuple(ScaledTensor(torch.zeros_like(template) if part is None else part, -power)
                           for part, power in zip(parts, powers))
            gradients.append(scaled[0] if len(scaled) == 1 else ScaledTensor._from_parts(*scaled))
        count = ctx.count
        coefficients = ctx.saved_tensors
        groups = tuple(tuple(c.conj() for c in coefficients[i:i + count]) for i in range(0, len(coefficients), count))
        query, key, value = groups[:3]
        mask = groups[3] if ctx.has_mask else None
        result = [None] * len(coefficients)
        # Coefficient k depends on input coefficient j through coefficient
        # k-j of the ordinary attention Jacobian. Conjugating its input jet
        # gives PyTorch's complex adjoint convention.
        for order in range(count):
            output_gradient, normalizer_gradient = gradients[order], gradients[count + order]
            if output_gradient is None and normalizer_gradient is None:
                continue
            if output_gradient is None:
                shape = torch.broadcast_shapes(query[0].shape[:-2], key[0].shape[:-2], value[0].shape[:-2])
                output_gradient = value[0].new_zeros((*shape, query[0].shape[-2], value[0].shape[-1]))
            output_gradient = ScaledTensor.from_tensor(output_gradient)
            G = (output_gradient,) + tuple(torch.zeros_like(output_gradient.mantissa) for _ in range(order))
            H = None if normalizer_gradient is None else (normalizer_gradient,) + tuple(torch.zeros_like(normalizer_gradient.mantissa) for _ in range(order))
            jets = _backward_scaled(
                G, H, query[:order + 1], key[:order + 1], value[:order + 1],
                None if mask is None else mask[:order + 1],
                ctx.is_causal, ctx.scale, ctx.multiplier,
            )
            for group, jet in enumerate(jets):
                for input_order in range(order + 1):
                    index = group * count + input_order
                    contribution = _sum_to_size(jet[order - input_order], coefficients[index].shape)
                    result[index] = contribution if result[index] is None else result[index] + contribution
        converted = []
        for coefficient, gradient in zip(coefficients, result):
            if gradient is None:
                converted.append(None)
            else:
                if not coefficient.is_complex():
                    gradient = gradient.real
                converted.append(gradient.to_tensor(dtype=coefficient.dtype))
        return (None, None, None, None, None, None, *converted)


def attention(query, key, value, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, _return_scaled=False):
    if not 0 <= dropout_p <= 1:
        raise ValueError("attention dropout probability must be between 0 and 1")
    output_dtype = query[0].dtype
    query, key, value, attn_mask = tuple(_working_precision(c) for c in (query, key, value, attn_mask))
    coefficients = (*query, *key, *value, *(attn_mask or ()))
    count = len(query)
    if torch.is_grad_enabled() and any(c.requires_grad for c in coefficients):
        layout = []
        mantissas = _AttentionSeries.apply(layout, count, attn_mask is not None, dropout_p, is_causal, scale, *coefficients)
        result = []
        position = 0
        for powers in layout:
            parts = tuple(ScaledTensor(mantissa, power) for mantissa, power in zip(mantissas[position:], powers))
            position += len(powers)
            result.append(parts[0] if len(parts) == 1 else ScaledTensor._from_parts(*parts))
        scaled_output, scaled_normalizers = result[:count], result[count:]
    else:
        scaled_output, scaled_normalizers, _ = _evaluate(query, key, value, attn_mask, dropout_p, is_causal, scale)
    output = _output_precision(tuple(c.to_tensor() for c in scaled_output), output_dtype)
    normalizers = tuple(c.to_tensor() for c in scaled_normalizers)
    if _return_scaled:
        return output, normalizers, tuple(scaled_output), tuple(scaled_normalizers)
    return output, normalizers


def attention_backward(grad_output, query, key, value, *, attn_mask=None, is_causal=False, scale=None, _return_scaled=False):
    """Real-primal attention backward, continued into complex directions."""
    output_dtypes = tuple(c[0].dtype for c in (query, key, value))
    grad_output, query, key, value, attn_mask = tuple(
        _working_precision(c) for c in (grad_output, query, key, value, attn_mask)
    )
    gradients = _backward_scaled(grad_output, None, query, key, value, attn_mask, is_causal, scale)
    if _return_scaled:
        return gradients[:3]
    return tuple(_output_precision(tuple(c.to_tensor() for c in jet), dtype) for jet, dtype in zip(gradients, output_dtypes))
