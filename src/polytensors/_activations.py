"""Activation jets with scaled tails and analytic coefficient Jacobians.

The exponent of a tiny tail is retained while multiplying by directions.
Consequently a rounded constant value (for example sigmoid(x) == 1) does
not erase representable higher coefficients.
"""

import math

import torch

from ._scaled import ScaledTensor as S


def _zeros(like):
    return S.from_tensor(torch.zeros_like(like))


def _convolve(left, right, degree):
    return [
        sum((left[i] * right[k - i] for i in range(k + 1)), _zeros(left[0].mantissa))
        for k in range(degree + 1)
    ]


def _compose(derivatives, delta, degree):
    """Compose normalized scalar derivatives with a zero-constant series."""
    power = [S.from_tensor(torch.ones_like(delta[0].mantissa))]
    power.extend(_zeros(delta[0].mantissa) for _ in range(degree))
    result = [_zeros(delta[0].mantissa) for _ in range(degree + 1)]
    for n in range(degree + 1):
        for k in range(n, degree + 1):
            result[k] = result[k] + derivatives[n] * power[k]
        if n < degree:
            power = _convolve(power, delta, degree)
    return result


def _sigmoid_taylor(base, degree):
    # r is always on the decaying side of the exponential. In particular,
    # q[0] is not reconstructed from an already rounded sigmoid value.
    sign = torch.where(base.real >= 0, 1.0, -1.0)
    r = S.from_log(-sign * base)
    denominator = 1 + r
    tail = r / denominator
    root = (sign > 0).to(base.dtype) - sign * tail
    values = [root]
    if degree == 0:
        return values
    q = [r / (denominator * denominator)]
    for n in range(degree):
        if n:
            coefficient = values[n] * (1 - 2 * root)
            for i in range(1, n):
                coefficient = coefficient - values[i] * values[n - i]
            q.append(coefficient)
        values.append(q[n] / (n + 1))
    return values


def _tanh_taylor(base, degree):
    sigmoid = _sigmoid_taylor(2 * base, degree)
    # Chain rule: sigmoid(2(x+u)).
    values = [2 * sigmoid[0] - 1]
    multiplier = S.from_tensor(torch.full_like(base, 2))
    for n in range(1, degree + 1):
        multiplier = multiplier * 2
        values.append(sigmoid[n] * multiplier)
    return values


def _gelu_taylor(base, degree):
    sign = torch.where(base >= 0, 1.0, -1.0)
    density = S.from_log(-0.5 * base.square() - 0.5 * math.log(2 * math.pi))
    # erfcx is bounded on this nonnegative argument. Keeping its Gaussian
    # factor scaled preserves the negative CDF tail even after erfc underflows.
    tail = density * (math.sqrt(math.pi / 2) * torch.special.erfcx(sign * base / math.sqrt(2)))
    cdf = (sign > 0).to(base.dtype) - sign * tail
    values = [base * cdf]
    if degree == 0:
        return values
    weighted = [density, density * base]
    values.append(cdf + weighted[1])
    inverse_factorial = S.from_tensor(torch.ones_like(base))
    for n in range(2, degree + 1):
        weighted.append(base * weighted[n - 1] - (n - 1) * weighted[n - 2])
        inverse_factorial = inverse_factorial / n
        values.append((weighted[n - 2] - weighted[n]) * ((-1) ** n) * inverse_factorial)
    return values


def _scalar_taylor(base, degree, kind):
    if kind == "gelu":
        return _gelu_taylor(base, degree)
    if kind == "sigmoid":
        return _sigmoid_taylor(base, degree)
    if kind == "tanh":
        return _tanh_taylor(base, degree)
    if kind == "silu":
        sigmoid = _sigmoid_taylor(base, degree)
        return [base * sigmoid[0]] + [
            base * sigmoid[n] + sigmoid[n - 1] for n in range(1, degree + 1)
        ]
    if kind == "gelu_tanh":
        scale = 2 * math.sqrt(2 / math.pi)
        cubic = 0.044715
        argument = scale * base * (1 + cubic * base.square())
        sigmoid = _sigmoid_taylor(argument, degree)
        delta = [_zeros(base) for _ in range(degree + 1)]
        if degree >= 1:
            delta[1] = S.from_tensor(scale * (1 + 3 * cubic * base.square()))
        if degree >= 2:
            delta[2] = S.from_tensor(scale * 3 * cubic * base)
        if degree >= 3:
            delta[3] = S.from_tensor(torch.full_like(base, scale * cubic))
        composed = _compose(sigmoid, delta, degree)
        return [base * composed[0]] + [
            base * composed[n] + composed[n - 1] for n in range(1, degree + 1)
        ]
    raise ValueError(f"Unknown activation {kind!r}")


def _output_dtypes(coefficients):
    dtype = coefficients[0].dtype
    result = []
    for coefficient in coefficients:
        dtype = torch.promote_types(dtype, coefficient.dtype)
        result.append(dtype)
    return result


def _log_amplification(tensors, like):
    dtype = torch.float32 if like.device.type == "mps" else torch.float64
    log_magnitude = torch.zeros_like(like.real, dtype=dtype)
    count = 0
    for tensor in tensors:
        if tensor is None:
            continue
        if isinstance(tensor, S):
            # Incoming gradients may themselves lie outside the public dtype's
            # exponent range. Measure them without materializing a tensor (and
            # handle complex components separately to avoid an overflowing
            # absolute value).
            parts = (tensor.real, tensor.imag) if tensor.mantissa.is_complex() else (tensor,)
            for part in parts:
                component = part.mantissa.detach().abs().to(dtype).log()
                component = component + part.exponent.to(dtype) * math.log(2)
                log_magnitude = torch.maximum(log_magnitude, component)
        else:
            value = tensor.detach()
            component = torch.maximum(value.real.abs(), value.imag.abs()) if value.is_complex() else value.abs()
            log_magnitude = torch.maximum(log_magnitude, component.to(dtype).log())
        count += 1
    return log_magnitude + math.log(2 * max(1, count))


def _tail_mask(coefficients, kind, derivative_order, log_amplification=0):
    """Prove a complete tail jet rounds to zero before forming squares/cubes.

    For |t| <= 1/(4 D M), M=max(1, |X_1|, ..., |X_D|), the input
    displacement is at most 1/4. An additional radius 1/4 accounts for
    derivative_order scalar derivatives. Cauchy's coefficient estimate gives
    the factor exp(B) below; thus this threshold depends on every direction,
    the order, and the smallest representable output value.
    """
    base = coefficients[0]
    # Complex bases can approach poles of sigmoid/tanh; a bound based on the
    # real-line tail does not apply. Their native analytic semantics remain.
    if base.is_complex():
        return torch.zeros_like(base.real, dtype=torch.bool)
    degree = len(coefficients) - 1
    order = max(1, degree + derivative_order)
    working_dtype = torch.float32 if base.device.type == "mps" else torch.float64
    magnitude = torch.ones_like(base, dtype=working_dtype)
    for coefficient in coefficients[1:]:
        value = coefficient.detach()
        component = torch.maximum(value.real.abs(), value.imag.abs()) if value.is_complex() else value.abs()
        magnitude = torch.maximum(magnitude, component.to(working_dtype))
    real_dtype = torch.empty((), dtype=_output_dtypes(coefficients)[-1]).real.dtype
    info = torch.finfo(real_dtype)
    log_half_subnormal = math.log(info.tiny) + math.log(info.eps) - math.log(2)
    # Twice the largest real/imaginary component bounds complex magnitude
    # without an overflowing hypot at the largest representable components.
    bound = order * (math.log(8 * order) + magnitude.log()) + math.lgamma(order + 1) + log_amplification
    absolute = base.detach().abs().to(working_dtype)
    if kind in ("gelu", "gelu_tanh"):
        threshold = 0.5 + torch.sqrt(2 * (bound - log_half_subnormal + math.log(2) + 0.125))
        if kind == "gelu_tanh":
            # On the complex disk of radius 1/2 at a>=64, the cubic
            # sigmoid tail is smaller than the Gaussian bound used above.
            threshold = threshold.clamp_min(64)
    else:
        # For a>=2 all three residuals (sigmoid, tanh, x*sigmoid)
        # on that disk are bounded by 2 exp(1/2-a/2).
        threshold = (2 * (bound - log_half_subnormal + 2)).clamp_min(2)
    # Retain a scaled analytic graph even if the current output rounds to
    # zero: a subsequent reverse pass may multiply its derivative by a large
    # gradient. Use the proved tail shortcut only when raw square/cube
    # arithmetic would otherwise become unsafe in the working dtype.
    safe_cubic_magnitude = torch.finfo(working_dtype).max ** (1 / 3) / 4
    threshold = threshold.clamp_min(safe_cubic_magnitude)
    return torch.isfinite(absolute) & (absolute > threshold)


def _evaluate_scaled(coefficients, kind, derivative_order, log_amplification=0):
    degree = len(coefficients) - 1
    mask = _tail_mask(coefficients, kind, derivative_order, log_amplification)
    # Double mantissas reduce cancellation in scalar derivative polynomials;
    # the separate exponent supplies range beyond float64.
    real_work_dtype = torch.float32 if coefficients[0].device.type == "mps" else torch.float64
    complex_work_dtype = torch.complex64 if real_work_dtype == torch.float32 else torch.complex128
    work = [coefficient.to(complex_work_dtype if coefficient.is_complex() else real_work_dtype)
            for coefficient in coefficients]
    safe = [torch.where(mask, torch.zeros_like(coefficient), coefficient) for coefficient in work]
    derivatives = _scalar_taylor(safe[0], degree + derivative_order, kind)
    normalized = []
    for n in range(degree + 1):
        value = derivatives[n + derivative_order]
        for multiplier in range(n + 1, n + derivative_order + 1):
            value = value * multiplier
        normalized.append(value)
    delta = [_zeros(safe[0])] + [S.from_tensor(value) for value in safe[1:]]
    result = _compose(normalized, delta, degree)
    positive = work[0].real >= 0
    for n in range(degree + 1):
        if kind in ("gelu", "gelu_tanh", "silu"):
            if derivative_order == 0:
                asymptote = work[n] * positive
            elif derivative_order == 1 and n == 0:
                asymptote = positive.to(work[0].dtype)
            else:
                asymptote = torch.zeros_like(work[n])
        elif derivative_order == 0 and n == 0:
            asymptote = positive.to(work[0].dtype)
            if kind == "tanh":
                asymptote = 2 * asymptote - 1
        else:
            asymptote = torch.zeros_like(work[n])
        # The branches were sanitized before evaluation; multiplying the
        # finite scaled result by the mask does not conceal invalid arithmetic.
        result[n] = result[n] * (~mask).to(work[0].real.dtype) + S.from_tensor(torch.where(mask, asymptote, 0))
    return result


def _native_value(base, kind):
    if kind == "gelu":
        return torch.nn.functional.gelu(base, approximate="none")
    if kind == "gelu_tanh":
        return torch.nn.functional.gelu(base, approximate="tanh")
    if kind == "silu":
        return torch.nn.functional.silu(base)
    return getattr(torch, kind)(base)


class _ActivationSeries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kind, derivative_order, *coefficients):
        ctx.kind = kind
        ctx.derivative_order = derivative_order
        ctx.save_for_backward(*coefficients)
        ctx.set_materialize_grads(False)
        result = [
            value.to_tensor(dtype)
            for value, dtype in zip(
                _evaluate_scaled(coefficients, kind, derivative_order),
                _output_dtypes(coefficients),
            )
        ]
        if derivative_order == 0:
            result[0] = _native_value(coefficients[0], kind)
        return tuple(result)

    @staticmethod
    def backward(ctx, *grad_outputs):
        coefficients = ctx.saved_tensors
        # d [t^k] f(X(t)) / d X_j = [t^(k-j)] f'(X(t)).
        # Evaluate this Jacobian analytically: differentiating native GELU's
        # tanh backward or a saturated native activation would reintroduce the
        # very overflow/cancellation that the scaled forward avoids.
        derivatives = _evaluate_scaled(coefficients, ctx.kind, ctx.derivative_order + 1,
                                       _log_amplification(grad_outputs, coefficients[0]))
        result = []
        for j, coefficient in enumerate(coefficients):
            if not ctx.needs_input_grad[j + 2]:
                result.append(None)
                continue
            value = _zeros(coefficient)
            for k in range(j, len(coefficients)):
                gradient = grad_outputs[k]
                if gradient is not None:
                    derivative = derivatives[k - j]
                    value = value + S.from_tensor(gradient) * derivative.conj()
            gradient = value.to_tensor()
            if not coefficient.is_complex():
                gradient = gradient.real
            result.append(gradient.to(coefficient.dtype))
        return None, None, *result


def activation_series(coefficients, kind, derivative_order=0):
    """Return coefficients of f^(derivative_order)(X) for a supported activation."""
    return list(_ActivationSeries.apply(kind, derivative_order, *coefficients))


def activation_backward_series(coefficients, grad_coefficients, kind, *, _return_scaled=False):
    """Multiply an activation's derivative jet by an upstream gradient jet.

    Keep both the derivative and the product scaled: even the first derivative
    can round to zero before a large upstream gradient makes it representable.
    """
    derivatives = _evaluate_scaled(coefficients, kind, 1,
                                   _log_amplification(grad_coefficients, coefficients[0]))
    product = _convolve([S.from_tensor(value) for value in grad_coefficients], derivatives,
                        len(coefficients) - 1)
    if _return_scaled:
        return product
    dtype = coefficients[0].dtype
    result = []
    for derivative, coefficient, gradient in zip(product, coefficients, grad_coefficients):
        gradient_dtype = gradient.mantissa.dtype if isinstance(gradient, S) else gradient.dtype
        dtype = torch.promote_types(dtype, torch.promote_types(coefficient.dtype, gradient_dtype))
        result.append(derivative.to_tensor(dtype))
    return result
