"""Analytic adjoints for scaled matrix products and polynomial products."""

import torch

from ._scaled import ScaledTensor as S


def _parts(value):
    return (value.real, value.imag) if value.mantissa.is_complex() else (value,)


def _restore(mantissas, powers):
    parts = tuple(S(mantissa, power) for mantissa, power in zip(mantissas, powers))
    return parts[0] if len(parts) == 1 else S._from_parts(*parts)


def _restore_groups(mantissas, layout):
    groups = []
    position = 0
    for group_layout in layout:
        group = []
        for powers in group_layout:
            group.append(_restore(mantissas[position:position + len(powers)], powers))
            position += len(powers)
        groups.append(tuple(group))
    return tuple(groups)


def _sum_to_size(value, shape):
    current = value.mantissa.shape
    leading = len(current) - len(shape)
    axes = tuple(range(leading)) + tuple(i + leading for i, size in enumerate(shape)
                                        if size == 1 and current[i + leading] != 1)
    if axes:
        value = value.sum(axes, keepdim=True)
    return value.map_tensor(lambda tensor: tensor.reshape(shape))


def _convolution(left, right, matrix):
    multiply = (lambda a, b: a.matmul(b)) if matrix else (lambda a, b: a * b)
    return tuple(sum(multiply(left[i], right[k - i]) for i in range(k + 1)) for k in range(len(left)))


def _matrix_adjoint(gradient, left, right, *, to_left):
    left_vector = left.mantissa.ndim == 1
    right_vector = right.mantissa.ndim == 1
    left_matrix = left.unsqueeze(0) if left_vector else left
    right_matrix = right.unsqueeze(-1) if right_vector else right
    if right_vector:
        gradient = gradient.unsqueeze(-1)
    if left_vector:
        gradient = gradient.unsqueeze(-2)
    if to_left:
        result = gradient.matmul(right_matrix.conj().transpose(-2, -1))
        return result.squeeze(-2) if left_vector else result
    result = left_matrix.conj().transpose(-2, -1).matmul(gradient)
    return result.squeeze(-1) if right_vector else result


class _ScaledProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, input_layout, output_layout, *mantissas):
        ctx.matrix = matrix
        ctx.input_layout = input_layout
        ctx.save_for_backward(*mantissas)
        ctx.set_materialize_grads(False)
        left, right = _restore_groups(mantissas, input_layout)
        result = _convolution(left, right, matrix)
        outputs = []
        for coefficient in result:
            parts = _parts(coefficient)
            output_layout.append(tuple(part.exponent for part in parts))
            outputs.extend(part.mantissa for part in parts)
        ctx.output_layout = output_layout
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *mantissa_gradients):
        gradients = []
        position = 0
        for powers in ctx.output_layout:
            parts = mantissa_gradients[position:position + len(powers)]
            position += len(powers)
            if all(part is None for part in parts):
                gradients.append(None)
                continue
            template = next(part for part in parts if part is not None)
            gradients.append(_restore(
                tuple(torch.zeros_like(template) if part is None else part for part in parts),
                tuple(-power for power in powers),
            ))
        mantissas = ctx.saved_tensors
        left, right = _restore_groups(mantissas, ctx.input_layout)
        requested = ctx.needs_input_grad[3:]
        result = []
        position = 0
        # Aggregate all orders before converting an input-mantissa adjoint.
        # A zero polynomial product can have an enormous intermediate adjoint
        # whose product with another zero coefficient is nevertheless zero.
        for group, coefficients in enumerate((left, right)):
            for order, (coefficient, powers) in enumerate(zip(coefficients, ctx.input_layout[group])):
                needed = requested[position:position + len(powers)]
                contribution = None
                if any(needed):
                    for output_order in range(order, len(gradients)):
                        gradient = gradients[output_order]
                        if gradient is None:
                            continue
                        a, b = ((left[order], right[output_order - order]) if group == 0
                                else (left[output_order - order], right[order]))
                        term = (_matrix_adjoint(gradient, a, b, to_left=group == 0) if ctx.matrix
                                else gradient * (b if group == 0 else a).conj())
                        contribution = term if contribution is None else contribution + term
                    if contribution is not None:
                        contribution = _sum_to_size(contribution, coefficient.mantissa.shape)
                components = ((None,) * len(powers) if contribution is None else
                              (contribution.real, contribution.imag) if len(powers) == 2 else (contribution.real,))
                for component, power, need in zip(components, powers, needed):
                    if component is None or not need:
                        result.append(None)
                    else:
                        scaled = component * S(torch.ones_like(mantissas[position]), power)
                        result.append(scaled.to_tensor(dtype=mantissas[position].dtype))
                    position += 1
        return (None, None, None, *result)


def scaled_polynomial_product(left, right, *, matrix=False):
    """Convolve scaled coefficient tuples with an analytic coefficient adjoint."""
    if not torch.is_grad_enabled() or not any(c.mantissa.requires_grad for c in (*left, *right)):
        return _convolution(left, right, matrix)
    parts = tuple(tuple(_parts(coefficient) for coefficient in group) for group in (left, right))
    input_layout = tuple(tuple(tuple(part.exponent for part in coefficient) for coefficient in group) for group in parts)
    output_layout = []
    mantissas = _ScaledProduct.apply(
        matrix, input_layout, output_layout,
        *(part.mantissa for group in parts for coefficient in group for part in coefficient),
    )
    return _restore_groups(mantissas, (output_layout,))[0]


def scaled_polynomial_matmul(left, right):
    return scaled_polynomial_product(left, right, matrix=True)


def scaled_matmul(left, right):
    """Multiply scaled tensors while retaining finite coefficient adjoints."""
    return scaled_polynomial_matmul((left,), (right,))[0]
