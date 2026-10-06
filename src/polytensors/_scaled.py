"""Internal tensor arithmetic with a detached per-element binary exponent.

The representation is mantissa * 2**exponent. Exponents are numerical scales,
not differentiable variables. Keep products and sums in this representation
until a coefficient is ready to return in the requested PyTorch dtype.
"""

import math

import torch


def _safe_ldexp(value, power):
    """value * 2**power without materializing an out-of-range 2**power.

    Some PyTorch releases implement ``torch.ldexp`` as ``value * 2**power``.
    For large |power| the power of two itself overflows to infinity or
    underflows to zero, giving 0 * inf = NaN or a premature zero. Applying the
    exponent in steps that are each representable is exact (multiplication by
    a power of two) except for the final, genuine overflow or underflow.
    """
    step, steps = _ldexp_steps(value.dtype)
    power = power.clamp(-step * steps, step * steps)
    out = value
    remaining = power
    # A fixed number of steps avoids a device synchronization per call. Each
    # step's power of two is finite and normal, so native ldexp is safe here.
    for _ in range(steps):
        part = remaining.clamp(-step, step)
        out = torch.ldexp(out, part)
        remaining = remaining - part
    return out


def _ldexp_steps(dtype):
    """Largest safe single power-of-two step and the number needed to span the range."""
    info = torch.finfo(dtype)
    max_exponent = math.frexp(info.max)[1]              # 2**(max_exponent-1) <= max
    min_exponent = math.frexp(info.tiny)[1]             # smallest normal exponent
    digits = -math.frexp(info.eps)[1] + 2               # mantissa bits (subnormal span)
    step = max_exponent - 2                             # 2**step is finite and normal
    span = max_exponent - min_exponent + digits + 2     # beyond this, rounds to 0 or inf
    return step, -(-span // step)


class _BinaryScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, power):
        ctx.save_for_backward(power)
        ctx.input_shape = value.shape
        return _safe_ldexp(value, power)

    @staticmethod
    def backward(ctx, gradient):
        (power,) = ctx.saved_tensors
        # Native ldexp backward materializes 2**power (in default float32 on
        # some PyTorch versions). Rescale the incoming gradient directly so
        # a finite float64 derivative does not spuriously overflow/underflow.
        return _BinaryScale.apply(gradient, power).sum_to_size(ctx.input_shape), None


def _ldexp(value, exponent):
    # Exponents outside this range cannot affect a finite normalized mantissa
    # in any supported dtype. Clamp only for conversion to the kernel's integer
    # exponent, not in the stored scaled representation.
    power = exponent.clamp(-32768, 32768).to(torch.int32)
    value, power = torch.broadcast_tensors(value, power)
    if value.is_complex():
        return torch.complex(_BinaryScale.apply(value.real, power), _BinaryScale.apply(value.imag, power))
    return _BinaryScale.apply(value, power)


def _magnitude(value):
    if value.is_complex():
        return torch.maximum(value.real.abs(), value.imag.abs())
    return value.abs()


class ScaledTensor:
    """Floating-point tensor with extended exponent range, including complex values."""

    def __init__(self, mantissa, exponent=0):
        if mantissa.is_complex():
            self._parts = (ScaledTensor(mantissa.real, exponent), ScaledTensor(mantissa.imag, exponent))
            self.mantissa = torch.complex(self._parts[0].mantissa, self._parts[1].mantissa)
            self.exponent = None
            return
        self._parts = None
        if mantissa.dtype in (torch.float16, torch.bfloat16):
            mantissa = mantissa.float()
        exponent_dtype = torch.float32 if mantissa.device.type == "mps" else torch.float64
        if isinstance(exponent, torch.Tensor):
            exponent = exponent.to(dtype=exponent_dtype, device=mantissa.device).detach()
        else:
            # Create Python scalars directly on the device; as_tensor would
            # perform a synchronous host-to-device copy on every construction.
            exponent = torch.full((), exponent, dtype=exponent_dtype, device=mantissa.device)
        magnitude = _magnitude(mantissa).detach()
        _, shift = torch.frexp(magnitude)
        # Use [1,2), rather than [1/2,1), so conversion of a finite maximum
        # coefficient does not require a nonrepresentable 2**max_exponent.
        shift = torch.where((magnitude != 0) & torch.isfinite(magnitude), shift - 1, 0)
        self.mantissa = _ldexp(mantissa, -shift)
        self.exponent = exponent + shift.to(exponent_dtype)

    @classmethod
    def _from_parts(cls, real, imaginary):
        out = object.__new__(cls)
        out._parts = (real, imaginary)
        dtype = torch.promote_types(real.mantissa.dtype, imaginary.mantissa.dtype)
        out.mantissa = torch.complex(real.mantissa.to(dtype), imaginary.mantissa.to(dtype))
        out.exponent = None
        return out

    @property
    def real(self):
        return self if self._parts is None else self._parts[0]

    @property
    def imag(self):
        return ScaledTensor(torch.zeros_like(self.mantissa)) if self._parts is None else self._parts[1]

    def conj(self):
        return self if self._parts is None else self._from_parts(self.real, -self.imag)

    def map_tensor(self, function):
        """Apply a shape/copy operation equally to mantissas and scales."""
        if self._parts is not None:
            return self._from_parts(self.real.map_tensor(function), self.imag.map_tensor(function))
        exponent = self.exponent.expand(self.mantissa.shape)
        return ScaledTensor(function(self.mantissa), function(exponent))

    def transpose(self, dim0, dim1):
        return self.map_tensor(lambda value: value.transpose(dim0, dim1))

    def unsqueeze(self, dim):
        return self.map_tensor(lambda value: value.unsqueeze(dim))

    def squeeze(self, dim):
        return self.map_tensor(lambda value: value.squeeze(dim))

    def masked_fill(self, mask, value=0):
        other = self._coerce(value)
        if self._parts is not None or other._parts is not None:
            return self._from_parts(self.real.masked_fill(mask, other.real),
                                    self.imag.masked_fill(mask, other.imag))
        return ScaledTensor(torch.where(mask, other.mantissa, self.mantissa),
                            torch.where(mask, other.exponent, self.exponent))

    def matmul(self, other):
        """Multiply before reducing, with bounded contraction-axis workspace.

        Independent row/column rescaling would erase a tiny factor whose
        matching factor is huge. Keep each product scaled until reduction.
        """
        other = self._coerce(other)
        if torch.is_grad_enabled() and (self.mantissa.requires_grad or other.mantissa.requires_grad):
            from ._scaled_matmul import scaled_matmul

            return scaled_matmul(self, other)
        if self._parts is not None or other._parts is not None:
            if self._parts is None:
                return self._from_parts(self.matmul(other.real), self.matmul(other.imag))
            if other._parts is None:
                return self._from_parts(self.real.matmul(other), self.imag.matmul(other))
            return self._from_parts(self.real.matmul(other.real) - self.imag.matmul(other.imag),
                                    self.real.matmul(other.imag) + self.imag.matmul(other.real))
        left_vector = self.mantissa.ndim == 1
        right_vector = other.mantissa.ndim == 1
        left = self.unsqueeze(0) if left_vector else self
        right = other.unsqueeze(-1) if right_vector else other
        if left.mantissa.ndim < 2 or right.mantissa.ndim < 2:
            raise RuntimeError("matmul requires tensors with at least one dimension")
        if left.mantissa.shape[-1] != right.mantissa.shape[-2]:
            raise RuntimeError("matmul contraction dimensions must match")
        if left.mantissa.numel() == 0 or right.mantissa.numel() == 0:
            # Preserve the autograd connection even for an empty contraction.
            dtype = torch.promote_types(self.mantissa.dtype, other.mantissa.dtype)
            return ScaledTensor(torch.matmul(self.mantissa.to(dtype), other.mantissa.to(dtype)))
        left_max, left_min = left.exponent.amax(), left.exponent.amin()
        right_max, right_min = right.exponent.amax(), right.exponent.amin()
        dtype = torch.promote_types(left.mantissa.dtype, right.mantissa.dtype)
        safe_span = -math.log2(torch.finfo(dtype).tiny) - 4
        if bool(((left_max - left_min + right_max - right_min) < safe_span)
                & ((left_max + right_max).abs() < safe_span)):
            # When all normalized products fit, retain the optimized native
            # matrix kernel. Include zero entries' exponents too: a zero
            # value can still have a nonzero derivative. Use explicit scaled
            # products outside the native exponent range: a fused multiply-add
            # cancellation residual must not be magnified into an infinity.
            a = _ldexp(left.mantissa, left.exponent - left_max).to(dtype)
            b = _ldexp(right.mantissa, right.exponent - right_max).to(dtype)
            result = ScaledTensor(torch.matmul(a, b), left_max + right_max)
            if left_vector:
                result = result.squeeze(-2)
            if right_vector:
                result = result.squeeze(-1)
            return result
        batch = torch.broadcast_shapes(left.mantissa.shape[:-2], right.mantissa.shape[:-2])
        shape = (*batch, left.mantissa.shape[-2], right.mantissa.shape[-1])
        result = ScaledTensor(torch.zeros(shape, dtype=dtype, device=left.mantissa.device))
        # Cap the temporary product at roughly 2**18 elements, independent
        # of the contraction dimension. Never form the full M x K x N tensor.
        block = max(1, min(32, 2**18 // max(1, math.prod(shape))))
        for start in range(0, left.mantissa.shape[-1], block):
            stop = start + block
            a = left.map_tensor(lambda value: value[..., :, start:stop]).unsqueeze(-1)
            b = right.map_tensor(lambda value: value[..., start:stop, :]).unsqueeze(-3)
            result = result + (a * b).sum(dim=-2)
        if left_vector:
            result = result.squeeze(-2)
        if right_vector:
            result = result.squeeze(-1)
        return result

    __matmul__ = matmul

    @classmethod
    def from_tensor(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.get_default_dtype())
        return cls(value)

    @classmethod
    def from_log(cls, logarithm):
        """Represent exp(logarithm) without first under/overflowing its value."""
        if logarithm.dtype in (torch.float16, torch.bfloat16):
            logarithm = logarithm.float()
        real = logarithm.real
        scale_dtype = torch.float32 if real.device.type == "mps" else torch.float64
        detached = real.detach().to(scale_dtype)
        finite = torch.isfinite(detached)
        safe = torch.where(finite, detached, 0)
        bound = torch.finfo(scale_dtype).max / 4
        exponent = (safe / math.log(2)).clamp(-bound, bound).floor()
        remainder = torch.remainder(safe, math.log(2)).to(real.dtype)
        # The detached remainder supplies the value; the zero-valued difference
        # keeps d exp(x)/dx in the mantissa graph without differentiating scales.
        safe_real = torch.where(finite, real, torch.zeros_like(real))
        residual = remainder + (safe_real - safe_real.detach())
        if logarithm.is_complex():
            residual = torch.complex(residual, logarithm.imag)
        mantissa = torch.exp(residual)
        # Preserve NaN/infinity semantics. This branch is sanitized before exp,
        # so ordinary finite inputs never evaluate an overflowing exponential.
        exceptional = torch.exp(torch.where(finite, torch.zeros_like(logarithm), logarithm))
        mantissa = torch.where(finite, mantissa, exceptional)
        return cls(mantissa, exponent)

    exp = from_log

    def _coerce(self, other):
        if isinstance(other, ScaledTensor):
            return other
        if not isinstance(other, torch.Tensor):
            other = torch.as_tensor(other, dtype=self.mantissa.dtype, device=self.mantissa.device)
        return ScaledTensor.from_tensor(other)

    def __neg__(self):
        if self._parts is not None:
            return self._from_parts(-self.real, -self.imag)
        return ScaledTensor(-self.mantissa, self.exponent)

    def __add__(self, other):
        other = self._coerce(other)
        if self._parts is not None or other._parts is not None:
            return self._from_parts(self.real + other.real, self.imag + other.imag)
        left_zero = self.mantissa.detach() == 0
        right_zero = other.mantissa.detach() == 0
        exponent = torch.maximum(self.exponent, other.exponent)
        exponent = torch.where(left_zero, other.exponent, exponent)
        exponent = torch.where(right_zero, self.exponent, exponent)
        left = _ldexp(self.mantissa, self.exponent - exponent)
        right = _ldexp(other.mantissa, other.exponent - exponent)
        return ScaledTensor(left + right, exponent)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        if self._parts is not None or other._parts is not None:
            if self._parts is None:
                return self._from_parts(self * other.real, self * other.imag)
            if other._parts is None:
                return self._from_parts(self.real * other, self.imag * other)
            return self._from_parts(self.real * other.real - self.imag * other.imag,
                                    self.real * other.imag + self.imag * other.real)
        return ScaledTensor(self.mantissa * other.mantissa, self.exponent + other.exponent)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        if self._parts is not None or other._parts is not None:
            if other._parts is None:
                return self._from_parts(self.real / other, self.imag / other)
            denominator = other.real * other.real + other.imag * other.imag
            numerator = self * other.conj()
            return self._from_parts(numerator.real / denominator, numerator.imag / denominator)
        return ScaledTensor(self.mantissa / other.mantissa, self.exponent - other.exponent)

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def sum(self, dim=None, keepdim=False):
        if self._parts is not None:
            return self._from_parts(self.real.sum(dim, keepdim), self.imag.sum(dim, keepdim))
        if self.mantissa.numel() == 0:
            return ScaledTensor(self.mantissa.sum(dim=dim, keepdim=keepdim))
        exponent = self.exponent + torch.zeros_like(self.mantissa.real)
        nonzero = self.mantissa.detach() != 0
        candidate = torch.where(nonzero, exponent, -float("inf"))
        maximum = torch.amax(candidate, dim=dim, keepdim=True)
        maximum = torch.where(torch.isneginf(maximum), 0, maximum)
        mantissa = _ldexp(self.mantissa, exponent - maximum).sum(dim=dim, keepdim=keepdim)
        if not keepdim:
            if dim is None:
                maximum = maximum.squeeze()
            elif isinstance(dim, (tuple, list)):
                for axis in sorted((d % maximum.ndim for d in dim), reverse=True):
                    maximum = maximum.squeeze(axis)
            else:
                maximum = maximum.squeeze(dim)
        return ScaledTensor(mantissa, maximum)

    def to_tensor(self, dtype=None):
        if dtype is None:
            dtype = self.mantissa.dtype
        if self._parts is not None:
            real_dtype = torch.empty((), dtype=dtype).real.dtype
            return torch.complex(self.real.to_tensor(real_dtype), self.imag.to_tensor(real_dtype)).to(dtype)
        # Cast after rescaling: narrowing a mantissa first needlessly loses
        # precision. Only this conversion is allowed to overflow/underflow.
        return _ldexp(self.mantissa, self.exponent).to(dtype=dtype)
