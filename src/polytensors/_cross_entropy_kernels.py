"""Triton kernels for fused degree-1 cross entropy (see _fused_cross_entropy). Imported lazily."""

import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # 2 / (1 + exp(-2x)) - 1 saturates correctly at both ends
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0

@triton.jit
def _capped(z0, z1, cap, HAS_CAP: tl.constexpr):
    if HAS_CAP:
        t = _tanh(z0 / cap)
        return cap * t, (1.0 - t * t) * z1
    return z0, z1

@triton.jit
def _forward(Z0, Z1, Y, LOSS0, LOSS1, LSE, PZ1, C, stride, ignore_index, cap,
             HAS_CAP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    y = tl.load(Y + row)
    base = row * stride
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    s = tl.zeros((BLOCK,), tl.float32)
    w = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, C, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inside = cols < C
        z0 = tl.load(Z0 + base + cols, mask=inside, other=0.0).to(tl.float32)
        z1 = tl.load(Z1 + base + cols, mask=inside, other=0.0).to(tl.float32)
        z0, z1 = _capped(z0, z1, cap, HAS_CAP)
        z0 = tl.where(inside, z0, float("-inf"))
        m_new = tl.maximum(m, z0)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m - m_safe)
        e = tl.exp(z0 - m_safe)
        s = s * alpha + e
        w = w * alpha + e * z1
        m = m_new
    # combine the per-lane partial sums
    mx = tl.max(m, 0)
    scale = tl.exp(m - mx)
    total = tl.sum(s * scale, 0)
    pz1 = tl.sum(w * scale, 0) / total
    lse = mx + tl.log(total)
    valid = y != ignore_index
    y_safe = tl.where(valid, y, 0)
    z0y, z1y = _capped(tl.load(Z0 + base + y_safe).to(tl.float32), tl.load(Z1 + base + y_safe).to(tl.float32),
                       cap, HAS_CAP)
    tl.store(LOSS0 + row, tl.where(valid, lse - z0y, 0.0))
    tl.store(LOSS1 + row, tl.where(valid, pz1 - z1y, 0.0))
    tl.store(LSE + row, lse)
    tl.store(PZ1 + row, pz1)

@triton.jit
def _backward(Z0, Z1, Y, LSE, PZ1, C0, C1, DZ0, DZ1, C, stride, ignore_index, cap,
              HAS_CAP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    y = tl.load(Y + row)
    valid = y != ignore_index
    c0 = tl.where(valid, tl.load(C0 + row).to(tl.float32), 0.0)
    c1 = tl.where(valid, tl.load(C1 + row).to(tl.float32), 0.0)
    lse = tl.load(LSE + row)
    pz1 = tl.load(PZ1 + row)
    base = row * stride
    out = row * C
    for start in range(0, C, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inside = cols < C
        x0 = tl.load(Z0 + base + cols, mask=inside, other=0.0).to(tl.float32)
        x1 = tl.load(Z1 + base + cols, mask=inside, other=0.0).to(tl.float32)
        z0, z1 = _capped(x0, x1, cap, HAS_CAP)
        p = tl.exp(z0 - lse)
        p_y = p - tl.where(cols == y, 1.0, 0.0)
        dz1 = c1 * p_y
        dz0 = c0 * p_y + c1 * p * (z1 - pz1)
        if HAS_CAP:
            # chain rule through z0 = cap tanh(x0 / cap), z1 = (1 - tanh^2) x1
            t = _tanh(x0 / cap)
            d = 1.0 - t * t
            dz0, dz1 = dz0 * d + dz1 * (-2.0 * t * d * x1 / cap), dz1 * d
        tl.store(DZ0 + out + cols, dz0.to(DZ0.dtype.element_ty), mask=inside)
        tl.store(DZ1 + out + cols, dz1.to(DZ1.dtype.element_ty), mask=inside)
