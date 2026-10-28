"""Elementwise activation values and gradient series, real first order."""
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _activation(X0, X1, G0, G1, Y0, Y1, N: tl.constexpr, KIND: tl.constexpr,
                BACKWARD: tl.constexpr, DOUBLE: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.float64 if DOUBLE else tl.float32
    x0 = tl.load(X0 + col, col < N, 0).to(acc)
    x1 = tl.load(X1 + col, col < N, 0).to(acc)
    if KIND == "gelu":
        phi = libdevice.exp(-0.5 * x0 * x0) * 0.3989422804014327
        cdf = 0.5 * (1.0 + libdevice.erf(x0 * 0.7071067811865475))
        value = x0 * cdf
        d1 = cdf + x0 * phi
        d2 = (2.0 - x0 * x0) * phi
    elif KIND == "tanh":
        value = x0 if BACKWARD else libdevice.tanh(x0)
        d1 = 1.0 - value * value
        # Native backward takes the forward output, so X1 is its tangent.
        d2 = -2.0 * value
    else:
        value = x0 if BACKWARD else 1.0 / (1.0 + libdevice.exp(-x0))
        d1 = value * (1.0 - value)
        d2 = 1.0 - 2.0 * value
    if BACKWARD:
        g0 = tl.load(G0 + col, col < N, 0).to(acc)
        g1 = tl.load(G1 + col, col < N, 0).to(acc)
        y0 = g0 * d1
        y1 = g1 * d1 + g0 * d2 * x1
    else:
        y0 = value
        y1 = d1 * x1
    tl.store(Y0 + col, y0, col < N)
    tl.store(Y1 + col, y1, col < N)
