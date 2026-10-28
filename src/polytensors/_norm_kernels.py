"""First-order RMSNorm kernels; one program per normalized row."""
import triton
import triton.language as tl


@triton.jit
def _forward(X0, X1, Y0, Y1, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
             DOUBLE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    acc = tl.float64 if DOUBLE else tl.float32
    x0 = tl.load(X0 + row * N + col, col < N, 0).to(acc)
    x1 = tl.load(X1 + row * N + col, col < N, 0).to(acc)
    variance = tl.sum(x0 * x0, 0) / N + EPS
    r = 1.0 / tl.sqrt(variance)
    r1 = -r * r * r * tl.sum(x0 * x1, 0) / N
    tl.store(Y0 + row * N + col, x0 * r, col < N)
    tl.store(Y1 + row * N + col, x1 * r + x0 * r1, col < N)


@triton.jit
def _backward(X0, X1, G0, G1, DX0, DX1, N: tl.constexpr, EPS: tl.constexpr,
              BLOCK: tl.constexpr, DOUBLE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    acc = tl.float64 if DOUBLE else tl.float32
    x0 = tl.load(X0 + row * N + col, col < N, 0).to(acc)
    x1 = tl.load(X1 + row * N + col, col < N, 0).to(acc)
    g0 = tl.load(G0 + row * N + col, col < N, 0).to(acc)
    g1 = tl.load(G1 + row * N + col, col < N, 0).to(acc)
    r = 1.0 / tl.sqrt(tl.sum(x0 * x0, 0) / N + EPS)
    r1 = -r * r * r * tl.sum(x0 * x1, 0) / N
    dot0 = tl.sum(x0 * g0, 0) / N
    dot1 = tl.sum(x1 * g0 + x0 * g1, 0) / N
    dx0 = r * g0 - x0 * r * r * r * dot0
    dx1 = r * g1 + r1 * g0 - x1 * r * r * r * dot0 - x0 * (3 * r * r * r1 * dot0 + r * r * r * dot1)
    tl.store(DX0 + row * N + col, dx0, col < N)
    tl.store(DX1 + row * N + col, dx1, col < N)
