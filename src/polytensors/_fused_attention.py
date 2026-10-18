"""Fused (flash-style) attention for real degree-1 series.

For queries, keys and values ``Q0 + t Q1``, ``K0 + t K1``, ``V0 + t V1`` the
attention output is ``O0 + t O1`` with

    S = s Q0 K0^T,   A = s (Q1 K0^T + Q0 K1^T)   (the tangent of the scores),
    P = softmax(S),  r = rowsum(P * A),
    O0 = P V0,       O1 = P V1 + (P * (A - r)) V0.

The forward kernel computes ``O0``, ``O1``, the row log-sum-exp of ``S`` and
``r`` block by block with an online softmax, never forming the L x S score
matrices, and skips key blocks that the causal mask or a boolean mask removes
entirely. The backward kernels compute the exact vector-Jacobian product of the
coefficient map ``(Q0, Q1, K0, K1, V0, V1) -> (O0, O1)``, recomputing ``P``
from the saved log-sum-exp. Accumulation is in float32 for half-precision and
float32 inputs and in float64 for float64 inputs.

For a truncated series map ``x(t) -> y(t)``, the series of the gradient
``J(x(t))^T G(t)`` has coefficient ``j`` equal to the derivative of
``sum_k <G_k, y_{D-k}>`` with respect to ``x_{D-j}``. For degree 1 the
PolyTensor backward is therefore this coefficient VJP with the cotangents
``(G1, G0)`` and the input orders swapped (see ``_SeriesAttention``).

Triton is imported lazily; without it (or on CPU) callers fall back to the
general attention path.
"""

import math

import torch


_BLOCK = 64


def available():
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


_kernels = None


def _build_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _load_rows(ptr, offs, n, D: tl.constexpr):
        d = tl.arange(0, D)
        return tl.load(ptr + offs[:, None] * D + d[None, :], mask=(offs < n)[:, None], other=0.0)

    @triton.jit
    def _valid(offs_m, offs_n, L, S, MASK, HAS_MASK: tl.constexpr, CAUSAL: tl.constexpr):
        valid = (offs_m < L)[:, None] & (offs_n < S)[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        if HAS_MASK:
            keep = tl.load(MASK + offs_m[:, None] * S + offs_n[None, :], mask=valid, other=0)
            valid = valid & (keep != 0)
        return valid

    @triton.jit
    def _dot(a, b, PREC: tl.constexpr, ACC: tl.constexpr):
        return tl.dot(a, b, input_precision=PREC, out_dtype=ACC)

    @triton.jit
    def _forward(Q0, Q1, K0, K1, V0, V1, O0, O1, LSE, R, MASK, NLO, NHI, SCALE, L, S,
                 HAS_MASK: tl.constexpr, CAUSAL: tl.constexpr, D: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr, ACC: tl.constexpr):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1).to(tl.int64)
        scale = tl.load(SCALE).to(ACC)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qbase = bh * L * D
        kbase = bh * S * D
        q0 = _load_rows(Q0 + qbase, offs_m, L, D)
        q1 = _load_rows(Q1 + qbase, offs_m, L, D)
        m_i = tl.full((BLOCK_M,), float("-inf"), ACC)
        l_i = tl.zeros((BLOCK_M,), ACC)
        r_i = tl.zeros((BLOCK_M,), ACC)
        acc0 = tl.zeros((BLOCK_M, D), ACC)
        acc1 = tl.zeros((BLOCK_M, D), ACC)
        lo = 0
        hi = S
        if CAUSAL:
            hi = tl.minimum(hi, (pid_m + 1) * BLOCK_M)
        if HAS_MASK:
            lo = tl.load(NLO + pid_m) * BLOCK_N
            hi = tl.minimum(hi, tl.load(NHI + pid_m) * BLOCK_N)
        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k0 = _load_rows(K0 + kbase, offs_n, S, D)
            k1 = _load_rows(K1 + kbase, offs_n, S, D)
            v0 = _load_rows(V0 + kbase, offs_n, S, D)
            v1 = _load_rows(V1 + kbase, offs_n, S, D)
            valid = _valid(offs_m, offs_n, L, S, MASK, HAS_MASK, CAUSAL)
            s = _dot(q0, tl.trans(k0), PREC, ACC) * scale
            a = (_dot(q1, tl.trans(k0), PREC, ACC) + _dot(q0, tl.trans(k1), PREC, ACC)) * scale
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(s - m_safe[:, None])
            pa = p * tl.where(valid, a, 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            r_i = r_i * alpha + tl.sum(pa, 1)
            acc0 = acc0 * alpha[:, None] + _dot(p.to(v0.dtype), v0, PREC, ACC)
            acc1 = acc1 * alpha[:, None] + _dot(p.to(v1.dtype), v1, PREC, ACC) + _dot(pa.to(v0.dtype), v0, PREC, ACC)
            m_i = m_new
        empty = l_i == 0.0
        l_safe = tl.where(empty, 1.0, l_i)
        o0 = acc0 / l_safe[:, None]
        r = r_i / l_safe
        o1 = acc1 / l_safe[:, None] - r[:, None] * o0
        # A row with no admissible key gets zero output and lse = +inf, so its P is zero in backward.
        lse = tl.where(empty, float("inf"), m_i + tl.log(l_safe))
        d = tl.arange(0, D)
        rows = offs_m < L
        tl.store(O0 + qbase + offs_m[:, None] * D + d[None, :], o0.to(O0.dtype.element_ty), mask=rows[:, None])
        tl.store(O1 + qbase + offs_m[:, None] * D + d[None, :], o1.to(O1.dtype.element_ty), mask=rows[:, None])
        tl.store(LSE + bh * L + offs_m, lse, mask=rows)
        tl.store(R + bh * L + offs_m, r, mask=rows)

    @triton.jit
    def _block_terms(q0, q1, c0, c1, k0, k1, v0, v1, lse, r, delta1, dsum, valid, scale,
                     PREC: tl.constexpr, ACC: tl.constexpr):
        # P, its tangent and the score cotangents dS (for S) and dA (for A) of one block.
        s = _dot(q0, tl.trans(k0), PREC, ACC) * scale
        a = (_dot(q1, tl.trans(k0), PREC, ACC) + _dot(q0, tl.trans(k1), PREC, ACC)) * scale
        p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        a = tl.where(valid, a, 0.0)
        u = _dot(c1, tl.trans(v0), PREC, ACC)
        w = _dot(c0, tl.trans(v0), PREC, ACC) + _dot(c1, tl.trans(v1), PREC, ACC)
        a_r = a - r[:, None]
        p_dot = p * a_r
        g = w + a_r * u - a * delta1[:, None]
        ds = p * (g - dsum[:, None])
        da = p * (u - delta1[:, None])
        return p, p_dot, ds, da

    @triton.jit
    def _backward_kv(Q0, Q1, K0, K1, V0, V1, C0, C1, LSE, R, DELTA1, DSUM, MASK, MLO, MHI,
                     DK0, DK1, DV0, DV1, SCALE, L, S,
                     HAS_MASK: tl.constexpr, CAUSAL: tl.constexpr, D: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr, ACC: tl.constexpr):
        pid_n = tl.program_id(0)
        bh = tl.program_id(1).to(tl.int64)
        scale = tl.load(SCALE).to(ACC)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        qbase = bh * L * D
        kbase = bh * S * D
        k0 = _load_rows(K0 + kbase, offs_n, S, D)
        k1 = _load_rows(K1 + kbase, offs_n, S, D)
        v0 = _load_rows(V0 + kbase, offs_n, S, D)
        v1 = _load_rows(V1 + kbase, offs_n, S, D)
        dk0 = tl.zeros((BLOCK_N, D), ACC)
        dk1 = tl.zeros((BLOCK_N, D), ACC)
        dv0 = tl.zeros((BLOCK_N, D), ACC)
        dv1 = tl.zeros((BLOCK_N, D), ACC)
        lo = 0
        hi = L
        if CAUSAL:
            lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
        if HAS_MASK:
            lo = tl.maximum(lo, tl.load(MLO + pid_n) * BLOCK_M)
            hi = tl.minimum(hi, tl.load(MHI + pid_n) * BLOCK_M)
        for start_m in range(lo, hi, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            rows = offs_m < L
            q0 = _load_rows(Q0 + qbase, offs_m, L, D)
            q1 = _load_rows(Q1 + qbase, offs_m, L, D)
            c0 = _load_rows(C0 + qbase, offs_m, L, D)
            c1 = _load_rows(C1 + qbase, offs_m, L, D)
            lse = tl.load(LSE + bh * L + offs_m, mask=rows, other=float("inf"))
            r = tl.load(R + bh * L + offs_m, mask=rows, other=0.0)
            delta1 = tl.load(DELTA1 + bh * L + offs_m, mask=rows, other=0.0)
            dsum = tl.load(DSUM + bh * L + offs_m, mask=rows, other=0.0)
            valid = _valid(offs_m, offs_n, L, S, MASK, HAS_MASK, CAUSAL)
            p, p_dot, ds, da = _block_terms(q0, q1, c0, c1, k0, k1, v0, v1, lse, r, delta1, dsum, valid, scale,
                                            PREC, ACC)
            pt = tl.trans(p).to(c0.dtype)
            dv0 += _dot(pt, c0, PREC, ACC) + _dot(tl.trans(p_dot).to(c1.dtype), c1, PREC, ACC)
            dv1 += _dot(pt, c1, PREC, ACC)
            dst = tl.trans(ds).to(q0.dtype)
            dat = tl.trans(da).to(q0.dtype)
            dk0 += (_dot(dst, q0, PREC, ACC) + _dot(dat, q1, PREC, ACC)) * scale
            dk1 += _dot(dat, q0, PREC, ACC) * scale
        d = tl.arange(0, D)
        cols = (offs_n < S)[:, None]
        out = kbase + offs_n[:, None] * D + d[None, :]
        tl.store(DK0 + out, dk0.to(DK0.dtype.element_ty), mask=cols)
        tl.store(DK1 + out, dk1.to(DK1.dtype.element_ty), mask=cols)
        tl.store(DV0 + out, dv0.to(DV0.dtype.element_ty), mask=cols)
        tl.store(DV1 + out, dv1.to(DV1.dtype.element_ty), mask=cols)

    @triton.jit
    def _backward_q(Q0, Q1, K0, K1, V0, V1, C0, C1, LSE, R, DELTA1, DSUM, MASK, NLO, NHI,
                    DQ0, DQ1, SCALE, L, S,
                    HAS_MASK: tl.constexpr, CAUSAL: tl.constexpr, D: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr, ACC: tl.constexpr):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1).to(tl.int64)
        scale = tl.load(SCALE).to(ACC)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = offs_m < L
        qbase = bh * L * D
        kbase = bh * S * D
        q0 = _load_rows(Q0 + qbase, offs_m, L, D)
        q1 = _load_rows(Q1 + qbase, offs_m, L, D)
        c0 = _load_rows(C0 + qbase, offs_m, L, D)
        c1 = _load_rows(C1 + qbase, offs_m, L, D)
        lse = tl.load(LSE + bh * L + offs_m, mask=rows, other=float("inf"))
        r = tl.load(R + bh * L + offs_m, mask=rows, other=0.0)
        delta1 = tl.load(DELTA1 + bh * L + offs_m, mask=rows, other=0.0)
        dsum = tl.load(DSUM + bh * L + offs_m, mask=rows, other=0.0)
        dq0 = tl.zeros((BLOCK_M, D), ACC)
        dq1 = tl.zeros((BLOCK_M, D), ACC)
        lo = 0
        hi = S
        if CAUSAL:
            hi = tl.minimum(hi, (pid_m + 1) * BLOCK_M)
        if HAS_MASK:
            lo = tl.load(NLO + pid_m) * BLOCK_N
            hi = tl.minimum(hi, tl.load(NHI + pid_m) * BLOCK_N)
        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k0 = _load_rows(K0 + kbase, offs_n, S, D)
            k1 = _load_rows(K1 + kbase, offs_n, S, D)
            v0 = _load_rows(V0 + kbase, offs_n, S, D)
            v1 = _load_rows(V1 + kbase, offs_n, S, D)
            valid = _valid(offs_m, offs_n, L, S, MASK, HAS_MASK, CAUSAL)
            p, p_dot, ds, da = _block_terms(q0, q1, c0, c1, k0, k1, v0, v1, lse, r, delta1, dsum, valid, scale,
                                            PREC, ACC)
            ds = ds.to(k0.dtype)
            da = da.to(k0.dtype)
            dq0 += (_dot(ds, k0, PREC, ACC) + _dot(da, k1, PREC, ACC)) * scale
            dq1 += _dot(da, k0, PREC, ACC) * scale
        d = tl.arange(0, D)
        out = qbase + offs_m[:, None] * D + d[None, :]
        tl.store(DQ0 + out, dq0.to(DQ0.dtype.element_ty), mask=rows[:, None])
        tl.store(DQ1 + out, dq1.to(DQ1.dtype.element_ty), mask=rows[:, None])

    return _forward, _backward_kv, _backward_q


def _get_kernels():
    global _kernels
    if _kernels is None:
        _kernels = _build_kernels()
    return _kernels


def _precision(dtype):
    import triton.language as tl

    if dtype == torch.float64:
        return "ieee", tl.float64
    if dtype == torch.float32:
        return ("tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"), tl.float32
    return "ieee", tl.float32


def _blocks(dtype, D):
    # (BLOCK_M, BLOCK_N, num_warps, num_stages) per kernel; wide types need smaller tiles for shared memory.
    if dtype == torch.float64:
        return (16, 16, 8, 1), (16, 16, 8, 1)
    if dtype == torch.float32 or D > 128:
        return (32, 32, 4, 2), (32, 32, 4, 1)
    return (64, 64, 4, 2), (64, 32, 8, 1)


def _mask_ranges(mask, L, S, bm, bn):
    """Per query block the first/last key block (in units of ``bn``) holding an admissible key, and per key
    block the first/last query block (units of ``bm``), from a boolean (L, S) mask."""
    mb, nb = -(-L // bm), -(-S // bn)
    padded = torch.zeros(mb * bm, nb * bn, dtype=torch.bool, device=mask.device)
    padded[:L, :S] = mask
    blocks = padded.view(mb, bm, nb, bn).any(dim=3).any(dim=1)          # (mb, nb)
    index_n = torch.arange(nb, device=mask.device)
    index_m = torch.arange(mb, device=mask.device)
    any_n = blocks.any(dim=1)
    nlo = torch.where(blocks, index_n, nb).amin(dim=1)
    nhi = torch.where(blocks, index_n + 1, 0).amax(dim=1)
    nlo = torch.where(any_n, nlo, 0)
    any_m = blocks.any(dim=0)
    mlo = torch.where(blocks, index_m[:, None], mb).amin(dim=0)
    mhi = torch.where(blocks, index_m[:, None] + 1, 0).amax(dim=0)
    mlo = torch.where(any_m, mlo, 0)
    return (nlo.to(torch.int32).contiguous(), nhi.to(torch.int32).contiguous(),
            mlo.to(torch.int32).contiguous(), mhi.to(torch.int32).contiguous())


class _Plan:
    """Shapes, mask and launch parameters shared by the forward and backward of one call."""

    def __init__(self, q, k, mask, causal, scale):
        self.B, self.H, self.L, self.D = q.shape
        self.S = k.shape[2]
        self.causal = causal
        self.scale = float(scale)
        self.scale_t = torch.full((1,), self.scale, dtype=torch.float64, device=q.device)
        self.dtype = q.dtype
        (self.fwd_m, self.fwd_n, self.fwd_warps, self.fwd_stages), (self.bwd_m, self.bwd_n, self.bwd_warps, self.bwd_stages) = \
            _blocks(q.dtype, self.D)
        self.mask = None
        dummy = torch.zeros(1, dtype=torch.int32, device=q.device)
        self.nlo = self.nhi = self.mlo = self.mhi = self.bnlo = self.bnhi = dummy
        if mask is not None:
            self.mask = mask.to(torch.int32).contiguous()
            self.nlo, self.nhi, _, _ = _mask_ranges(mask, self.L, self.S, self.fwd_m, self.fwd_n)
            self.bnlo, self.bnhi, self.mlo, self.mhi = _mask_ranges(mask, self.L, self.S, self.bwd_m, self.bwd_n)

    def mask_arg(self, device):
        return self.mask if self.mask is not None else torch.zeros(1, dtype=torch.int32, device=device)


def forward(q0, q1, k0, k1, v0, v1, plan):
    """Fused forward on contiguous (B, H, T, D) coefficients; returns (o0, o1, lse, r)."""
    fwd, _, _ = _get_kernels()
    B, H, L, S, D = plan.B, plan.H, plan.L, plan.S, plan.D
    o0 = torch.empty_like(q0)
    o1 = torch.empty_like(q0)
    acc_dtype = torch.float64 if plan.dtype == torch.float64 else torch.float32
    lse = torch.empty(B, H, L, dtype=acc_dtype, device=q0.device)
    r = torch.empty_like(lse)
    prec, acc = _precision(plan.dtype)
    grid = (-(-L // plan.fwd_m), B * H)
    fwd[grid](q0, q1, k0, k1, v0, v1, o0, o1, lse, r, plan.mask_arg(q0.device), plan.nlo, plan.nhi,
              plan.scale_t, L, S, HAS_MASK=plan.mask is not None, CAUSAL=plan.causal, D=D,
              BLOCK_M=plan.fwd_m, BLOCK_N=plan.fwd_n, PREC=prec, ACC=acc,
              num_warps=plan.fwd_warps, num_stages=plan.fwd_stages)
    return o0, o1, lse, r


def backward(q0, q1, k0, k1, v0, v1, o0, o1, lse, r, c0, c1, plan):
    """VJP of (q0, q1, k0, k1, v0, v1) -> (o0, o1) with cotangents (c0, c1)."""
    _, bwd_kv, bwd_q = _get_kernels()
    B, H, L, S, D = plan.B, plan.H, plan.L, plan.S, plan.D
    acc_dtype = lse.dtype
    c0 = c0.to(q0.dtype).contiguous()
    c1 = c1.to(q0.dtype).contiguous()
    delta1 = (c1.to(acc_dtype) * o0.to(acc_dtype)).sum(-1)
    dsum = (c0.to(acc_dtype) * o0.to(acc_dtype)).sum(-1) + (c1.to(acc_dtype) * o1.to(acc_dtype)).sum(-1) - r * delta1
    dq0, dq1 = torch.empty_like(q0), torch.empty_like(q0)
    dk0, dk1, dv0, dv1 = (torch.empty_like(k0) for _ in range(4))
    prec, acc = _precision(plan.dtype)
    mask = plan.mask_arg(q0.device)
    common = dict(HAS_MASK=plan.mask is not None, CAUSAL=plan.causal, D=D, BLOCK_M=plan.bwd_m, BLOCK_N=plan.bwd_n,
                  PREC=prec, ACC=acc, num_warps=plan.bwd_warps, num_stages=plan.bwd_stages)
    bwd_kv[(-(-S // plan.bwd_n), B * H)](q0, q1, k0, k1, v0, v1, c0, c1, lse, r, delta1, dsum, mask, plan.mlo, plan.mhi,
                                        dk0, dk1, dv0, dv1, plan.scale_t, L, S, **common)
    bwd_q[(-(-L // plan.bwd_m), B * H)](q0, q1, k0, k1, v0, v1, c0, c1, lse, r, delta1, dsum, mask, plan.bnlo, plan.bnhi,
                                       dq0, dq1, plan.scale_t, L, S, **common)
    return dq0, dq1, dk0, dk1, dv0, dv1


class CoefficientAttention(torch.autograd.Function):
    """The coefficient map (q0, q1, k0, k1, v0, v1) -> (o0, o1) on ordinary tensors (for tests and
    coefficient-autograd use)."""

    @staticmethod
    def forward(ctx, q0, q1, k0, k1, v0, v1, mask, causal, scale):
        plan = _Plan(q0, k0, mask, causal, scale)
        inputs = [x.contiguous() for x in (q0, q1, k0, k1, v0, v1)]
        o0, o1, lse, r = forward(*inputs, plan)
        ctx.save_for_backward(*inputs, o0, o1, lse, r)
        ctx.plan = plan
        return o0, o1

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, c0, c1):
        q0, q1, k0, k1, v0, v1, o0, o1, lse, r = ctx.saved_tensors
        c0 = torch.zeros_like(o0) if c0 is None else c0
        c1 = torch.zeros_like(o1) if c1 is None else c1
        return (*backward(q0, q1, k0, k1, v0, v1, o0, o1, lse, r, c0, c1, ctx.plan), None, None, None)


def default_scale(head_dim):
    return 1 / math.sqrt(head_dim)


# ---------------------------------------------------------------------------
# PolyTensor integration: torch.nn.functional.scaled_dot_product_attention on
# degree-1 series dispatches here (from _decompositions.maybe_decompose).

import contextvars

_enabled = contextvars.ContextVar("polytensor_fused_attention", default=True)

_HEAD_DIMS = {torch.float16: (16, 32, 64, 128), torch.bfloat16: (16, 32, 64, 128),
              torch.float32: (16, 32, 64, 128), torch.float64: (16, 32, 64)}


def _coefficients(x, like):
    from ._tensor import PolyTensor

    if isinstance(x, PolyTensor):
        return x.coeffs
    return (x, torch.zeros_like(x))


def _flat(c, shape):
    return c.expand(shape).reshape(-1, 1, *shape[-2:]).contiguous()


def _square_mask(attn_mask, L, S):
    """A boolean attention mask as an (L, S) tensor, or None if it is not one (e.g. batch-dependent)."""
    if attn_mask is None:
        return None
    if attn_mask.dtype != torch.bool or attn_mask.shape[-2:] != (L, S):
        return NotImplemented
    if any(size != 1 for size in attn_mask.shape[:-2]):
        return NotImplemented
    return attn_mask.reshape(L, S)


def eligible(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
    """Whether this attention call can use the fused degree-1 kernels."""
    from ._plain import plain_range_enabled
    from ._tensor import PolyTensor

    if not _enabled.get() or not plain_range_enabled() or not available():
        return False
    if dropout_p != 0.0 or enable_gqa:
        return False
    tensors = (query, key, value)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        return False
    polys = [x for x in tensors if isinstance(x, PolyTensor)]
    if not polys or any(p.degree != 1 for p in polys):
        return False
    if any(not isinstance(x, PolyTensor) and x.requires_grad for x in tensors):
        return False
    if isinstance(attn_mask, PolyTensor):
        return False
    coefficients = [c for x in tensors for c in _coefficients(x, query)]
    dtype = coefficients[0].dtype
    if any(c.dtype != dtype or c.is_complex() or not c.is_cuda for c in coefficients):
        return False
    if query.ndim < 3 or key.shape != value.shape or query.shape[:-2] != key.shape[:-2]:
        return False
    if query.shape[-1] != key.shape[-1] or query.shape[-1] not in _HEAD_DIMS.get(dtype, ()):
        return False
    if is_causal and attn_mask is not None:
        return False   # PyTorch rejects this combination too
    return _square_mask(attn_mask, query.shape[-2], key.shape[-2]) is not NotImplemented


def _run_forward(q, k, v, mask, causal, scale):
    """Flattened (N, 1, T, D) contiguous coefficients, the plan, and flat (o0, o1, lse, r)."""
    qc, kc, vc = (_coefficients(x, q) for x in (q, k, v))
    flat = [_flat(c, q.shape) for c in qc] + [_flat(c, k.shape) for c in (*kc, *vc)]
    plan = _Plan(flat[0], flat[2], mask, causal, scale)
    return flat, plan, *forward(*flat, plan)


class _SeriesAttention(torch.autograd.Function):
    """Degree-1 attention on PolyTensors (wrapper autograd)."""

    @staticmethod
    def forward(ctx, query, key, value, mask, causal, scale):
        from ._tensor import PolyTensor

        flat, plan, o0, o1, lse, r = _run_forward(query, key, value, mask, causal, scale)
        ctx.save_for_backward(*flat, o0, o1, lse, r)
        ctx.plan = plan
        ctx.shapes = (query.shape, key.shape)
        ctx.poly = tuple(isinstance(x, PolyTensor) for x in (query, key, value))
        return PolyTensor((o0.view(query.shape), o1.view(query.shape)))

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        from ._tensor import PolyTensor

        q0, q1, k0, k1, v0, v1, o0, o1, lse, r = ctx.saved_tensors
        shape_q, shape_k = ctx.shapes
        g0, g1 = gradient.coeffs if isinstance(gradient, PolyTensor) else (gradient, torch.zeros_like(gradient))
        g0, g1 = (_flat(g, shape_q) for g in (g0, g1))
        # Gradient series coefficient j = VJP w.r.t. input coefficient 1 - j, with cotangents (c0, c1) = (G1, G0).
        dq0, dq1, dk0, dk1, dv0, dv1 = backward(q0, q1, k0, k1, v0, v1, o0, o1, lse, r, g1, g0, ctx.plan)
        grads = (PolyTensor((dq1.view(shape_q), dq0.view(shape_q))),
                 PolyTensor((dk1.view(shape_k), dk0.view(shape_k))),
                 PolyTensor((dv1.view(shape_k), dv0.view(shape_k))))
        return (*(g if is_poly else None for g, is_poly in zip(grads, ctx.poly)), None, None, None)


def attention(query, key, value, attn_mask=None, is_causal=False, scale=None):
    """Fused degree-1 attention for an eligible call (see ``eligible``)."""
    from ._tensor import PolyTensor, _coefficient_autograd_enabled

    L, S = query.shape[-2], key.shape[-2]
    mask = _square_mask(attn_mask, L, S)
    scale = default_scale(query.shape[-1]) if scale is None else scale
    if _coefficient_autograd_enabled.get():
        # Ordinary coefficient graphs: differentiate the coefficient map itself.
        shape_q, shape_k = query.shape, key.shape
        qc, kc, vc = (_coefficients(x, query) for x in (query, key, value))
        flat = [c.expand(shape_q).reshape(-1, 1, L, shape_q[-1]) for c in qc] + \
               [c.expand(shape_k).reshape(-1, 1, S, shape_k[-1]) for c in (*kc, *vc)]
        o0, o1 = CoefficientAttention.apply(*flat, mask, is_causal, scale)
        return PolyTensor((o0.view(shape_q), o1.view(shape_q)), requires_grad=False)
    if torch.is_grad_enabled() and any(isinstance(x, PolyTensor) and x.requires_grad for x in (query, key, value)):
        return _SeriesAttention.apply(query, key, value, mask, is_causal, scale)
    _, _, o0, o1, _, _ = _run_forward(query, key, value, mask, is_causal, scale)
    return PolyTensor((o0.view(query.shape), o1.view(query.shape)))
