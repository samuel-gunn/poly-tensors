"""Triton kernels for fused degree-1 attention (see _fused_attention). Imported lazily."""

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
