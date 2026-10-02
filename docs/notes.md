# Implementation and numerical notes

This is an early package, not a drop-in replacement for arbitrary PyTorch
programs. The library is separated into `_tensor.py` (datatype and autograd
contexts), `_dispatch.py` (PyTorch operator rules), and `_series.py` (coefficient
arithmetic). Only `PolyTensor` is public.

## What is implemented

The tested core includes polynomial arithmetic, nonnegative integer powers,
exp/log/sqrt/rsqrt, sine/cosine, sigmoid/tanh/SiLU/GELU, softmax/log-softmax and
single-axis logsumexp, matrix and vector products, basic shape operations,
reductions, and small forward-through-SGD calculations. Additional inherited
rules cover convolution, embedding, pooling, dropout, and some loss and attention
operations; their presence is not a claim of complete forward/backward coverage.
An unsupported dispatched operation raises `NotImplementedError`.

There is no hard-coded maximum order. Each result contains `degree + 1` tensors.
A polynomial product generally needs O(degree²) tensor operations. Increasing
tensor size and derivative order increases both memory and computation.

## Not implemented or not established

- A general adapter that makes arbitrary modules, parameters, optimizer states,
  and training pipelines polynomial-aware. Replacing only data inputs is
  insufficient when ordinary parameters receive in-place polynomial updates.
  With complex input directions, mutable parameters and state also need complex
  higher-coefficient storage, including initially zero coefficients.
  Tiny adapted modules and SGD with `foreach=False` are tested. Adam, fused or
  foreach optimizers, and arbitrary optimizer state are not supported generally.
- Full PyTorch operator coverage: examples of missing rules include negative or
  fractional powers, linear solves/determinants, native layer/batch normalization,
  and several indexing backward operations (`scatter_add`, `index_add`, and
  related updates). Forward support does not imply backward support.
- General fused attention support. The inherited CPU attention decomposition
  has no corresponding fused backward rule. Fully masked rows can produce NaNs
  because it uses ordinary softmax on an all-negative-infinity row. Prefer an
  explicitly implemented, tested attention calculation; this is a known bug.
- General loss options. The direct cross-entropy rule rejects soft targets,
  label smoothing, and polynomial class weights/targets. PyTorch may decompose a
  public loss differently across releases; no complete loss-option compatibility
  guarantee is made.
- Multivariate coefficient storage for independent directions or a complete
  derivative tensor. All polynomial inputs use the same scalar variable.
- Arbitrary precision or custom numeric backends. Precision is limited to the
  floating-point/complex dtypes and kernels available in PyTorch. Basic arithmetic
  is tested in float16, bfloat16, float32, float64, complex64, and complex128;
  this does not imply every operation/device supports every dtype.
- Comprehensive compatibility with CUDA/MPS, distributed execution, autocast,
  `torch.compile`, `torch.func`/`vmap`, custom autograd functions, serialization,
  or arbitrary views and strides. Storage-dependent operations such as
  `as_strided` are especially sensitive to coefficients with different layouts.
- A PyTorch version matrix. This extraction was tested on Python 3.12 and
  PyTorch 2.14.1, CPU. The declared minimum is PyTorch 2.4, the source's intended
  baseline, but it has not been revalidated here. Private wrapper/dispatch APIs
  can change between PyTorch releases.

## Complex directions

For a real base value, complex directions extend the derivatives of the real
computation multilinearly to complex vectors. The wrapper retains a real dtype
for its constant coefficient, while higher coefficients may be complex. This
allows native real softmax and activation kernels at the expansion point.
Precision conversions such as `.double()` preserve the imaginary parts of these
higher coefficients. Explicit complex-to-real coefficient storage writes are
rejected rather than silently discarding imaginary parts.

This convention is not differentiation of every non-holomorphic complex
function along a complex-valued curve. For example, PyTorch treats conjugation
and `.real` of a real-typed wrapper as identities, including its directions;
`.imag` is unavailable on a real-typed wrapper. Access ordinary `.coeffs` to
project individual coefficients explicitly. A complex constant coefficient uses
PyTorch's complex semantics instead: native real-only kernels may reject it,
and reverse-mode gradients involve conjugation. Complex-base training is not
interchangeable with complex directions through a real training computation.

## Correctness and stability fixes made during extraction

- Mixed degrees previously truncated silently or failed depending on operand
  order; they now raise consistently.
- `full_like` previously filled every coefficient, creating false derivatives.
  Constant factories now set higher coefficients to zero and honor dtype/device.
- Gradient-buffer allocation previously cast complex directions into real
  storage, silently losing imaginary gradients during SGD. Allocations and
  conversions now preserve each coefficient's real/complex precision.
- Matrix products could confuse the coefficient axis with batch axes or reject
  complex directions multiplied by real matrices. Batch alignment and
  coefficientwise dtype promotion are corrected.
- Reusing the same coefficient storage for multiple orders could corrupt
  in-place updates. Construction now gives overlapping orders independent
  storage and materializes coefficients expanded by constructor broadcasting.
- Python scalar directions no longer round through float32 before conversion
  to float64/complex128.
- `addmm(..., beta=0)` now ignores nonfinite bias values as PyTorch does.
- Loss rules no longer silently project polynomial targets/class weights onto
  their constants. Unsupported polynomial weights/targets raise explicitly.
- Reciprocal and inverse-square-root recurrences previously squared/cubed a
  large inverse before multiplying a tiny direction. For example, float32
  `x(t)=1e-20*(1+t)` produced overflow despite representable reciprocal
  coefficients. Normalized recurrences avoid that intermediate overflow.
- Logarithm/exponential recurrences avoid some unnecessary order-scaled
  intermediates. Logsumexp derivatives now use normalized softmax probabilities:
  `exp(x - logsumexp(x))` can sum to two instead of one for two equal logits
  near `1e20`, corrupting even the first derivative.

Native constant coefficients are retained where supplied by an existing native
kernel. Higher coefficients need not match the exact rounding order of PyTorch's
native backward formulas; the tests check numerical correctness instead.

## Remaining sources of infinity and NaN

1. **Genuine derivative growth.** Small denominators amplify higher orders.
   A reciprocal along a line contains powers of the ratio of direction to base
   value. Long sequences of updates multiply derivatives through each step;
   unstable update dynamics can therefore create enormous coefficients even
   when parameter values stay finite. No finite floating-point dtype prevents
   this in general.
2. **Avoidable intermediate overflow remains in GELU.** With float32
   `x(t)=1e20 + 1e20*t`, degree 3, exact GELU produces NaNs in higher orders and
   its tanh approximation fails earlier. Squaring/cubing the input overflows,
   then saturated exponential/tanh terms produce `0 * inf`. This is a confirmed
   implementation issue; it has not been fixed by the recurrence changes above.
3. **Domain boundaries and masks.** Reciprocal at zero, square root at zero,
   logarithms outside their real domain, and all-masked softmax rows can have
   undefined or singular derivatives. Sqrt's recurrence divides by the base
   square root. No clipping or replacement of nonfinite coefficients is applied.
4. **Rounding and cancellation.** Low precision, many reductions, and subtracting
   similar coefficient values lose accuracy. float64/complex128 offer more
   precision and range but cannot cure an ill-conditioned computation.
5. **Converting coefficients to derivatives.** Multiplication by `k!` may
   overflow even when the stored coefficient is finite. Keep coefficient form
   when the calculation does not need raw derivatives.
6. **Wrapper lifetime and memory.** A retention scope holds every intermediate
   wrapper until exit. One scope around a long loop can consume much more memory
   than one per step. An uninitialized wrapper raises rather than substituting
   zeros. Coefficient autograd avoids that wrapper retention but still retains
   ordinary autograd graphs as requested by the caller.

To locate a failure, inspect `torch.isfinite(c).all()` for each ordinary tensor
in `.coeffs` after successive operations or updates. Compare degree zero with an
ordinary run using the same data and randomness, and check small-order results
against an independent derivative calculation. Scaling the direction by a
nonzero factor `s` scales coefficient `k` by `s**k` for a line input; this can
reduce intermediate growth, but undoing the scale can reintroduce overflow.
No automatic direction normalization or rescaling is performed.

## Validation

Run `pip install -e '.[test]'` followed by `python -m pytest`. Tests compare
coefficients with analytic formulas, ordinary PyTorch higher derivatives, and
independent functional training calculations. They include float32 overflow
regressions, degree-12 series, complex directions, and real/complex scalar and
small-module SGD. The README example and wheel/source-archive builds were also
checked. These small tests do not establish stability for long or large runs.
