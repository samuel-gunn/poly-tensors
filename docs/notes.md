# PolyTensors implementation and numerical notes

This is an early package, not a drop-in replacement for arbitrary PyTorch
programs. `_tensor.py` defines the datatype and autograd contexts;
`_dispatch.py` supplies PyTorch operator rules. Private modules implement series
arithmetic, activations, normalization, and attention. Only `PolyTensor` is public.

## What is implemented

The tested core includes polynomial arithmetic, nonnegative integer powers,
exp/log/sqrt/rsqrt, sine/cosine, sigmoid/tanh/SiLU/GELU, softmax/log-softmax and
single-axis logsumexp, matrix and vector products, basic shape operations,
reductions, and small forward-through-SGD calculations. Additional inherited
rules cover convolution, embedding, pooling, dropout, and some loss and attention
operations; their presence is not a claim of complete forward/backward coverage.
An unsupported dispatched operation raises `NotImplementedError`.

There is no hard-coded maximum order. Each result contains `degree + 1` tensors.
A polynomial product generally needs O(degree²) tensor operations; the current
activation composition can require O(degree³). Increasing tensor size and
derivative order increases both memory and computation. Stable activation
evaluation also uses wider internal mantissas where the device supports them.

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
- General fused attention support. PyTorch's math attention path and CPU flash
  attention forward/backward are tested. CUDA fused attention and grouped-query
  attention are not established. CPU flash backward requires zero dropout.
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

## Numerical stability changes

- Exact and tanh-approximate GELU evaluate derivative terms with a separate
  binary exponent. Gaussian factors therefore do not round to zero before
  multiplication by large directions. Extreme inputs are replaced by their
  limiting linear/constant behavior only after a bound involving the degree,
  direction magnitudes, and dtype establishes that the requested tail cannot
  be represented. Unsafe intermediate squares/cubes are avoided.
- Sigmoid, tanh, and SiLU derivatives use the input and the decaying exponential
  tail, rather than a rounded output such as `sigmoid(x) == 1`. This preserves
  representable higher coefficients in saturated regions. Their backward
  rules and GELU's multiply incoming gradients before rounding tiny derivatives.
- Exponential and square-root series keep intermediate products scaled, avoiding
  premature underflow and overflow. For example, float64
  `exp(-1000 + 1e150*t)` has a zero stored constant but nonzero first and second
  coefficients.
- Softmax, log-softmax, and logsumexp remove common offsets from higher input
  coefficients and retain tiny probabilities in scaled form during coefficient
  calculations. Analytic coefficient-autograd rules preserve these properties
  in backward. Softmax/log-softmax wrapper backward also saves the input series
  instead of reconstructing derivatives from rounded probabilities.
- Attention excludes masked entries before normalization. Fully masked rows
  produce zero output coefficients and zero gradients; their CPU flash
  auxiliary log-normalizer is zero, following PyTorch's convention. Both Boolean
  masks and additive negative-infinity masks are covered. An ordinary unmasked
  softmax of all negative infinities remains undefined.
- Attention applies its scale before the query/key product to avoid overflowing
  an unscaled score. Half/bfloat16 inputs use float32 internal accumulation.
  Complex higher coefficients retain their imaginary parts in dtype conversions.
- Wrapper autograd for exp and single-axis logsumexp now saves the input and
  uses explicit backward rules. These run above PyTorch's native autograd layer;
  changing the forward dispatch rule alone could not repair its saved-output
  formulas. Equal float32 logits near `1e20` now give logsumexp gradients
  `[0.5, 0.5]`, and large incoming gradients can recover underflowed exp tails.
  First, second, and third reverse derivatives are tested.
- Extended-range coefficients now survive supported operator boundaries.
  Arithmetic, matrix products, sums/means, ordinary shape/index operations,
  clones, detaches, and dtype conversions carry private binary scales.
  Exp, activations, softmax/log-softmax, and attention provide those scales.
  Thus `exp(-1000) * 1e300` and attention with scores `[0, -1000]` and values
  `[0, 1e300]` recover contributions near `5.076e-135` in float64.
  Attention's value and score gradients also keep products scaled, including
  cases where the incoming gradient times a value exceeds the dtype's range.
- Scaled matrix products use the native matrix kernel when the exponent range
  is safe and bounded blocks of scaled products otherwise. The slower path
  avoids a full query-by-key-by-value temporary tensor. Retaining scales adds
  memory and arithmetic overhead; large-workload performance is not established.
- Native conjugate/negative view handling and coefficient version counters are
  enabled within dispatch. Complex backward values are interpreted correctly,
  and writes through aliases invalidate cached scales. In-place arithmetic
  computes from the saved scales and updates the destination's scale cache.

Native constant coefficients are retained where supplied by an existing native
kernel, but later scaled arithmetic can recover information that an ordinary
tensor operation already rounded away. Results therefore need not match native
PyTorch's rounding order, including at coefficient zero; the tests check
numerical correctness instead.

## Remaining sources of infinity and NaN

1. **Genuine derivative growth.** Small denominators amplify higher orders.
   A reciprocal along a line contains powers of the ratio of direction to base
   value. Long sequences of updates multiply derivatives through each step;
   unstable update dynamics can therefore create enormous coefficients even
   when parameter values stay finite. No finite floating-point dtype prevents
   this in general.
2. **Boundaries of extended-range storage.** Public `.value` and `.coeffs`
   tensors still have their declared dtype's finite range. Using those ordinary
   tensors in subsequent computations, or constructing a new PolyTensor from
   them, loses the private scales. Rules without scale propagation, including
   convolution and storage-based `as_strided`, also form rounding boundaries.
   Nonlinear rules generally consume ordinary input coefficients; this does not
   enable expressions such as `sin(exp(1000))` to evaluate an out-of-range
   argument. Manual coefficient writes invalidate the scales; inference-mode
   tensors do not retain them because PyTorch disables mutation version counters.
   Other inherited native backward formulas have not all been audited.
   Ordinary coefficient autograd also still has finite-range internal adjoints:
   for float64 `a = b = 1e300`, the coefficient-mode calculation
   `a*b - b*b` has a recovered zero value but can give an infinite gradient with
   respect to `a`, although the mathematical gradient is `1e300`. Its backward
   graph would need to carry scaled gradients through every intermediate, not
   just the analytic boundaries implemented here. Wrapper autograd and
   coefficient autograd therefore have different numerical limits.
3. **Domain boundaries and invalid inputs.** Reciprocal at zero, square root at
   zero, and logarithms outside their real domain have undefined or singular
   derivatives. Sqrt's recurrence divides by the base square root. Unmasked NaNs
   and positive infinities are not repaired. There is no blanket clipping or
   replacement of nonfinite coefficients.
4. **Rounding and cancellation.** Low precision, many reductions, and subtracting
   similar coefficient values lose accuracy. float64/complex128 offer more
   precision and range but cannot cure an ill-conditioned computation. Scaled
   arithmetic extends exponent range, not mantissa precision, and does not
   guarantee stability through an arbitrary number of reverse passes.
5. **Converting coefficients to derivatives.** Multiplication by `k!` may
   overflow even when the stored coefficient is finite. Keep coefficient form
   when the calculation does not need raw derivatives.
6. **Wrapper lifetime and memory.** A retention scope holds every intermediate
   wrapper until exit. One scope around a long loop can consume much more memory
   than one per step. An uninitialized wrapper raises rather than substituting
   zeros. Coefficient autograd avoids that wrapper retention but still retains
   ordinary autograd graphs as requested by the caller.

To locate a failure, inspect `torch.isfinite(c).all()` for each ordinary tensor
in `.coeffs` after successive operations or updates; a nonfinite materialized
coefficient can still have a recoverable private scale. Compare degree zero with an
ordinary run using the same data and randomness, and check small-order results
against an independent derivative calculation. Scaling the direction by a
nonzero factor `s` scales coefficient `k` by `s**k` for a line input; this can
reduce intermediate growth, but undoing the scale can reintroduce overflow.
No automatic direction normalization or rescaling is performed.

## Validation

Run `pip install -e '.[test]'` followed by `python -m pytest`. Tests compare
coefficients with analytic formulas, ordinary PyTorch higher derivatives, and
independent functional training calculations. They include float32 overflow
regressions in float32/float64, degree-12 series, saturated activation tails,
fully masked attention, complex directions, coefficient gradients and second
reverse derivatives, and real/complex scalar and small-module SGD. The README
example and wheel/source-archive builds were also checked. These small tests do
not establish stability for long or large runs or on untested devices.
