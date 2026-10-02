# API

```python
from polytensors import PolyTensor
```

`PolyTensor` is a `torch.Tensor` subclass. Internally it stores ordinary tensors
for the coefficients of a truncated polynomial in one scalar variable:

`p(t) = c[0] + c[1]*t + ... + c[n]*t**n`.

Operations keep coefficients through degree `n`. When the input is `x + t*v`,
coefficient `k` of the result is `D^k f(x)[v, ..., v] / k!`. Several polynomial
inputs share the same `t`. This does not compute a full derivative tensor or
all mixed derivatives in several independent directions at once.

## Construction and inspection

- `PolyTensor(coeffs, *, degree=None, requires_grad=False)` accepts a nonempty
  sequence of floating-point or complex tensors or Python numbers. Coefficients
  broadcast to a common shape. `degree` defaults to `len(coeffs) - 1`; a larger
  degree pads with zero coefficients. Negative degrees and truncating a supplied
  sequence are errors.
- `PolyTensor.constant(value, degree)` sets all higher coefficients to zero.
- `.coeffs` is the coefficient tuple; `.degree` is the truncation order;
  `.value` is coefficient zero; `.tangent` is coefficient one, and is unavailable
  at degree zero. These are ordinary tensors. Extracting `.value` discards the
  directional information from subsequent operations.
- Polynomial operands must have equal degrees. Ordinary tensors and scalars are
  treated as constants.

Coefficients use one device and a common real precision. A real constant may
have complex higher coefficients; `.dtype` describes the constant coefficient.
For example, a float64 value and complex64 direction use float64 / complex128
coefficients. Python complex directions are preserved. Integer tensor
coefficients are rejected; integer Python literals are converted to floating
point. Sparse and quantized coefficients are unsupported.

Construction generally shares storage with supplied tensors, so in-place updates
may modify those tensors. Supply clones when needed. Coefficients sharing storage
with another order are copied to keep orders independently mutable. Mutating
`.coeffs` manually, or replacing coefficients with different shapes/devices,
bypasses the wrapper's metadata and is unsupported.

## Reverse differentiation inside a forward calculation

For supported reverse-mode operations, construct polynomial parameters with
`requires_grad=True`. Their `.grad` is a `PolyTensor`, so an optimizer update can
propagate derivatives through the update. Use a separate
`with PolyTensor.retain_wrappers():` scope around each forward/backward pair
(and its update), as in the README. The scope prevents PyTorch from retaining a
wrapper's tensor metadata after its Python coefficient storage has been freed.
It releases its references on exit. Keep a scope alive through every backward
that needs its intermediates, including repeated backward calls.

Scalar `.backward()` seeds coefficient zero with one and higher coefficients
with zero. Nonscalar outputs require an explicit gradient. For SGD, use
`foreach=False`; optimizer and operator coverage is limited. Ordinary parameters
cannot receive an in-place polynomial update: parameters and optimizer state
that acquire derivatives must also be represented as polynomials.
For complex input directions, preallocate complex higher coefficients in every
mutable parameter and optimizer state, even when their initial derivatives are
zero: `PolyTensor((base, torch.zeros_like(base, dtype=torch.complex128)), degree=n)`
for a float64 real base. In-place updates cannot change an existing coefficient's
storage dtype.

An alternative is `with PolyTensor.coefficient_autograd():`. Make ordinary
coefficient tensors independent autograd leaves, and leave the wrapper's
`requires_grad=False`. Operations inside the context build graphs for those
ordinary tensors. Differentiating loss coefficient `k` with respect to a
parameter's coefficient zero gives coefficient `k` of its gradient, provided
other parameter coefficients are independent of coefficient zero. For repeated
updates, detach and recreate those independent leaves at each step; otherwise
this partial derivative also follows unwanted dependencies between coefficients.

For a complex loss coefficient and a real parameter base, obtain both components:

```python
real = torch.autograd.grad(coefficient.real, base, retain_graph=True)[0]
imag = torch.autograd.grad(coefficient.imag, base, retain_graph=True)[0]
gradient_coefficient = torch.complex(real, imag)
```

Use this split only for complex coefficients; ordinary real coefficients need
one gradient call. `tests/test_training.py` contains complete, independently
checked examples. Do not combine wrapper autograd with coefficient autograd, or
wrap coefficient-autograd values in `nn.Parameter`, whose construction detaches
the coefficients. General module adaptation is not provided yet.
