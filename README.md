# PolyTensors

Forward-mode automatic differentiation in PyTorch.
Warning: All code is written by AI.

`PolyTensor((x, v), degree=n)` represents `x + t*v`, keeping powers of the scalar
`t` through order `n`. For a supported smooth function `f`, `f(...).coeffs[k]`
is the `k`th-order Taylor coefficient (i.e., the directional derivative divided by `k!`), and `.value` is the constant term.
Directions may be complex.

Install from this checkout: `pip install .`. The planned PyPI command is
`pip install polytensors`; the package has not been published yet.

For training, replace all trainable parameters with `PolyTensor`s **before
constructing the optimizer**, using the same degree as the polynomial inputs.
Keep their initial values and set higher coefficients to zero unless you are
perturbing the initialization. Enable gradients; for `nn.Module` parameters,
assign replacements as `torch.nn.Parameter(PolyTensor(...))`. This provides
storage for derivatives acquired during in-place optimizer updates. For complex
directions, initialize higher coefficients with complex zeros. Any existing
optimizer state that acquires derivatives must also have polynomial storage.

This example differentiates two gradient-descent updates with respect to the
initial weight:

```python
import torch
from polytensors import PolyTensor

w = PolyTensor((torch.tensor(0.5), torch.tensor(1.0)),
               degree=3, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1, foreach=False)
for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    with PolyTensor.retain_wrappers():
        loss = (w - 1).pow(4) / 4
        loss.backward()
        optimizer.step()

print(w.value)            # approximately 0.5241
print(w.coeffs[1])        # first derivative: approximately 0.8591
print(2 * w.coeffs[2])    # second derivative: approximately 0.5289
```

`retain_wrappers()` keeps intermediate Python objects and their coefficients
alive through backward, releasing its extra references on exit. Use one scope
per forward/backward step; wrapping the entire loop retains every intermediate.

Support is incomplete: arbitrary PyTorch pipelines are not yet supported.
All polynomial operands must have the same degree. Available precision depends
on PyTorch; use `float64` / `complex128` for more demanding calculations.
See [API details](docs/api.md) and [limitations and numerical notes](docs/notes.md).
