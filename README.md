# PolyTensors

Higher-order directional automatic differentiation for PyTorch. The only public
export is `PolyTensor`.

`PolyTensor((x, v), degree=n)` represents `x + t*v`, keeping powers of the scalar
`t` through order `n`. For a supported smooth function `f`, `f(...).coeffs[k]`
is the order-`k` directional derivative divided by `k!`; `.value` is coefficient
zero. Directions may be complex. There is no fixed maximum order.

Install from this checkout: `pip install .`. The planned PyPI command is
`pip install polytensors`; the package has not been published yet.

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

Support is incomplete: arbitrary PyTorch pipelines are not yet supported.
All polynomial operands must have the same degree. Available precision depends
on PyTorch; use `float64` / `complex128` for more demanding calculations.
See [API details](docs/api.md) and [limitations and numerical notes](docs/notes.md).
