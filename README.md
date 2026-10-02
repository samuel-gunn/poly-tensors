# PolyTensors

Forward-mode automatic differentiation in PyTorch.
Warning: All code is written by AI.

`PolyTensor((x, v), degree=n)` represents `x + t*v`, keeping powers of the scalar
`t` through order `n`. For a supported smooth function `f`, `f(...).coeffs[k]`
is the `k`th-order Taylor coefficient (i.e., the directional derivative divided by `k!`), and `.value` is the constant term.
Directions may be complex.

Install from this checkout: `pip install .`. The planned PyPI command is
`pip install polytensors`; the package has not been published yet.

Compute the Taylor coefficients of `exp(t)` at zero:

```python
import torch
from polytensors import PolyTensor

Z = PolyTensor((0, 1), degree=5)
Y = torch.exp(Z)
print(torch.stack(Y.coeffs))  # coefficients 1/k! for k = 0, ..., 5
```

For training, replace all trainable parameters with `PolyTensor`s **before
constructing the optimizer**, using the same degree as the polynomial inputs.
Keep their initial values and set higher coefficients to zero unless you are
perturbing the initialization. Enable gradients; for `nn.Module` parameters,
assign replacements as `torch.nn.Parameter(PolyTensor(...))`. This provides
storage for derivatives acquired during in-place optimizer updates. For complex
directions, initialize higher coefficients with complex zeros. Any existing
optimizer state that acquires derivatives must also have polynomial storage.

This example fits a linear regression to fixed signal data `Y = X @ A + b + E`
and independent Gaussian data `X_prime, Y_prime`. Their variances match when
averaged over the random signal and data. The independent examples receive loss
weight `1 - Z`; derivatives at `Z = 0` describe reducing their contribution,
holding the datasets and sampled minibatches fixed.

```python
import torch
from polytensors import PolyTensor

Z = PolyTensor((0, 1), degree=5)  # the formal variable t
n, d, noise_std = 64, 3, 0.1
A, b = torch.randn(d), torch.randn(())  # secret weights and bias
X = torch.randn(n, d)
E = noise_std * torch.randn(n)
Y = X @ A + b + E
X_prime = torch.randn(n, d)
Y_prime = (d + 1 + noise_std**2)**0.5 * torch.randn(n)  # same variance as Y

w = (torch.zeros(d) + 0 * Z).requires_grad_()
c = (torch.zeros(()) + 0 * Z).requires_grad_()
optimizer = torch.optim.SGD([w, c], lr=0.1, foreach=False)

for _ in range(10):
    i, j = torch.randint(n, (16,)), torch.randint(n, (16,))
    optimizer.zero_grad(set_to_none=True)
    with PolyTensor.retain_wrappers():
        signal_loss = (X[i] @ w + c - Y[i]).square().mean()
        noise_loss = (X_prime[j] @ w + c - Y_prime[j]).square().mean()
        loss = (signal_loss + (1 - Z) * noise_loss) / 2
        loss.backward()
        optimizer.step()

print(w.value)          # fitted weights with both datasets equally weighted
print(w.coeffs[1])      # first derivative with respect to Z at zero
print(2 * w.coeffs[2])  # second derivative with respect to Z at zero
```

`retain_wrappers()` keeps intermediate Python objects and their coefficients
alive through backward, releasing its extra references on exit. Use one scope
per forward/backward step; wrapping the entire loop retains every intermediate.

Support is incomplete: arbitrary PyTorch pipelines are not yet supported.
All polynomial operands must have the same degree. Available precision depends
on PyTorch; use `float64` / `complex128` for more demanding calculations.
See [API details](docs/api.md) and [limitations and numerical notes](docs/notes.md).
