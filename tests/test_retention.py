import torch

from polytensors import PolyTensor


def _poly(seed, shape=(4, 3), degree=2, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    coeffs = tuple(torch.randn(shape, generator=g, dtype=torch.float64) for _ in range(degree + 1))
    return PolyTensor(coeffs, degree=degree, requires_grad=requires_grad)


def test_isolated_scope_releases_its_wrappers_inside_an_outer_scope():
    x = _poly(0, requires_grad=True)
    with PolyTensor.retain_wrappers() as outer:
        y = torch.tanh(x)
        before = len(outer)
        with PolyTensor.retain_wrappers(isolated=True) as inner:
            (torch.sin(y) * y).sum().backward(retain_graph=True)
            assert len(inner) > 0
        assert len(inner) == 0
        assert len(outer) == before
        with PolyTensor.retain_wrappers() as nested:
            (torch.sin(y) * y).sum()
            assert len(nested) > 0
        assert len(outer) > before


def test_chunked_backward_through_a_detached_leaf_matches_direct_backward():
    """A loss split into chunks, each back-propagated in an isolated scope into a detached copy of a shared
    intermediate, then once through the shared part, gives the direct gradient."""
    def loss_of(h, cols):
        return (torch.exp(torch.sin(h[:, cols])) * h[:, cols]).sum()

    x = _poly(1, requires_grad=True)
    with PolyTensor.retain_wrappers():
        loss_of(torch.tanh(x) * x, slice(None)).backward()
    direct = x.grad.coeffs

    x2 = _poly(1, requires_grad=True)
    with PolyTensor.retain_wrappers():
        h = torch.tanh(x2) * x2
        with torch.no_grad():
            leaf = torch.nn.Parameter(h.detach(), requires_grad=True)
        for c in range(3):
            with PolyTensor.retain_wrappers(isolated=True):
                loss_of(leaf, slice(c, c + 1)).backward()
        h.backward(leaf.grad)
    for a, b in zip(direct, x2.grad.coeffs):
        torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-12)
