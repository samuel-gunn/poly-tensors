import pytest
import torch

import polytensors
from polytensors import PolyTensor


def test_only_public_export():
    assert polytensors.__all__ == ["PolyTensor"]


def test_validation():
    with pytest.raises(ValueError, match="at least one"):
        PolyTensor([])
    for degree in (-1, -3):
        with pytest.raises(ValueError):
            PolyTensor([1.0], degree=degree)
    for degree in (True, 1.2):
        with pytest.raises(TypeError):
            PolyTensor([1.0], degree=degree)
    with pytest.raises(ValueError, match="at least"):
        PolyTensor([1.0, 2.0], degree=0)
    with pytest.raises(TypeError, match="floating-point"):
        PolyTensor([torch.tensor(1)])
    with pytest.raises(ValueError, match="same device"):
        PolyTensor([torch.tensor(1.0), torch.empty((), device="meta")])
    with pytest.raises(TypeError, match="nested"):
        PolyTensor([PolyTensor([1.0])])


def test_integer_literals_and_constant_padding():
    x = PolyTensor([1, 2], degree=4)
    assert x.dtype == torch.get_default_dtype()
    assert [c.item() for c in x.coeffs] == [1, 2, 0, 0, 0]
    constant = PolyTensor.constant(2, degree=3)
    assert [c.item() for c in constant.coeffs] == [2, 0, 0, 0]
    with pytest.raises(ValueError, match="no tangent"):
        _ = PolyTensor([1.0]).tangent


def test_precision_and_complex_direction_preserved():
    x = PolyTensor([torch.tensor(1.0, dtype=torch.float64), 2 + 3j], degree=3)
    assert x.dtype == torch.float64
    assert all(c.dtype == torch.complex128 for c in x.coeffs[1:])
    assert x.tangent == 2 + 3j
    y = x.exp()
    for k, coefficient in enumerate(y.coeffs):
        import math
        expected = torch.tensor(math.e * (2 + 3j) ** k / math.factorial(k))
        torch.testing.assert_close(coefficient.to(torch.complex128), expected.to(torch.complex128))


def test_reused_storage_does_not_couple_mutable_orders():
    zero = torch.tensor(0.0)
    x = PolyTensor([torch.tensor(1.0), zero, zero], degree=2)
    x.add_(PolyTensor([0.0, 2.0, 3.0]))
    assert [c.item() for c in x.coeffs] == [1.0, 2.0, 3.0]


def test_coefficient_broadcasting():
    x = PolyTensor([torch.tensor([1.0, 2.0]), torch.tensor(3.0)], degree=2)
    square = x.square()
    torch.testing.assert_close(square.value, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(square.tangent, torch.tensor([6.0, 12.0]))
    torch.testing.assert_close(square.coeffs[2], torch.tensor([9.0, 9.0]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                  torch.float64, torch.complex64, torch.complex128])
def test_basic_arithmetic_preserves_dtype(dtype):
    x = PolyTensor([torch.tensor(2.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)], degree=3)
    actual = (x * x + 1).coeffs
    for c, expected in zip(actual, (5, 4, 1, 0)):
        assert c.dtype == dtype
        assert c.item() == expected


def test_backward_requires_explicit_gradient_for_vector():
    x = PolyTensor([torch.tensor([1.0, 2.0])], degree=1, requires_grad=True)
    with pytest.raises(RuntimeError, match="scalar"):
        x.backward()


def test_unimplemented_operation_fails_explicitly():
    x = PolyTensor([torch.eye(2)], degree=1)
    with pytest.raises(NotImplementedError, match="PolyTensor"):
        torch.linalg.det(x)


def test_python_direction_does_not_round_through_float32():
    direction = 0.12345678912345678
    x = PolyTensor((torch.tensor(0.0, dtype=torch.float64), direction))
    assert x.tangent.item() == direction
    z = PolyTensor((torch.tensor(0.0, dtype=torch.float64), complex(direction, direction)))
    assert z.tangent.item() == complex(direction, direction)


def test_constructor_broadcasting_is_safe_for_inplace_updates():
    x = PolyTensor((torch.tensor([1.0, 2.0]), torch.tensor(1.0)), degree=2)
    x.mul_(2)
    torch.testing.assert_close(x.tangent, torch.tensor([2.0, 2.0]))
