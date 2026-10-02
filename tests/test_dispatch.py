import pytest
import torch

from polytensors import PolyTensor
from polytensors._dispatch import dispatch


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("operation", [torch.add, torch.mul, torch.cat])
def test_mixed_degrees_raise_in_both_argument_orders(reverse, operation):
    operands = [PolyTensor((torch.ones(2),), degree=d) for d in (1, 3)]
    if reverse:
        operands.reverse()
    with pytest.raises(ValueError, match="same degree"):
        operation(operands) if operation is torch.cat else operation(*operands)


@pytest.mark.parametrize("factory, value", [(torch.ones_like, 1), (torch.full_like, 2.5)])
def test_constant_factories_have_zero_derivatives_and_honor_dtype(factory, value):
    x = PolyTensor((torch.tensor([0.1, 0.2]), torch.tensor([3.0, 4.0])), degree=3)
    args = (x, value) if factory is torch.full_like else (x,)
    result = factory(*args, dtype=torch.float64, device="cpu")
    assert result.dtype == torch.float64
    torch.testing.assert_close(result.value, torch.full((2,), value, dtype=torch.float64))
    for coefficient in result.coeffs[1:]:
        assert coefficient.dtype == torch.float64
        assert torch.count_nonzero(coefficient) == 0


@pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3)])
@pytest.mark.parametrize("poly_on_left", [False, True])
def test_matmul_order_axis_does_not_overlap_batch_axes(batch_shape, poly_on_left):
    generator = torch.Generator().manual_seed(11)
    coefficients = tuple(torch.randn((3, 3), generator=generator) for _ in range(2))
    polynomial = PolyTensor(coefficients)
    ordinary = torch.randn((*batch_shape, 3, 3), generator=generator)
    operands = (polynomial, ordinary) if poly_on_left else (ordinary, polynomial)
    # Exercise both public decomposition and the specialized operator rule.
    for result in (torch.matmul(*operands), dispatch(torch.ops.aten.matmul.default, (), operands)):
        for coefficient, actual in zip(coefficients, result.coeffs):
            expected = coefficient @ ordinary if poly_on_left else ordinary @ coefficient
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("operation, shape", [(torch.matmul, (2, 3, 3)), (torch.bmm, (2, 3, 3)), (torch.mm, (3, 3))])
@pytest.mark.parametrize("poly_on_left", [False, True])
def test_matrix_products_preserve_real_base_and_complex_direction(operation, shape, poly_on_left):
    generator = torch.Generator().manual_seed(9)
    base = torch.randn(shape, generator=generator, dtype=torch.float64)
    direction = torch.randn(shape, generator=generator, dtype=torch.complex128)
    ordinary = torch.randn(shape, generator=generator, dtype=torch.float64)
    polynomial = PolyTensor((base, direction))
    operands = (polynomial, ordinary) if poly_on_left else (ordinary, polynomial)
    result = operation(*operands)
    expected_base = operation(base, ordinary) if poly_on_left else operation(ordinary, base)
    complex_ordinary = ordinary.to(torch.complex128)
    expected_direction = operation(direction, complex_ordinary) if poly_on_left else operation(complex_ordinary, direction)
    assert result.value.dtype == torch.float64
    torch.testing.assert_close(result.value, expected_base)
    torch.testing.assert_close(result.tangent, expected_direction)


def test_two_polynomial_matrices_include_complex_cross_terms():
    a = PolyTensor((torch.eye(2, dtype=torch.float64), torch.full((2, 2), 1j, dtype=torch.complex128)), degree=2)
    b = PolyTensor((torch.eye(2, dtype=torch.float64) * 2, torch.full((2, 2), 2j, dtype=torch.complex128)), degree=2)
    result = a @ b
    torch.testing.assert_close(result.value, 2 * torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(result.tangent, torch.full((2, 2), 4j, dtype=torch.complex128))
    torch.testing.assert_close(result.coeffs[2], torch.full((2, 2), -4 + 0j, dtype=torch.complex128))


@pytest.mark.parametrize("polynomial_bias", [False, True])
def test_addmm_beta_zero_ignores_nonfinite_bias(polynomial_bias):
    x = torch.full((2, 2), float("nan"))
    if polynomial_bias:
        x = PolyTensor((x, torch.full_like(x, float("inf"))))
    a = PolyTensor((torch.eye(2), torch.eye(2) * 3))
    b = torch.eye(2) * 2
    result = torch.addmm(x, a, b, beta=0)
    torch.testing.assert_close(result.value, torch.eye(2) * 2)
    torch.testing.assert_close(result.tangent, torch.eye(2) * 6)


def test_vector_products_propagate_derivatives():
    a = PolyTensor((torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])))
    b = PolyTensor((torch.tensor([2.0, 3.0]), torch.tensor([1.0, 2.0])))
    result = torch.dot(a, b)
    torch.testing.assert_close(result.value, torch.tensor(8.0))
    torch.testing.assert_close(result.tangent, torch.tensor(23.0))
    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    result = torch.mv(matrix, a)
    torch.testing.assert_close(result.tangent, matrix @ a.tangent)


@pytest.mark.parametrize("operation", [torch.conj, torch.conj_physical, torch.real, torch.imag, torch.view_as_real])
def test_complex_operations_apply_to_every_coefficient(operation):
    coefficients = (torch.tensor([1 + 2j, 3 + 4j]), torch.tensor([5 + 6j, 7 + 8j]))
    result = operation(PolyTensor(coefficients))
    for actual, coefficient in zip(result.coeffs, coefficients):
        torch.testing.assert_close(actual, operation(coefficient))


def test_view_as_complex_preserves_directions():
    coefficients = (torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
    result = torch.view_as_complex(PolyTensor(coefficients))
    torch.testing.assert_close(result.value, torch.tensor([1 + 2j]))
    torch.testing.assert_close(result.tangent, torch.tensor([3 + 4j]))


@pytest.mark.parametrize("operation", [torch.sin, torch.cos])
def test_trigonometric_functions_accept_complex_directions(operation):
    base = torch.tensor([0.1, 0.5], dtype=torch.float64)
    direction = torch.tensor([1 + 2j, -2 + 1j], dtype=torch.complex128)
    result = operation(PolyTensor((base, direction), degree=2))
    derivative = torch.cos(base) if operation is torch.sin else -torch.sin(base)
    torch.testing.assert_close(result.value, operation(base))
    torch.testing.assert_close(result.tangent, derivative * direction)
    torch.testing.assert_close(result.coeffs[2], -operation(base) * direction.square() / 2)


def test_log_softmax_half_to_float_converts_every_coefficient():
    x = PolyTensor((torch.tensor([0.0, 1.0], dtype=torch.float16), torch.ones(2, dtype=torch.float16)))
    result = dispatch(torch.ops.aten._log_softmax.default, (), (x, 0, True))
    assert all(c.dtype == torch.float32 for c in result.coeffs)
    torch.testing.assert_close(result.value, torch.log_softmax(x.value.float(), dim=0))
    torch.testing.assert_close(result.tangent, torch.zeros(2))


@pytest.mark.parametrize("operation", [torch.ops.aten.cross_entropy_loss.default, torch.ops.aten.nll_loss_forward.default])
def test_loss_rejects_polynomial_class_weights(operation):
    x = PolyTensor((torch.tensor([[0.1, 0.2]]), torch.ones(1, 2)))
    weights = PolyTensor((torch.ones(2), torch.ones(2)))
    with pytest.raises(NotImplementedError, match="ordinary targets and class weights"):
        dispatch(operation, (), (x, torch.tensor([1]), weights))


def test_cross_entropy_rejects_polynomial_targets():
    x = PolyTensor((torch.tensor([[0.1, 0.2]]), torch.ones(1, 2)))
    target = PolyTensor((torch.tensor([[0.2, 0.8]]), torch.zeros(1, 2)))
    with pytest.raises(NotImplementedError, match="ordinary targets and class weights"):
        dispatch(torch.ops.aten.cross_entropy_loss.default, (), (x, target))


@pytest.mark.parametrize("operation", [lambda x: x.double(), lambda x: x.to(dtype=torch.float64), lambda x: x.to(torch.zeros((), dtype=torch.float64))])
def test_precision_conversion_preserves_complex_directions(operation):
    x = PolyTensor((torch.tensor([0.5]), torch.tensor([0.3 + 0.2j])), degree=2)
    result = operation(x)
    assert result.value.dtype == torch.float64
    assert result.tangent.dtype == torch.complex128
    torch.testing.assert_close(result.tangent, x.tangent.to(torch.complex128))


@pytest.mark.parametrize("factory", [torch.zeros_like, torch.empty_like])
def test_allocations_preserve_complex_direction_storage(factory):
    x = PolyTensor((torch.tensor([0.5]), torch.tensor([0.3 + 0.2j])))
    result = factory(x, dtype=torch.float64)
    assert result.value.dtype == torch.float64
    assert result.tangent.dtype == torch.complex128


@pytest.mark.parametrize("operation", [lambda a, b: a.copy_(b), lambda a, b: a.mul_(b), lambda a, b: a.div_(b)])
def test_inplace_copy_does_not_silently_discard_complex_directions(operation):
    destination = PolyTensor((torch.tensor([2.0]), torch.tensor([3.0])))
    source = PolyTensor((torch.tensor([0.5]), torch.tensor([0.3 + 0.2j])))
    before = tuple(c.clone() for c in destination.coeffs)
    with pytest.raises(RuntimeError, match="complex PolyTensor coefficients"):
        operation(destination, source)
    for actual, original in zip(destination.coeffs, before):
        torch.testing.assert_close(actual, original)


@pytest.mark.parametrize("method", ["add_", "sub_"])
def test_complex_update_rejection_does_not_partially_mutate_real_parameter(method):
    parameter = PolyTensor((torch.tensor(1.0), torch.tensor(0.0)), degree=2)
    update = PolyTensor((torch.tensor(2.0), torch.tensor(1.0 + 2.0j)), degree=2)
    before = tuple(c.clone() for c in parameter.coeffs)
    with pytest.raises(RuntimeError, match="complex PolyTensor coefficients"):
        getattr(parameter, method)(update)
    for actual, expected in zip(parameter.coeffs, before):
        torch.testing.assert_close(actual, expected)
