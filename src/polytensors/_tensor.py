"""Tensor subclass and public PolyTensor API."""

import contextlib
import contextvars
import operator

import torch

from ._dispatch import dispatch


_retention_scopes = contextvars.ContextVar(
    "polytensor_retention_scopes",
    default=(),
)
_coefficient_autograd_enabled = contextvars.ContextVar(
    "polytensor_coefficient_autograd_enabled",
    default=False,
)


class PolyTensor(torch.Tensor):
    """A truncated tensor-valued power series in one scalar variable.

    ``PolyTensor((x, v), degree=n)`` represents ``x + t*v`` through order
    ``n``. After applying a supported smooth function, coefficient ``k`` is
    its order-k directional derivative divided by ``k!``. All operands must
    have the same degree. Real values may have complex directions; ``dtype``
    describes the constant coefficient, and higher coefficients can be complex.

    Coefficients broadcast to one shape and use a common device and real
    precision. Integer Python literals become floating-point tensors; integer
    tensor coefficients are rejected. Missing orders are filled with zeros.
    """

    @staticmethod
    def _coerce_coeffs(coeffs, degree=None):
        if isinstance(coeffs, PolyTensor):
            coeffs = coeffs.coeffs
        coeffs = tuple(coeffs)
        if not coeffs:
            raise ValueError("at least one coefficient is required")
        if degree is not None:
            if isinstance(degree, bool):
                raise TypeError("degree must be a nonnegative integer")
            degree = operator.index(degree)
            if degree < 0:
                raise ValueError("degree must be nonnegative")
        base = coeffs[0] if isinstance(coeffs[0], torch.Tensor) else torch.as_tensor(coeffs[0])
        if not isinstance(coeffs[0], torch.Tensor) and not (base.is_floating_point() or base.is_complex()):
            base = base.to(torch.get_default_dtype())
        coeffs = tuple(
            c if isinstance(c, torch.Tensor)
            else torch.as_tensor(
                c, device=base.device,
                dtype=torch.promote_types(base.dtype, torch.as_tensor(c).dtype),
            )
            for c in coeffs
        )
        for c in coeffs:
            if isinstance(c, PolyTensor):
                raise TypeError("nested PolyTensor coefficients are not supported")
            if c.layout != torch.strided:
                raise TypeError("coefficients must be dense strided tensors")
            if not (c.is_floating_point() or c.is_complex()):
                raise TypeError("tensor coefficients must have floating-point or complex dtype")
            if c.device != base.device:
                raise ValueError("all coefficients must be on the same device")
        dtype = coeffs[0].dtype
        for c in coeffs[1:]:
            dtype = torch.promote_types(dtype, c.dtype)
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        complex_directions = any(c.is_complex() for c in coeffs[1:])
        coeffs = tuple(
            c.to(dtype=dtype if c.is_complex() or (k > 0 and complex_directions) else real_dtype)
            for k, c in enumerate(coeffs)
        )
        base = coeffs[0]
        if degree is not None:
            if degree < len(coeffs) - 1:
                raise ValueError("degree must be at least the degree of coeffs")
            tangent_template = coeffs[-1] if len(coeffs) > 1 else base
            coeffs = coeffs + tuple(torch.zeros_like(tangent_template) for _ in range(degree + 1 - len(coeffs)))
        shape = torch.broadcast_shapes(*(c.shape for c in coeffs))
        coeffs = tuple(c if c.shape == shape else c.expand(shape).clone() for c in coeffs)
        # Each order is independently mutable. Reusing a zero tensor for several
        # orders must not make optimizer updates write to those orders repeatedly.
        independent = []
        for c in coeffs:
            if any(torch._C._overlaps(c, previous) for previous in independent):
                c = c.clone()
            independent.append(c)
        return tuple(independent)

    def _set_coeffs(self, coeffs):
        self.coeffs = self._coerce_coeffs(coeffs)
        self.degree = len(self.coeffs) - 1

    @staticmethod
    def __new__(cls, coeffs, *, degree=None, requires_grad=False):
        coeffs = cls._coerce_coeffs(coeffs, degree)
        base = coeffs[0]

        out = torch.Tensor._make_wrapper_subclass(
            cls,
            base.shape,
            strides=base.stride(),
            storage_offset=base.storage_offset(),
            dtype=base.dtype,
            layout=base.layout,
            device=base.device,
            requires_grad=requires_grad,
        )
        out.coeffs = coeffs
        out.degree = len(coeffs) - 1
        # PyTorch autograd can retain only the C++ TensorImpl for a wrapper
        # subclass saved for backward.  If the corresponding Python wrapper is
        # collected, a later dispatch sees a PolyTensor without ``coeffs``.
        # A retention scope keeps the original wrapper objects alive through
        # forward and backward.  Append to every active scope so nested helper
        # scopes cannot accidentally release objects needed by an outer one.
        for refs in _retention_scopes.get():
            refs.append(out)
        return out

    def __init__(self, coeffs, *, degree=None, requires_grad=False):
        pass

    @property
    def value(self):
        return self.coeffs[0]

    @property
    def tangent(self):
        """The first coefficient (the first directional derivative)."""
        if self.degree == 0:
            raise ValueError("a degree-zero PolyTensor has no tangent")
        return self.coeffs[1]

    @classmethod
    def constant(cls, x, degree):
        """Construct a value whose higher coefficients are all zero."""
        return cls((x,), degree=degree)

    @classmethod
    @contextlib.contextmanager
    def retain_wrappers(cls):
        """Keep intermediate PolyTensor wrappers alive through backward.

        Wrapper-subclass autograd graphs may otherwise outlive the Python
        objects that carry ``coeffs``.  Enclose each polynomial forward and its
        backward pass in this scope::

            with PolyTensor.retain_wrappers():
                loss = model(...)
                loss.backward()

        References are context-local, support nesting, and are released when
        each scope exits (an outer scope also retains its nested wrappers).
        The yielded list is intended only for diagnostics
        such as checking how many wrappers were retained.
        """

        refs = []
        token = _retention_scopes.set((*_retention_scopes.get(), refs))
        try:
            yield refs
        finally:
            _retention_scopes.reset(token)
            refs.clear()

    @classmethod
    @contextlib.contextmanager
    def coefficient_autograd(cls):
        """Build an ordinary autograd graph for polynomial coefficients.

        This mode avoids retaining intermediate wrapper objects. Construct the
        PolyTensor wrappers with ``requires_grad=False`` and make their ordinary
        coefficient tensors autograd leaves.  Within this scope, operations on
        those coefficients acquire ordinary ``grad_fn`` nodes while the
        storage-less wrapper stays outside wrapper-subclass autograd::

            coefficients = tuple(c.requires_grad_() for c in coefficients)
            parameter = PolyTensor(coefficients, requires_grad=False)
            with PolyTensor.coefficient_autograd():
                loss = model(...)
            gradient_k = torch.autograd.grad(
                loss.coeffs[k], parameter.coeffs[0]
            )[0]

        Differentiating loss coefficient ``k`` with respect to each parameter's
        coefficient zero gives coefficient ``k`` of the parameter gradient,
        provided the input coefficients are independent autograd leaves. A
        complex output coefficient with a real base requires separate gradients
        of its real and imaginary parts. See ``docs/api.md`` for the convention.
        Do not wrap such a PolyTensor in ``nn.Parameter``: Parameter construction
        detaches its internal coefficients.  Assign it directly to a module's
        ``_parameters`` mapping when adapting an existing module.
        """

        token = _coefficient_autograd_enabled.set(True)
        try:
            yield
        finally:
            _coefficient_autograd_enabled.reset(token)

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            if self.is_complex():
                raise RuntimeError("an explicit gradient is required for a complex output")
            gradient = PolyTensor.constant(torch.ones_like(self.value), self.degree)
        elif not isinstance(gradient, PolyTensor):
            gradient = PolyTensor.constant(gradient, self.degree)

        torch.autograd.backward(
            self, gradient, retain_graph=retain_graph,
            create_graph=create_graph, inputs=inputs,
        )

    def __repr__(self):
        if not hasattr(self, "coeffs"):
            return f"PolyTensor(<uninitialized>, requires_grad={self.requires_grad})"
        return f"PolyTensor({self.coeffs}, requires_grad={self.requires_grad})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        coefficient_autograd = _coefficient_autograd_enabled.get()
        if coefficient_autograd:
            # Tensor-subclass dispatch normally runs below Autograd.  Re-enable
            # the relevant keys so operations on the ordinary coefficient
            # tensors build their own graph.  The guards are thread-local, and
            # the ContextVar above makes the public mode safe for nested and
            # asynchronous callers.
            with contextlib.ExitStack() as stack:
                for key in (
                    torch._C.DispatchKey.Autograd,
                    torch._C.DispatchKey.AutogradFunctionality,
                    torch._C.DispatchKey.AutogradOther,
                    torch._C.DispatchKey.AutogradCPU,
                    torch._C.DispatchKey.AutogradCUDA,
                ):
                    stack.enter_context(
                        torch._C._SetExcludeDispatchKeyGuard(key, False)
                    )
                return dispatch(
                    func,
                    types,
                    args,
                    kwargs,
                    coefficient_autograd=True,
                )

        return dispatch(
            func,
            types,
            args,
            kwargs,
            coefficient_autograd=False,
        )
