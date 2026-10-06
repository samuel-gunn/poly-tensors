"""Private extended-range coefficients carried between supported operators.

Public coefficient tensors remain ordinary PyTorch tensors. A sidecar retains
binary exponents until a consumer needs an ordinary tensor. Version snapshots
ensure that a mutation never reuses a stale hidden value.
"""

def _snapshot(coefficients):
    return tuple((id(coefficient), coefficient._version) for coefficient in coefficients)


def _discard(value):
    for attribute in ("_scaled_coefficients", "_scaled_versions"):
        if hasattr(value, attribute):
            delattr(value, attribute)


def attach_scaled(value, coefficients):
    from ._plain import plain_range_enabled

    if plain_range_enabled():
        return value
    # Inference tensors have no version counter, so mutation cannot be tracked.
    if any(c.is_inference() for c in value.coeffs):
        _discard(value)
        return value
    # Constructors and in-place copies can broadcast a scalar coefficient to
    # the public shape. Views must see that same shape in the hidden storage.
    value._scaled_coefficients = tuple(
        coefficient if coefficient.mantissa.shape == stored.shape
        else coefficient.map_tensor(lambda tensor, shape=stored.shape: tensor.expand(shape))
        for coefficient, stored in zip(coefficients, value.coeffs)
    )
    value._scaled_versions = _snapshot(value.coeffs)
    return value


def get_scaled(value):
    from ._plain import plain_range_enabled

    if plain_range_enabled():
        return None
    coefficients = getattr(value, "_scaled_coefficients", None)
    if coefficients is None:
        return None
    if (any(c.is_inference() for c in value.coeffs)
            or getattr(value, "_scaled_versions", None) != _snapshot(value.coeffs)):
        # An alias or direct coefficient write invalidates the extra range.
        # Falling back to current public storage is preferable to stale data.
        _discard(value)
        return None
    return coefficients
