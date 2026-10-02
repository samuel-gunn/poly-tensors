"""PyTorch operator rules for PolyTensor."""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from ._series import SeriesOps

aten = torch.ops.aten


def dispatch(func, types, args=(), kwargs=None, *, coefficient_autograd=False):
    from ._tensor import PolyTensor

    kwargs = {} if kwargs is None else kwargs

    polys = []
    poly_wrappers = []

    def is_poly_wrapper(x):
        return type(x) is PolyTensor

    def is_initialized_poly(x):
        return is_poly_wrapper(x) and hasattr(x, "coeffs")

    def collect(x):
        if is_initialized_poly(x):
            polys.append(x)
            poly_wrappers.append(x)
        elif is_poly_wrapper(x):
            poly_wrappers.append(x)
        elif isinstance(x, (tuple, list)):
            for y in x:
                collect(y)
        elif isinstance(x, dict):
            for y in x.values():
                collect(y)

    collect(args)
    collect(kwargs)

    if len(polys) != len(poly_wrappers):
        raise RuntimeError(
            "PolyTensor wrapper reached dispatch without coefficient storage. "
            "Keep every PolyTensor wrapper alive through forward and backward; "
            "silently replacing this value with zero would corrupt derivatives."
        )

    if not polys:
        return func(*args, **kwargs)

    D = polys[0].degree
    if any(poly.degree != D for poly in polys[1:]):
        raise ValueError("PolyTensor operands must have the same degree")
    series = SeriesOps(D)
    like = polys[0].value

    def lift(x):
        if is_initialized_poly(x):
            return x.coeffs
        if isinstance(x, torch.Tensor):
            return (x,) + tuple(torch.zeros_like(x) for _ in range(D))
        return (x,) + tuple(torch.zeros_like(like) for _ in range(D))

    def wrap(cs):
        cs = tuple(cs)
        # In coefficient-autograd mode, differentiation belongs exclusively
        # to the ordinary coefficient tensors.  Making the storage-less
        # wrapper differentiable would re-enter wrapper-subclass autograd and
        # recreate its lifetime and aliasing constraints.
        # Wrapper autograd belongs to the native outer operation as well:
        # inferring it from coefficient views pre-populates the output's
        # metadata before native view autograd can install its own graph.
        return PolyTensor(cs, requires_grad=False)

    def correct(out):
        if coefficient_autograd:
            # Every coefficient already owns its ordinary view/alias graph.
            # Wrapper alias correction performs a differentiable ``set_`` on
            # a storage-less leaf in this mode and is both unnecessary and
            # rejected by autograd.
            return out
        # Ordinary coefficient views need ADInplaceOrView, but alias correction
        # here operates on the wrapper itself, whose outer native kernel owns
        # that metadata. Re-entering the view key on this level would attach it
        # twice to decomposed view operations.
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.ADInplaceOrView, True):
            return return_and_correct_aliasing(func, args, kwargs, out)

    def preserve_saved_inputs(source, output):
        # Some native backwards save an activation's output, which can round
        # to exactly zero/one. Keep its original series for a stable derivative.
        for attribute in (
            "_activation_input_coeffs", "_normalization_input_coeffs",
            "_normalization_dim", "_normalization_excluded",
        ):
            if hasattr(source, attribute):
                setattr(output, attribute, getattr(source, attribute))
        return output

    def wrap_normalization(coefficients, dim, *, log=False):
        from ._normalization import normalization_scaled_series
        from ._range import attach_scaled

        values = (series.poly_log_softmax(coefficients, dim) if log
                  else series.poly_softmax(coefficients, dim))
        output = wrap(values)
        probabilities, normalizers = normalization_scaled_series(coefficients, dim, log)
        attach_scaled(output, normalizers if log else probabilities)
        output._normalization_input_coeffs = tuple(coefficients)
        output._normalization_dim = dim
        return output

    def wrap_activation(coefficients, kind):
        from ._activations import _evaluate_scaled, activation_series
        from ._range import attach_scaled

        output = attach_scaled(wrap(activation_series(coefficients, kind)),
                               _evaluate_scaled(coefficients, kind, 0))
        output._activation_input_coeffs = coefficients
        return output

    def wrap_activation_backward(coefficients, gradient, kind):
        from ._activations import activation_backward_series
        from ._range import attach_scaled, get_scaled

        stored_gradients = lift(gradient)
        values = activation_backward_series(coefficients, get_scaled(gradient) or stored_gradients,
                                            kind, _return_scaled=True)
        targets = [torch.promote_types(x.dtype, g.dtype) for x, g in zip(coefficients, stored_gradients)]
        return attach_scaled(wrap(c.to_tensor(dtype) for c, dtype in zip(values, targets)), values)

    def is_poly(x):
        return is_initialized_poly(x)

    def has_poly(x):
        if is_initialized_poly(x):
            return True
        if isinstance(x, (tuple, list)):
            return any(has_poly(y) for y in x)
        if isinstance(x, dict):
            return any(has_poly(y) for y in x.values())
        return False

    def plain(x):
        if is_poly_wrapper(x):
            return x.value
        if isinstance(x, tuple):
            return tuple(plain(y) for y in x)
        if isinstance(x, list):
            return [plain(y) for y in x]
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        return x

    def coefficient_dtype(coefficient, dtype):
        if not coefficient.is_complex() or dtype is None or dtype.is_complex:
            return dtype
        complex_dtype = {
            torch.float16: torch.complex32,
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
        }.get(dtype)
        if complex_dtype is None:
            raise TypeError(f"{dtype} cannot represent complex PolyTensor directions")
        return complex_dtype

    def coefficient_options(coefficient, options):
        options = dict(options)
        if "dtype" in options:
            options["dtype"] = coefficient_dtype(coefficient, options["dtype"])
        return options

    schema = getattr(func, "_schema", None)
    name = getattr(schema, "name", "")
    mutates_first_arg = bool(getattr(schema, "is_mutable", False)) or name.rsplit("::", 1)[-1].endswith("_")
    if args and mutates_first_arg and not is_poly(args[0]) and has_poly((args[1:], kwargs)):
        target = args[0]
        source_types = ", ".join(type(arg).__name__ for arg in args[1:4])
        raise RuntimeError(
            "cannot update a regular Tensor in-place with a PolyTensor; "
            f"func={func}, target_shape={getattr(target, 'shape', None)}, "
            f"target_dtype={getattr(target, 'dtype', None)}, "
            f"source_types=[{source_types}]"
        )

    def add_terms(xs):
        s = xs[0]
        for x in xs[1:]:
            s = s + x
        return s

    def bilinear(op, a, b):
        def apply(left, right):
            # Real base values may carry complex directions.  Matrix kernels
            # require matching dtypes, unlike elementwise multiplication.
            dtype = torch.promote_types(left.dtype, right.dtype)
            return op(left.to(dtype=dtype), right.to(dtype=dtype))

        a_is_poly = is_poly(a)
        b_is_poly = is_poly(b)
        if a_is_poly and not b_is_poly:
            A = a.coeffs
            return [apply(A[k], b) for k in range(D + 1)]
        if b_is_poly and not a_is_poly:
            B = b.coeffs
            return [apply(a, B[k]) for k in range(D + 1)]
        if not a_is_poly and not b_is_poly:
            y0 = apply(a, b)
            return [y0] + [torch.zeros_like(y0) for _ in range(D)]

        A, B = lift(a), lift(b)

        out = []
        for k in range(D + 1):
            s = apply(A[0], B[k])
            for i in range(1, k + 1):
                s = s + apply(A[i], B[k - i])
            out.append(s)
        return out
    def poly_cross_entropy_loss(X, target, weight, reduction, ignore_index):
        if is_poly(target) or is_poly(weight):
            raise NotImplementedError(
                "PolyTensor cross entropy requires ordinary targets and class weights"
            )
        if isinstance(reduction, str):
            reduction_name = reduction
            reduction = {"none": 0, "mean": 1, "sum": 2}[reduction]
        else:
            reduction_name = {0: "none", 1: "mean", 2: "sum"}[reduction]
        if target.shape == X[0].shape:
            raise NotImplementedError("PolyTensor cross entropy does not implement soft labels")
        if X[0].dim() < 2:
            raise NotImplementedError("PolyTensor cross entropy expects a class dimension")

        target = plain(target)
        weight = plain(weight)
        valid = target != ignore_index
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        gather_index = safe_target.unsqueeze(1)

        # Cross entropy's native backward is expressed through the native
        # log-softmax output.  Use that same rooted jet rather than a
        # separately evaluated logsumexp-minus-target identity, whose
        # float32 gradient can differ by several ulps.
        log_probabilities = series.poly_log_softmax(X, 1)
        target_log_probabilities = [
            x.gather(1, gather_index).squeeze(1)
            for x in log_probabilities
        ]
        losses = [
            torch.where(
                valid,
                -target_log_probabilities[k],
                torch.zeros_like(target_log_probabilities[k]),
            )
            for k in range(D + 1)
        ]

        if weight is not None:
            sample_weight = weight.gather(
                0,
                safe_target.reshape(-1),
            ).reshape_as(safe_target)
            sample_weight = torch.where(valid, sample_weight, torch.zeros_like(sample_weight))
            losses = [loss * sample_weight for loss in losses]
            total_weight = sample_weight.sum()
        else:
            total_weight = valid.sum().to(device=losses[0].device, dtype=losses[0].dtype)

        if reduction == 0:
            out = losses
        if reduction == 1:
            out = [loss.sum() / total_weight for loss in losses]
        elif reduction == 2:
            out = [loss.sum() for loss in losses]
        elif reduction != 0:
            raise ValueError(f"unknown reduction {reduction!r}")

        native_loss = torch.nn.functional.cross_entropy(
            X[0],
            target,
            weight=weight,
            reduction=reduction_name,
            ignore_index=ignore_index,
        )
        return [native_loss, *out[1:]]

    from ._range_ops import NOT_HANDLED, dispatch_range

    ranged = dispatch_range(func, args, kwargs, polys=polys, degree=D,
                            wrap=wrap, correct=correct, preserve=preserve_saved_inputs)
    if ranged is not NOT_HANDLED:
        return ranged

    if func is aten.scaled_dot_product_attention.default:
        from ._attention import attention
        from ._range import attach_scaled

        query, key, value = args[:3]
        attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
        is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
        scale = kwargs.get("scale")
        enable_gqa = kwargs.get("enable_gqa", False)
        if enable_gqa:
            raise NotImplementedError("PolyTensor does not implement GQA head repetition")

        output, _, scaled, _ = attention(
            lift(query), lift(key), lift(value),
            attn_mask=None if attn_mask is None else lift(attn_mask),
            dropout_p=dropout_p, is_causal=is_causal, scale=scale, _return_scaled=True,
        )
        return attach_scaled(wrap(output), scaled)

    if func is aten._scaled_dot_product_flash_attention_for_cpu.default:
        from ._attention import attention
        from ._range import attach_scaled

        query, key, value = args[:3]
        dropout_p = args[3] if len(args) > 3 else kwargs.get("dropout_p", 0.0)
        is_causal = args[4] if len(args) > 4 else kwargs.get("is_causal", False)
        attn_mask = kwargs.get("attn_mask")
        scale = kwargs.get("scale")

        if dropout_p != 0:
            raise NotImplementedError("CPU flash attention requires dropout_p=0")
        output, logsumexp, scaled, scaled_logsumexp = attention(
            lift(query), lift(key), lift(value),
            attn_mask=None if attn_mask is None else lift(attn_mask),
            is_causal=is_causal, scale=scale, _return_scaled=True,
        )
        return attach_scaled(wrap(output), scaled), attach_scaled(wrap(logsumexp), scaled_logsumexp)

    if func is aten._scaled_dot_product_flash_attention_for_cpu_backward.default:
        from ._attention import attention_backward
        from ._range import attach_scaled, get_scaled

        grad_output, query, key, value = args[:4]
        dropout_p = args[6] if len(args) > 6 else kwargs.get("dropout_p", 0.0)
        is_causal = args[7] if len(args) > 7 else kwargs.get("is_causal", False)
        attn_mask = kwargs.get("attn_mask")
        if dropout_p != 0:
            raise NotImplementedError("CPU flash attention backward requires dropout_p=0")
        gradients = attention_backward(
            get_scaled(grad_output) or lift(grad_output), lift(query), lift(key), lift(value),
            attn_mask=None if attn_mask is None else lift(attn_mask),
            is_causal=is_causal, scale=kwargs.get("scale"), _return_scaled=True,
        )
        return tuple(
            attach_scaled(wrap(c.to_tensor(coefficient_dtype(c.mantissa, source.dtype)) for c in gradient), gradient)
            for gradient, source in zip(gradients, (query, key, value))
        )

    if func in (aten.convolution.default, aten.conv2d.default):
        x, weight, bias = args[:3]
        rest = args[3:]
        Y = bilinear(lambda a, b: func(a, b, None, *rest, **kwargs), x, weight)
        if bias is not None:
            B = lift(bias)
            Y = [
                Y[k] + B[k].reshape(1, -1, *([1] * (Y[k].dim() - 2)))
                for k in range(D + 1)
            ]
        return wrap(Y)

    if func is aten.pow.Tensor_Scalar:
        x, n = args[:2]
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        if not isinstance(n, int) or n < 0:
            raise NotImplementedError("only nonnegative integer powers")

        y = PolyTensor.constant(torch.ones_like(lift(x)[0]), D)
        for _ in range(n):
            y = y * x
        return y

    if func is aten.exp.default:
        from ._range import attach_scaled

        coefficients = lift(args[0])
        return attach_scaled(wrap(series.poly_exp(coefficients)), series._poly_exp_scaled(coefficients))

    if func is aten.sin.default:
        return wrap(series.poly_sin(lift(args[0])))

    if func is aten.cos.default:
        return wrap(series.poly_cos(lift(args[0])))

    if func is aten.log.default:
        return wrap(series.poly_log(lift(args[0])))

    if func is aten.sqrt.default:
        return wrap(series.poly_sqrt(lift(args[0])))

    if func is aten.rsqrt.default:
        return wrap(series.poly_rsqrt(lift(args[0])))

    if func is aten.logsumexp.default:
        x = args[0]
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        return wrap(series.poly_logsumexp(lift(x), dim, keepdim))

    if func is aten.sigmoid.default:
        return wrap_activation(lift(args[0]), "sigmoid")

    if func is aten.tanh.default:
        return wrap_activation(lift(args[0]), "tanh")

    if func is aten.silu.default:
        return wrap_activation(lift(args[0]), "silu")

    if func is aten.gelu.default:
        approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
        if approximate not in ("none", "tanh"):
            raise ValueError("GELU approximate must be 'none' or 'tanh'")
        return wrap_activation(lift(args[0]), "gelu" if approximate == "none" else "gelu_tanh")

    if func is aten._log_softmax.default:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        half_to_float = args[2] if len(args) > 2 else kwargs.get("half_to_float", False)
        coefficients = lift(args[0])
        if half_to_float:
            coefficients = [c.to(dtype=coefficient_dtype(c, torch.float32)) for c in coefficients]
        return wrap_normalization(coefficients, dim, log=True)

    if func is aten.log_softmax.int:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
        coefficients = lift(args[0])
        if dtype is not None:
            coefficients = [c.to(dtype=coefficient_dtype(c, dtype)) for c in coefficients]
        return wrap_normalization(coefficients, dim, log=True)

    if func is getattr(getattr(aten, "_safe_softmax", None), "default", None):
        from ._attention import _safe_scores, safe_softmax

        dim = args[1] if len(args) > 1 else kwargs["dim"]
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
        coefficients = tuple(
            c.to(dtype=coefficient_dtype(c, dtype)) if dtype is not None else c
            for c in lift(args[0])
        )
        probabilities, _ = safe_softmax(coefficients, dim)
        output = wrap(probabilities)
        safe, excluded, _ = _safe_scores(coefficients, dim)
        from ._normalization import normalization_scaled_series
        from ._range import attach_scaled

        scaled, _ = normalization_scaled_series(safe, dim)
        attach_scaled(output, (c.masked_fill(excluded, 0) for c in scaled))
        output._normalization_input_coeffs = safe
        output._normalization_dim = dim
        output._normalization_excluded = excluded
        return output

    if func is aten._softmax.default:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        half_to_float = args[2] if len(args) > 2 else kwargs.get("half_to_float", False)
        coefficients = lift(args[0])
        if half_to_float:
            coefficients = [c.to(dtype=coefficient_dtype(c, torch.float32)) for c in coefficients]
        return wrap_normalization(coefficients, dim)

    if func is aten.softmax.int:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
        coefficients = lift(args[0])
        if dtype is not None:
            coefficients = [c.to(dtype=coefficient_dtype(c, dtype)) for c in coefficients]
        return wrap_normalization(coefficients, dim)

    if func is aten.embedding.default:
        weight = args[0]
        return wrap(func(w, *plain(args[1:]), **plain(kwargs)) for w in lift(weight))

    if func is aten.avg_pool2d.default:
        return wrap(func(c, *args[1:], **kwargs) for c in lift(args[0]))

    if func is aten.nll_loss_forward.default:
        if has_poly((args[1:], kwargs)):
            raise NotImplementedError(
                "PolyTensor NLL loss requires ordinary targets and class weights"
            )
        outs = [func(c, *plain(args[1:]), **plain(kwargs)) for c in lift(args[0])]
        return wrap(out[0] for out in outs), outs[0][1]

    if func is aten.cross_entropy_loss.default:
        x, target = args[:2]
        weight = args[2] if len(args) > 2 else kwargs.get("weight")
        reduction = args[3] if len(args) > 3 else kwargs.get("reduction", 1)
        ignore_index = args[4] if len(args) > 4 else kwargs.get("ignore_index", -100)
        label_smoothing = args[5] if len(args) > 5 else kwargs.get("label_smoothing", 0.0)
        if label_smoothing != 0.0:
            raise NotImplementedError("PolyTensor does not implement label-smoothed cross entropy")

        return wrap(poly_cross_entropy_loss(lift(x), target, weight, reduction, ignore_index))

    if func is aten.nll_loss_backward.default:
        if has_poly((args[2:], kwargs)):
            raise NotImplementedError(
                "PolyTensor NLL loss requires ordinary targets, class weights, and total weight"
            )
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten._log_softmax_backward_data.default:
        grad_output, output, dim = args[:3]
        input_dtype = args[3] if len(args) > 3 else kwargs.get("input_dtype")
        if hasattr(output, "_normalization_input_coeffs"):
            from ._normalization import normalization_backward
            from ._range import attach_scaled, get_scaled

            gradients = get_scaled(grad_output) or lift(grad_output)
            values = normalization_backward(gradients, output._normalization_input_coeffs, dim, log=True, _return_scaled=True)
            return attach_scaled(wrap(c.to_tensor(coefficient_dtype(c.mantissa, input_dtype)) for c in values), values)
        G = lift(grad_output)
        O = lift(output)

        S = [g.sum(dim=dim, keepdim=True) for g in G]
        E = series.poly_exp(O)

        out = []
        for k in range(D + 1):
            s = E[0] * S[k]
            for i in range(1, k + 1):
                s = s + E[i] * S[k - i]
            y = G[k] - s
            out.append(y.to(dtype=coefficient_dtype(y, input_dtype)) if input_dtype is not None else y)

        return wrap(out)

    if func is aten._softmax_backward_data.default:
        grad_output, output, dim = args[:3]
        input_dtype = args[3] if len(args) > 3 else kwargs.get("input_dtype")
        if hasattr(output, "_normalization_input_coeffs"):
            from ._normalization import normalization_backward
            from ._range import attach_scaled, get_scaled

            gradients = get_scaled(grad_output) or lift(grad_output)
            excluded = getattr(output, "_normalization_excluded", None)
            if excluded is not None:
                gradients = tuple(c.masked_fill(excluded, 0) for c in gradients)
            values = normalization_backward(gradients, output._normalization_input_coeffs, dim, _return_scaled=True)
            if excluded is not None:
                values = [c.masked_fill(excluded, 0) for c in values]
            return attach_scaled(wrap(c.to_tensor(coefficient_dtype(c.mantissa, input_dtype)) for c in values), values)
        G = lift(grad_output)
        O = lift(output)
        GO = series.conv(G, O)
        S = [go.sum(dim=dim, keepdim=True) for go in GO]
        out = series.conv(O, [G[k] - S[k] for k in range(D + 1)])
        return wrap(y.to(dtype=coefficient_dtype(y, input_dtype)) if input_dtype is not None else y for y in out)

    if func is aten.sigmoid_backward.default:
        G = lift(args[0])
        if hasattr(args[1], "_activation_input_coeffs"):
            return wrap_activation_backward(args[1]._activation_input_coeffs, args[0], "sigmoid")
        O = lift(args[1])
        one_minus_o = [1 - O[0]] + [-O[k] for k in range(1, D + 1)]
        return wrap(series.conv(G, series.conv(O, one_minus_o)))

    if func is aten.tanh_backward.default:
        G = lift(args[0])
        if hasattr(args[1], "_activation_input_coeffs"):
            return wrap_activation_backward(args[1]._activation_input_coeffs, args[0], "tanh")
        O = lift(args[1])
        O2 = series.conv(O, O)
        one_minus_o2 = [1 - O2[0]] + [-O2[k] for k in range(1, D + 1)]
        return wrap(series.conv(G, one_minus_o2))

    if func is aten.silu_backward.default:
        return wrap_activation_backward(lift(args[1]), args[0], "silu")

    if func is aten.gelu_backward.default:
        grad_output, x = args[:2]
        approximate = args[2] if len(args) > 2 else kwargs.get("approximate", "none")
        if approximate not in ("none", "tanh"):
            raise ValueError("GELU approximate must be 'none' or 'tanh'")
        kind = "gelu" if approximate == "none" else "gelu_tanh"
        return wrap_activation_backward(lift(x), grad_output, kind)

    if func is aten.isnan.default:
        out = torch.isnan(lift(args[0])[0])
        for c in lift(args[0])[1:]:
            out = out | torch.isnan(c)
        return out

    if func is aten.native_dropout.default:
        x = args[0]
        p = args[1] if len(args) > 1 else kwargs["p"]
        train = args[2] if len(args) > 2 else kwargs["train"]
        X = lift(x)
        y0, mask = func(X[0], p, train)
        if not train or p == 0:
            return correct((wrap((y0, *X[1:])), mask))
        if p == 1:
            return wrap(torch.zeros_like(c) for c in X), mask
        scale = 1.0 / (1.0 - p)
        mask_values = mask.to(dtype=X[0].dtype)
        return wrap((y0, *(c * mask_values * scale for c in X[1:]))), mask

    if func is aten.dropout.default:
        x = args[0]
        p = args[1] if len(args) > 1 else kwargs["p"]
        train = args[2] if len(args) > 2 else kwargs["train"]
        X = lift(x)
        if not train or p == 0:
            y = func(X[0], p, train)
            return correct(wrap((y, *X[1:])))
        if p == 1:
            return wrap(torch.zeros_like(c) for c in X)
        y, mask = aten.native_dropout.default(X[0], p, train)
        mask = mask.to(dtype=X[0].dtype)
        scale = 1.0 / (1.0 - p)
        return wrap((y, *(c * mask * scale for c in X[1:])))

    if func is aten.native_dropout_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.avg_pool2d_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.embedding_dense_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.slice_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.select_backward.default:
        grad_output = args[0]
        return wrap(func(g, *plain(args[1:]), **plain(kwargs)) for g in lift(grad_output))

    if func is aten.convolution_backward.default:
        grad_output, x, weight = args[:3]
        rest = plain(args[3:])
        call_kwargs = plain(kwargs)
        if rest:
            output_mask = rest[-1]
            call_args = rest[:-1]
        else:
            output_mask = call_kwargs.pop("output_mask")
            call_args = ()
        G = lift(grad_output)
        X = lift(x)
        W = lift(weight)
        outputs = []

        if output_mask[0]:
            outputs.append(
                wrap(
                    add_terms([
                        func(
                            G[i],
                            X[0],
                            W[k - i],
                            *call_args,
                            (True, False, False),
                            **call_kwargs,
                        )[0]
                        for i in range(k + 1)
                    ])
                    for k in range(D + 1)
                )
            )
        else:
            outputs.append(None)

        if output_mask[1]:
            outputs.append(
                wrap(
                    add_terms([
                        func(
                            G[i],
                            X[k - i],
                            W[0],
                            *call_args,
                            (False, True, False),
                            **call_kwargs,
                        )[1]
                        for i in range(k + 1)
                    ])
                    for k in range(D + 1)
                )
            )
        else:
            outputs.append(None)

        if output_mask[2]:
            outputs.append(
                wrap(
                    func(
                        G[k],
                        X[0],
                        W[0],
                        *call_args,
                        (False, False, True),
                        **call_kwargs,
                    )[2]
                    for k in range(D + 1)
                )
            )
        else:
            outputs.append(None)

        return tuple(outputs)

    # Minimal autograd / optimizer plumbing.
    if func is aten.ones_like.default:
        x0 = lift(args[0])[0]
        out_zero = func(x0, *args[1:], **kwargs)
        return wrap((out_zero, *(torch.zeros_like(out_zero) for _ in range(D))))

    if func is aten.zeros_like.default:
        return wrap(torch.zeros_like(c, *args[1:], **coefficient_options(c, kwargs)) for c in lift(args[0]))

    if func is aten.empty_like.default:
        return wrap(torch.empty_like(c, *args[1:], **coefficient_options(c, kwargs)) for c in lift(args[0]))

    if func is aten.full_like.default:
        call_kwargs = dict(kwargs)
        fill_value = args[1] if len(args) > 1 else call_kwargs.pop("fill_value")
        out_zero = torch.full_like(lift(args[0])[0], fill_value, *args[2:], **call_kwargs)
        return wrap((out_zero, *(torch.zeros_like(out_zero) for _ in range(D))))

    if func is aten.new_empty_strided.default:
        return wrap(func(c, *args[1:], **coefficient_options(c, kwargs)) for c in lift(args[0]))

    if func is aten.detach_.default:
        x = args[0]
        x._set_coeffs(c.detach() for c in lift(x))
        return x

    if func in (
        aten._conj.default,
        aten._conj_physical.default,
        aten.conj_physical.default,
        aten.real.default,
        aten.imag.default,
        aten.view_as_real.default,
        aten.view_as_complex.default,
    ):
        return correct(wrap(func(c, *args[1:], **kwargs) for c in lift(args[0])))

    if func in (
        aten._to_copy.default,
        aten.to.dtype,
        aten.to.device,
        aten.to.other,
        aten.to.dtype_layout,
    ):
        def convert(c):
            other_args = list(args[1:])
            options = coefficient_options(c, kwargs)
            if func in (aten.to.dtype, aten.to.dtype_layout) and other_args:
                other_args[0] = coefficient_dtype(c, other_args[0])
            elif func is aten.to.device and len(other_args) > 1:
                other_args[1] = coefficient_dtype(c, other_args[1])
            elif func is aten.to.other:
                other = other_args[0] if other_args else options.pop("other")
                non_blocking = other_args[1] if len(other_args) > 1 else options.get("non_blocking", False)
                copy = other_args[2] if len(other_args) > 2 else options.get("copy", False)
                return c.to(
                    device=other.device,
                    dtype=coefficient_dtype(c, other.dtype),
                    non_blocking=non_blocking,
                    copy=copy,
                    memory_format=options.get("memory_format", torch.preserve_format),
                )
            return func(c, *other_args, **options)

        from ._range import attach_scaled, get_scaled
        from ._scaled import ScaledTensor

        result = wrap(convert(c) for c in lift(args[0]))
        scaled = get_scaled(args[0])
        if scaled is not None:
            def convert_scaled(value, target):
                if value.mantissa.is_complex():
                    return ScaledTensor._from_parts(convert_scaled(value.real, target.real),
                                                    convert_scaled(value.imag, target.real))
                return ScaledTensor(value.mantissa.to(device=target.device, dtype=target.dtype),
                                    value.exponent.to(device=target.device))
            attach_scaled(result, (convert_scaled(c, target) for c, target in zip(scaled, result.coeffs)))
        return result

    if func in (aten.bernoulli_.float, aten.bernoulli_.Tensor):
        x = args[0]
        x.coeffs[0].bernoulli_(*plain(args[1:]), **plain(kwargs))
        for coeff in x.coeffs[1:]:
            coeff.zero_()
        return x

    if func is aten.as_strided.default:
        return correct(wrap(func(c, *args[1:], **kwargs) for c in lift(args[0])))

    raise NotImplementedError(f"PolyTensor does not implement {func}")
