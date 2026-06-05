import argparse
import json
from pathlib import Path

import torch
from torch import nn

from fig2_precompute import (
    EXPT_NAME,
    METADATA_FILENAME,
    TRAIN_CONFIG_FILENAME,
    default_precompute_dir,
)
from train import (
    cleanup_model,
    format_generation_prompt,
    load_tokenizer,
    make_model,
    validate_train_config,
    validate_training_args,
)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGPROB_SUFFIX = "logprob"
PROB_SUFFIX = "prob"


def is_poly_tensor(value):
    return hasattr(value, "coeffs") and hasattr(value, "degree")


def evaluate_polynomial(coefficients, zs, degree):
    ys = torch.zeros_like(zs)
    for power in range(degree + 1):
        ys = ys + coefficients[power] * zs.pow(power)
    return ys


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"saved {path}")


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")


def args_from_metadata(metadata, metadata_path, download, quiet):
    config = metadata.get("train_config")
    if config is None:
        config_path = metadata_path.parent / metadata.get(
            "train_config_path",
            TRAIN_CONFIG_FILENAME,
        )
        config = read_json(config_path)
    validate_train_config(config, f"{metadata_path}: train_config")

    args = argparse.Namespace(**config)
    args.train_config = config
    args.config = metadata.get("config")
    args.expt_name = metadata["expt_name"]
    args.download = download
    args.quiet = quiet
    args.degree = int(metadata["degree"])
    args.num_dots = int(metadata["num_dots"])
    validate_training_args(args, allow_zero_eval_examples=True)
    return args


def expected_trainable_names(model):
    return {
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def check_state_names(model, state):
    expected = expected_trainable_names(model)
    actual = set(state)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise KeyError(f"state is missing trainable parameters: {', '.join(missing[:10])}")
    if extra:
        raise KeyError(f"state has unknown trainable parameters: {', '.join(extra[:10])}")


def load_tensor_trainable_state(model, state):
    check_state_names(model, state)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_poly_tensor(param):
                raise TypeError(f"expected tensor parameter for {name!r}, found PolyTensor")
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def load_poly_trainable_state(model, state):
    check_state_names(model, state)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not is_poly_tensor(param):
                raise TypeError(f"expected PolyTensor parameter for {name!r}")
            coeffs = state[name]
            if len(coeffs) != param.degree + 1:
                raise ValueError(
                    f"{name!r} has {len(coeffs)} saved coefficients, "
                    f"expected {param.degree + 1}"
                )
            param._set_coeffs(
                coeff.to(device=param.device, dtype=param.dtype)
                for coeff in coeffs
            )


def encode_answer_prefix(tokenizer, question, answer_prefix, answer, device):
    if not answer:
        raise ValueError("--answer must not be empty")

    prompt, add_special_tokens = format_generation_prompt(tokenizer, question)
    masked_text = prompt + answer_prefix
    full_text = masked_text + answer
    masked_ids = tokenizer(
        masked_text,
        add_special_tokens=add_special_tokens,
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    masked_length = min(len(masked_ids), labels.shape[-1])
    labels[:, :masked_length] = -100

    num_answer_tokens = int((labels[:, 1:] != -100).sum().item())
    if num_answer_tokens == 0:
        raise ValueError("the question and answer produced no answer tokens to score")

    encoded["labels"] = labels
    return (
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        },
        num_answer_tokens,
    )


def raw_answer_log_probability(model, batch):
    loss_fn = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = batch["labels"][:, 1:]
            loss = loss_fn(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
            )
            return -loss
    finally:
        if was_training:
            model.train()


def answer_metric_from_log_probability(log_probability, use_probs):
    if use_probs:
        return torch.exp(log_probability.double())
    return log_probability


def metric_name(use_probs):
    return "probability" if use_probs else "log_probability"


def metric_label(use_probs):
    if use_probs:
        return "Pr(answer)"
    return "log Pr(answer)"


def poly_answer_coefficients(
    args,
    tokenizer,
    question,
    answer_prefix,
    answer,
    state_path,
    device,
    use_probs,
):
    from gen_fig2 import make_poly_model

    checkpoint = load_checkpoint(state_path)
    model = make_poly_model(args, device)
    try:
        load_poly_trainable_state(model, checkpoint["state"])
        batch, num_answer_tokens = encode_answer_prefix(
            tokenizer,
            question,
            answer_prefix,
            answer,
            device,
        )
        log_probability = raw_answer_log_probability(model, batch)
        if not is_poly_tensor(log_probability):
            raise TypeError("expected PolyTensor log probability")
        metric = answer_metric_from_log_probability(log_probability, use_probs)
        coefficients = [
            coeff.detach().float().cpu().item()
            for coeff in metric.coeffs
        ]
        return coefficients, num_answer_tokens
    finally:
        cleanup_model(model)


def retrained_answer_values(
    args,
    tokenizer,
    question,
    answer_prefix,
    answer,
    entries,
    base_dir,
    device,
    use_probs,
):
    model = make_model(args, device)
    values = []
    try:
        batch, num_answer_tokens = encode_answer_prefix(
            tokenizer,
            question,
            answer_prefix,
            answer,
            device,
        )
        for entry in entries:
            checkpoint = load_checkpoint(base_dir / entry["path"])
            load_tensor_trainable_state(model, checkpoint["state"])
            log_probability = raw_answer_log_probability(model, batch)
            metric = answer_metric_from_log_probability(log_probability, use_probs)
            values.append(metric.detach().float().cpu().item())
        return values, num_answer_tokens
    finally:
        cleanup_model(model)


def plot_results(
    zs,
    retrained,
    coefficients,
    question,
    answer,
    metadata,
    args,
    output_path,
    use_probs,
    no_polys,
):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(zs, retrained, color="black", label="retrained", zorder=3)

    if not no_polys:
        grid = torch.linspace(0.0, 1.0, 301, dtype=torch.float64)
        for degree in range(1, args.degree + 1):
            approx = evaluate_polynomial(coefficients, grid, degree)
            ax.plot(grid, approx, label=f"degree {degree}", linewidth=2)

    ax.set_xlabel("downweight z")
    ax.set_ylabel(metric_label(use_probs))
    ax.set_title(f"TOFU answer-prefix {metric_name(use_probs).replace('_', ' ')}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    ax.text(
        0.01,
        0.99,
        (
            f"n={metadata['num_trained_examples']}, "
            f"d={metadata['num_downweighted_examples']}, "
            f"batch size={effective_batch_size}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    print(f"saved {output_path}")


def default_output_path(expt_name, use_probs, no_polys):
    suffix = PROB_SUFFIX if use_probs else LOGPROB_SUFFIX
    if no_polys:
        suffix = f"{suffix}_no_polys"
    return Path("results") / expt_name / f"{suffix}.png"


def plot_json_payload(
    precompute_dir,
    question,
    answer_prefix,
    answer,
    answer_token_count,
    degree,
    zs,
    coefficients,
    retrained,
    use_probs,
    no_polys,
):
    metric = metric_name(use_probs)
    payload = {
        "format": "fig2_answer_prefix_plot",
        "metric": metric,
        "no_polys": no_polys,
        "precompute_dir": str(precompute_dir),
        "question": question,
        "answer_prefix": answer_prefix,
        "answer": answer,
        "full_answer": answer_prefix + answer,
        "scored_answer": answer,
        "conditioned_answer_prefix": answer_prefix,
        "answer_token_count": answer_token_count,
        "zs": zs,
        "retrained_values": retrained,
    }
    if use_probs:
        payload["retrained_probabilities"] = retrained
    else:
        payload["retrained_log_probabilities"] = retrained
    if not no_polys:
        payload["degree"] = degree
        payload["poly_coefficients"] = coefficients
        if use_probs:
            payload["poly_probability_coefficients"] = coefficients
        else:
            payload["poly_log_probability_coefficients"] = coefficients
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Figure-2 TOFU answer-prefix log probabilities from precomputed states."
    )
    parser.add_argument(
        "--expt-name",
        default=EXPT_NAME,
        help="Experiment name to load from data/<expt-name>.",
    )
    parser.add_argument(
        "--precompute-dir",
        help="Directory containing precomputed states. Default: data/<expt-name>.",
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--answer-prefix",
        default="",
        help="Condition on this answer prefix before scoring --answer.",
    )
    parser.add_argument("--answer", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--probs",
        action="store_true",
        help="Plot actual answer-prefix probability instead of log probability.",
    )
    parser.add_argument(
        "--no-polys",
        action="store_true",
        help="Skip PolyTensor/Taylor evaluation and plot only retrained dots.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    precompute_dir = (
        Path(cli_args.precompute_dir)
        if cli_args.precompute_dir is not None
        else default_precompute_dir(cli_args.expt_name)
    )
    metadata_path = precompute_dir / METADATA_FILENAME
    metadata = read_json(metadata_path)
    args = args_from_metadata(
        metadata,
        metadata_path,
        cli_args.download,
        cli_args.quiet,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args)
    entries = metadata["retrained_states"]
    zs = [float(entry["z"]) for entry in entries]
    no_polys = cli_args.no_polys or bool(metadata.get("no_polys", False))

    coefficients = None
    poly_answer_tokens = None
    if not no_polys:
        poly_state_path = metadata.get("poly_state_path")
        if poly_state_path is None:
            raise ValueError(
                "metadata has no poly_state_path; pass --no-polys to plot only retrained dots"
            )
        coefficients, poly_answer_tokens = poly_answer_coefficients(
            args,
            tokenizer,
            cli_args.question,
            cli_args.answer_prefix,
            cli_args.answer,
            precompute_dir / poly_state_path,
            device,
            cli_args.probs,
        )
    retrained, retrained_answer_tokens = retrained_answer_values(
        args,
        tokenizer,
        cli_args.question,
        cli_args.answer_prefix,
        cli_args.answer,
        entries,
        precompute_dir,
        device,
        cli_args.probs,
    )
    if poly_answer_tokens is not None and poly_answer_tokens != retrained_answer_tokens:
        raise RuntimeError("PolyTensor and retrained paths scored different token counts")

    output_path = (
        Path(cli_args.output)
        if cli_args.output is not None
        else default_output_path(metadata["expt_name"], cli_args.probs, no_polys)
    )
    plot_results(
        zs,
        retrained,
        coefficients,
        cli_args.question,
        cli_args.answer,
        metadata,
        args,
        output_path,
        cli_args.probs,
        no_polys,
    )
    write_json(
        output_path.with_suffix(".json"),
        plot_json_payload(
            precompute_dir,
            cli_args.question,
            cli_args.answer_prefix,
            cli_args.answer,
            retrained_answer_tokens,
            args.degree,
            zs,
            coefficients,
            retrained,
            cli_args.probs,
            no_polys,
        ),
    )


if __name__ == "__main__":
    main()
