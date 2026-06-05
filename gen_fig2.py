import argparse
import json

from train import (
    CausalLMCollator,
    DEFAULT_CONFIG_PATH,
    cleanup_model,
    eval_loss,
    format_run_summary,
    load_args_from_config,
    load_tokenizer,
    make_epoch_indices,
    make_model,
    make_train_eval_data,
    result_paths,
    train_model,
    validate_training_args,
)

import matplotlib
import torch
from torch import nn
from torch.utils.data import Subset

from PolyTensor import PolyTensor


matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOWNWEIGHT_FRACTION = 0.02
DEGREE = 3
NUM_DOTS = 6
EXPT_NAME = "llm_figure_2"

# TO DO
# 2. try with TOFU, see if it's actually memorizing stuff
# 3. get figure 2 for some interpretable TOFU task
# 4. get figure 1 (for alpaca and/or TOFU)
# 5. get figure 3


def tensor_value(x):
    return x.value if isinstance(x, PolyTensor) else x


def make_poly_parameters(module, degree):
    for child in module.children():
        make_poly_parameters(child, degree)

    for name, param in list(module._parameters.items()):
        if param is None or not param.requires_grad:
            continue
        coeffs = (param.detach().clone(),) + tuple(torch.zeros_like(param) for _ in range(degree))
        module._parameters[name] = nn.Parameter(
            PolyTensor(coeffs, requires_grad=param.requires_grad),
            requires_grad=param.requires_grad,
        )


def assert_only_poly_parameters_trainable(module):
    bad_params = [
        (name, type(param).__name__)
        for name, param in module.named_parameters()
        if param.requires_grad and not isinstance(param, PolyTensor)
    ]
    if bad_params:
        preview = ", ".join(f"{name} ({type_name})" for name, type_name in bad_params[:10])
        if len(bad_params) > 10:
            preview += f", ... and {len(bad_params) - 10} more"
        raise RuntimeError(
            "expected polynomial training to leave only PolyTensor parameters trainable; "
            f"found trainable non-PolyTensor parameters: {preview}"
        )


def make_poly_model(args, device):
    model = make_model(args, device)
    make_poly_parameters(model, args.degree)
    assert_only_poly_parameters_trainable(model)
    return model


def zeros_like_parameter(param):
    if isinstance(param, PolyTensor):
        return PolyTensor(tuple(torch.zeros_like(coeff) for coeff in param.coeffs))
    return torch.zeros_like(param)


class AdamWForPolyTensor:
    def __init__(self, params, betas, eps, weight_decay):
        self.params = list(params)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def step(self, lr):
        with torch.no_grad():
            for param in self.params:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state.get(id(param))
                if state is None:
                    state = {
                        "step": 0,
                        "exp_avg": zeros_like_parameter(param),
                        "exp_avg_sq": zeros_like_parameter(param),
                    }
                    self.state[id(param)] = state

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if self.weight_decay != 0.0:
                    param.mul_(1.0 - lr * self.weight_decay)

                exp_avg.mul_(self.beta1)
                exp_avg.add_(grad, alpha=1.0 - self.beta1)
                exp_avg_sq.mul_(self.beta2)
                exp_avg_sq.add_(grad * grad, alpha=1.0 - self.beta2)

                bias_correction1 = 1.0 - self.beta1 ** step
                bias_correction2 = 1.0 - self.beta2 ** step
                denom = (exp_avg_sq / bias_correction2 + self.eps).sqrt()
                update = (exp_avg / bias_correction1) / denom
                param.add_(update, alpha=-lr)


def make_poly_optimizer(model, args):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return AdamWForPolyTensor(
        trainable_params,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )


def polynomial_coefficients(train_data, eval_data, selected, epoch_indices, collator, args, device):
    z = PolyTensor(
        [torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)],
        degree=args.degree,
    )
    model = make_poly_model(args, device)

    print("training PolyTensor LLM fine-tune for Taylor coefficients")
    train_model(
        model,
        train_data,
        selected,
        z,
        epoch_indices,
        collator,
        args,
        device,
        make_optimizer_fn=make_poly_optimizer,
        loss_value_fn=tensor_value,
    )
    loss = eval_loss(model, eval_data, collator, device)
    if not isinstance(loss, PolyTensor):
        raise TypeError("expected the evaluation loss to be a PolyTensor")
    coefficients = [coeff.detach().float().cpu().item() for coeff in loss.coeffs]
    cleanup_model(model)
    return coefficients


def retrained_losses(train_data, eval_data, selected, epoch_indices, collator, args, device):
    zs = torch.linspace(0.0, 1.0, args.num_dots).tolist()
    losses = []

    for z in zs:
        print(f"retraining LLM fine-tune at z={z:.2f}")
        model = make_model(args, device)
        train_model(model, train_data, selected, z, epoch_indices, collator, args, device)
        loss = eval_loss(model, eval_data, collator, device)
        after_loss = tensor_value(loss).detach().float().cpu().item()
        losses.append(after_loss)
        cleanup_model(model)

    return zs, losses


def evaluate_polynomial(coefficients, zs, degree):
    ys = torch.zeros_like(zs)
    for power in range(degree + 1):
        ys = ys + coefficients[power] * zs.pow(power)
    return ys


def plot_results(zs, retrained, coefficients, num_trained_examples, num_downweighted, args):
    grid = torch.linspace(0.0, 1.0, 301, dtype=torch.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(zs, retrained, color="black", label="retrained", zorder=3)

    for degree in range(1, args.degree + 1):
        approx = evaluate_polynomial(coefficients, grid, degree)
        ax.plot(grid, approx, label=f"degree {degree}", linewidth=2)

    ax.set_xlabel("downweight z")
    ax.set_ylabel("loss f(z 1_D)")
    ax.set_title("LLM fine-tuning deletion Taylor approximations")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    ax.text(
        0.01,
        0.99,
        (
            f"n={num_trained_examples}, batch size={effective_batch_size}, "
            f"deletions={num_downweighted}\n"
            f"LoRA r={args.lora_r}, epochs={args.epochs}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    figure_path, config_path = result_paths(args.expt_name)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=args.dpi)
    plt.close(fig)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(args.train_config, handle, indent=2)
        handle.write("\n")
    print(f"saved {figure_path}")
    print(f"saved {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce a Figure-2-style Taylor approximation plot for LLM fine-tuning."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--expt-name", default=EXPT_NAME)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--downweight-fraction", type=float, default=DOWNWEIGHT_FRACTION)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--num-dots", type=int, default=NUM_DOTS)
    cli_args = parser.parse_args()

    args = load_args_from_config(
        cli_args.config,
        overrides={
            "expt_name": cli_args.expt_name,
            "download": cli_args.download,
            "quiet": cli_args.quiet,
            "downweight_fraction": cli_args.downweight_fraction,
            "degree": cli_args.degree,
            "num_dots": cli_args.num_dots,
        },
    )
    return args


def main():
    args = parse_args()
    result_paths(args.expt_name)
    if args.degree != DEGREE:
        print(f"using degree {args.degree}; pass no --degree flag for the requested degree 3")
    validate_training_args(args, validate_deletion_args=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args)
    collator = CausalLMCollator(tokenizer, args.pad_to_multiple_of)
    train_data, heldout_data, selected, eval_range = make_train_eval_data(
        args,
        tokenizer,
        args.num_eval_examples,
    )
    eval_data = Subset(heldout_data, [0])
    epoch_indices = make_epoch_indices(len(train_data), args.epochs, args.seed + 50)
    eval_description = f"{eval_range[0]} (1 of {len(heldout_data)} held-out examples)"

    print(format_run_summary(args, train_data, selected, eval_description, epoch_indices))

    coefficients = polynomial_coefficients(
        train_data,
        eval_data,
        selected,
        epoch_indices,
        collator,
        args,
        device,
    )
    print("Taylor coefficients:", ", ".join(f"{c:.6g}" for c in coefficients))

    zs, empirical = retrained_losses(
        train_data,
        eval_data,
        selected,
        epoch_indices,
        collator,
        args,
        device,
    )
    num_trained_examples = sum(len(indices) for indices in epoch_indices)
    plot_results(zs, empirical, coefficients, num_trained_examples, selected.sum().item(), args)


if __name__ == "__main__":
    main()
