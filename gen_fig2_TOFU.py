import argparse
import json

from gen_fig2 import DEGREE, NUM_DOTS, evaluate_polynomial, polynomial_coefficients, retrained_losses
from train import (
    CausalLMCollator,
    DEFAULT_CONFIG_PATH,
    encode_example,
    format_example,
    format_run_summary,
    load_args_from_config,
    load_raw_dataset,
    load_tokenizer,
    make_epoch_indices,
    result_paths,
    validate_training_args,
)

import matplotlib
import torch
from torch.utils.data import Dataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPT_NAME = "llm_figure_2_TOFU"


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def split_deleted_names(value):
    names = [name.strip() for name in value.split(";") if name.strip()]
    if not names:
        raise ValueError("--deleted must include at least one author name")
    return names


def text_or_empty(value):
    if value is None:
        return ""
    return str(value).strip()


def require_tofu_columns(example):
    if "question" not in example or "answer" not in example:
        raise ValueError(
            "gen_fig2_TOFU.py expects examples with question and answer columns"
        )


def example_question_answer_text(example):
    require_tofu_columns(example)
    return "\n".join(
        [
            text_or_empty(example["question"]),
            text_or_empty(example["answer"]),
        ]
    )


def matching_deleted_names(example, deleted_names):
    text = example_question_answer_text(example).casefold()
    return [name for name in deleted_names if name.casefold() in text]


class TokenizedSelectedDataset(Dataset):
    def __init__(self, examples, selected_flags, tokenizer, args):
        if len(examples) != len(selected_flags):
            raise ValueError("examples and selected flags must have the same length")

        self.items = []
        self.selected_flags = []
        for example, selected in zip(examples, selected_flags):
            prompt_text, full_text = format_example(example, tokenizer, args)
            encoded = encode_example(prompt_text, full_text, tokenizer, args)
            has_target = len(encoded["input_ids"]) >= 2 and any(
                label != -100 for label in encoded["labels"][1:]
            )
            if has_target:
                self.items.append(encoded)
                self.selected_flags.append(bool(selected))

        if not self.items:
            raise ValueError("all tokenized examples had no supervised target tokens")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = dict(self.items[index])
        item["index"] = index
        return item

    def selected_tensor(self):
        return torch.tensor(self.selected_flags, dtype=torch.bool)


def make_tofu_author_deletion_data(args, tokenizer):
    raw = load_raw_dataset(args)
    if len(raw) < 1:
        raise ValueError("need at least one training example")

    train_indices = list(range(len(raw)))
    generator = make_generator(args.seed + 10)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    train_indices = [train_indices[position] for position in order]
    train_indices = train_indices[: args.max_train_examples]
    if not train_indices:
        raise ValueError("no training examples selected")

    train_examples = [raw[int(idx)] for idx in train_indices]
    train_selected_flags = [
        bool(matching_deleted_names(example, args.deleted_names))
        for example in train_examples
    ]
    eval_example = {
        "question": args.eval_question,
        "answer": args.eval_answer,
    }

    train_data = TokenizedSelectedDataset(
        train_examples,
        train_selected_flags,
        tokenizer,
        args,
    )
    eval_data = TokenizedSelectedDataset([eval_example], [False], tokenizer, args)
    selected = train_data.selected_tensor()
    if selected.sum().item() == 0:
        names = "; ".join(args.deleted_names)
        raise ValueError(
            f"no selected training examples mention deleted author names: {names}"
        )

    return train_data, eval_data, selected


def plot_results(
    zs,
    retrained,
    coefficients,
    num_trained_examples,
    selected,
    args,
):
    grid = torch.linspace(0.0, 1.0, 301, dtype=torch.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(zs, retrained, color="black", label="retrained", zorder=3)

    for degree in range(1, args.degree + 1):
        approx = evaluate_polynomial(coefficients, grid, degree)
        ax.plot(grid, approx, label=f"degree {degree}", linewidth=2)

    ax.set_xlabel("downweight z")
    ax.set_ylabel("loss f(z 1_D)")
    ax.set_title("TOFU author deletion Taylor approximations")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    ax.text(
        0.01,
        0.99,
        (
            f"n={num_trained_examples}, batch size={effective_batch_size}, "
            f"downweighted={selected.sum().item()}"
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

    run_config = dict(args.train_config)
    run_config.update(
        {
            "expt_name": args.expt_name,
            "deleted": args.deleted,
            "deleted_names": args.deleted_names,
            "eval_question": args.eval_question,
            "eval_answer": args.eval_answer,
            "degree": args.degree,
            "num_dots": args.num_dots,
            "num_downweighted_examples": int(selected.sum().item()),
        }
    )
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")
    print(f"saved {figure_path}")
    print(f"saved {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce a Figure-2-style Taylor plot for TOFU author deletion."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--expt-name", default=EXPT_NAME)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--deleted",
        required=True,
        help='Semicolon-separated author names to downweight, e.g. "A;B".',
    )
    parser.add_argument(
        "--eval-question",
        required=True,
        help="Question to use for the custom evaluation loss.",
    )
    parser.add_argument(
        "--eval-answer",
        required=True,
        help="Answer to use for the custom evaluation loss.",
    )
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--num-dots", type=int, default=NUM_DOTS)
    cli_args = parser.parse_args()

    deleted_names = split_deleted_names(cli_args.deleted)
    args = load_args_from_config(
        cli_args.config,
        overrides={
            "expt_name": cli_args.expt_name,
            "download": cli_args.download,
            "quiet": cli_args.quiet,
            "deleted": cli_args.deleted,
            "deleted_names": deleted_names,
            "eval_question": cli_args.eval_question,
            "eval_answer": cli_args.eval_answer,
            "degree": cli_args.degree,
            "num_dots": cli_args.num_dots,
        },
    )
    return args


def validate_args(args):
    if args.degree < 1:
        raise ValueError("--degree must be at least 1")
    if args.num_dots < 2:
        raise ValueError("--num-dots must be at least 2")
    if not text_or_empty(args.eval_question):
        raise ValueError("--eval-question must not be empty")
    if not text_or_empty(args.eval_answer):
        raise ValueError("--eval-answer must not be empty")
    validate_training_args(args, allow_zero_eval_examples=True)


def main():
    args = parse_args()
    result_paths(args.expt_name)
    if args.degree != DEGREE:
        print(f"using degree {args.degree}; pass no --degree flag for the requested degree 3")
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args)
    collator = CausalLMCollator(tokenizer, args.pad_to_multiple_of)
    train_data, eval_data, selected = make_tofu_author_deletion_data(
        args,
        tokenizer,
    )
    epoch_indices = make_epoch_indices(len(train_data), args.epochs, args.seed + 50)
    eval_description = "custom question/answer"

    print(f"deleted authors: {'; '.join(args.deleted_names)}")
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
    plot_results(
        zs,
        empirical,
        coefficients,
        num_trained_examples,
        selected,
        args,
    )


if __name__ == "__main__":
    main()
