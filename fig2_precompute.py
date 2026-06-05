import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from train import (
    CausalLMCollator,
    DEFAULT_CONFIG_PATH,
    cleanup_model,
    encode_example,
    format_example,
    format_run_summary,
    load_args_from_config,
    load_raw_dataset,
    load_tokenizer,
    make_epoch_indices,
    make_model,
    result_paths,
    text_or_empty,
    train_model,
    validate_training_args,
)


EXPT_NAME = "llm_figure_2_TOFU"
ARTIFACT_VERSION = 1
DEGREE = 3
NUM_DOTS = 6
METADATA_FILENAME = "metadata.json"
POLY_STATE_FILENAME = "poly.pt"
TRAIN_CONFIG_FILENAME = "train_config.json"


def default_precompute_dir(expt_name):
    return Path("data") / expt_name


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def is_poly_tensor(value):
    return hasattr(value, "coeffs") and hasattr(value, "degree")


def tensor_value(value):
    return value.value if is_poly_tensor(value) else value


def split_deleted_names(value):
    names = [name.strip() for name in value.split(";") if name.strip()]
    if not names:
        raise ValueError("--deleted must include at least one author name")
    return names


def require_question_answer_columns(example):
    if "question" not in example or "answer" not in example:
        raise ValueError(
            "fig2_precompute.py expects dataset rows with question and answer columns"
        )


def question_answer_text(example):
    require_question_answer_columns(example)
    return "\n".join(
        [
            text_or_empty(example["question"]),
            text_or_empty(example["answer"]),
        ]
    )


def matches_train_only(example, train_only):
    if train_only is None:
        return True
    return train_only.casefold() in question_answer_text(example).casefold()


def matching_deleted_names(example, deleted_names):
    text = question_answer_text(example).casefold()
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


def make_tofu_author_training_data(args, tokenizer):
    raw = load_raw_dataset(args)
    if len(raw) < 1:
        raise ValueError("need at least one training example")

    train_indices = list(range(len(raw)))
    order = torch.randperm(
        len(train_indices),
        generator=make_generator(args.seed + 10),
    ).tolist()
    train_indices = [train_indices[position] for position in order]
    train_indices = [
        idx
        for idx in train_indices
        if matches_train_only(raw[int(idx)], args.train_only)
    ]
    train_indices = train_indices[: args.max_train_examples]
    if not train_indices:
        if args.train_only is None:
            raise ValueError("no training examples selected")
        raise ValueError(
            f"no training examples matched --train-only={args.train_only!r}"
        )

    train_examples = [raw[int(idx)] for idx in train_indices]
    train_selected_flags = [
        bool(matching_deleted_names(example, args.deleted_names))
        for example in train_examples
    ]
    train_data = TokenizedSelectedDataset(
        train_examples,
        train_selected_flags,
        tokenizer,
        args,
    )
    selected = train_data.selected_tensor()
    if selected.sum().item() == 0:
        names = "; ".join(args.deleted_names)
        raise ValueError(
            f"no selected training examples mention deleted author names: {names}"
        )

    return train_data, selected


def tensor_trainable_state(model):
    state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_poly_tensor(param):
            raise TypeError(f"expected tensor parameter for {name!r}, found PolyTensor")
        state[name] = param.detach().cpu().clone()
    return state


def poly_trainable_state(model):
    state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not is_poly_tensor(param):
            raise TypeError(f"expected PolyTensor parameter for {name!r}")
        state[name] = [coeff.detach().cpu().clone() for coeff in param.coeffs]
    return state


def train_poly_state(train_data, selected, epoch_indices, collator, args, device):
    from gen_fig2 import make_poly_model, make_poly_optimizer
    from PolyTensor import PolyTensor

    z = PolyTensor(
        [torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)],
        degree=args.degree,
    )
    model = make_poly_model(args, device)
    try:
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
        return poly_trainable_state(model)
    finally:
        cleanup_model(model)


def train_retrained_state(train_data, selected, z, epoch_indices, collator, args, device):
    model = make_model(args, device)
    try:
        print(f"retraining LLM fine-tune at z={z:.2f}")
        train_model(model, train_data, selected, z, epoch_indices, collator, args, device)
        return tensor_trainable_state(model)
    finally:
        cleanup_model(model)


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"saved {path}")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute Figure-2 TOFU training states without evaluating a plot loss."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--expt-name", default=EXPT_NAME)
    parser.add_argument(
        "--output-dir",
        help="Directory for precomputed states. Default: data/<expt-name>.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--deleted",
        required=True,
        help='Semicolon-separated author names to downweight, e.g. "A;B".',
    )
    parser.add_argument(
        "--train-only",
        default=None,
        help=(
            "Restrict training to rows whose question or answer contains this "
            "case-insensitive text."
        ),
    )
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--num-dots", type=int, default=NUM_DOTS)
    parser.add_argument(
        "--no-polys",
        action="store_true",
        help="Skip PolyTensor/Taylor training and save only retrained model states.",
    )
    cli_args = parser.parse_args()

    deleted_names = split_deleted_names(cli_args.deleted)
    train_only = None
    if cli_args.train_only is not None:
        train_only = cli_args.train_only.strip()
        if not train_only:
            raise ValueError("--train-only must not be empty")

    args = load_args_from_config(
        cli_args.config,
        overrides={
            "expt_name": cli_args.expt_name,
            "download": cli_args.download,
            "quiet": cli_args.quiet,
            "deleted": cli_args.deleted,
            "deleted_names": deleted_names,
            "degree": cli_args.degree,
            "num_dots": cli_args.num_dots,
            "output_dir": cli_args.output_dir,
            "train_only": train_only,
            "no_polys": cli_args.no_polys,
        },
    )
    return args


def validate_args(args):
    if args.degree < 1:
        raise ValueError("--degree must be at least 1")
    if args.num_dots < 2:
        raise ValueError("--num-dots must be at least 2")
    validate_training_args(args, allow_zero_eval_examples=True)


def main():
    args = parse_args()
    precompute_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_precompute_dir(args.expt_name)
    )
    write_json(precompute_dir / TRAIN_CONFIG_FILENAME, args.train_config)

    result_paths(args.expt_name)
    if args.degree != DEGREE:
        print(f"using degree {args.degree}; pass no --degree flag for the requested degree 3")
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args)
    collator = CausalLMCollator(tokenizer, args.pad_to_multiple_of)
    train_data, selected = make_tofu_author_training_data(args, tokenizer)
    epoch_indices = make_epoch_indices(len(train_data), args.epochs, args.seed + 50)
    num_trained_examples = sum(len(indices) for indices in epoch_indices)

    print(f"deleted authors: {'; '.join(args.deleted_names)}")
    summary = format_run_summary(
        args,
        train_data,
        selected,
        "none (precompute only)",
        epoch_indices,
    )
    if args.train_only is not None:
        summary += f", train only={args.train_only!r}"
    if args.no_polys:
        summary += ", no polys"
    print(summary)

    if not args.no_polys:
        poly_path = precompute_dir / POLY_STATE_FILENAME
        poly_state = train_poly_state(
            train_data,
            selected,
            epoch_indices,
            collator,
            args,
            device,
        )
        save_checkpoint(
            poly_path,
            {
                "format": "fig2_poly_lora_coefficients",
                "version": ARTIFACT_VERSION,
                "degree": args.degree,
                "state": poly_state,
            },
        )
        del poly_state

    zs = torch.linspace(0.0, 1.0, args.num_dots).tolist()
    retrained_states = []
    for index, z in enumerate(zs):
        state_path = precompute_dir / f"retrained_{index:03d}.pt"
        state = train_retrained_state(
            train_data,
            selected,
            float(z),
            epoch_indices,
            collator,
            args,
            device,
        )
        save_checkpoint(
            state_path,
            {
                "format": "fig2_retrained_lora_state",
                "version": ARTIFACT_VERSION,
                "z": float(z),
                "state": state,
            },
        )
        del state
        retrained_states.append(
            {
                "z": float(z),
                "path": state_path.name,
            }
        )

    metadata = {
        "format": "fig2_precompute",
        "version": ARTIFACT_VERSION,
        "expt_name": args.expt_name,
        "config": args.config,
        "train_config_path": TRAIN_CONFIG_FILENAME,
        "train_config": args.train_config,
        "deleted": args.deleted,
        "deleted_names": args.deleted_names,
        "train_only": args.train_only,
        "no_polys": bool(args.no_polys),
        "degree": args.degree,
        "num_dots": args.num_dots,
        "num_train_examples": len(train_data),
        "num_trained_examples": num_trained_examples,
        "num_downweighted_examples": int(selected.sum().item()),
        "retrained_states": retrained_states,
    }
    if not args.no_polys:
        metadata["poly_state_path"] = POLY_STATE_FILENAME
    write_json(precompute_dir / METADATA_FILENAME, metadata)


if __name__ == "__main__":
    main()
