import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if "--download" not in sys.argv[1:]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import matplotlib
import torch
from datasets import DownloadConfig, load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from PolyTensor import PolyTensor


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "tatsu-lab/alpaca"
OUTPUT_PATH = "results/llm_figure_2.png"
BATCH_SIZE = 1
MICRO_BATCH_SIZE = 1
MAX_LOGIT_TOKENS = 16
EPOCHS = 1
LEARNING_RATE = 1e-3
DOWNWEIGHT_FRACTION = 0.10
DEGREE = 3
NUM_DOTS = 6
MAX_TRAIN_EXAMPLES = 64
MAX_LENGTH = 128
SEED = 0


def tensor_value(x):
    return x.value if isinstance(x, PolyTensor) else x


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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


def clone_parameters(module):
    for child in module.children():
        clone_parameters(child)

    for name, param in list(module._parameters.items()):
        if param is not None:
            module._parameters[name] = nn.Parameter(
                param.detach().clone(),
                requires_grad=param.requires_grad,
            )


def get_submodule(module, path):
    current = module
    for part in path.split("."):
        current = getattr(current, part)
    return current


def freeze_except(module, trainable_module_name):
    for param in module.parameters():
        param.requires_grad_(False)

    trainable_module = get_submodule(module, trainable_module_name)
    for param in trainable_module.parameters():
        param.requires_grad_(True)

    return trainable_module


def freeze_non_poly_parameters(module):
    for param in module.parameters():
        param.requires_grad_(isinstance(param, PolyTensor))


def dtype_from_name(name):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unknown dtype {name!r}")


class TokenizedTextDataset(Dataset):
    def __init__(self, examples, tokenizer, args):
        self.items = []
        for example in examples:
            text = format_example(example, tokenizer, args)
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                max_length=args.max_length,
                truncation=True,
            )
            if len(encoded["input_ids"]) >= 2:
                self.items.append(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                    }
                )

        if not self.items:
            raise ValueError("all tokenized examples were shorter than two tokens")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = dict(self.items[index])
        item["index"] = index
        return item


class CausalLMCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        indices = torch.tensor([feature["index"] for feature in features], dtype=torch.long)
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        batch["indices"] = indices
        return batch


def text_or_empty(value):
    if value is None:
        return ""
    return str(value).strip()


def format_example(example, tokenizer, args):
    if args.text_column is not None:
        if args.text_column not in example:
            raise ValueError(f"text column {args.text_column!r} was not found")
        return text_or_empty(example[args.text_column])

    if "messages" in example and hasattr(tokenizer, "apply_chat_template"):
        messages = example["messages"]
        if isinstance(messages, list):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

    if "instruction" in example and "output" in example:
        instruction = text_or_empty(example.get("instruction"))
        input_text = text_or_empty(example.get("input"))
        output = text_or_empty(example.get("output"))
        if input_text:
            return (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Input:\n"
                f"{input_text}\n\n"
                "### Response:\n"
                f"{output}"
            )
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
            f"{output}"
        )

    for prompt_key, response_key in (("prompt", "completion"), ("question", "answer")):
        if prompt_key in example and response_key in example:
            return (
                f"{text_or_empty(example[prompt_key])}\n\n"
                f"{text_or_empty(example[response_key])}"
            )

    if "text" in example:
        return text_or_empty(example["text"])

    raise ValueError(
        "could not infer how to format dataset rows; pass --text-column for this dataset"
    )


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=not args.download,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_raw_dataset(args):
    download_config = DownloadConfig(local_files_only=not args.download)

    if args.jsonl is not None:
        return load_dataset(
            "json",
            data_files=args.jsonl,
            split="train",
            cache_dir=args.cache_dir,
            download_config=download_config,
        )

    dataset_kwargs = {
        "path": args.dataset_name,
        "split": args.dataset_split,
        "cache_dir": args.cache_dir,
        "download_config": download_config,
    }
    if args.dataset_config is not None:
        dataset_kwargs["name"] = args.dataset_config
    return load_dataset(**dataset_kwargs)


def make_data(args, tokenizer):
    raw = load_raw_dataset(args)
    if len(raw) < 2:
        raise ValueError("need at least two examples: one for training and one for evaluation")

    raw = raw.shuffle(seed=args.seed + 10)
    eval_index = (
        min(args.max_train_examples, len(raw) - 1)
        if args.eval_index is None
        else args.eval_index
    )
    if not 0 <= eval_index < len(raw):
        raise ValueError("--eval-index is outside the dataset")

    train_indices = [idx for idx in range(len(raw)) if idx != eval_index]
    train_indices = train_indices[: args.max_train_examples]
    if not train_indices:
        raise ValueError("no training examples selected")

    train_examples = [raw[int(idx)] for idx in train_indices]
    eval_examples = [raw[int(eval_index)]]
    train_data = TokenizedTextDataset(train_examples, tokenizer, args)
    eval_data = TokenizedTextDataset(eval_examples, tokenizer, args)

    selection_generator = make_generator(args.seed + 20)
    num_downweighted = int(round(args.downweight_fraction * len(train_data)))
    num_downweighted = min(len(train_data), max(1, num_downweighted))
    selected = torch.zeros(len(train_data), dtype=torch.bool)
    selected_indices = torch.randperm(
        len(train_data),
        generator=selection_generator,
    )[:num_downweighted]
    selected[selected_indices] = True

    return train_data, eval_data, selected, int(eval_index)


def make_epoch_indices(num_examples, epochs, seed):
    generator = make_generator(seed)
    return [
        torch.randperm(num_examples, generator=generator).tolist()
        for _ in range(epochs)
    ]


def make_model(args, device, degree=None):
    torch.manual_seed(args.seed + 40)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 40)

    load_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": not args.download,
        "trust_remote_code": args.trust_remote_code,
        "dtype": dtype_from_name(args.dtype),
    }
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.config.use_cache = False
    model.to(device)

    clone_parameters(get_submodule(model, args.trainable_module))
    trainable_module = freeze_except(model, args.trainable_module)
    if degree is not None:
        make_poly_parameters(trainable_module, degree)
        freeze_non_poly_parameters(model)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_trainable == 0:
        raise ValueError(f"{args.trainable_module!r} did not expose trainable parameters")

    return model


def batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def trim_right_padding(batch):
    attention_mask = batch.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 2:
        return batch

    max_length = int(attention_mask.sum(dim=1).max().item())
    max_length = max(2, min(max_length, attention_mask.shape[1]))
    if max_length == attention_mask.shape[1]:
        return batch

    trimmed = dict(batch)
    for key in ("input_ids", "attention_mask", "labels"):
        value = trimmed.get(key)
        if isinstance(value, torch.Tensor) and value.dim() == 2:
            trimmed[key] = value[:, :max_length].contiguous()
    return trimmed


def iter_micro_batches(batch, selected_batch, micro_batch_size):
    batch_size = batch["input_ids"].shape[0]
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        micro_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
                micro_batch[key] = value[start:end]
            else:
                micro_batch[key] = value
        yield trim_right_padding(micro_batch), selected_batch[start:end]


def release_cuda_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def format_cuda_memory(device):
    if device.type != "cuda":
        return ""
    return (
        f"alloc={torch.cuda.memory_allocated(device) / 2**30:.2f}GiB, "
        f"reserved={torch.cuda.memory_reserved(device) / 2**30:.2f}GiB, "
        f"peak={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB"
    )


def shifted_causal_lm_per_example_loss(shift_logits, shift_labels, loss_fn):
    flat_loss = loss_fn(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
    )
    token_loss = flat_loss.reshape(shift_labels.shape)
    token_mask = (shift_labels != -100).to(dtype=token_loss.dtype)
    token_counts = token_mask.sum(dim=1).clamp_min(1)
    return (token_loss * token_mask).sum(dim=1) * token_counts.reciprocal()


def causal_lm_per_example_loss(logits, labels, loss_fn):
    return shifted_causal_lm_per_example_loss(logits[:, :-1, :], labels[:, 1:], loss_fn)


def forward_per_example_loss(model, batch, loss_fn):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    return causal_lm_per_example_loss(outputs.logits, batch["labels"], loss_fn)


def output_head_chunking_available(model, trainable_module):
    if trainable_module != "lm_head":
        return False
    if model.get_output_embeddings() is None:
        return False
    base_model_prefix = getattr(model, "base_model_prefix", None)
    return isinstance(base_model_prefix, str) and hasattr(model, base_model_prefix)


def forward_base_hidden_states(model, batch):
    base_model = getattr(model, model.base_model_prefix)
    with torch.no_grad():
        outputs = base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0]


def iter_hidden_micro_batches(hidden_states, batch, selected_batch, micro_batch_size):
    batch_size = hidden_states.shape[0]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        micro_hidden = hidden_states[start:end]
        micro_labels = labels[start:end]
        micro_attention_mask = attention_mask[start:end]

        max_length = int(micro_attention_mask.sum(dim=1).max().item())
        max_length = max(2, min(max_length, micro_hidden.shape[1]))
        micro_hidden = micro_hidden[:, :max_length].contiguous()
        micro_labels = micro_labels[:, :max_length].contiguous()

        yield micro_hidden, micro_labels, selected_batch[start:end]


def forward_lm_head_per_example_loss(model, hidden_states, labels, loss_fn):
    shift_hidden_states = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_logits = model.get_output_embeddings()(shift_hidden_states)
    return shifted_causal_lm_per_example_loss(shift_logits, shift_labels, loss_fn)


def backward_lm_head_loss_chunks(
    model,
    hidden_states,
    labels,
    selected_batch,
    downweight,
    loss_fn,
    batch_size,
    micro_batch_size,
    max_logit_tokens,
):
    batch_loss = 0.0

    for micro_hidden, micro_labels, micro_selected in iter_hidden_micro_batches(
        hidden_states,
        {"labels": labels, "attention_mask": labels != -100},
        selected_batch,
        micro_batch_size,
    ):
        shift_hidden = micro_hidden[:, :-1, :]
        shift_labels = micro_labels[:, 1:]
        token_mask = shift_labels != -100
        token_counts = token_mask.sum(dim=1).clamp_min(1).to(dtype=shift_hidden.dtype)
        weights = 1 - micro_selected.float() * downweight
        example_scales = weights * token_counts.reciprocal()

        for token_start in range(0, shift_hidden.shape[1], max_logit_tokens):
            token_end = min(token_start + max_logit_tokens, shift_hidden.shape[1])
            hidden_chunk = shift_hidden[:, token_start:token_end, :].contiguous()
            labels_chunk = shift_labels[:, token_start:token_end].contiguous()
            mask_chunk = token_mask[:, token_start:token_end].to(dtype=shift_hidden.dtype)

            logits_chunk = model.get_output_embeddings()(hidden_chunk)
            flat_loss = loss_fn(
                logits_chunk.reshape(-1, logits_chunk.shape[-1]),
                labels_chunk.reshape(-1),
            )
            token_loss = flat_loss.reshape(labels_chunk.shape)
            weighted_loss = (token_loss * mask_chunk * example_scales[:, None]).sum()
            loss = weighted_loss * (1.0 / batch_size)

            loss.backward()
            batch_loss += tensor_value(weighted_loss).detach().float().item()
            del hidden_chunk, labels_chunk, mask_chunk
            del logits_chunk, flat_loss, token_loss, weighted_loss, loss

        del micro_hidden, micro_labels, micro_selected
        del shift_hidden, shift_labels, token_mask, token_counts, weights, example_scales

    return batch_loss


def make_optimizer(model, args):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.SGD(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def train_one_epoch(
    model,
    loader,
    selected,
    downweight,
    loss_fn,
    optimizer,
    device,
    quiet,
    micro_batch_size,
    max_logit_tokens,
    chunk_output_head,
    memory_report_every,
):
    model.train()
    total_loss = 0.0
    total = 0

    progress = tqdm(loader, disable=quiet, leave=False)
    for batch_step, batch in enumerate(progress, start=1):
        if memory_report_every and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        batch = batch_to_device(batch, device)
        batch = trim_right_padding(batch)
        selected_batch = selected[batch["indices"].cpu()].to(device)
        batch_size = batch["input_ids"].shape[0]
        batch_loss = 0.0

        if chunk_output_head:
            hidden_states = forward_base_hidden_states(model, batch)
            batch_loss = backward_lm_head_loss_chunks(
                model,
                hidden_states,
                batch["labels"],
                selected_batch,
                downweight,
                loss_fn,
                batch_size,
                micro_batch_size,
                max_logit_tokens,
            )
            del hidden_states
        else:
            for micro_batch, micro_selected in iter_micro_batches(
                batch,
                selected_batch,
                micro_batch_size,
            ):
                per_example_loss = forward_per_example_loss(model, micro_batch, loss_fn)
                weights = 1 - micro_selected.float() * downweight
                weighted_loss = (per_example_loss * weights).sum()
                loss = weighted_loss * (1.0 / batch_size)

                loss.backward()
                batch_loss += tensor_value(weighted_loss).detach().float().item()
                del micro_batch, micro_selected, per_example_loss
                del weights, weighted_loss, loss

        optimizer.step()

        total_loss += batch_loss
        total += batch_size
        progress.set_postfix(train_loss=f"{total_loss / total:.4f}")
        if (
            memory_report_every
            and device.type == "cuda"
            and batch_step % memory_report_every == 0
        ):
            print(
                f"  batch {batch_step}: length={batch['input_ids'].shape[1]}, "
                f"{format_cuda_memory(device)}"
            )
        optimizer.zero_grad(set_to_none=True)
        del batch, selected_batch
        release_cuda_cache(device)

    return total_loss / total


def train_model(model, train_data, selected, downweight, epoch_indices, collator, args, device):
    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    optimizer = make_optimizer(model, args)
    chunk_output_head = output_head_chunking_available(model, args.trainable_module)

    for epoch, indices in enumerate(epoch_indices, start=1):
        loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=indices,
            collate_fn=collator,
            num_workers=args.num_workers,
        )
        train_loss = train_one_epoch(
            model,
            loader,
            selected,
            downweight,
            loss_fn,
            optimizer,
            device,
            args.quiet,
            args.micro_batch_size,
            args.max_logit_tokens,
            chunk_output_head,
            args.memory_report_every,
        )
        if not args.quiet:
            print(f"  epoch {epoch}: train token loss {train_loss:.4f}")


def eval_loss(model, eval_data, collator, device):
    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    loader = DataLoader(eval_data, batch_size=1, collate_fn=collator)
    model.eval()

    with torch.no_grad():
        batch = next(iter(loader))
        batch = batch_to_device(batch, device)
        per_example_loss = forward_per_example_loss(model, batch, loss_fn)
        return per_example_loss.sum()


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def polynomial_coefficients(train_data, eval_data, selected, epoch_indices, collator, args, device):
    z = PolyTensor(
        [torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)],
        degree=args.degree,
    )
    model = make_model(args, device, degree=args.degree)

    print("training PolyTensor LLM fine-tune for Taylor coefficients")
    train_model(model, train_data, selected, z, epoch_indices, collator, args, device)
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
        print(f"retraining empirical LLM fine-tune at z={z:.2f}")
        model = make_model(args, device)
        if z == 0.0:
            before_loss = eval_loss(model, eval_data, collator, device).detach().float().cpu().item()
            print(f"  z=0 eval token loss before training: {before_loss:.4f}")
        train_model(model, train_data, selected, z, epoch_indices, collator, args, device)
        loss = eval_loss(model, eval_data, collator, device)
        after_loss = tensor_value(loss).detach().float().cpu().item()
        if z == 0.0:
            print(f"  z=0 eval token loss after training: {after_loss:.4f}")
        losses.append(after_loss)
        cleanup_model(model)

    return zs, losses


def evaluate_polynomial(coefficients, zs, degree):
    ys = torch.zeros_like(zs)
    for power in range(degree + 1):
        ys = ys + coefficients[power] * zs.pow(power)
    return ys


def plot_results(zs, empirical, coefficients, eval_index, num_downweighted, args):
    grid = torch.linspace(0.0, 1.0, 301, dtype=torch.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(zs, empirical, color="black", label="empirical", zorder=3)

    for degree in range(1, args.degree + 1):
        approx = evaluate_polynomial(coefficients, grid, degree)
        ax.plot(grid, approx, label=f"degree {degree}", linewidth=2)

    ax.set_xlabel("downweight z")
    ax.set_ylabel("held-out token loss f(z 1_D)")
    ax.set_title("LLM fine-tuning deletion Taylor approximations")
    ax.text(
        0.01,
        0.99,
        (
            f"{args.model_name}, {num_downweighted} downweighted examples, "
            f"eval row {eval_index}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    print(f"saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce a Figure-2-style Taylor approximation plot for LLM fine-tuning."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--jsonl", help="Optional local JSONL dataset; overrides --dataset-name.")
    parser.add_argument("--text-column")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=MICRO_BATCH_SIZE,
        help="Examples to keep live at once inside each SGD batch.",
    )
    parser.add_argument(
        "--max-logit-tokens",
        type=int,
        default=MAX_LOGIT_TOKENS,
        help="Maximum sequence positions to pass through lm_head at once.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--downweight-fraction", type=float, default=DOWNWEIGHT_FRACTION)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--num-dots", type=int, default=NUM_DOTS)
    parser.add_argument("--max-train-examples", type=int, default=MAX_TRAIN_EXAMPLES)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-index", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--trainable-module", default="lm_head")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--memory-report-every",
        type=int,
        default=0,
        help="Print CUDA memory stats every N training batches.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.degree < 1:
        raise ValueError("--degree must be at least 1")
    if args.degree != DEGREE:
        print(f"using degree {args.degree}; pass no --degree flag for the requested degree 3")
    if not 0.0 <= args.downweight_fraction <= 1.0:
        raise ValueError("--downweight-fraction must be between 0 and 1")
    if args.num_dots < 2:
        raise ValueError("--num-dots must be at least 2")
    if args.max_train_examples < 1:
        raise ValueError("--max-train-examples must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be at least 1")
    if args.max_logit_tokens < 1:
        raise ValueError("--max-logit-tokens must be at least 1")
    if args.memory_report_every < 0:
        raise ValueError("--memory-report-every must be nonnegative")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args)
    collator = CausalLMCollator(tokenizer, args.pad_to_multiple_of)
    train_data, eval_data, selected, eval_index = make_data(args, tokenizer)
    epoch_indices = make_epoch_indices(len(train_data), args.epochs, args.seed + 50)

    print(
        f"device={device}, model={args.model_name}, train examples={len(train_data)}, "
        f"downweighted={selected.sum().item()} ({selected.float().mean().item():.1%}), "
        f"eval row={eval_index}, trainable={args.trainable_module}, "
        f"batch size={args.batch_size}, micro batch size={args.micro_batch_size}, "
        f"max logit tokens={args.max_logit_tokens}, "
        f"epochs={args.epochs}, "
        f"steps per epoch={(len(epoch_indices[0]) + args.batch_size - 1) // args.batch_size}, "
        f"download={'on' if args.download else 'off'}"
    )

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
    plot_results(zs, empirical, coefficients, eval_index, selected.sum().item(), args)


if __name__ == "__main__":
    main()
