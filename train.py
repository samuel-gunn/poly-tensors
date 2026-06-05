import argparse
import gc
import json
import os
import sys
from pathlib import Path

DEFAULT_CONFIG_PATH = "train_config.json"
CONFIG_KEYS = (
    "model_name",
    "dataset_name",
    "dataset_config",
    "dataset_split",
    "jsonl",
    "text_column",
    "cache_dir",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "eps",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "warmup_steps",
    "min_learning_rate_ratio",
    "max_train_examples",
    "num_eval_examples",
    "max_length",
    "seed",
    "eval_index",
    "num_workers",
    "pad_to_multiple_of",
    "dpi",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "gradient_checkpointing",
    "dtype",
    "attn_implementation",
    "trust_remote_code",
)


def download_enabled_from_argv(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--download", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args.download


def read_train_config(config_path):
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return config


def validate_train_config(config, config_path):
    expected = set(CONFIG_KEYS)
    actual = set(config)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(
            f"{config_path} is missing required config keys: {', '.join(missing)}"
        )
    if extra:
        raise ValueError(
            f"{config_path} contains unknown config keys: {', '.join(extra)}"
        )


def configure_offline_from_argv(argv=None):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if argv is None:
        argv = sys.argv[1:]
    if not download_enabled_from_argv(argv):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


configure_offline_from_argv()

import torch
from datasets import DownloadConfig, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_args_from_config(config_path, overrides=None):
    config = read_train_config(config_path)
    validate_train_config(config, config_path)
    args = argparse.Namespace(**config)
    args.train_config = config
    args.config = config_path
    if overrides:
        for name, value in overrides.items():
            setattr(args, name, value)
    return args


def result_paths(expt_name):
    if (
        not expt_name
        or expt_name in (".", "..")
        or "/" in expt_name
        or "\\" in expt_name
    ):
        raise ValueError("expt_name must be a non-empty filename stem")
    results_dir = Path("results")
    return results_dir / f"{expt_name}.png", results_dir / f"{expt_name}.json"


def validate_training_args(
    args,
    validate_deletion_args=False,
    allow_zero_eval_examples=False,
):
    if validate_deletion_args:
        if args.degree < 1:
            raise ValueError("--degree must be at least 1")
        if not 0.0 <= args.downweight_fraction <= 1.0:
            raise ValueError("--downweight-fraction must be between 0 and 1")
        if args.num_dots < 2:
            raise ValueError("--num-dots must be at least 2")
    if args.max_train_examples < 1:
        raise ValueError("max_train_examples must be at least 1")
    min_eval_examples = 0 if allow_zero_eval_examples else 1
    if args.num_eval_examples < min_eval_examples:
        raise ValueError(f"num_eval_examples must be at least {min_eval_examples}")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.max_length < 2:
        raise ValueError("max_length must be at least 2")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if not 0.0 <= args.min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio must be between 0 and 1")
    if args.lora_r < 1:
        raise ValueError("lora_r must be at least 1")
    if args.lora_alpha < 1:
        raise ValueError("lora_alpha must be at least 1")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not split_csv(args.lora_target_modules):
        raise ValueError("lora_target_modules must include at least one module name")
    if args.dtype not in ("auto", "float32", "bfloat16", "float16"):
        raise ValueError("dtype must be one of: auto, float32, bfloat16, float16")


def format_run_summary(args, train_data, selected, eval_index, epoch_indices, include_downweighted=True):
    eval_prefix = "eval rows" if isinstance(eval_index, str) else "eval row"
    parts = [
        f"device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}",
        f"model={args.model_name}",
        f"train examples={len(train_data)}",
    ]
    if include_downweighted:
        parts.append(
            f"downweighted={selected.sum().item()} ({selected.float().mean().item():.1%})"
        )
    parts.extend(
        [
            f"{eval_prefix}={eval_index}",
            f"LoRA r={args.lora_r}",
            f"alpha={args.lora_alpha}",
            f"targets={args.lora_target_modules}",
            f"micro batch size={args.batch_size}",
            f"grad accum={args.gradient_accumulation_steps}",
            f"effective batch size={args.batch_size * args.gradient_accumulation_steps}",
            f"epochs={args.epochs}",
            f"steps per epoch={(len(epoch_indices[0]) + args.batch_size - 1) // args.batch_size}",
            f"optimizer steps per run={((args.epochs * ((len(epoch_indices[0]) + args.batch_size - 1) // args.batch_size)) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps}",
            f"download={'on' if args.download else 'off'}",
        ]
    )
    return ", ".join(parts)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def reset_training_rng(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_csv(values):
    return [value.strip() for value in values.split(",") if value.strip()]


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
            prompt_text, full_text = format_example(example, tokenizer, args)
            encoded = encode_example(prompt_text, full_text, tokenizer, args)
            if len(encoded["input_ids"]) >= 2 and any(label != -100 for label in encoded["labels"][1:]):
                self.items.append(encoded)

        if not self.items:
            raise ValueError("all tokenized examples had no supervised target tokens")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = dict(self.items[index])
        item["index"] = index
        return item


def encode_example(prompt_text, full_text, tokenizer, args):
    encoded = tokenizer(
        full_text,
        add_special_tokens=prompt_text is None,
        max_length=args.max_length,
        truncation=True,
    )
    labels = list(encoded["input_ids"])

    if prompt_text is not None:
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            max_length=args.max_length,
            truncation=True,
        )["input_ids"]
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

    encoded["labels"] = labels
    return encoded


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
        labels = torch.full_like(batch["input_ids"], -100)
        for row, feature in enumerate(features):
            feature_labels = torch.tensor(feature["labels"], dtype=torch.long)
            labels[row, : feature_labels.numel()] = feature_labels
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        batch["indices"] = indices
        return batch


def text_or_empty(value):
    if value is None:
        return ""
    return str(value).strip()


def with_eos(text, tokenizer):
    if tokenizer.eos_token is None or text.endswith(tokenizer.eos_token):
        return text
    return text + tokenizer.eos_token


def has_chat_template(tokenizer):
    return (
        callable(getattr(tokenizer, "apply_chat_template", None))
        and getattr(tokenizer, "chat_template", None) is not None
    )


def apply_chat_template_text(tokenizer, messages, add_generation_prompt):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as error:
        if "enable_thinking" not in str(error):
            raise
        del kwargs["enable_thinking"]
        return tokenizer.apply_chat_template(messages, **kwargs)


def format_chat_pair(tokenizer, user_content, assistant_content):
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": assistant_content}
    ]
    prompt = apply_chat_template_text(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_text = apply_chat_template_text(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
    )
    return prompt, full_text


def format_generation_prompt(tokenizer, question):
    if has_chat_template(tokenizer):
        prompt = apply_chat_template_text(
            tokenizer,
            [{"role": "user", "content": text_or_empty(question)}],
            add_generation_prompt=True,
        )
        return prompt, False
    return question.strip() + "\n\n", True


def format_example(example, tokenizer, args):
    if args.text_column is not None:
        if args.text_column not in example:
            raise ValueError(f"text column {args.text_column!r} was not found")
        return None, text_or_empty(example[args.text_column])

    if "messages" in example and has_chat_template(tokenizer):
        messages = example["messages"]
        if isinstance(messages, list):
            if (
                len(messages) > 1
                and isinstance(messages[-1], dict)
                and messages[-1].get("role") == "assistant"
            ):
                prompt = apply_chat_template_text(
                    tokenizer,
                    messages[:-1],
                    add_generation_prompt=True,
                )
                full_text = apply_chat_template_text(
                    tokenizer,
                    messages,
                    add_generation_prompt=False,
                )
                return prompt, full_text
            return (
                None,
                apply_chat_template_text(
                    tokenizer,
                    messages,
                    add_generation_prompt=False,
                ),
            )

    if "instruction" in example and "output" in example:
        instruction = text_or_empty(example.get("instruction"))
        input_text = text_or_empty(example.get("input"))
        output = text_or_empty(example.get("output"))
        if has_chat_template(tokenizer):
            user_content = instruction
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            return format_chat_pair(tokenizer, user_content, output)
        if input_text:
            prompt = (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Input:\n"
                f"{input_text}\n\n"
                "### Response:\n"
            )
            return prompt, with_eos(prompt + output, tokenizer)
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )
        return prompt, with_eos(prompt + output, tokenizer)

    for prompt_key, response_key in (("prompt", "completion"), ("question", "answer")):
        if prompt_key in example and response_key in example:
            prompt_text = text_or_empty(example[prompt_key])
            response_text = text_or_empty(example[response_key])
            if has_chat_template(tokenizer):
                return format_chat_pair(tokenizer, prompt_text, response_text)
            prompt = f"{prompt_text}\n\n"
            return prompt, with_eos(prompt + response_text, tokenizer)

    if "text" in example:
        return None, text_or_empty(example["text"])

    raise ValueError(
        "could not infer how to format dataset rows; pass text_column in the config for this dataset"
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

    selected = make_downweight_selection(len(train_data), args)

    return train_data, eval_data, selected, int(eval_index)


def make_train_eval_data(args, tokenizer, num_eval_examples):
    raw = load_raw_dataset(args)
    if len(raw) < 2:
        raise ValueError("need at least two examples: one for training and one for evaluation")
    if num_eval_examples < 1:
        raise ValueError("num_eval_examples must be at least 1")
    if num_eval_examples >= len(raw):
        raise ValueError("num_eval_examples must leave at least one training example")

    raw = raw.shuffle(seed=args.seed + 10)
    eval_start = 0 if args.eval_index is None else args.eval_index
    eval_stop = eval_start + num_eval_examples
    if not 0 <= eval_start < len(raw) or eval_stop > len(raw):
        raise ValueError("validation range is outside the dataset")

    eval_indices = set(range(eval_start, eval_stop))
    train_indices = [idx for idx in range(len(raw)) if idx not in eval_indices]
    train_indices = train_indices[: args.max_train_examples]
    if not train_indices:
        raise ValueError("no training examples selected")

    train_examples = [raw[int(idx)] for idx in train_indices]
    eval_examples = [raw[int(idx)] for idx in range(eval_start, eval_stop)]
    train_data = TokenizedTextDataset(train_examples, tokenizer, args)
    eval_data = TokenizedTextDataset(eval_examples, tokenizer, args)
    selected = make_downweight_selection(len(train_data), args)

    return train_data, eval_data, selected, (int(eval_start), int(eval_stop - 1))


def make_downweight_selection(num_examples, args):
    selection_generator = make_generator(args.seed + 20)
    num_downweighted = int(round(args.downweight_fraction * num_examples))
    num_downweighted = min(num_examples, max(1, num_downweighted))
    selected = torch.zeros(num_examples, dtype=torch.bool)
    selected_indices = torch.randperm(
        num_examples,
        generator=selection_generator,
    )[:num_downweighted]
    selected[selected_indices] = True
    return selected


def make_epoch_indices(num_examples, epochs, seed):
    generator = make_generator(seed)
    return [
        torch.randperm(num_examples, generator=generator).tolist()
        for _ in range(epochs)
    ]


def make_model(args, device):
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

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=split_csv(args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.to(device)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_trainable == 0:
        raise ValueError("LoRA configuration did not expose trainable parameters")

    return model


def batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def causal_lm_per_example_loss(logits, labels, loss_fn):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_loss = loss_fn(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
    )
    token_loss = flat_loss.reshape(shift_labels.shape)
    token_mask = (shift_labels != -100).to(dtype=token_loss.dtype)
    token_counts = token_mask.sum(dim=1).clamp_min(1)
    return (token_loss * token_mask).sum(dim=1) * token_counts.reciprocal()


def forward_per_example_loss(model, batch, loss_fn):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    return causal_lm_per_example_loss(outputs.logits, batch["labels"], loss_fn)


class AdamWForParameters:
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
                        "exp_avg": torch.zeros_like(param),
                        "exp_avg_sq": torch.zeros_like(param),
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


def scheduled_learning_rate(args, optimizer_step, total_optimizer_steps, warmup_steps):
    if warmup_steps > 0 and optimizer_step <= warmup_steps:
        return args.learning_rate * optimizer_step / warmup_steps

    remaining_steps = max(1, total_optimizer_steps - warmup_steps)
    completed_decay_steps = min(
        remaining_steps,
        max(0, optimizer_step - warmup_steps - 1),
    )
    decay = 1.0 - completed_decay_steps / remaining_steps
    return args.learning_rate * max(args.min_learning_rate_ratio, decay)


def make_optimizer(model, args):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return AdamWForParameters(
        trainable_params,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
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
    args,
    optimizer_step,
    total_optimizer_steps,
    warmup_steps,
    loss_value_fn=None,
):
    model.train()
    total_loss = 0.0
    total = 0

    if loss_value_fn is None:
        loss_value_fn = lambda value: value

    progress = tqdm(loader, disable=quiet, leave=False)
    optimizer.zero_grad(set_to_none=True)
    current_lr = 0.0
    for batch_step, batch in enumerate(progress, start=1):
        batch = batch_to_device(batch, device)
        selected_batch = selected[batch["indices"].cpu()].to(device)

        per_example_loss = forward_per_example_loss(model, batch, loss_fn)
        weights = 1 - selected_batch.float() * downweight
        loss = (per_example_loss * weights).sum() * (1.0 / per_example_loss.shape[0])
        scaled_loss = loss / args.gradient_accumulation_steps

        scaled_loss.backward()

        batch_size = per_example_loss.shape[0]
        total_loss += loss_value_fn(loss).detach().float().item() * batch_size
        total += batch_size
        if batch_step % args.gradient_accumulation_steps == 0 or batch_step == len(loader):
            optimizer_step += 1
            lr = scheduled_learning_rate(
                args,
                optimizer_step,
                total_optimizer_steps,
                warmup_steps,
            )
            optimizer.step(lr)
            optimizer.zero_grad(set_to_none=True)
            current_lr = lr
        progress.set_postfix(train_loss=f"{total_loss / total:.4f}", lr=f"{current_lr:.2e}")
        del batch, selected_batch, per_example_loss, weights, loss, scaled_loss

    return total_loss / total, optimizer_step


def train_model(
    model,
    train_data,
    selected,
    downweight,
    epoch_indices,
    collator,
    args,
    device,
    make_optimizer_fn=make_optimizer,
    loss_value_fn=None,
):
    reset_training_rng(args.seed + 60)
    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    optimizer = make_optimizer_fn(model, args)
    batches_per_epoch = (len(train_data) + args.batch_size - 1) // args.batch_size
    total_batches = args.epochs * batches_per_epoch
    total_optimizer_steps = (
        total_batches + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(round(args.warmup_ratio * total_optimizer_steps))
    optimizer_step = 0

    for epoch, indices in enumerate(epoch_indices, start=1):
        loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=indices,
            collate_fn=collator,
            num_workers=args.num_workers,
        )
        train_loss, optimizer_step = train_one_epoch(
            model,
            loader,
            selected,
            downweight,
            loss_fn,
            optimizer,
            device,
            args.quiet,
            args,
            optimizer_step,
            total_optimizer_steps,
            warmup_steps,
            loss_value_fn=loss_value_fn,
        )
        if not args.quiet:
            print(f"  epoch {epoch}: train token loss {train_loss:.4f}")


def eval_loss(model, eval_data, collator, device, batch_size=1):
    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    loader = DataLoader(eval_data, batch_size=batch_size, collate_fn=collator)
    model.eval()

    total_loss = None
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            per_example_loss = forward_per_example_loss(model, batch, loss_fn)
            batch_loss = per_example_loss.sum()
            total_loss = batch_loss if total_loss is None else total_loss + batch_loss
            total += per_example_loss.shape[0]

    if total == 0:
        raise ValueError("validation dataset is empty")
    return total_loss / total


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
