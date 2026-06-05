import argparse

import torch

from train import (
    CausalLMCollator,
    DEFAULT_CONFIG_PATH,
    cleanup_model,
    eval_loss,
    format_generation_prompt,
    format_run_summary,
    load_args_from_config,
    load_tokenizer,
    make_epoch_indices,
    make_model,
    make_train_eval_data,
    train_model,
    validate_training_args,
)


DOWNWEIGHT_FRACTION = 0.02
TOP_K = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the real z=0 LLM fine-tune and compare next-token "
            "probabilities before and after training."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of post-training next-token candidates to print. Default: {TOP_K}.",
    )
    cli_args = parser.parse_args()
    if cli_args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    args = load_args_from_config(
        cli_args.config,
        overrides={
            "downweight_fraction": DOWNWEIGHT_FRACTION,
            "download": cli_args.download,
            "quiet": cli_args.quiet,
        },
    )
    args.top_k = cli_args.top_k
    return args


def trainable_state(model):
    return {
        name: param.detach().to(device="cpu", copy=True)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_state(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in state:
                raise KeyError(f"missing trainable parameter {name!r} in saved state")
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def encode_probe(tokenizer, question, answer_prefix, device):
    prompt, add_special_tokens = format_generation_prompt(tokenizer, question)
    inputs = tokenizer(
        prompt + answer_prefix,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    if inputs["input_ids"].shape[-1] == 0:
        raise ValueError("the question and answer prefix produced an empty prompt")
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def next_token_probabilities(model, inputs):
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits[0, -1].float()
            return torch.softmax(logits, dim=-1)
    finally:
        if was_training:
            model.train()


def decode_token(tokenizer, token_id):
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def format_probability(probability):
    return f"{probability:.6g}"


def print_probability_table(tokenizer, token_ids, trained_probs, before_probs):
    header = f"{'rank':>4}  {'token_id':>8}  {'token':<18}  {'trained':>12}  {'before':>12}"
    print(header)
    print("-" * len(header))
    for rank, (token_id, trained_prob, before_prob) in enumerate(
        zip(token_ids, trained_probs, before_probs),
        start=1,
    ):
        token_text = repr(decode_token(tokenizer, int(token_id)))
        print(
            f"{rank:>4}  "
            f"{int(token_id):>8}  "
            f"{token_text:<18}  "
            f"{format_probability(float(trained_prob)):>12}  "
            f"{format_probability(float(before_prob)):>12}"
        )


def compare_next_tokens(
    model,
    tokenizer,
    device,
    question,
    answer_prefix,
    before_state,
    trained_state,
    top_k,
):
    inputs = encode_probe(tokenizer, question, answer_prefix, device)

    trained_distribution = next_token_probabilities(model, inputs)
    trained_probs, token_ids = torch.topk(
        trained_distribution,
        k=min(top_k, trained_distribution.shape[-1]),
    )
    token_ids = token_ids.cpu()
    trained_probs = trained_probs.cpu()
    del trained_distribution

    try:
        load_trainable_state(model, before_state)
        before_distribution = next_token_probabilities(model, inputs)
        before_probs = before_distribution[token_ids.to(before_distribution.device)].cpu()
        del before_distribution
    finally:
        load_trainable_state(model, trained_state)

    print_probability_table(tokenizer, token_ids, trained_probs, before_probs)


def interactive_phase(model, tokenizer, device, before_state, trained_state, top_k):
    print()
    print("Interactive next-token probes. Type /done at either prompt to exit.")
    while True:
        try:
            question = input("question> ")
        except EOFError:
            print()
            return

        question = question.strip()
        if question == "/done":
            return
        if not question:
            continue

        try:
            answer_prefix = input("answer prefix> ")
        except EOFError:
            print()
            return
        if answer_prefix == "/done":
            return

        compare_next_tokens(
            model,
            tokenizer,
            device,
            question,
            answer_prefix,
            before_state,
            trained_state,
            top_k,
        )


def main():
    args = parse_args()
    validate_training_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args)
    collator = CausalLMCollator(tokenizer, args.pad_to_multiple_of)
    train_data, eval_data, selected, eval_range = make_train_eval_data(
        args,
        tokenizer,
        args.num_eval_examples,
    )
    epoch_indices = make_epoch_indices(len(train_data), args.epochs, args.seed + 50)
    eval_description = f"{eval_range[0]}..{eval_range[1]} ({len(eval_data)} examples)"

    print(
        format_run_summary(
            args,
            train_data,
            selected,
            eval_description,
            epoch_indices,
            include_downweighted=False,
        )
    )

    model = make_model(args, device)
    before_state = trainable_state(model)

    before_loss = eval_loss(
        model,
        eval_data,
        collator,
        device,
        batch_size=args.batch_size,
    ).detach().float().cpu().item()
    print(f"validation token loss before training: {before_loss:.6g}")

    train_model(model, train_data, selected, 0.0, epoch_indices, collator, args, device)
    trained_state = trainable_state(model)

    after_loss = eval_loss(
        model,
        eval_data,
        collator,
        device,
        batch_size=args.batch_size,
    ).detach().float().cpu().item()
    print(f"validation token loss after training: {after_loss:.6g}")

    interactive_phase(
        model,
        tokenizer,
        device,
        before_state,
        trained_state,
        args.top_k,
    )
    cleanup_model(model)


if __name__ == "__main__":
    main()
