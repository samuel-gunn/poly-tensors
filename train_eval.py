import argparse

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

import torch

DOWNWEIGHT_FRACTION = 0.02
INTERACTIVE_MAX_NEW_TOKENS = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the real z=0 LLM fine-tune and report validation loss before and after."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask TOFU-style questions before and after training. Type /done to continue.",
    )
    cli_args = parser.parse_args()
    args = load_args_from_config(
        cli_args.config,
        overrides={
            "downweight_fraction": DOWNWEIGHT_FRACTION,
            "download": cli_args.download,
            "quiet": cli_args.quiet,
        },
    )
    args.interactive = cli_args.interactive
    return args


def generate_tofu_answer(model, tokenizer, device, question):
    prompt, add_special_tokens = format_generation_prompt(tokenizer, question)
    inputs = tokenizer(
        prompt,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    was_training = model.training
    previous_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if previous_use_cache is not None:
        model.config.use_cache = True

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=INTERACTIVE_MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
        if was_training:
            model.train()

    prompt_length = inputs["input_ids"].shape[-1]
    answer_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def interactive_phase(label, model, tokenizer, device):
    print()
    print(f"{label} interactive questions. Type /done to continue.")
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

        answer = generate_tofu_answer(model, tokenizer, device, question)
        print(f"answer> {answer}")


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
    before_loss = eval_loss(
        model,
        eval_data,
        collator,
        device,
        batch_size=args.batch_size,
    ).detach().float().cpu().item()
    print(f"validation token loss before training: {before_loss:.6g}")

    if args.interactive:
        interactive_phase("Before training", model, tokenizer, device)

    train_model(model, train_data, selected, 0.0, epoch_indices, collator, args, device)

    after_loss = eval_loss(
        model,
        eval_data,
        collator,
        device,
        batch_size=args.batch_size,
    ).detach().float().cpu().item()
    print(f"validation token loss after training: {after_loss:.6g}")

    if args.interactive:
        interactive_phase("After training", model, tokenizer, device)

    cleanup_model(model)


if __name__ == "__main__":
    main()
