import argparse
import os

import config
from checkpoint import load_story_model
from generate import build_anchored_prompt, generate_text
from runtime import non_negative_int, positive_float, positive_int, set_random_seed


PROMPT_FILE = "sample_prompts.txt"
OUTPUT_FILE = "outputs/batch_outputs.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate stories for every prompt in a text file."
    )
    parser.add_argument(
        "--prompt_file",
        default=PROMPT_FILE,
        help="Text file containing one prompt per line.",
    )
    parser.add_argument(
        "--output_file",
        default=OUTPUT_FILE,
        help="File path for saving generated stories.",
    )
    parser.add_argument(
        "--max_tokens",
        type=positive_int,
        default=200,
        help="Maximum number of generated subword tokens per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=positive_float,
        default=0.8,
        help="Sampling temperature used for every prompt.",
    )
    parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=config.RANDOM_SEED,
        help="Random seed used to make generation reproducible.",
    )
    return parser.parse_args()


def load_prompts(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines()]

    return [prompt for prompt in prompts if prompt]


def main():
    args = parse_args()
    set_random_seed(args.seed)

    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model, tokenizer, _ = load_story_model(config.BEST_MODEL_PATH)
    prompts = load_prompts(args.prompt_file)

    if not prompts:
        raise ValueError("Prompt file does not contain any non-empty prompts.")

    results = [f"Random Seed: {args.seed}\n{'=' * 80}\n"]

    for i, prompt in enumerate(prompts, start=1):
        formatted_prompt = build_anchored_prompt(prompt)

        story = generate_text(
            model=model,
            start_text=formatted_prompt,
            tokenizer=tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        result = (
            f"Example {i}\n"
            f"Prompt: {prompt}\n"
            f"Generated Story:\n{story}\n"
            f"{'-' * 80}\n"
        )

        print(result)
        results.append(result)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"Batch generation saved to {args.output_file}")


if __name__ == "__main__":
    main()
