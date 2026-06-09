import argparse
import os

import config
from checkpoint import load_story_model
from generate import build_anchored_prompt, generate_text
from runtime import non_negative_int, positive_float, positive_int, set_random_seed


OUTPUT_FILE = "outputs/temperature_comparison.txt"
DEFAULT_PROMPT = "A student discovers that the school library is alive."
DEFAULT_TEMPERATURES = [0.4, 0.7, 1.0, 1.3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ChaosWriter outputs across sampling temperatures."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Story prompt to use for every temperature.",
    )

    parser.add_argument(
        "--temperatures",
        type=positive_float,
        nargs="+",
        default=DEFAULT_TEMPERATURES,
        help="One or more sampling temperatures to compare.",
    )

    parser.add_argument(
        "--max_tokens",
        type=positive_int,
        default=200,
        help="Maximum number of generated subword tokens per temperature.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=OUTPUT_FILE,
        help="File path for saving the comparison report.",
    )

    parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=config.RANDOM_SEED,
        help="Random seed used to make generation reproducible.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    prompt = args.prompt
    temperatures = args.temperatures

    set_random_seed(args.seed)

    model, tokenizer, _ = load_story_model(config.BEST_MODEL_PATH)

    results = [f"Random Seed: {args.seed}\n{'=' * 80}\n"]

    for temperature in temperatures:
        formatted_prompt = build_anchored_prompt(prompt)

        story = generate_text(
            model=model,
            start_text=formatted_prompt,
            tokenizer=tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=temperature,
        )

        result = (
            f"Temperature: {temperature}\n"
            f"Prompt: {prompt}\n"
            f"Generated Story:\n{story}\n"
            f"{'-' * 80}\n"
        )

        print(result)
        results.append(result)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"Temperature comparison saved to {args.output_file}")


if __name__ == "__main__":
    main()
