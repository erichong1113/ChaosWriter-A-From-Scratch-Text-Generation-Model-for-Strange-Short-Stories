import argparse
import os
import torch

import config
from checkpoint import load_story_model
from generate import generate_text


OUTPUT_FILE = "outputs/temperature_comparison.txt"
DEFAULT_PROMPT = "A student discovers that the school library is alive."
DEFAULT_TEMPERATURES = [0.4, 0.7, 1.0, 1.3]
DEFAULT_SEED = 42


def positive_float(value):
    parsed_value = float(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("temperature must be greater than 0")
    return parsed_value


def positive_int(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("max_chars must be greater than 0")
    return parsed_value


def non_negative_int(value):
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("seed must be 0 or greater")
    return parsed_value


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
        "--max_chars",
        type=positive_int,
        default=600,
        help="Maximum number of generated characters per temperature.",
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
        default=DEFAULT_SEED,
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

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, stoi, itos, _ = load_story_model(config.BEST_MODEL_PATH)

    results = [f"Random Seed: {args.seed}\n{'=' * 80}\n"]

    for temperature in temperatures:
        formatted_prompt = f"Prompt: {prompt}\nStory:"

        story = generate_text(
            model=model,
            start_text=formatted_prompt,
            stoi=stoi,
            itos=itos,
            max_new_chars=args.max_chars,
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
