import argparse
import os
import torch

import config
from model import LSTMStoryModel
from generate import generate_text


OUTPUT_FILE = "outputs/temperature_comparison.txt"
DEFAULT_PROMPT = "A student discovers that the school library is alive."
DEFAULT_TEMPERATURES = [0.4, 0.7, 1.0, 1.3]


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

    return parser.parse_args()


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    model_config = checkpoint.get("config", {})

    model = LSTMStoryModel(
        vocab_size=len(stoi),
        embed_dim=model_config.get("embed_dim", config.EMBED_DIM),
        hidden_dim=model_config.get("hidden_dim", config.HIDDEN_DIM),
        num_layers=model_config.get("num_layers", config.NUM_LAYERS),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, stoi, itos


def main():
    args = parse_args()

    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {config.BEST_MODEL_PATH}. "
            "Please train the model first."
        )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    prompt = args.prompt
    temperatures = args.temperatures

    model, stoi, itos = load_model()

    results = []

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
