import argparse
import os

import config
from checkpoint import load_story_model


def generate_text(model, start_text, stoi, itos, max_new_chars=500, temperature=0.8):
    """
    Generate text from a trained ChaosWriter model.

    Args:
        model: Trained LSTMStoryModel.
        start_text: Initial prompt text.
        stoi: Character-to-index vocabulary.
        itos: Index-to-character vocabulary.
        max_new_chars: Number of new characters to generate.
        temperature: Controls randomness. Higher values make output more random.

    Returns:
        A generated text string.
    """
    device = next(model.parameters()).device

    ids = [stoi[ch] for ch in start_text if ch in stoi]

    if len(ids) == 0:
        raise ValueError("The prompt does not contain any known characters.")

    x = torch.tensor([ids], dtype=torch.long).to(device)

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_chars):
            logits, _ = model(x)

            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)

    output_ids = x[0].tolist()
    return "".join([itos[i] for i in output_ids])

def main():
    parser = argparse.ArgumentParser(description="Generate a short story with ChaosWriter.")

    parser.add_argument(
        "--prompt",
        type=str,
        default="A student finds a notebook that writes back.",
        help="Writing prompt for the model."
    )

    parser.add_argument(
        "--max_chars",
        type=int,
        default=500,
        help="Maximum number of characters to generate."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Higher values make output more random."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional file path to save the generated story."
    )

    args = parser.parse_args()

    model, stoi, itos, _ = load_story_model(config.BEST_MODEL_PATH)

    formatted_prompt = f"Prompt: {args.prompt}\nStory:"

    output = generate_text(
        model,
        formatted_prompt,
        stoi,
        itos,
        max_new_chars=args.max_chars,
        temperature=args.temperature
    )

    print("\nGenerated Story:\n")
    print(output)

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"\nGenerated story saved to {args.output_file}")


if __name__ == "__main__":
    main()
