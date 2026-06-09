import argparse
import os

import torch

import config
from checkpoint import load_story_model
from runtime import non_negative_int, positive_float, positive_int, set_random_seed


SENTENCE_ENDINGS = ".!?"
MAX_SENTENCE_COMPLETION_TOKENS = 40


def build_anchored_prompt(prompt):
    """Repeat the user's idea as the first sentence of the generated story."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("The prompt must not be empty.")

    if prompt[-1] not in ".!?":
        prompt = f"{prompt}."

    return f"Prompt: {prompt}\nStory: {prompt}"


def generate_text(
    model,
    start_text,
    tokenizer,
    max_new_tokens=200,
    temperature=0.8,
    finish_sentence=True,
    max_completion_tokens=MAX_SENTENCE_COMPLETION_TOKENS,
):
    """
    Generate text from a trained ChaosWriter model.

    Args:
        model: Trained LSTMStoryModel.
        start_text: Initial prompt text.
        tokenizer: Character or SentencePiece tokenizer.
        max_new_tokens: Number of new tokens to generate.
        temperature: Controls randomness. Higher values make output more random.
        finish_sentence: Continue briefly until the story reaches punctuation.
        max_completion_tokens: Maximum extra tokens used to finish the sentence.

    Returns:
        A generated text string.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0.")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")
    if max_completion_tokens < 0:
        raise ValueError("max_completion_tokens must not be negative.")

    device = next(model.parameters()).device
    ids = tokenizer.encode(start_text)

    if len(ids) == 0:
        raise ValueError("The prompt does not contain any known characters.")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    output_ids = list(ids)

    model.eval()

    with torch.no_grad():
        logits, hidden = model(x)
        token_limit = max_new_tokens
        if finish_sentence:
            token_limit += max_completion_tokens

        for generated_count in range(token_limit):
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)
            output_ids.append(next_id.item())
            logits, hidden = model(next_id, hidden)

            if generated_count + 1 < max_new_tokens:
                continue

            generated_text = tokenizer.decode(output_ids[len(ids):]).rstrip()
            if not finish_sentence or generated_text.endswith(
                tuple(SENTENCE_ENDINGS)
            ):
                break

    output = tokenizer.decode(output_ids).rstrip()
    if not finish_sentence or output.endswith(tuple(SENTENCE_ENDINGS)):
        return output

    prompt_text = tokenizer.decode(ids)
    continuation = output[len(prompt_text):]
    last_ending = max(continuation.rfind(mark) for mark in SENTENCE_ENDINGS)
    if last_ending >= 0:
        return f"{prompt_text}{continuation[:last_ending + 1]}".rstrip()

    return f"{output}."

def main():
    parser = argparse.ArgumentParser(description="Generate a short story with ChaosWriter.")

    parser.add_argument(
        "--prompt",
        type=str,
        default="A student finds a notebook that writes back.",
        help="Writing prompt for the model."
    )

    parser.add_argument(
        "--max_tokens",
        type=positive_int,
        default=200,
        help="Maximum number of subword tokens to generate."
    )

    parser.add_argument(
        "--temperature",
        type=positive_float,
        default=0.8,
        help="Sampling temperature. Higher values make output more random."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional file path to save the generated story."
    )

    parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=config.RANDOM_SEED,
        help="Random seed used to make generation reproducible."
    )

    args = parser.parse_args()
    set_random_seed(args.seed)

    model, tokenizer, _ = load_story_model(config.BEST_MODEL_PATH)

    formatted_prompt = build_anchored_prompt(args.prompt)

    output = generate_text(
        model,
        formatted_prompt,
        tokenizer,
        max_new_tokens=args.max_tokens,
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
