import os
import torch

import config
from model import LSTMStoryModel
from generate import generate_text


PROMPT_FILE = "sample_prompts.txt"
OUTPUT_FILE = "outputs/batch_outputs.txt"


def load_checkpoint():
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


def load_prompts(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines()]

    return [prompt for prompt in prompts if prompt]


def main():
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {config.BEST_MODEL_PATH}. "
            "Please train the model first."
        )

    os.makedirs("outputs", exist_ok=True)

    model, stoi, itos = load_checkpoint()
    prompts = load_prompts(PROMPT_FILE)

    results = []

    for i, prompt in enumerate(prompts, start=1):
        formatted_prompt = f"Prompt: {prompt}\nStory:"

        story = generate_text(
            model=model,
            start_text=formatted_prompt,
            stoi=stoi,
            itos=itos,
            max_new_chars=600,
            temperature=0.8,
        )

        result = (
            f"Example {i}\n"
            f"Prompt: {prompt}\n"
            f"Generated Story:\n{story}\n"
            f"{'-' * 80}\n"
        )

        print(result)
        results.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"Batch generation saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()