import os

import config
from checkpoint import load_story_model
from generate import generate_text


PROMPT_FILE = "sample_prompts.txt"
OUTPUT_FILE = "outputs/batch_outputs.txt"


def load_prompts(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines()]

    return [prompt for prompt in prompts if prompt]


def main():
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    os.makedirs("outputs", exist_ok=True)

    model, stoi, itos, _ = load_story_model(config.BEST_MODEL_PATH)
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
