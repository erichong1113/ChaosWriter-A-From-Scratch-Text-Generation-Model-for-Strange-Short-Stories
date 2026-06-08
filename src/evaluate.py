import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from checkpoint import get_device, load_story_model
from dataset import CharVocab, StoryDataset, load_writing_prompts_sample
from runtime import set_random_seed, split_dataset
import config


def evaluate_model():
    device = get_device()
    set_random_seed(config.RANDOM_SEED)

    print("Loading checkpoint...")
    model, stoi, itos, checkpoint = load_story_model(
        config.BEST_MODEL_PATH,
        device=device,
    )
    split_seed = checkpoint.get("config", {}).get(
        "random_seed",
        config.RANDOM_SEED,
    )

    print("Loading dataset...")
    text = load_writing_prompts_sample(
        max_examples=config.MAX_EXAMPLES,
        max_story_chars=config.MAX_STORY_CHARS
    )

    vocab = CharVocab.from_mappings(stoi, itos)
    dataset = StoryDataset(text, vocab, block_size=config.BLOCK_SIZE)

    _, val_dataset = split_dataset(
        dataset,
        train_fraction=config.TRAIN_SPLIT,
        seed=split_seed,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    evaluate_model()
