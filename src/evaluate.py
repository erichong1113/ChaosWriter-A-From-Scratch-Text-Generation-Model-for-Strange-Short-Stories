import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from checkpoint import get_device, load_story_model
from dataset import CharVocab, StoryDataset, load_writing_prompts_sample
import config


def evaluate_model():
    device = get_device()

    print("Loading checkpoint...")
    model, stoi, itos, _ = load_story_model(
        config.BEST_MODEL_PATH,
        device=device,
    )

    print("Loading dataset...")
    text = load_writing_prompts_sample(
        max_examples=config.MAX_EXAMPLES,
        max_story_chars=config.MAX_STORY_CHARS
    )

    vocab = CharVocab.from_mappings(stoi, itos)
    dataset = StoryDataset(text, vocab, block_size=config.BLOCK_SIZE)

    train_size = int(len(dataset) * config.TRAIN_SPLIT)
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(dataset, [train_size, val_size])

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
