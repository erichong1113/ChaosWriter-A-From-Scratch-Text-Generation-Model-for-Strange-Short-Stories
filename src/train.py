import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CharVocab, StoryDataset, load_writing_prompts_sample
from model import LSTMStoryModel
from runtime import set_random_seed, split_dataset
import config


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(config.RANDOM_SEED)

    print("Loading dataset...")
    text = load_writing_prompts_sample(
        max_examples=config.MAX_EXAMPLES,
        max_story_chars=config.MAX_STORY_CHARS
    )

    print("Building vocabulary...")
    vocab = CharVocab(text)

    full_dataset = StoryDataset(text, vocab, block_size=config.BLOCK_SIZE)

    train_dataset, val_dataset = split_dataset(
        full_dataset,
        train_fraction=config.TRAIN_SPLIT,
        seed=config.RANDOM_SEED,
    )
    train_size = len(train_dataset)
    val_size = len(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    model = LSTMStoryModel(
        vocab_size=len(vocab.stoi),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    print("Training model...")

    with open(config.LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("ChaosWriter Training Log\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Examples: {config.MAX_EXAMPLES}\n")
        log_file.write(f"Train size: {train_size}\n")
        log_file.write(f"Validation size: {val_size}\n")
        log_file.write(f"Block size: {config.BLOCK_SIZE}\n")
        log_file.write(f"Batch size: {config.BATCH_SIZE}\n")
        log_file.write(f"Random seed: {config.RANDOM_SEED}\n")
        log_file.write(f"Model: LSTM\n\n")

        for epoch in range(config.EPOCHS):
            model.train()
            total_train_loss = 0

            for x, y in tqdm(train_loader):
                x = x.to(device)
                y = y.to(device)

                logits, _ = model(x)

                train_loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_train_loss += train_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_perplexity = math.exp(avg_train_loss)

            val_loss, val_perplexity = evaluate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device
            )

            message = (
                f"Epoch {epoch + 1}/{config.EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train PPL: {train_perplexity:.2f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_perplexity:.2f}"
            )

            print(message)
            log_file.write(message + "\n")

            checkpoint = {
                "model_state": model.state_dict(),
                "stoi": vocab.stoi,
                "itos": vocab.itos,
                "config": {
                    "embed_dim": config.EMBED_DIM,
                    "hidden_dim": config.HIDDEN_DIM,
                    "num_layers": config.NUM_LAYERS,
                    "block_size": config.BLOCK_SIZE,
                    "random_seed": config.RANDOM_SEED,
                },
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity
            }

            torch.save(checkpoint, config.MODEL_PATH)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, config.BEST_MODEL_PATH)

                best_message = (
                    f"New best model saved with Val Loss: {best_val_loss:.4f}"
                )

                print(best_message)
                log_file.write(best_message + "\n")

    print(f"Last model saved to {config.MODEL_PATH}")
    print(f"Best model saved to {config.BEST_MODEL_PATH}")
    print(f"Training log saved to {config.LOG_PATH}")


if __name__ == "__main__":
    train()
