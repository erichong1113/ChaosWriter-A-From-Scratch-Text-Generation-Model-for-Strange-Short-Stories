import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CharVocab, StoryDataset, load_writing_prompts_sample
from model import LSTMStoryModel
import config


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    text = load_writing_prompts_sample(max_examples=config.MAX_EXAMPLES)

    print("Building vocabulary...")
    vocab = CharVocab(text)

    dataset = StoryDataset(text, vocab, block_size=config.BLOCK_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    model = LSTMStoryModel(
        vocab_size=len(vocab.stoi),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print("Training model...")

    with open(config.LOG_PATH, "w") as log_file:
        log_file.write("ChaosWriter Training Log\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Examples: {config.MAX_EXAMPLES}\n")
        log_file.write(f"Block size: {config.BLOCK_SIZE}\n")
        log_file.write(f"Batch size: {config.BATCH_SIZE}\n\n")

        for epoch in range(config.EPOCHS):
            total_loss = 0

            for x, y in tqdm(dataloader):
                x = x.to(device)
                y = y.to(device)

                logits, _ = model(x)

                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            message = f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {avg_loss:.4f}"

            print(message)
            log_file.write(message + "\n")

    torch.save({
        "model_state": model.state_dict(),
        "stoi": vocab.stoi,
        "itos": vocab.itos,
        "config": {
            "embed_dim": config.EMBED_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "num_layers": config.NUM_LAYERS,
            "block_size": config.BLOCK_SIZE
        }
    }, config.MODEL_PATH)

    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Training log saved to {config.LOG_PATH}")


if __name__ == "__main__":
    train()