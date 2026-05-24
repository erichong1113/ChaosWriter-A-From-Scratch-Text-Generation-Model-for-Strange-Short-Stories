import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CharVocab, StoryDataset, load_writing_prompts_sample
from model import LSTMStoryModel


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    text = load_writing_prompts_sample(max_examples=1000)

    print("Building vocabulary...")
    vocab = CharVocab(text)

    dataset = StoryDataset(text, vocab, block_size=256)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMStoryModel(vocab_size=len(vocab.stoi)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 3

    print("Training model...")
    for epoch in range(epochs):
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "stoi": vocab.stoi,
        "itos": vocab.itos
    }, "chaoswriter_lstm.pt")

    print("Model saved to chaoswriter_lstm.pt")


if __name__ == "__main__":
    train()