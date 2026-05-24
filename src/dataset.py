from datasets import load_dataset
import torch
from torch.utils.data import Dataset


class CharVocab:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])


class StoryDataset(Dataset):
    def __init__(self, text, vocab, block_size=256):
        self.vocab = vocab
        self.block_size = block_size
        self.data = torch.tensor(vocab.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_writing_prompts_sample(max_examples=1000):
    dataset = load_dataset("euclaise/writingprompts", split="train")

    texts = []
    for i, row in enumerate(dataset):
        if i >= max_examples:
            break

        prompt = row["prompt"]
        story = row["story"]

        text = f"Prompt: {prompt}\nStory: {story}\n\n"
        texts.append(text)

    return "".join(texts)