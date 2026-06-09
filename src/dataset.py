from datasets import load_dataset
import torch
from torch.utils.data import Dataset

from preprocess import format_prompt_story
from tokenizer import CharTokenizer


CharVocab = CharTokenizer


class StoryDataset(Dataset):
    def __init__(self, text, vocab, block_size=256):
        self.vocab = vocab
        self.block_size = block_size
        self.data = torch.tensor(vocab.encode(text), dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_writing_prompts_sample(max_examples=1000, max_story_chars=2000):
    dataset = load_dataset("euclaise/writingprompts", split="train")

    texts = []

    for i, row in enumerate(dataset):
        if i >= max_examples:
            break

        text = format_prompt_story(
            prompt=row["prompt"],
            story=row["story"],
            max_story_chars=max_story_chars
        )

        texts.append(text)

    return "".join(texts)
