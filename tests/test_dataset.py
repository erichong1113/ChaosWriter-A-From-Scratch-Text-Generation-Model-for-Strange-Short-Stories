import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import CharVocab, StoryDataset


class StoryDatasetTests(unittest.TestCase):
    def test_dataset_uses_non_overlapping_blocks(self):
        text = "abcdefghij"
        vocab = CharVocab(text)
        dataset = StoryDataset(text, vocab, block_size=3)

        self.assertEqual(len(dataset), 3)

        first_x, first_y = dataset[0]
        second_x, second_y = dataset[1]

        self.assertEqual(vocab.decode(first_x.tolist()), "abc")
        self.assertEqual(vocab.decode(first_y.tolist()), "bcd")
        self.assertEqual(vocab.decode(second_x.tolist()), "def")
        self.assertEqual(vocab.decode(second_y.tolist()), "efg")

    def test_dataset_requires_enough_text_for_a_full_block(self):
        text = "abc"
        vocab = CharVocab(text)
        dataset = StoryDataset(text, vocab, block_size=3)

        self.assertEqual(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
