import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tokenizer import SentencePieceTokenizer, tokenizer_from_checkpoint


class SentencePieceTokenizerTests(unittest.TestCase):
    def test_sentencepiece_round_trip_and_checkpoint_restore(self):
        text = "\n".join(
            [
                "Prompt: A monster waits below the school.",
                "Story: The student opens the hidden door.",
            ]
            * 20
        )
        tokenizer = SentencePieceTokenizer.train(text, vocab_size=300)
        ids = tokenizer.encode("A monster waits below the school.")
        restored = tokenizer_from_checkpoint(tokenizer.checkpoint_data())

        self.assertGreater(len(ids), 0)
        self.assertEqual(restored.decode(ids), tokenizer.decode(ids))
        self.assertEqual(restored.vocab_size, tokenizer.vocab_size)


if __name__ == "__main__":
    unittest.main()
