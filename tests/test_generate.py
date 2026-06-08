import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from generate import generate_text
from model import LSTMStoryModel


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.stoi = {"a": 0, "b": 1}
        self.itos = {0: "a", 1: "b"}
        self.model = LSTMStoryModel(
            vocab_size=2,
            embed_dim=4,
            hidden_dim=6,
            num_layers=1,
        )

    def test_generate_text_returns_prompt_and_requested_characters(self):
        output = generate_text(
            model=self.model,
            start_text="ab",
            stoi=self.stoi,
            itos=self.itos,
            max_new_chars=3,
            temperature=0.8,
        )

        self.assertEqual(len(output), 5)
        self.assertTrue(output.startswith("ab"))

    def test_generate_text_rejects_invalid_sampling_values(self):
        with self.assertRaisesRegex(ValueError, "max_new_chars"):
            generate_text(
                self.model,
                "ab",
                self.stoi,
                self.itos,
                max_new_chars=0,
            )

        with self.assertRaisesRegex(ValueError, "temperature"):
            generate_text(
                self.model,
                "ab",
                self.stoi,
                self.itos,
                temperature=0,
            )

    def test_generate_text_rejects_unknown_prompt(self):
        with self.assertRaisesRegex(ValueError, "known characters"):
            generate_text(
                self.model,
                "?",
                self.stoi,
                self.itos,
            )


if __name__ == "__main__":
    unittest.main()
