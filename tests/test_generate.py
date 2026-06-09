import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from generate import build_anchored_prompt, generate_text
from model import LSTMStoryModel
from tokenizer import CharTokenizer


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.stoi = {"a": 0, "b": 1}
        self.itos = {0: "a", 1: "b"}
        self.tokenizer = CharTokenizer(stoi=self.stoi, itos=self.itos)
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
            tokenizer=self.tokenizer,
            max_new_tokens=3,
            temperature=0.8,
            finish_sentence=False,
        )

        self.assertEqual(len(output), 5)
        self.assertTrue(output.startswith("ab"))

    def test_generate_text_adds_sentence_ending_as_fallback(self):
        output = generate_text(
            model=self.model,
            start_text="ab",
            tokenizer=self.tokenizer,
            max_new_tokens=1,
            temperature=0.8,
            max_completion_tokens=0,
        )

        self.assertTrue(output.endswith("."))

    def test_build_anchored_prompt_repeats_prompt_as_story_opening(self):
        output = build_anchored_prompt("A monster lives under a school")

        self.assertEqual(
            output,
            "Prompt: A monster lives under a school.\n"
            "Story: A monster lives under a school.",
        )

    def test_build_anchored_prompt_preserves_terminal_punctuation(self):
        output = build_anchored_prompt("Who opened the door?")

        self.assertTrue(output.endswith("Story: Who opened the door?"))

    def test_build_anchored_prompt_rejects_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_anchored_prompt("  ")

    def test_generate_text_rejects_invalid_sampling_values(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            generate_text(
                self.model,
                "ab",
                self.tokenizer,
                max_new_tokens=0,
            )

        with self.assertRaisesRegex(ValueError, "temperature"):
            generate_text(
                self.model,
                "ab",
                self.tokenizer,
                temperature=0,
            )

        with self.assertRaisesRegex(ValueError, "max_completion_tokens"):
            generate_text(
                self.model,
                "ab",
                self.tokenizer,
                max_completion_tokens=-1,
            )

    def test_generate_text_rejects_unknown_prompt(self):
        with self.assertRaisesRegex(ValueError, "known characters"):
            generate_text(
                self.model,
                "?",
                self.tokenizer,
            )


if __name__ == "__main__":
    unittest.main()
