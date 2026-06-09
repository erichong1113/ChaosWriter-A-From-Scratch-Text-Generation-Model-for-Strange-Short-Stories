import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from checkpoint import get_device, load_story_model
from dataset import CharVocab
from model import LSTMStoryModel
from tokenizer import SentencePieceTokenizer


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temp_dir.name) / "model.pt"
        self.stoi = {"a": 0, "b": 1, "c": 2}
        self.itos = {index: char for char, index in self.stoi.items()}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_story_model_restores_model_and_metadata(self):
        original_model = LSTMStoryModel(
            vocab_size=len(self.stoi),
            embed_dim=8,
            hidden_dim=12,
            num_layers=1,
        )
        checkpoint = {
            "model_state": original_model.state_dict(),
            "stoi": self.stoi,
            "itos": self.itos,
            "config": {
                "embed_dim": 8,
                "hidden_dim": 12,
                "num_layers": 1,
            },
            "epoch": 3,
        }
        torch.save(checkpoint, self.checkpoint_path)

        model, tokenizer, metadata = load_story_model(
            self.checkpoint_path,
            device="cpu",
        )

        self.assertFalse(model.training)
        self.assertEqual(model.embedding.embedding_dim, 8)
        self.assertEqual(model.lstm.hidden_size, 12)
        self.assertEqual(tokenizer.stoi, self.stoi)
        self.assertEqual(tokenizer.itos, self.itos)
        self.assertEqual(metadata["epoch"], 3)

    def test_load_story_model_rejects_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "Please train the model"):
            load_story_model(self.checkpoint_path, device="cpu")

    def test_load_story_model_rejects_incomplete_checkpoint(self):
        torch.save({"model_state": {}}, self.checkpoint_path)

        with self.assertRaisesRegex(ValueError, "itos, stoi"):
            load_story_model(self.checkpoint_path, device="cpu")

    def test_load_story_model_restores_sentencepiece_tokenizer(self):
        text = "\n".join(["A monster waits under the school."] * 30)
        tokenizer = SentencePieceTokenizer.train(text, vocab_size=300)
        original_model = LSTMStoryModel(
            vocab_size=tokenizer.vocab_size,
            embed_dim=8,
            hidden_dim=12,
            num_layers=1,
        )
        checkpoint = {
            "model_state": original_model.state_dict(),
            "config": {
                "embed_dim": 8,
                "hidden_dim": 12,
                "num_layers": 1,
            },
        }
        checkpoint.update(tokenizer.checkpoint_data())
        torch.save(checkpoint, self.checkpoint_path)

        model, restored, _ = load_story_model(
            self.checkpoint_path,
            device="cpu",
        )

        self.assertFalse(model.training)
        self.assertEqual(restored.tokenizer_type, "sentencepiece")
        self.assertEqual(
            restored.decode(restored.encode("A monster waits.")),
            "A monster waits.",
        )

    def test_char_vocab_can_be_restored_from_checkpoint_mappings(self):
        vocab = CharVocab.from_mappings(self.stoi, self.itos)

        self.assertEqual(vocab.encode("cab?"), [2, 0, 1])
        self.assertEqual(vocab.decode([1, 0, 2]), "bac")

    @patch("checkpoint.torch.backends.mps.is_available", return_value=True)
    @patch("checkpoint.torch.cuda.is_available", return_value=False)
    def test_get_device_uses_mps_when_available(self, _, __):
        self.assertEqual(get_device(), "mps")

    @patch("checkpoint.torch.backends.mps.is_available", return_value=False)
    @patch("checkpoint.torch.cuda.is_available", return_value=False)
    def test_get_device_falls_back_to_cpu(self, _, __):
        self.assertEqual(get_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
