import sys
import tempfile
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from checkpoint import load_story_model
from dataset import CharVocab
from model import LSTMStoryModel


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

        model, stoi, itos, metadata = load_story_model(
            self.checkpoint_path,
            device="cpu",
        )

        self.assertFalse(model.training)
        self.assertEqual(model.embedding.embedding_dim, 8)
        self.assertEqual(model.lstm.hidden_size, 12)
        self.assertEqual(stoi, self.stoi)
        self.assertEqual(itos, self.itos)
        self.assertEqual(metadata["epoch"], 3)

    def test_load_story_model_rejects_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "Please train the model"):
            load_story_model(self.checkpoint_path, device="cpu")

    def test_load_story_model_rejects_incomplete_checkpoint(self):
        torch.save({"model_state": {}}, self.checkpoint_path)

        with self.assertRaisesRegex(ValueError, "itos, stoi"):
            load_story_model(self.checkpoint_path, device="cpu")

    def test_char_vocab_can_be_restored_from_checkpoint_mappings(self):
        vocab = CharVocab.from_mappings(self.stoi, self.itos)

        self.assertEqual(vocab.encode("cab?"), [2, 0, 1])
        self.assertEqual(vocab.decode([1, 0, 2]), "bac")


if __name__ == "__main__":
    unittest.main()
