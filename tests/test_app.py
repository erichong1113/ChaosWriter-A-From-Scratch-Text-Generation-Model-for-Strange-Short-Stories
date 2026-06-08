import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import config


class AppTests(unittest.TestCase):
    def test_app_shows_training_message_without_checkpoint(self):
        original_path = config.BEST_MODEL_PATH
        config.BEST_MODEL_PATH = "missing-test-checkpoint.pt"
        try:
            app = AppTest.from_file(ROOT_DIR / "src" / "app.py").run(timeout=10)
        finally:
            config.BEST_MODEL_PATH = original_path

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "ChaosWriter")
        self.assertIn("Run `python src/train.py`", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
