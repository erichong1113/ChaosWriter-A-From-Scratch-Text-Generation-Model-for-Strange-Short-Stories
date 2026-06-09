import os

import torch

import config
from model import LSTMStoryModel
from tokenizer import tokenizer_from_checkpoint


REQUIRED_CHECKPOINT_KEYS = {"model_state"}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_story_model(checkpoint_path=None, device=None):
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH
    device = device or get_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            "Please train the model first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    missing_keys = REQUIRED_CHECKPOINT_KEYS - checkpoint.keys()

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    tokenizer = tokenizer_from_checkpoint(checkpoint)
    if tokenizer.vocab_size == 0:
        raise ValueError("Checkpoint vocabulary cannot be empty.")

    model_config = checkpoint.get("config", {})
    model = LSTMStoryModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=model_config.get("embed_dim", config.EMBED_DIM),
        hidden_dim=model_config.get("hidden_dim", config.HIDDEN_DIM),
        num_layers=model_config.get("num_layers", config.NUM_LAYERS),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer, checkpoint
