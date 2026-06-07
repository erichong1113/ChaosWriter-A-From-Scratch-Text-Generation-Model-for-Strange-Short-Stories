import os

import torch

import config
from model import LSTMStoryModel


REQUIRED_CHECKPOINT_KEYS = {"model_state", "stoi", "itos"}


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    if not stoi or not itos:
        raise ValueError("Checkpoint vocabulary cannot be empty.")

    model_config = checkpoint.get("config", {})
    model = LSTMStoryModel(
        vocab_size=len(stoi),
        embed_dim=model_config.get("embed_dim", config.EMBED_DIM),
        hidden_dim=model_config.get("hidden_dim", config.HIDDEN_DIM),
        num_layers=model_config.get("num_layers", config.NUM_LAYERS),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, stoi, itos, checkpoint
