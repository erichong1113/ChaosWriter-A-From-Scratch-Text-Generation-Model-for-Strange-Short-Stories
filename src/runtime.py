import argparse
import random

import torch
from torch.utils.data import random_split


def positive_float(value):
    parsed_value = float(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed_value


def positive_int(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed_value


def non_negative_int(value):
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed_value


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, train_fraction, seed):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two training sequences.")

    train_size = int(len(dataset) * train_fraction)
    train_size = min(max(train_size, 1), len(dataset) - 1)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)

    return random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
