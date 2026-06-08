import argparse
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from runtime import (
    non_negative_int,
    positive_float,
    positive_int,
    set_random_seed,
    split_dataset,
)


class RuntimeTests(unittest.TestCase):
    def test_numeric_argument_validators_accept_valid_values(self):
        self.assertEqual(positive_int("5"), 5)
        self.assertEqual(positive_float("0.8"), 0.8)
        self.assertEqual(non_negative_int("0"), 0)

    def test_numeric_argument_validators_reject_invalid_values(self):
        for validator, value in (
            (positive_int, "0"),
            (positive_float, "-0.1"),
            (non_negative_int, "-1"),
        ):
            with self.subTest(validator=validator.__name__):
                with self.assertRaises(argparse.ArgumentTypeError):
                    validator(value)

    def test_split_dataset_is_reproducible(self):
        dataset = TensorDataset(torch.arange(20))

        first_train, first_val = split_dataset(dataset, 0.8, seed=42)
        second_train, second_val = split_dataset(dataset, 0.8, seed=42)

        self.assertEqual(first_train.indices, second_train.indices)
        self.assertEqual(first_val.indices, second_val.indices)
        self.assertEqual(len(first_train), 16)
        self.assertEqual(len(first_val), 4)

    def test_split_dataset_rejects_too_little_data(self):
        dataset = TensorDataset(torch.arange(1))

        with self.assertRaisesRegex(ValueError, "at least two"):
            split_dataset(dataset, 0.9, seed=42)

    def test_set_random_seed_repeats_torch_samples(self):
        set_random_seed(7)
        first = torch.rand(4)
        set_random_seed(7)
        second = torch.rand(4)

        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
