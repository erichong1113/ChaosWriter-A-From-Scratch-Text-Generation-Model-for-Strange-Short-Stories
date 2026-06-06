# ChaosWriter

ChaosWriter is a from-scratch text generation project for strange short stories.

The goal is to train a small LSTM-based language model on prompt-story pairs. Given a writing prompt, the model will generate a short story continuation.

This is the first version of the repo. It currently includes:

- dataset loading
- simple text preprocessing
- basic training script
- basic generation script
- batch prompt generation
- temperature comparison experiments

## Useful commands

Train the model:

```bash
python src/train.py
```

Generate one story:

```bash
python src/generate.py --prompt "A student finds a notebook that writes back."
```

Compare different sampling temperatures:

```bash
python src/compare_temperature.py --temperatures 0.4 0.8 1.2 --seed 42
```
