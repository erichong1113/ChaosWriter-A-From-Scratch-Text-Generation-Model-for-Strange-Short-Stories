# ChaosWriter

ChaosWriter is a from-scratch character-level text generation model for strange
short stories. It trains an LSTM language model on prompt-story pairs and can
generate continuations from the command line, in batches, or through a
Streamlit chatbot.

## Extra Criterion

This project includes a stateful chatbot GUI built with Streamlit. Users send
story prompts through a chat interface, receive generated stories, and can
control story length, sampling temperature, and random seed.

## How It Works

### Dataset and preprocessing

The training script downloads the `euclaise/writingprompts` dataset from
Hugging Face. Each example is converted into a consistent form:

```text
Prompt: <writing prompt>
Story: <story text>
```

Stories are truncated to `MAX_STORY_CHARS`, joined into one training corpus,
and encoded one character at a time. The vocabulary maps every unique
character to an integer ID.

### Model architecture

`LSTMStoryModel` contains three layers:

1. An embedding layer converts character IDs into dense vectors.
2. A multi-layer LSTM processes the sequence and carries information from
   earlier characters.
3. A linear layer produces a score for every possible next character.

The model is implemented directly with PyTorch rather than using a pretrained
language model.

### Training and evaluation

Training uses next-character prediction. For each sequence, the input contains
all characters except the last one and the target contains the same sequence
shifted by one character. Cross-entropy loss measures prediction error.

The dataset is split into training and validation subsets with a fixed random
seed. The same seed is saved in the checkpoint and reused by the standalone
evaluation script, so both scripts evaluate the same validation subset.

The training log reports:

- training loss
- training perplexity
- validation loss
- validation perplexity

The latest checkpoint is saved to `chaoswriter_lstm.pt`. The checkpoint with
the lowest validation loss is saved to `chaoswriter_best.pt`.

### Text generation

Generation repeatedly samples one next character from the model's probability
distribution. Temperature controls that distribution:

- lower values favor likely characters and more predictable text
- higher values increase variety and randomness

Random seeds make generation experiments reproducible. The temperature
comparison script generates the same prompt at several temperatures for direct
comparison.

## Project Structure

| Path | Responsibility |
| --- | --- |
| `src/model.py` | Character-level LSTM architecture |
| `src/dataset.py` | Dataset download, vocabulary, and training sequences |
| `src/preprocess.py` | Prompt-story formatting and text cleanup |
| `src/train.py` | Training loop, validation, logging, and checkpoints |
| `src/evaluate.py` | Standalone validation loss and perplexity |
| `src/checkpoint.py` | Shared, validated checkpoint loading |
| `src/runtime.py` | Random seeds, argument validation, and dataset splitting |
| `src/generate.py` | Single-prompt command-line generation |
| `src/batch_generate.py` | Generation from a prompt file |
| `src/compare_temperature.py` | Reproducible temperature experiments |
| `src/app.py` | Streamlit chatbot GUI |
| `tests/` | Unit tests for shared project behavior |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python src/train.py
```

Evaluate the best checkpoint:

```bash
python src/evaluate.py
```

Generate one story:

```bash
python src/generate.py \
  --prompt "A student finds a notebook that writes back." \
  --max_chars 500 \
  --temperature 0.8 \
  --seed 42
```

Generate stories from `sample_prompts.txt`:

```bash
python src/batch_generate.py \
  --prompt_file sample_prompts.txt \
  --output_file outputs/batch_outputs.txt
```

Compare sampling temperatures:

```bash
python src/compare_temperature.py \
  --temperatures 0.4 0.8 1.2 \
  --seed 42
```

Launch the chatbot GUI:

```bash
streamlit run src/app.py
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Configuration

Training hyperparameters are stored in `src/config.py`, including dataset size,
sequence length, batch size, model dimensions, learning rate, number of epochs,
train-validation split, and random seed.

## Limitations

- Character-level generation is slower and less semantically aware than a
  pretrained token-level language model.
- Training uses a limited dataset sample by default so it can run on modest
  hardware.
- Generated stories may lose long-range consistency because the model is a
  small LSTM trained from scratch.
- Model checkpoints and generated outputs are intentionally excluded from Git
  because they can be large or machine-specific.
