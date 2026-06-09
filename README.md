# ChaosWriter

## Project Overview

ChaosWriter is a from-scratch text generation system for strange short
stories. Given a writing prompt, it uses a PyTorch LSTM language model to
generate an original continuation. The project includes the complete machine
learning pipeline: dataset download, preprocessing, SentencePiece BPE
tokenization, model training, validation, checkpoint management, command-line
generation, reproducible sampling experiments, automated tests, and a
Streamlit chatbot.

The goal was to build and understand a text generator rather than call a
pretrained language model. The model learns next-token prediction from 10,000
prompt-story pairs in the `euclaise/writingprompts` dataset. The current model
was trained for 12 epochs on an Apple M1 GPU using PyTorch MPS.

## Extra Criterion: Chatbot GUI

The extra criterion pursued is a **chatbot GUI**. `src/app.py` provides a
stateful Streamlit chat interface where a user can:

- enter a custom story prompt
- control generated story length
- adjust sampling creativity with temperature
- choose a random seed for reproducible output
- clear the conversation and generate another story

The interface loads the best validation checkpoint and uses the same
generation pipeline as the command-line tools.

## Difficulties and Solutions

### Character-level output was often unreadable

The first version predicted one character at a time. It learned punctuation
and word-like patterns, but frequently invented misspelled words. I replaced
the character vocabulary with a 2,000-token SentencePiece BPE tokenizer.
Subword tokens allow the model to learn complete words and reusable word
fragments while still handling unfamiliar text.

### Generated text could ignore the prompt

A small LSTM does not have the semantic understanding of a large pretrained
transformer. To keep short generations visibly connected to the requested
topic, I added prompt anchoring. The user's prompt becomes the opening sentence
of the story, then the LSTM continues from that context. This improves topic
continuity without retraining or adding another model.

### Stories could stop in the middle of a sentence

A strict token limit sometimes produced endings such as "he was afraid to".
After reaching the requested length, generation can now continue for up to 40
additional tokens and stop when it reaches `.`, `!`, or `?`. A fallback also
prevents an unfinished final sentence when no ending punctuation is sampled.

### Reproducibility and checkpoint compatibility

Training and generation use fixed random seeds, and checkpoints store the
tokenizer, model dimensions, split settings, and model state. Shared checkpoint
loading validates required data and remains compatible with earlier
character-level checkpoints. Temperature comparison and batch-generation
scripts make output experiments repeatable.

## Model and Code Explanation

### Data pipeline

`src/dataset.py` downloads the WritingPrompts dataset. `src/preprocess.py`
cleans each example and formats it as:

```text
Prompt: <writing prompt>
Story: <story text>
```

Stories are truncated to a configurable maximum length. `src/tokenizer.py`
trains SentencePiece BPE directly from the corpus and converts text into token
IDs. The encoded corpus is divided into consecutive, non-overlapping blocks.
Each input block is paired with the same sequence shifted by one token as its
target.

### Model architecture

`LSTMStoryModel` in `src/model.py` contains:

1. An embedding layer that converts token IDs into 128-dimensional vectors.
2. A two-layer LSTM with a hidden size of 256.
3. A linear output layer that predicts a score for every token in the
   2,000-token vocabulary.

The model is implemented with PyTorch and is trained from scratch. It does not
use pretrained weights or an external text-generation API.

### Training

`src/train.py` minimizes cross-entropy loss for next-token prediction using
Adam with a learning rate of `0.001`. Data is split 90/10 into training and
validation subsets with random seed `42`. After every epoch, the script records
loss and perplexity and saves both the latest checkpoint and the checkpoint
with the lowest validation loss.

### Generation

`src/generate.py` processes the prompt once through the LSTM and then reuses
its hidden state while sampling each new token. Temperature changes the
probability distribution:

- lower temperature produces safer, more predictable text
- higher temperature produces more varied but less stable text

A random seed makes the sampling result reproducible. Prompt anchoring keeps
the opening connected to the user's idea, and sentence completion avoids
cutting off the final sentence.

## Training Results

Final training configuration:

| Setting           |             Value |
| ----------------- | ----------------: |
| Training examples |            10,000 |
| Tokenizer         | SentencePiece BPE |
| Vocabulary size   |             2,000 |
| Sequence length   |        128 tokens |
| Batch size        |                32 |
| Epochs            |                12 |
| Training device   |      Apple M1 MPS |

Final validation results:

| Metric                |  Value |
| --------------------- | -----: |
| Validation loss       | 3.6217 |
| Validation perplexity |  37.40 |

Validation loss improved during every epoch, from `4.4897` in epoch 1 to
`3.6217` in epoch 12. The complete record is available in
`training_log_bpe.txt`.

The result contains recognizable English sentence structure and preserves the
prompt as its subject. Because this is a compact model trained from scratch,
some sentences are still unusual

## Code Quality and Development

The repository was developed through incremental commits rather than a single
final upload. Separate commits introduced preprocessing, command-line
generation, saved outputs, batch generation, temperature experiments,
checkpoint loading, the Streamlit interface, and tests.

Responsibilities are separated across small modules instead of one large
script. Shared behavior such as device selection, random seeds, checkpoint
loading, tokenization, and validation is reused by training, evaluation, and
generation tools. The test suite currently contains 23 tests covering dataset
blocks, tokenizer restoration, checkpoint validation, generation arguments,
prompt anchoring, sentence completion, runtime utilities, and application
imports.

## Project Structure

| Path                         | Responsibility                                       |
| ---------------------------- | ---------------------------------------------------- |
| `src/model.py`               | LSTM language model architecture                     |
| `src/dataset.py`             | Dataset download and training sequences              |
| `src/tokenizer.py`           | SentencePiece BPE and legacy tokenizer support       |
| `src/preprocess.py`          | Prompt-story formatting and cleanup                  |
| `src/train.py`               | Training, validation, logging, and checkpoint saving |
| `src/evaluate.py`            | Standalone validation loss and perplexity            |
| `src/checkpoint.py`          | Shared checkpoint validation and loading             |
| `src/runtime.py`             | Device-independent seeds and argument validation     |
| `src/generate.py`            | Prompt anchoring, sampling, and sentence completion  |
| `src/batch_generate.py`      | Generation for multiple prompts                      |
| `src/compare_temperature.py` | Reproducible sampling comparison                     |
| `src/app.py`                 | Streamlit chatbot GUI                                |
| `tests/`                     | Automated unit tests                                 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On an Apple Silicon Mac, MPS support can be checked with:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

ChaosWriter automatically selects CUDA when available, then MPS, and otherwise
uses CPU.

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
  --max_tokens 200 \
  --temperature 0.5 \
  --seed 42
```

Launch the chatbot:

```bash
streamlit run src/app.py
```

Generate stories from a prompt file:

```bash
python src/batch_generate.py \
  --prompt_file sample_prompts.txt \
  --output_file outputs/batch_outputs.txt
```

Compare temperatures using the same prompt and random seed:

```bash
python src/compare_temperature.py \
  --temperatures 0.4 0.8 1.2 \
  --seed 42
```

Run all tests:

```bash
python -m unittest discover -s tests -v
```

## Limitations

- The model is much smaller than a pretrained transformer and may lose the
  topic or logical consistency over long generations.
- Prompt anchoring guarantees a relevant opening but does not give the LSTM
  full semantic understanding of the prompt.
- Checkpoints and generated output files are excluded from Git because they
  are large or machine-specific.
