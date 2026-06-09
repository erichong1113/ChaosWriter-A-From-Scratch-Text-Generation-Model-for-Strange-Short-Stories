# Basic configuration for ChaosWriter
DATASET_NAME = "euclaise/writingprompts"
MAX_EXAMPLES = 10000

TOKENIZER_TYPE = "sentencepiece"
VOCAB_SIZE = 2000

BLOCK_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 12

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2

LEARNING_RATE = 0.001

TRAIN_SPLIT = 0.9
RANDOM_SEED = 42

MODEL_PATH = "chaoswriter_bpe_lstm.pt"
BEST_MODEL_PATH = "chaoswriter_bpe_best.pt"
LOG_PATH = "training_log_bpe.txt"

MAX_STORY_CHARS = 2000
