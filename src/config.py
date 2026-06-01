# Basic configuration for ChaosWriter
DATASET_NAME = "euclaise/writingprompts"
MAX_EXAMPLES = 1000

BLOCK_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 3

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2

LEARNING_RATE = 0.001

TRAIN_SPLIT = 0.9

MODEL_PATH = "chaoswriter_lstm.pt"
BEST_MODEL_PATH = "chaoswriter_best.pt"
LOG_PATH = "training_log.txt"

MAX_STORY_CHARS = 2000

