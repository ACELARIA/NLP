"""Configuration for Assignment 5."""

INPUT_FILE = "indic_tokenized.txt"

# Assignment requirement: use at least 100,000 sentences.
MAX_SENTENCES = 100000

TRAIN_SIZE = 98000
DEV_SIZE = 1000
TEST_SIZE = 1000

# Assignment 5: Add-K smoothing.
ADD_K = 0.3

NGRAM_ORDERS = (1, 2, 3, 4)

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"

PROGRESS_EVERY = 1000

RESULTS_FILE = "results.txt"
SPLIT_FILE = "dataset_split.txt"
