from pathlib import Path


# =========================================================
# PROJECT DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# INPUT / OUTPUT FILES
# =========================================================

# Assignment 1 tokenized data copied into Assignment 4
INPUT_FILE = BASE_DIR / "indic_tokenized.txt"

# Output files will also be created directly in Assignment_4
RESULT_FILE = BASE_DIR / "results.txt"
SPLIT_FILE = BASE_DIR / "dataset_split.txt"

# Database containing n-gram counts
DATABASE_FILE = BASE_DIR / "ngram_counts.db"


# =========================================================
# DATASET SETTINGS
# =========================================================

TOTAL_SENTENCES = 100000

TRAIN_SENTENCES = 98000
DEV_SENTENCES = 1000
TEST_SENTENCES = 1000

RANDOM_SEED = 42


# =========================================================
# N-GRAM SETTINGS
# =========================================================

MAX_N = 4

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<UNK>"


# =========================================================
# PROCESSING SETTINGS
# =========================================================

BATCH_SIZE = 1000