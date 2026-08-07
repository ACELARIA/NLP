"""
process_indic.py

Reads IndicCorpV2 text corpus,
tokenizes it,
stores tokenized output,
creates Parquet file,
computes corpus statistics.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from tokenizer import tokenize_paragraph
from statistics import CorpusStatistics

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

MAX_LINES = 100000      # Use subset

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "indic" / "hi-1.txt"

OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PARQUET_DIR = BASE_DIR / "data" / "parquet"
PARQUET_DIR.mkdir(exist_ok=True)

TOKENIZED_FILE = OUTPUT_DIR / "indic_tokenized.txt"

PARQUET_FILE = PARQUET_DIR / "indic.parquet"

STATISTICS_FILE = OUTPUT_DIR / "indic_statistics.txt"

# -------------------------------------------------------
# Check input file
# -------------------------------------------------------

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nCannot find\n{INPUT_FILE}\n"
    )

print("=" * 60)
print("Processing IndicCorpV2")
print("=" * 60)

stats = CorpusStatistics()

rows = []

# -------------------------------------------------------
# Read corpus
# -------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(TOKENIZED_FILE, "w", encoding="utf-8") as outfile:

    for line_number, paragraph in enumerate(
        tqdm(infile, desc="Reading Corpus"),
        start=1
    ):

        if line_number > MAX_LINES:
            break

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        tokenized = tokenize_paragraph(paragraph)

        for sentence in tokenized:

            if not sentence:
                continue

            stats.add_sentence(sentence)

            tokenized_sentence = " ".join(sentence)

            outfile.write(tokenized_sentence + "\n")

            rows.append(
                {
                    "sentence": tokenized_sentence
                }
            )

# -------------------------------------------------------
# Save Parquet
# -------------------------------------------------------

df = pd.DataFrame(rows)

df.to_parquet(
    PARQUET_FILE,
    compression="snappy",
    index=False
)

# -------------------------------------------------------
# Save statistics
# -------------------------------------------------------

stats.print_statistics()

stats.save_statistics(STATISTICS_FILE)

print()

print("Tokenized File :")
print(TOKENIZED_FILE)

print()

print("Parquet File :")
print(PARQUET_FILE)

print()

print("Statistics :")
print(STATISTICS_FILE)

print()

print("Finished Processing Indic Corpus")