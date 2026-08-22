"""Load and split the tokenized corpus."""

from pathlib import Path

from config import INPUT_FILE, MAX_SENTENCES, TRAIN_SIZE, DEV_SIZE, TEST_SIZE


def load_sentences(filename=INPUT_FILE, max_sentences=MAX_SENTENCES):
    """Read one tokenized sentence per non-empty line."""
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path.resolve()}\n"
            "Keep indic_tokenized.txt in the Assignment_5 folder."
        )

    sentences = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            if tokens:
                sentences.append(tokens)

            if len(sentences) >= max_sentences:
                break

    required = TRAIN_SIZE + DEV_SIZE + TEST_SIZE
    if len(sentences) < required:
        raise ValueError(
            f"At least {required} non-empty sentences are required, "
            f"but only {len(sentences)} were found."
        )

    return sentences


def split_dataset(sentences):
    """Use the first 98,000 sentences for training, next 1,000 for dev,
    and next 1,000 for test."""
    train_end = TRAIN_SIZE
    dev_end = TRAIN_SIZE + DEV_SIZE
    test_end = dev_end + TEST_SIZE

    return (
        sentences[:train_end],
        sentences[train_end:dev_end],
        sentences[dev_end:test_end],
    )


def save_split(train, dev, test, filename):
    """Save the exact split used by the experiment."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write("=== TRAINING SET ===\n")
        for sentence in train:
            file.write(" ".join(sentence) + "\n")

        file.write("\n=== DEVELOPMENT SET ===\n")
        for sentence in dev:
            file.write(" ".join(sentence) + "\n")

        file.write("\n=== TEST SET ===\n")
        for sentence in test:
            file.write(" ".join(sentence) + "\n")
