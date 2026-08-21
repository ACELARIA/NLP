import random

from config import (
    TOTAL_SENTENCES,
    TRAIN_SENTENCES,
    DEV_SENTENCES,
    TEST_SENTENCES,
    RANDOM_SEED
)


# =========================================================
# LOAD TOKENIZED SENTENCES
# =========================================================

def load_sentences(file_path):

    """
    Load tokenized sentences from the Assignment-1 file.

    Assumption:
    One sentence per line.
    Words/tokens are separated by spaces.
    """

    sentences = []

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    print(f"Input file: {file_path}")

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Split sentence into tokens
            tokens = line.split()

            if tokens:
                sentences.append(tokens)

            # We only need 100,000 sentences
            if len(sentences) >= TOTAL_SENTENCES:
                break

    print(f"Sentences loaded: {len(sentences)}")

    return sentences


# =========================================================
# SPLIT DATASET
# =========================================================

def split_dataset(sentences):

    """
    Split 100,000 sentences into:

    Training    = 98,000
    Development = 1,000
    Testing     = 1,000
    """

    if len(sentences) < TOTAL_SENTENCES:

        raise ValueError(
            f"Assignment requires at least "
            f"{TOTAL_SENTENCES} sentences, "
            f"but only {len(sentences)} were found."
        )

    # Take exactly 100,000
    sentences = sentences[:TOTAL_SENTENCES]

    # Reproducible random split
    random.seed(RANDOM_SEED)

    random.shuffle(sentences)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    train = sentences[
        :TRAIN_SENTENCES
    ]

    # -----------------------------------------------------
    # Development
    # -----------------------------------------------------

    dev_start = TRAIN_SENTENCES

    dev_end = (
        TRAIN_SENTENCES +
        DEV_SENTENCES
    )

    dev = sentences[
        dev_start:dev_end
    ]

    # -----------------------------------------------------
    # Testing
    # -----------------------------------------------------

    test = sentences[
        dev_end:
        dev_end + TEST_SENTENCES
    ]

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Total sentences      : "
        f"{len(sentences)}"
    )

    print(
        f"Training sentences   : "
        f"{len(train)}"
    )

    print(
        f"Development sentences: "
        f"{len(dev)}"
    )

    print(
        f"Testing sentences    : "
        f"{len(test)}"
    )

    return train, dev, test


# =========================================================
# SAVE SPLIT INFORMATION
# =========================================================

def save_split_information(
    train,
    dev,
    test,
    output_file
):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ASSIGNMENT 4 - DATASET SPLIT\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Training sentences    : "
            f"{len(train)}\n"
        )

        file.write(
            f"Development sentences : "
            f"{len(dev)}\n"
        )

        file.write(
            f"Testing sentences     : "
            f"{len(test)}\n"
        )

        file.write("\n")

        if train:

            file.write(
                "Sample training sentence:\n"
            )

            file.write(
                " ".join(train[0]) + "\n\n"
            )

        if dev:

            file.write(
                "Sample development sentence:\n"
            )

            file.write(
                " ".join(dev[0]) + "\n\n"
            )

        if test:

            file.write(
                "Sample testing sentence:\n"
            )

            file.write(
                " ".join(test[0]) + "\n"
            )