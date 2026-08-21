import time

from config import (
    INPUT_FILE,
    RESULT_FILE,
    SPLIT_FILE,
    DATABASE_FILE,
    TOTAL_SENTENCES,
    TRAIN_SENTENCES,
    DEV_SENTENCES,
    TEST_SENTENCES
)

from data_loader import (
    load_sentences,
    split_dataset,
    save_split_information
)

from ngram_model import NGramModel

from evaluation import (
    evaluate_model,
    print_sample_probability
)


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    start_time = time.time()

    # =====================================================
    # HEADER
    # =====================================================

    print()
    print("=" * 70)
    print("ASSIGNMENT 4")
    print("N-GRAM LANGUAGE MODELS")
    print("=" * 70)

    # =====================================================
    # CHECK INPUT FILE
    # =====================================================

    if not INPUT_FILE.exists():

        print()
        print("ERROR!")
        print("-" * 70)

        print(
            "Tokenized input file was not found:"
        )

        print(
            INPUT_FILE
        )

        print()
        print(
            "Make sure indic_tokenized.txt "
            "is inside the Assignment_4 folder."
        )

        return

    # =====================================================
    # LOAD DATA
    # =====================================================

    sentences = load_sentences(
        INPUT_FILE
    )

    # =====================================================
    # SPLIT DATA
    # =====================================================

    train, dev, test = (
        split_dataset(
            sentences
        )
    )

    # Save split information
    save_split_information(
        train,
        dev,
        test,
        SPLIT_FILE
    )

    # =====================================================
    # CREATE MODEL
    # =====================================================

    model = NGramModel(
        DATABASE_FILE
    )

    # =====================================================
    # TRAIN
    # =====================================================

    model.train(
        train
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    results = {}

    print()
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    model_names = {
        1: "Unigram",
        2: "Bigram",
        3: "Trigram",
        4: "Quadrigram"
    }

    for n in range(1, 5):

        model_name = model_names[n]

        print()
        print("-" * 70)

        print(
            f"{model_name} MODEL"
        )

        print("-" * 70)

        # Calculate perplexity
        dev_perplexity, test_perplexity = (
            evaluate_model(
                model,
                dev,
                test,
                n
            )
        )

        results[n] = {
            "name": model_name,
            "dev": dev_perplexity,
            "test": test_perplexity
        }

        print(
            f"Development Perplexity : "
            f"{dev_perplexity:.4f}"
        )

        print(
            f"Test Perplexity        : "
            f"{test_perplexity:.4f}"
        )

        # -------------------------------------------------
        # MOST FREQUENT N-GRAMS
        # -------------------------------------------------

        print()
        print(
            "Top 5 n-grams:"
        )

        samples = (
            model.show_sample_ngrams(
                n,
                limit=5
            )
        )

        for key, count in samples:

            words = key.split(
                "\t"
            )

            print(
                f"{' '.join(words)} "
                f"-> {count}"
            )

    # =====================================================
    # SAMPLE TEST SENTENCE
    # =====================================================

    if test:

        sample_sentence = test[0]

        print()
        print("=" * 70)
        print("SAMPLE TEST SENTENCE")
        print("=" * 70)

        print(
            " ".join(sample_sentence)
        )

        print()
        print(
            "Sentence probabilities:"
        )

        for n in range(1, 5):

            print_sample_probability(
                model,
                sample_sentence,
                n
            )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ASSIGNMENT 4 - "
            "N-GRAM LANGUAGE MODELS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        # Dataset information
        file.write(
            "DATASET INFORMATION\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Total sentences : "
            f"{TOTAL_SENTENCES}\n"
        )

        file.write(
            f"Training        : "
            f"{TRAIN_SENTENCES}\n"
        )

        file.write(
            f"Development     : "
            f"{DEV_SENTENCES}\n"
        )

        file.write(
            f"Testing         : "
            f"{TEST_SENTENCES}\n"
        )

        file.write(
            f"Vocabulary size : "
            f"{model.vocabulary_size}\n"
        )

        file.write(
            f"Training tokens : "
            f"{model.total_tokens}\n"
        )

        file.write("\n")

        # Smoothing formula
        file.write(
            "LAPLACE / ADD-ONE SMOOTHING\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            "P(w_n | w_1,...,w_(n-1)) = "
            "(C(w_1,...,w_n) + 1) / "
            "(C(w_1,...,w_(n-1)) + V)\n"
        )

        file.write("\n")

        # Model results
        file.write(
            "MODEL RESULTS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for n in range(1, 5):

            result = results[n]

            file.write(
                f"\n{result['name']} Model\n"
            )

            file.write(
                f"Development Perplexity : "
                f"{result['dev']:.4f}\n"
            )

            file.write(
                f"Test Perplexity        : "
                f"{result['test']:.4f}\n"
            )

    # =====================================================
    # CLOSE MODEL
    # =====================================================

    model.close()

    # =====================================================
    # EXECUTION TIME
    # =====================================================

    elapsed_time = (
        time.time() -
        start_time
    )

    print()
    print("=" * 70)
    print("ASSIGNMENT COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Results saved to:"
    )

    print(
        RESULT_FILE
    )

    print()
    print(
        f"Dataset split saved to:"
    )

    print(
        SPLIT_FILE
    )

    print()
    print(
        f"Database saved to:"
    )

    print(
        DATABASE_FILE
    )

    print()
    print(
        f"Execution time: "
        f"{elapsed_time / 60:.2f} minutes"
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":

    main()