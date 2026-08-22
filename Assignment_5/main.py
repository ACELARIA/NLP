"""Assignment 5: N-gram Language Models with Add-K smoothing."""

import time
from pathlib import Path

from config import (
    ADD_K,
    INPUT_FILE,
    NGRAM_ORDERS,
    RESULTS_FILE,
    SPLIT_FILE,
)
from data_loader import load_sentences, save_split, split_dataset
from evaluation import evaluate_model, format_top_ngrams
from ngram_model import NGramModel


def model_name(n):
    return {
        1: "Unigram",
        2: "Bigram",
        3: "Trigram",
        4: "Quadrigram",
    }[n]


def main():
    start_time = time.time()

    print("=" * 70)
    print("ASSIGNMENT 5")
    print("N-GRAM LANGUAGE MODELS")
    print("ADD-K SMOOTHING")
    print("=" * 70)
    print(f"Add-K value: K = {ADD_K}")

    # ------------------------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------------------------
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    print(f"Input file: {Path(INPUT_FILE).resolve()}")

    sentences = load_sentences()
    print(f"Sentences loaded: {len(sentences)}")

    # ------------------------------------------------------------
    # 2. SPLIT DATASET
    # ------------------------------------------------------------
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    train, dev, test = split_dataset(sentences)

    print(f"Total sentences      : {len(sentences)}")
    print(f"Training sentences   : {len(train)}")
    print(f"Development sentences: {len(dev)}")
    print(f"Testing sentences    : {len(test)}")

    save_split(train, dev, test, SPLIT_FILE)

    # ------------------------------------------------------------
    # 3. BUILD ALL FOUR MODELS
    # ------------------------------------------------------------
    print("=" * 70)
    print("BUILDING N-GRAM MODELS")
    print("=" * 70)

    models = {}
    for n in NGRAM_ORDERS:
        model = NGramModel(n, ADD_K)
        model.train(train, show_progress=True)
        models[n] = model

    unigram = models[1]

    print()
    print("Training completed.")
    print(f"Vocabulary size : {unigram.vocabulary_size}")
    print(f"Total word tokens : {unigram.total_word_tokens}")
    print(f"Total unigram tokens : {unigram.total_tokens}")

    # ------------------------------------------------------------
    # 4. EVALUATE
    # ------------------------------------------------------------
    print("=" * 70)
    print("ADD-K MODEL EVALUATION")
    print("=" * 70)

    result_lines = [
        "=" * 70,
        "ASSIGNMENT 5",
        "N-GRAM LANGUAGE MODELS",
        "ADD-K SMOOTHING",
        f"Add-K value: K = {ADD_K}",
        "=" * 70,
        "",
        "DATASET SPLIT",
        f"Total sentences      : {len(sentences)}",
        f"Training sentences   : {len(train)}",
        f"Development sentences: {len(dev)}",
        f"Testing sentences    : {len(test)}",
        "",
        "TRAINING STATISTICS",
        f"Vocabulary size : {unigram.vocabulary_size}",
        f"Total word tokens : {unigram.total_word_tokens}",
        f"Total unigram tokens : {unigram.total_tokens}",
        "",
    ]

    for n in NGRAM_ORDERS:
        model = models[n]
        name = model_name(n)

        print()
        print("-" * 70)
        print(f"{name} MODEL")
        print("-" * 70)

        dev_ppl, test_ppl = evaluate_model(model, dev, test)

        print(f"Development Perplexity : {dev_ppl:.4f}")
        print(f"Test Perplexity        : {test_ppl:.4f}")
        print()
        print("Top 5 n-grams:")

        top_lines = format_top_ngrams(model, 5)
        for line in top_lines:
            print(line)

        result_lines.extend([
            "-" * 70,
            f"{name} MODEL",
            "-" * 70,
            f"Development Perplexity : {dev_ppl:.4f}",
            f"Test Perplexity        : {test_ppl:.4f}",
            "",
            "Top 5 n-grams:",
            *top_lines,
            "",
        ])

    # ------------------------------------------------------------
    # 5. SAMPLE SENTENCE
    # ------------------------------------------------------------
    sample_sentence = ["नई", "दिल्ली।"]

    print()
    print("=" * 70)
    print("SAMPLE TEST SENTENCE")
    print("=" * 70)
    print(" ".join(sample_sentence))
    print()
    print("Sentence probabilities:")

    result_lines.extend([
        "=" * 70,
        "SAMPLE TEST SENTENCE",
        "=" * 70,
        " ".join(sample_sentence),
        "",
        "Sentence probabilities:",
    ])

    for n in NGRAM_ORDERS:
        probability = models[n].sentence_probability(sample_sentence)
        line = f"N={n} Sentence Probability: {probability:.12e}"
        print(line)
        result_lines.append(line)

    # ------------------------------------------------------------
    # 6. SAVE RESULTS
    # ------------------------------------------------------------
    elapsed_minutes = (time.time() - start_time) / 60.0

    result_lines.extend([
        "",
        "=" * 70,
        "ASSIGNMENT 5 COMPLETED",
        "=" * 70,
        "",
        f"Execution time: {elapsed_minutes:.2f} minutes",
    ])

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(result_lines) + "\n")

    print()
    print("=" * 70)
    print("ASSIGNMENT 5 COMPLETED")
    print("=" * 70)
    print()
    print("Results saved to:")
    print(Path(RESULTS_FILE).resolve())
    print()
    print("Dataset split saved to:")
    print(Path(SPLIT_FILE).resolve())
    print()
    print(f"Execution time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
