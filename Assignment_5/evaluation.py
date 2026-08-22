"""Evaluation functions for Assignment 5."""

import math


def calculate_perplexity(model, sentences):
    """Calculate perplexity using log probabilities.

    PP = exp(-sum(log P(sentence)) / number_of_predicted_tokens)
    """
    total_log_probability = 0.0
    total_predicted_tokens = 0

    for sentence in sentences:
        log_probability, token_count, _ = model.sentence_log_probability(sentence)
        total_log_probability += log_probability
        total_predicted_tokens += token_count

    if total_predicted_tokens == 0:
        return float("inf")

    exponent = -total_log_probability / total_predicted_tokens

    # Avoid math.exp overflow for pathological inputs.
    if exponent > 709:
        return float("inf")

    return math.exp(exponent)


def evaluate_model(model, dev_sentences, test_sentences):
    dev_perplexity = calculate_perplexity(model, dev_sentences)
    test_perplexity = calculate_perplexity(model, test_sentences)
    return dev_perplexity, test_perplexity


def format_top_ngrams(model, limit=5):
    return [
        f"{' '.join(ngram)} -> {count}"
        for ngram, count in model.top_ngrams(limit)
    ]
