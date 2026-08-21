import math


# =========================================================
# CALCULATE PERPLEXITY
# =========================================================

def calculate_perplexity(
    model,
    sentences,
    n
):

    total_log_probability = 0.0

    total_predictions = 0

    for sentence in sentences:

        log_probability, predictions = (
            model.sentence_log_probability(
                sentence,
                n
            )
        )

        total_log_probability += (
            log_probability
        )

        total_predictions += (
            predictions
        )

    if total_predictions == 0:

        return float("inf")

    # Perplexity:
    #
    # PP = exp(
    #       - total_log_probability /
    #       total_predictions
    #      )

    perplexity = math.exp(
        -total_log_probability /
        total_predictions
    )

    return perplexity


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    model,
    dev_sentences,
    test_sentences,
    n
):

    dev_perplexity = (
        calculate_perplexity(
            model,
            dev_sentences,
            n
        )
    )

    test_perplexity = (
        calculate_perplexity(
            model,
            test_sentences,
            n
        )
    )

    return (
        dev_perplexity,
        test_perplexity
    )


# =========================================================
# PRINT SAMPLE SENTENCE PROBABILITY
# =========================================================

def print_sample_probability(
    model,
    sentence,
    n
):

    probability = (
        model.sentence_probability(
            sentence,
            n
        )
    )

    print(
        f"N={n} "
        f"Sentence Probability: "
        f"{probability:.12e}"
    )