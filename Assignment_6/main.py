# ============================================================
# ASSIGNMENT 6 - FINAL CORRECTED + OPTIMIZED
# N-GRAM SMOOTHING
# ============================================================

import math
import time
from collections import Counter, defaultdict
from functools import lru_cache


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "indic_tokenized.txt"

TRAIN_SIZE = 98000
DEV_SIZE = 1000
TEST_SIZE = 1000

MAX_ORDER = 4

# Good-Turing is applied only to low-frequency counts.
GT_CUTOFF = 5

# Stupid Backoff parameter
STUPID_ALPHA = 0.4

# Kneser-Ney discount
KN_DISCOUNT = 0.75

EPSILON = 1e-12


# Linear interpolation weights
INTERPOLATION_WEIGHTS = {
    2: [0.4, 0.6],
    3: [0.2, 0.3, 0.5],
    4: [0.1, 0.2, 0.3, 0.4]
}


# ============================================================
# LOAD DATA
# ============================================================

def load_sentences(filename):

    sentences = []

    with open(filename, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                sentences.append(line.split())

    return sentences


# ============================================================
# BUILD VOCABULARY
# ============================================================

def prepare_vocabulary(training_sentences):

    word_counts = Counter()

    for sentence in training_sentences:
        word_counts.update(sentence)

    # Words occurring only once are replaced by <UNK>.
    vocabulary = {
        word
        for word, count in word_counts.items()
        if count >= 2
    }

    vocabulary.add("<UNK>")

    return vocabulary


def convert_unknown_words(sentences, vocabulary):

    return [
        [
            word if word in vocabulary else "<UNK>"
            for word in sentence
        ]
        for sentence in sentences
    ]


# ============================================================
# N-GRAM MODEL
# ============================================================

class NGramModel:

    def __init__(self, training_sentences, vocabulary):

        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)

        # N-gram frequency counts
        self.counts = {
            n: Counter()
            for n in range(1, MAX_ORDER + 1)
        }

        # Context counts
        self.context_counts = {
            n: Counter()
            for n in range(2, MAX_ORDER + 1)
        }

        # followers[n][context][word] = count
        self.followers = {
            n: defaultdict(dict)
            for n in range(2, MAX_ORDER + 1)
        }

        self.total_unigram_tokens = 0

        self.build_counts()

        self.prepare_good_turing()

        self.prepare_katz()

        self.prepare_kneser_ney()


    # ========================================================
    # BUILD N-GRAM COUNTS
    # ========================================================

    def build_counts(self):

        print()
        print("Building N-gram counts...")

        for number, sentence in enumerate(
            self.training_sentences,
            1
        ):

            pass


    # ========================================================
    # ACTUAL COUNT BUILDING
    # ========================================================

    def build_counts(self):

        print()
        print("Building N-gram counts...")

        for number, sentence in enumerate(
            self._training_sentences,
            1
        ):

            tokens = ["<s>"] + sentence + ["</s>"]

            # Unigrams
            for word in tokens:
                self.counts[1][(word,)] += 1

            self.total_unigram_tokens += len(tokens)

            # Higher order n-grams
            for n in range(2, MAX_ORDER + 1):

                for i in range(len(tokens) - n + 1):

                    ngram = tuple(tokens[i:i + n])

                    context = ngram[:-1]
                    word = ngram[-1]

                    self.counts[n][ngram] += 1

                    self.context_counts[n][context] += 1

        # Build fast follower dictionaries
        for n in range(2, MAX_ORDER + 1):

            for ngram, count in self.counts[n].items():

                context = ngram[:-1]
                word = ngram[-1]

                self.followers[n][context][word] = count


    # ========================================================
    # GOOD-TURING
    # ========================================================

    def prepare_good_turing(self):

        print("Preparing Good-Turing statistics...")

        self.freq_of_freq = {}

        for n in range(1, MAX_ORDER + 1):

            self.freq_of_freq[n] = Counter(
                self.counts[n].values()
            )

        # -----------------------------------------------
        # Good-Turing discounts
        # -----------------------------------------------

        self.gt_discount = {
            n: {}
            for n in range(1, MAX_ORDER + 1)
        }

        for n in range(1, MAX_ORDER + 1):

            freq = self.freq_of_freq[n]

            for r in range(1, GT_CUTOFF + 1):

                nr = freq.get(r, 0)
                nr1 = freq.get(r + 1, 0)

                if nr > 0 and nr1 > 0:

                    adjusted = (
                        (r + 1) * nr1 / nr
                    )

                    # Never make a count larger than
                    # the original count.
                    adjusted = min(
                        adjusted,
                        float(r)
                    )

                    discount = adjusted / r

                else:

                    # If N(r+1) does not exist,
                    # retain the original count.
                    discount = 1.0

                self.gt_discount[n][r] = max(
                    min(discount, 1.0),
                    EPSILON
                )

        # -----------------------------------------------
        # Good-Turing unigram probabilities
        # -----------------------------------------------

        adjusted_counts = {}

        for ngram, count in self.counts[1].items():

            if count <= GT_CUTOFF:

                discount = self.gt_discount[1].get(
                    count,
                    1.0
                )

                adjusted = count * discount

            else:

                adjusted = float(count)

            adjusted_counts[ngram] = adjusted

        total = sum(adjusted_counts.values())

        self.gt_unigram_probability = {
            ngram: max(
                count / total,
                EPSILON
            )
            for ngram, count
            in adjusted_counts.items()
        }

        # -----------------------------------------------
        # Good-Turing conditional probabilities
        # -----------------------------------------------

        self.gt_seen_probability = {
            n: {}
            for n in range(2, MAX_ORDER + 1)
        }

        self.gt_unseen_probability = {
            n: {}
            for n in range(2, MAX_ORDER + 1)
        }

        for n in range(2, MAX_ORDER + 1):

            for context, followers in (
                self.followers[n].items()
            ):

                context_total = (
                    self.context_counts[n][context]
                )

                adjusted = {}

                for word, count in followers.items():

                    if count <= GT_CUTOFF:

                        discount = (
                            self.gt_discount[n].get(
                                count,
                                1.0
                            )
                        )

                        adjusted[word] = (
                            count * discount
                        )

                    else:

                        adjusted[word] = float(count)

                adjusted_total = sum(
                    adjusted.values()
                )

                # Number of singleton types
                singleton_types = sum(
                    1
                    for count in followers.values()
                    if count == 1
                )

                unseen_types = max(
                    self.vocab_size - len(followers),
                    0
                )

                # Estimate unseen probability mass.
                if unseen_types > 0:

                    unseen_mass = (
                        singleton_types
                        / context_total
                    )

                    # Prevent unstable values.
                    unseen_mass = min(
                        unseen_mass,
                        0.25
                    )

                else:

                    unseen_mass = 0.0

                observed_mass = (
                    1.0 - unseen_mass
                )

                # Normalize observed events.
                if adjusted_total > 0:

                    scale = (
                        observed_mass
                        / adjusted_total
                    )

                else:

                    scale = 1.0

                for word, value in adjusted.items():

                    ngram = context + (word,)

                    self.gt_seen_probability[n][
                        ngram
                    ] = max(
                        value * scale,
                        EPSILON
                    )

                if unseen_types > 0:

                    self.gt_unseen_probability[n][
                        context
                    ] = max(
                        unseen_mass / unseen_types,
                        EPSILON
                    )

                else:

                    self.gt_unseen_probability[n][
                        context
                    ] = EPSILON


    # ========================================================
    # MAXIMUM LIKELIHOOD PROBABILITY
    # ========================================================

    def mle_probability(self, ngram):

        n = len(ngram)

        count = self.counts[n].get(
            ngram,
            0
        )

        if n == 1:

            if count == 0:
                return 0.0

            return (
                count
                / self.total_unigram_tokens
            )

        context_count = (
            self.context_counts[n].get(
                ngram[:-1],
                0
            )
        )

        if context_count == 0:
            return 0.0

        return count / context_count


    # ========================================================
    # INTERPOLATED SMOOTHING
    # ========================================================

    @lru_cache(maxsize=500000)
    def interpolated_probability(self, ngram):

        n = len(ngram)

        if n == 1:

            p = self.mle_probability(ngram)

            if p == 0:

                p = self.gt_unigram_probability.get(
                    ngram,
                    EPSILON
                )

            return max(p, EPSILON)

        weights = INTERPOLATION_WEIGHTS[n]

        probability = 0.0

        for order, weight in enumerate(
            weights,
            1
        ):

            suffix = ngram[-order:]

            p = self.mle_probability(suffix)

            if p == 0:

                if order == 1:

                    p = self.gt_unigram_probability.get(
                        suffix,
                        EPSILON
                    )

                else:

                    p = self.gt_unseen_probability[
                        order
                    ].get(
                        suffix[:-1],
                        EPSILON
                    )

            probability += weight * p

        return max(
            probability,
            EPSILON
        )


    # ========================================================
    # GOOD-TURING PROBABILITY
    # ========================================================

    @lru_cache(maxsize=500000)
    def good_turing_probability(self, ngram):

        n = len(ngram)

        # Unigram
        if n == 1:

            return max(
                self.gt_unigram_probability.get(
                    ngram,
                    EPSILON
                ),
                EPSILON
            )

        # Seen n-gram
        if ngram in self.gt_seen_probability[n]:

            return max(
                self.gt_seen_probability[n][ngram],
                EPSILON
            )

        context = ngram[:-1]

        # Unseen n-gram with known context
        if context in self.gt_unseen_probability[n]:

            return max(
                self.gt_unseen_probability[n][context],
                EPSILON
            )

        # Unknown context -> lower order
        return max(
            self.good_turing_probability(
                ngram[1:]
            ),
            EPSILON
        )


    # ========================================================
    # KATZ BACKOFF
    # ========================================================

    def prepare_katz(self):

        print("Preparing Katz Backoff statistics...")

        self.katz_seen_probability = {
            n: {}
            for n in range(2, MAX_ORDER + 1)
        }

        self.katz_alpha = {
            n: {}
            for n in range(2, MAX_ORDER + 1)
        }

        # -----------------------------------------------
        # Process lower orders first.
        # -----------------------------------------------

        for n in range(2, MAX_ORDER + 1):

            for context, followers in (
                self.followers[n].items()
            ):

                context_total = (
                    self.context_counts[n][context]
                )

                seen_mass = 0.0

                # ---------------------------------------
                # Discounted seen probabilities
                # ---------------------------------------

                for word, count in followers.items():

                    if count <= GT_CUTOFF:

                        discount = (
                            self.gt_discount[n].get(
                                count,
                                1.0
                            )
                        )

                    else:

                        discount = 1.0

                    p = (
                        count
                        * discount
                        / context_total
                    )

                    p = max(
                        p,
                        EPSILON
                    )

                    ngram = context + (word,)

                    self.katz_seen_probability[n][
                        ngram
                    ] = p

                    seen_mass += p

                # ---------------------------------------
                # Lower-order probability mass
                # ---------------------------------------

                lower_seen_mass = 0.0

                for word in followers:

                    lower_ngram = (
                        context[1:]
                        + (word,)
                    )

                    if n == 2:

                        lower_p = (
                            self.gt_unigram_probability.get(
                                (word,),
                                EPSILON
                            )
                        )

                    else:

                        lower_p = (
                            self.katz_probability(
                                lower_ngram
                            )
                        )

                    lower_seen_mass += lower_p

                remaining_mass = max(
                    1.0 - seen_mass,
                    EPSILON
                )

                lower_remaining_mass = max(
                    1.0 - lower_seen_mass,
                    EPSILON
                )

                alpha = (
                    remaining_mass
                    / lower_remaining_mass
                )

                # Keep alpha numerically stable.
                self.katz_alpha[n][
                    context
                ] = min(
                    max(alpha, EPSILON),
                    1.0
                )


    # ========================================================
    # KATZ PROBABILITY
    # ========================================================

    @lru_cache(maxsize=500000)
    def katz_probability(self, ngram):

        n = len(ngram)

        if n == 1:

            return max(
                self.gt_unigram_probability.get(
                    ngram,
                    EPSILON
                ),
                EPSILON
            )

        # Seen event
        if ngram in self.katz_seen_probability[n]:

            return max(
                self.katz_seen_probability[n][ngram],
                EPSILON
            )

        context = ngram[:-1]

        alpha = self.katz_alpha[n].get(
            context,
            1.0
        )

        lower_probability = (
            self.katz_probability(
                ngram[1:]
            )
        )

        return max(
            alpha * lower_probability,
            EPSILON
        )


    # ========================================================
    # STUPID BACKOFF
    # ========================================================

    @lru_cache(maxsize=500000)
    def stupid_backoff_score(self, ngram):

        n = len(ngram)

        if n == 1:

            p = self.mle_probability(ngram)

            if p == 0:

                p = self.gt_unigram_probability.get(
                    ngram,
                    EPSILON
                )

            return max(p, EPSILON)

        # If n-gram exists, use MLE probability.
        if self.counts[n].get(
            ngram,
            0
        ) > 0:

            return max(
                self.mle_probability(ngram),
                EPSILON
            )

        # Otherwise back off.
        return max(
            STUPID_ALPHA
            * self.stupid_backoff_score(
                ngram[1:]
            ),
            EPSILON
        )


    # ========================================================
    # KNESER-NEY
    # ========================================================

    def prepare_kneser_ney(self):

        print("Preparing Kneser-Ney statistics...")

        # -----------------------------------------------
        # Unique predecessor count of each word
        # -----------------------------------------------

        predecessors = defaultdict(set)

        for bigram in self.counts[2]:

            previous_word = bigram[0]
            word = bigram[1]

            predecessors[word].add(
                previous_word
            )

        self.continuation_count = {
            word: len(values)
            for word, values
            in predecessors.items()
        }

        self.total_unique_bigrams = len(
            self.counts[2]
        )

        # -----------------------------------------------
        # Lambda values
        # -----------------------------------------------

        self.kn_lambda = {
            n: {}
            for n in range(2, MAX_ORDER + 1)
        }

        for n in range(2, MAX_ORDER + 1):

            for context, followers in (
                self.followers[n].items()
            ):

                context_count = (
                    self.context_counts[n][context]
                )

                if context_count > 0:

                    self.kn_lambda[n][
                        context
                    ] = (
                        KN_DISCOUNT
                        * len(followers)
                        / context_count
                    )

                else:

                    self.kn_lambda[n][
                        context
                    ] = 1.0


    @lru_cache(maxsize=500000)
    def kneser_ney_probability(self, ngram):

        n = len(ngram)

        # -----------------------------------------------
        # Continuation probability
        # -----------------------------------------------

        if n == 1:

            word = ngram[0]

            count = self.continuation_count.get(
                word,
                0
            )

            if count > 0:

                return max(
                    count
                    / self.total_unique_bigrams,
                    EPSILON
                )

            return max(
                1.0 / self.vocab_size,
                EPSILON
            )

        context = ngram[:-1]

        context_count = (
            self.context_counts[n].get(
                context,
                0
            )
        )

        # Unknown context -> lower order
        if context_count == 0:

            return max(
                self.kneser_ney_probability(
                    ngram[1:]
                ),
                EPSILON
            )

        count = self.counts[n].get(
            ngram,
            0
        )

        # -----------------------------------------------
        # Discounted observed component
        # -----------------------------------------------

        first_component = (
            max(
                count - KN_DISCOUNT,
                0
            )
            / context_count
        )

        # -----------------------------------------------
        # Backoff component
        # -----------------------------------------------

        lambda_value = self.kn_lambda[n][
            context
        ]

        lower_probability = (
            self.kneser_ney_probability(
                ngram[1:]
            )
        )

        probability = (
            first_component
            + lambda_value
            * lower_probability
        )

        return max(
            probability,
            EPSILON
        )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(
    model,
    sentences,
    method,
    order
):

    if method == "Interpolated":

        probability_function = (
            model.interpolated_probability
        )

    elif method == "Good-Turing":

        probability_function = (
            model.good_turing_probability
        )

    elif method == "Katz Backoff":

        probability_function = (
            model.katz_probability
        )

    elif method == "Stupid Backoff":

        probability_function = (
            model.stupid_backoff_score
        )

    elif method == "Kneser-Ney":

        probability_function = (
            model.kneser_ney_probability
        )

    else:

        raise ValueError(
            "Unknown smoothing technique"
        )

    log_probability = 0.0
    token_count = 0
    failed_probability = 0

    for sentence in sentences:

        tokens = (
            ["<s>"]
            + sentence
            + ["</s>"]
        )

        for i in range(
            order - 1,
            len(tokens)
        ):

            ngram = tuple(
                tokens[
                    i - order + 1:
                    i + 1
                ]
            )

            probability = (
                probability_function(
                    ngram
                )
            )

            if (
                probability <= 0
                or not math.isfinite(probability)
            ):

                failed_probability += 1

                continue

            log_probability += math.log(
                probability
            )

            token_count += 1

    if token_count == 0:

        return (
            float("inf"),
            failed_probability
        )

    perplexity = math.exp(
        -log_probability / token_count
    )

    return (
        perplexity,
        failed_probability
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 90)

    print(
        "ASSIGNMENT 6 - FINAL CORRECTED + "
        "OPTIMIZED N-GRAM SMOOTHING"
    )

    print("=" * 90)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        f"Loading: {DATA_FILE}"
    )

    all_sentences = load_sentences(
        DATA_FILE
    )

    required = (
        TRAIN_SIZE
        + DEV_SIZE
        + TEST_SIZE
    )

    if len(all_sentences) < required:

        raise ValueError(
            f"Only {len(all_sentences)} sentences found. "
            f"{required} are required."
        )

    # ========================================================
    # SPLIT DATA
    # ========================================================

    training_raw = all_sentences[
        :TRAIN_SIZE
    ]

    development_raw = all_sentences[
        TRAIN_SIZE:
        TRAIN_SIZE + DEV_SIZE
    ]

    testing_raw = all_sentences[
        TRAIN_SIZE + DEV_SIZE:
        required
    ]

    print()
    print("DATASET SPLIT")
    print("-" * 90)

    print(
        f"Total Sentences Loaded : "
        f"{len(all_sentences)}"
    )

    print(
        f"Training Sentences     : "
        f"{len(training_raw)}"
    )

    print(
        f"Development Sentences  : "
        f"{len(development_raw)}"
    )

    print(
        f"Testing Sentences      : "
        f"{len(testing_raw)}"
    )

    # ========================================================
    # UNKNOWN WORD HANDLING
    # ========================================================

    print()
    print("Preparing <UNK> handling...")

    vocabulary = prepare_vocabulary(
        training_raw
    )

    training_sentences = (
        convert_unknown_words(
            training_raw,
            vocabulary
        )
    )

    development_sentences = (
        convert_unknown_words(
            development_raw,
            vocabulary
        )
    )

    testing_sentences = (
        convert_unknown_words(
            testing_raw,
            vocabulary
        )
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model_start = time.time()

    # Store training sentences so build_counts can use them.
    model = NGramModel.__new__(
        NGramModel
    )

    model._training_sentences = (
        training_sentences
    )

    model.vocabulary = vocabulary
    model.vocab_size = len(vocabulary)

    model.counts = {
        n: Counter()
        for n in range(1, MAX_ORDER + 1)
    }

    model.context_counts = {
        n: Counter()
        for n in range(2, MAX_ORDER + 1)
    }

    model.followers = {
        n: defaultdict(dict)
        for n in range(2, MAX_ORDER + 1)
    }

    model.total_unigram_tokens = 0

    model.build_counts()

    model.prepare_good_turing()

    model.prepare_katz()

    model.prepare_kneser_ney()

    model_preparation_time = (
        time.time() - model_start
    )

    # ========================================================
    # TRAINING STATISTICS
    # ========================================================

    print()
    print("TRAINING STATISTICS")
    print("-" * 90)

    for n in range(1, MAX_ORDER + 1):

        print(
            f"{n}-gram Types         : "
            f"{len(model.counts[n])}"
        )

    print(
        f"Vocabulary Size       : "
        f"{model.vocab_size}"
    )

    print(
        f"Model Preparation Time: "
        f"{model_preparation_time:.2f} seconds"
    )

    # ========================================================
    # EXPERIMENTS
    # ========================================================

    experiments = [

        ("Interpolated", 2),
        ("Interpolated", 3),
        ("Interpolated", 4),

        ("Good-Turing", 1),
        ("Good-Turing", 2),
        ("Good-Turing", 3),
        ("Good-Turing", 4),

        ("Katz Backoff", 1),
        ("Katz Backoff", 2),
        ("Katz Backoff", 3),
        ("Katz Backoff", 4),

        ("Stupid Backoff", 2),
        ("Stupid Backoff", 3),
        ("Stupid Backoff", 4),

        ("Kneser-Ney", 2),
        ("Kneser-Ney", 3),
        ("Kneser-Ney", 4)
    ]

    results = []

    print()
    print("=" * 120)
    print("EVALUATION")
    print("=" * 120)

    # ========================================================
    # RUN ALL EXPERIMENTS
    # ========================================================

    for number, (
        method,
        order
    ) in enumerate(
        experiments,
        1
    ):

        print()
        print(
            f"[{number}/{len(experiments)}] "
            f"{method} {order}-gram"
        )

        evaluation_start = time.time()

        development_perplexity, development_failed = (
            evaluate(
                model,
                development_sentences,
                method,
                order
            )
        )

        testing_perplexity, testing_failed = (
            evaluate(
                model,
                testing_sentences,
                method,
                order
            )
        )

        evaluation_time = (
            time.time()
            - evaluation_start
        )

        failed_probability = (
            development_failed
            + testing_failed
        )

        result = {
            "method": method,
            "order": order,
            "development_perplexity":
                development_perplexity,
            "testing_perplexity":
                testing_perplexity,
            "failed_probability":
                failed_probability,
            "evaluation_time":
                evaluation_time
        }

        results.append(result)

        print(
            f"  Development Perplexity : "
            f"{development_perplexity:.4f}"
        )

        print(
            f"  Testing Perplexity      : "
            f"{testing_perplexity:.4f}"
        )

        print(
            f"  Failed Probability      : "
            f"{failed_probability}"
        )

        print(
            f"  Evaluation Time         : "
            f"{evaluation_time:.2f} seconds"
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print()
    print("=" * 120)
    print("FINAL RESULTS")
    print("=" * 120)

    print(
        f"{'Smoothing Technique':<24}"
        f"{'N-gram Model':<16}"
        f"{'Development Perplexity':<26}"
        f"{'Testing Perplexity':<24}"
        f"{'Failed Probability':<20}"
    )

    print("-" * 120)

    for result in results:

        print(
            f"{result['method']:<24}"
            f"{str(result['order']) + '-gram':<16}"
            f"{result['development_perplexity']:<26.4f}"
            f"{result['testing_perplexity']:<24.4f}"
            f"{result['failed_probability']:<20}"
        )

    print("=" * 120)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    total_time = (
        time.time()
        - start_time
    )

    with open(
        "results.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ASSIGNMENT 6 - N-GRAM SMOOTHING RESULTS\n"
        )

        f.write("=" * 120 + "\n")

        f.write(
            f"{'Smoothing Technique':<24}"
            f"{'N-gram Model':<16}"
            f"{'Development Perplexity':<26}"
            f"{'Testing Perplexity':<24}"
            f"{'Failed Probability':<20}\n"
        )

        f.write("-" * 120 + "\n")

        for result in results:

            f.write(
                f"{result['method']:<24}"
                f"{str(result['order']) + '-gram':<16}"
                f"{result['development_perplexity']:<26.4f}"
                f"{result['testing_perplexity']:<24.4f}"
                f"{result['failed_probability']:<20}\n"
            )

        f.write("=" * 120 + "\n")

        f.write(
            f"Total Execution Time: "
            f"{total_time:.2f} seconds\n"
        )

        f.write(
            "\nNOTE:\n"
        )

        f.write(
            "Stupid Backoff is a scoring method rather than "
            "a normalized probability model. Therefore, its "
            "reported values are perplexity-style scores.\n"
        )

    print()
    print(
        "Results saved to: results.txt"
    )

    print(
        f"Total Execution Time: "
        f"{total_time:.2f} seconds"
    )

    print()
    print("DONE.")


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()