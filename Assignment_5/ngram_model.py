"""Unigram, Bigram, Trigram and Quadrigram models with Add-K smoothing.

For an n-gram w_1 ... w_n:

    P(w_n | w_1 ... w_(n-1))
      = (C(w_1 ... w_n) + K)
        / (C(w_1 ... w_(n-1)) + K * V)

For a unigram:

    P(w) = (C(w) + K) / (N + K * V)

The model predicts normal words plus the end-of-sentence token </s>.
The start token <s> is used only as context for n > 1.
"""

import math
from collections import Counter

from config import (
    ADD_K,
    END_TOKEN,
    PROGRESS_EVERY,
    START_TOKEN,
    UNK_TOKEN,
)


class NGramModel:
    def __init__(self, n, k=ADD_K):
        if n not in (1, 2, 3, 4):
            raise ValueError("n must be 1, 2, 3, or 4")
        if k <= 0:
            raise ValueError("Add-K value must be greater than 0")

        self.n = n
        self.k = float(k)

        # ngram -> count
        self.ngram_counts = Counter()
        # (n-1)-gram context -> count
        self.context_counts = Counter()

        # V contains tokens that may be predicted. <s> is deliberately
        # excluded because it is never predicted.
        self.vocabulary = set()

        self.total_word_tokens = 0
        self.total_predicted_tokens = 0

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    @property
    def total_tokens(self):
        """Compatibility name used in earlier Assignment 4 code."""
        return self.total_predicted_tokens

    def _normalise_for_evaluation(self, tokens):
        """Map words unseen during training to <UNK>."""
        return [token if token in self.vocabulary else UNK_TOKEN for token in tokens]

    def train(self, sentences, show_progress=True):
        """Train this n-gram model from the training sentences."""
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.total_word_tokens = 0
        self.total_predicted_tokens = 0

        # Build V from training words, plus </s> and <UNK>.
        # <s> is a context marker and is not part of V.
        word_vocabulary = set()
        for sentence in sentences:
            word_vocabulary.update(sentence)

        self.vocabulary = word_vocabulary
        self.vocabulary.add(END_TOKEN)
        self.vocabulary.add(UNK_TOKEN)

        total = len(sentences)

        for index, sentence in enumerate(sentences, start=1):
            tokens = list(sentence)
            self.total_word_tokens += len(tokens)

            if self.n == 1:
                # Every word and </s> is a predicted token.
                sequence = tokens + [END_TOKEN]
                for token in sequence:
                    self.ngram_counts[(token,)] += 1
                self.total_predicted_tokens += len(sequence)
            else:
                # <s> is context only; </s> is predicted.
                sequence = [START_TOKEN] * (self.n - 1) + tokens + [END_TOKEN]

                for i in range(self.n - 1, len(sequence)):
                    ngram = tuple(sequence[i - self.n + 1 : i + 1])
                    context = ngram[:-1]
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1
                    self.total_predicted_tokens += 1

            if show_progress and index % PROGRESS_EVERY == 0:
                print(f"Processed {index}/{total} training sentences")

        return self

    def get_count(self, ngram):
        return self.ngram_counts.get(tuple(ngram), 0)

    def get_context_count(self, context):
        return self.context_counts.get(tuple(context), 0)

    def probability(self, ngram):
        """Return the Add-K-smoothed probability of one n-gram."""
        ngram = tuple(ngram)

        if len(ngram) != self.n:
            raise ValueError(
                f"Expected {self.n}-gram, received {len(ngram)}-gram."
            )

        if self.n == 1:
            numerator = self.get_count(ngram) + self.k
            denominator = self.total_predicted_tokens + self.k * self.vocabulary_size
        else:
            context = ngram[:-1]
            numerator = self.get_count(ngram) + self.k
            denominator = self.get_context_count(context) + self.k * self.vocabulary_size

        return numerator / denominator

    def sentence_log_probability(self, sentence):
        """Return (log_probability, predicted_token_count, probability).

        The 3-value return is intentional and is used consistently by
        evaluation.py. This fixes the earlier unpacking errors.
        """
        tokens = self._normalise_for_evaluation(sentence)

        if self.n == 1:
            sequence = tokens + [END_TOKEN]
            log_probability = 0.0
            token_count = 0

            for token in sequence:
                p = self.probability((token,))
                log_probability += math.log(p)
                token_count += 1

        else:
            sequence = [START_TOKEN] * (self.n - 1) + tokens + [END_TOKEN]
            log_probability = 0.0
            token_count = 0

            for i in range(self.n - 1, len(sequence)):
                ngram = tuple(sequence[i - self.n + 1 : i + 1])
                p = self.probability(ngram)
                log_probability += math.log(p)
                token_count += 1

        # For normal assignment-sized sentences this is safe. If the
        # probability underflows, returning 0 is mathematically consistent.
        probability = math.exp(log_probability) if log_probability > -745 else 0.0

        return log_probability, token_count, probability

    def sentence_probability(self, sentence):
        """Return ordinary sentence probability."""
        return self.sentence_log_probability(sentence)[2]

    def top_ngrams(self, limit=5):
        return self.ngram_counts.most_common(limit)
