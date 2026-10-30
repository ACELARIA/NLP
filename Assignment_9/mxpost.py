# -*- coding: utf-8 -*-
"""
MEMM POS tagger using NLTK's MaxentClassifier.

This is an adapted, Python-3-compatible implementation based on the
nltk-maxent-pos-tagger project. It retains the NLTK maximum-entropy
classifier but uses the exact features specified in the assignment:

- Prefixes of lengths 1, 2, 3 and 4 (NULL when unavailable)
- Suffixes of lengths 1, 2, 3 and 4 (NULL when unavailable)
- Previous, current and next word
- Binary word-length feature: LESS (<= 4), MORE (> 4)
- Previous predicted POS tag (gold previous tag during training)

The model uses greedy left-to-right decoding, as is standard for an MEMM.
"""

from nltk import TaggerI
from nltk.classify import MaxentClassifier


class MaxentPosTagger(TaggerI):
    """Maximum Entropy Markov Model POS tagger."""

    def __init__(self):
        self.classifier = None
        self.tags = []

    @staticmethod
    def extract_feats(sentence, index, history):
        """Build the assignment-required feature dictionary for one token."""
        word = sentence[index]
        previous_word = sentence[index - 1] if index > 0 else "<BOS>"
        next_word = sentence[index + 1] if index + 1 < len(sentence) else "<EOS>"

        features = {
            "previous_word": previous_word,
            "current_word": word,
            "next_word": next_word,
            "length_feature": "LESS" if len(word) <= 4 else "MORE",
            "previous_tag": history[-1] if history else "<BOS_TAG>",
        }

        for length in range(1, 5):
            features[f"prefix_{length}"] = (
                word[:length] if len(word) >= length else "NULL"
            )
            features[f"suffix_{length}"] = (
                word[-length:] if len(word) >= length else "NULL"
            )

        return features

    def train(
        self,
        train_sents,
        algorithm="iis",
        trace=1,
        max_iter=10,
        **cutoffs,
    ):
        """
        Train NLTK's MaxentClassifier.

        train_sents: list of sentences; each sentence is [(word, tag), ...].
        algorithm: NLTK maximum-entropy algorithm, e.g. 'iis' or 'megam'.
        max_iter: maximum number of optimization iterations.
        """
        if not train_sents:
            raise ValueError("Training data is empty.")

        featuresets = []
        tag_set = set()

        for tagged_sentence in train_sents:
            words = [word for word, _ in tagged_sentence]
            history = []

            for index, (_, gold_tag) in enumerate(tagged_sentence):
                features = self.extract_feats(words, index, history)
                featuresets.append((features, gold_tag))
                history.append(gold_tag)
                tag_set.add(gold_tag)

        self.tags = sorted(tag_set)

        # NLTK's MaxentClassifier is the classifier used by the referenced
        # maximum-entropy tagger. IIS is available without an external binary.
        self.classifier = MaxentClassifier.train(
            featuresets,
            algorithm=algorithm.upper(),
            trace=trace,
            max_iter=max_iter,
            min_lldelta=0.01,
            **cutoffs,
        )
        return self

    def tag(self, sentence):
        """Tag a list of words, using previously predicted tags as history."""
        if self.classifier is None:
            raise ValueError("The MEMM has not been trained.")

        history = []
        tagged_output = []

        for index, word in enumerate(sentence):
            features = self.extract_feats(sentence, index, history)
            predicted_tag = self.classifier.classify(features)
            history.append(predicted_tag)
            tagged_output.append((word, predicted_tag))

        return tagged_output

    def predict(self, sentences):
        """Return predicted tag sequences for [(word, gold_tag), ...] sentences."""
        predictions = []

        for tagged_sentence in sentences:
            words = [word for word, _ in tagged_sentence]
            predictions.append([tag for _, tag in self.tag(words)])

        return predictions
