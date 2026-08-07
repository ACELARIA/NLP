"""
statistics.py

Computes corpus statistics for the NLP Assignment.
"""

from collections import Counter


class CorpusStatistics:

    def __init__(self):

        self.total_sentences = 0
        self.total_words = 0
        self.total_characters = 0

        self.vocabulary = Counter()

    # -------------------------------------
    # Add one tokenized sentence
    # -------------------------------------

    def add_sentence(self, tokens):

        if not tokens:
            return

        self.total_sentences += 1

        self.total_words += len(tokens)

        for token in tokens:

            self.total_characters += len(token)

            self.vocabulary[token] += 1

    # -------------------------------------
    # Compute statistics
    # -------------------------------------

    def get_statistics(self):

        unique_words = len(self.vocabulary)

        average_sentence_length = (
            self.total_words / self.total_sentences
            if self.total_sentences > 0
            else 0
        )

        average_word_length = (
            self.total_characters / self.total_words
            if self.total_words > 0
            else 0
        )

        ttr = (
            unique_words / self.total_words
            if self.total_words > 0
            else 0
        )

        return {
            "Total Sentences": self.total_sentences,
            "Total Words": self.total_words,
            "Total Characters": self.total_characters,
            "Average Sentence Length": round(
                average_sentence_length,
                2
            ),
            "Average Word Length": round(
                average_word_length,
                2
            ),
            "Unique Tokens": unique_words,
            "Type Token Ratio": round(
                ttr,
                4
            ),
        }

    # -------------------------------------
    # Print statistics
    # -------------------------------------

    def print_statistics(self):

        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("CORPUS STATISTICS")
        print("=" * 60)

        for key, value in stats.items():

            print(f"{key:<30}: {value}")

        print("=" * 60)

    # -------------------------------------
    # Save statistics to file
    # -------------------------------------

    def save_statistics(self, filename):

        stats = self.get_statistics()

        with open(filename, "w", encoding="utf-8") as file:

            file.write("Corpus Statistics\n")
            file.write("=" * 40 + "\n\n")

            for key, value in stats.items():

                file.write(f"{key}: {value}\n")


# ---------------------------------------------------
# Test
# ---------------------------------------------------

if __name__ == "__main__":

    stats = CorpusStatistics()

    stats.add_sentence(
        ["Hello", "world", "!"]
    )

    stats.add_sentence(
        ["This", "is", "NLP", "."]
    )

    stats.print_statistics()

    stats.save_statistics("sample_statistics.txt")