import sqlite3
import math

from collections import Counter

from config import (
    START_TOKEN,
    END_TOKEN,
    UNKNOWN_TOKEN,
    MAX_N,
    BATCH_SIZE
)


class NGramModel:

    def __init__(self, database_file):

        self.database_file = database_file

        # Connect to SQLite database
        self.connection = sqlite3.connect(
            database_file
        )

        self.cursor = self.connection.cursor()

        # Create tables
        self.create_tables()

        # Vocabulary
        self.vocabulary = set()

        self.vocabulary_size = 0

        # Total number of training tokens
        self.total_tokens = 0
        # Total number of tokens used by the unigram model,
        # including </s>
        self.total_unigram_tokens = 0

    # =====================================================
    # CREATE DATABASE TABLES
    # =====================================================

    def create_tables(self):

        for n in range(1, MAX_N + 1):

            table_name = f"ngram_{n}"

            self.cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    ngram TEXT PRIMARY KEY,
                    count INTEGER NOT NULL
                )
                """
            )

        self.connection.commit()

    # =====================================================
    # CLEAR OLD COUNTS
    # =====================================================

    def clear_database(self):

        for n in range(1, MAX_N + 1):

            self.cursor.execute(
                f"DELETE FROM ngram_{n}"
            )

        self.connection.commit()

    # =====================================================
    # CONVERT N-GRAM TO DATABASE KEY
    # =====================================================

    @staticmethod
    def make_key(tokens):

        return "\t".join(tokens)

    # =====================================================
    # ADD START / END TOKENS
    # =====================================================

    @staticmethod
    def prepare_sentence(sentence, n):

        """
        Example sentence:

        I love NLP

        Bigram:

        <s> I love NLP </s>

        Trigram:

        <s> <s> I love NLP </s>
        """

        tokens = list(sentence)

        start_tokens = [
            START_TOKEN
        ] * (n - 1)

        return (
            start_tokens +
            tokens +
            [END_TOKEN]
        )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    def train(self, train_sentences):

        print()
        print("=" * 70)
        print("BUILDING N-GRAM MODELS")
        print("=" * 70)

        # Remove old counts
        self.clear_database()

        self.vocabulary = set()

        self.total_tokens = 0
        self.total_unigram_tokens = 0

        total_sentences = len(
            train_sentences
        )

        # Process data in batches
        for batch_start in range(
            0,
            total_sentences,
            BATCH_SIZE
        ):

            batch_end = min(
                batch_start + BATCH_SIZE,
                total_sentences
            )

            batch = train_sentences[
                batch_start:batch_end
            ]

            # Counters for this batch
            counters = {
                n: Counter()
                for n in range(1, MAX_N + 1)
            }

            # -------------------------------------------------
            # PROCESS SENTENCES
            # -------------------------------------------------

            for sentence in batch:

                # Add words to vocabulary
                for word in sentence:

                    self.vocabulary.add(
                        word
                    )

                # Original tokens
                self.total_tokens += len(sentence)

                # Unigram model also contains the end-of-sentence token </s>
                self.total_unigram_tokens += len(sentence) + 1

                # Generate n-grams
                for n in range(
                    1,
                    MAX_N + 1
                ):

                    tokens = (
                        self.prepare_sentence(
                            sentence,
                            n
                        )
                    )

                    for i in range(
                        len(tokens) - n + 1
                    ):

                        ngram = tuple(
                            tokens[
                                i:i + n
                            ]
                        )

                        key = self.make_key(
                            ngram
                        )

                        counters[n][key] += 1

            # -------------------------------------------------
            # STORE COUNTS
            # -------------------------------------------------

            for n in range(
                1,
                MAX_N + 1
            ):

                self.insert_counts(
                    n,
                    counters[n]
                )

            print(
                f"Processed "
                f"{batch_end}/{total_sentences} "
                f"training sentences"
            )

        # Add special tokens
        self.vocabulary.add(
            START_TOKEN
        )

        self.vocabulary.add(
            END_TOKEN
        )

        self.vocabulary.add(
            UNKNOWN_TOKEN
        )

        self.vocabulary_size = len(
            self.vocabulary
        )

        self.connection.commit()

        print()
        print("Training completed.")

        print(
            f"Vocabulary size : "
            f"{self.vocabulary_size}"
        )

        print(
            f"Total tokens    : "
            f"{self.total_tokens}"
        )

    # =====================================================
    # INSERT COUNTS INTO DATABASE
    # =====================================================

    def insert_counts(
        self,
        n,
        counts
    ):

        if not counts:
            return

        table_name = f"ngram_{n}"

        data = list(
            counts.items()
        )

        self.cursor.executemany(
            f"""
            INSERT INTO {table_name}
                (ngram, count)

            VALUES (?, ?)

            ON CONFLICT(ngram)
            DO UPDATE SET
                count = count + excluded.count
            """,
            data
        )

        self.connection.commit()

    # =====================================================
    # GET N-GRAM COUNT
    # =====================================================

    def get_count(self, ngram):

        n = len(ngram)

        key = self.make_key(
            ngram
        )

        self.cursor.execute(
            f"""
            SELECT count
            FROM ngram_{n}
            WHERE ngram = ?
            """,
            (key,)
        )

        result = self.cursor.fetchone()

        if result is None:
            return 0

        return result[0]

    # =====================================================
    # CALCULATE ADD-ONE PROBABILITY
    # =====================================================

    def probability(self, ngram):

        """
        Laplace / Add-One Smoothing:

        P(w_n | w_1,...,w_(n-1))

        =
        [C(w_1,...,w_n) + 1]
        ---------------------
        [C(w_1,...,w_(n-1)) + V]
        """

        n = len(ngram)

        # -------------------------------------------------
        # UNIGRAM
        # -------------------------------------------------

        if n == 1:

            count = self.get_count(
                ngram
            )

            probability = (
                count + 1
            ) / (
                self.total_unigram_tokens +
                self.vocabulary_size
            )

            return probability

        # -------------------------------------------------
        # BIGRAM / TRIGRAM / QUADRIGRAM
        # -------------------------------------------------

        ngram_count = self.get_count(
            ngram
        )

        # Prefix
        prefix = ngram[:-1]

        prefix_count = self.get_count(
            prefix
        )

        probability = (
            ngram_count + 1
        ) / (
            prefix_count +
            self.vocabulary_size
        )

        return probability

    # =====================================================
    # NORMALIZE UNKNOWN WORDS
    # =====================================================

    def normalize_word(self, word):

        if word in self.vocabulary:

            return word

        return UNKNOWN_TOKEN

    # =====================================================
    # SENTENCE LOG PROBABILITY
    # =====================================================

    def sentence_log_probability(
        self,
        sentence,
        n
    ):

        # Replace unknown words
        words = [
            self.normalize_word(word)
            for word in sentence
        ]

        # Add start/end tokens
        tokens = self.prepare_sentence(
            words,
            n
        )

        log_probability = 0.0

        number_of_predictions = 0

        # Generate n-grams
        for i in range(
            n - 1,
            len(tokens)
        ):

            ngram = tuple(
                tokens[
                    i - n + 1:
                    i + 1
                ]
            )

            probability = self.probability(
                ngram
            )

            log_probability += math.log(
                probability
            )

            number_of_predictions += 1

        return (
            log_probability,
            number_of_predictions
        )

    # =====================================================
    # SENTENCE PROBABILITY
    # =====================================================

    def sentence_probability(
        self,
        sentence,
        n
    ):

        log_probability, _ = (
            self.sentence_log_probability(
                sentence,
                n
            )
        )

        return math.exp(
            log_probability
        )

    # =====================================================
    # GET MOST FREQUENT N-GRAMS
    # =====================================================

    def show_sample_ngrams(
        self,
        n,
        limit=10
    ):

        table_name = f"ngram_{n}"

        self.cursor.execute(
            f"""
            SELECT ngram, count
            FROM {table_name}
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,)
        )

        return self.cursor.fetchall()

    # =====================================================
    # CLOSE DATABASE
    # =====================================================

    def close(self):

        self.connection.close()