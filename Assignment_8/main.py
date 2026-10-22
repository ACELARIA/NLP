
import os
import re
import math
from collections import defaultdict

# ============================================================
# EXPERIMENT: SPELL CORRECTION USING THE NOISY CHANNEL MODEL
# ============================================================

FOLDER = os.path.dirname(os.path.abspath(__file__))

ERROR_FILE = os.path.join(FOLDER, "spell-errors.txt")
EDIT_FILE = os.path.join(FOLDER, "count_1edit.txt")
UNIGRAM_FILE = os.path.join(FOLDER, "count_unigram.txt")


# ------------------------------------------------------------
# 1. Read the spelling-error file
# Format example:
# raining: rainning, raning
# ------------------------------------------------------------
def load_spelling_errors(filename):
    errors = defaultdict(set)

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                continue

            correct, wrong_words = line.split(":", 1)
            correct = correct.strip().lower()

            wrong_words = re.split(r"[,;\s]+", wrong_words.strip())

            for wrong in wrong_words:
                wrong = wrong.strip().lower()
                if wrong and wrong != correct:
                    errors[correct].add(wrong)

    return errors


# ------------------------------------------------------------
# 2. Read unigram word frequencies
# Supports formats: word count OR count word
# ------------------------------------------------------------
def load_unigrams(filename):
    frequencies = {}

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            first, second = parts[0], parts[1]

            if first.isdigit():
                word, count = second.lower(), int(first)
            elif second.isdigit():
                word, count = first.lower(), int(second)
            else:
                continue

            frequencies[word] = frequencies.get(word, 0) + count

    return frequencies


# ------------------------------------------------------------
# 3. Read one-edit-distance candidate information
# Supports word: candidate1, candidate2
# or a file listing individual words.
# ------------------------------------------------------------
def load_edit_file(filename):
    candidates = defaultdict(set)
    vocabulary = set()

    with open(filename, "r", encoding="latin-1") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if ":" in line:
                word, values = line.split(":", 1)
                word = word.strip().lower()
                vocabulary.add(word)

                for candidate in re.split(r"[,;\s]+", values.strip()):
                    candidate = candidate.strip().lower()
                    if candidate:
                        candidates[word].add(candidate)
                        vocabulary.add(candidate)
            else:
                parts = line.split()

                # If the file is a word list, retain its vocabulary.
                # Ignore a numeric count after the word.
                vocabulary.add(parts[0].lower())

                if len(parts) >= 2 and not parts[1].isdigit():
                    for candidate in parts[1:]:
                        candidates[parts[0].lower()].add(
                            candidate.lower()
                        )

    return candidates, vocabulary


# ------------------------------------------------------------
# 4. Generate all words one edit away
# Deletion, insertion, replacement and transposition
# ------------------------------------------------------------
def edits_one(word):
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word))]
    results = set()

    for left, right in splits:
        if right:
            # Deletion
            results.add(left + right[1:])

        if len(right) > 1:
            # Transposition
            results.add(left + right[1] + right[0] + right[2:])

        if right:
            # Replacement
            for letter in letters:
                results.add(left + letter + right[1:])

        # Insertion
        for letter in letters:
            results.add(left + letter + right)

    return results


# ------------------------------------------------------------
# 5. Noisy Channel Model
# Score = log P(typo | correct) + log P(correct)
# ------------------------------------------------------------
def correct_word(wrong, spelling_errors, frequencies,
                 edit_candidates, edit_vocabulary):

    wrong = wrong.lower()

    # Words that could be correct according to the supplied files
    possible = set(edit_candidates.get(wrong, set()))

    # Reverse the known error dictionary:
    # correct -> wrong words
    for correct, mistakes in spelling_errors.items():
        if wrong in mistakes:
            possible.add(correct)

    # Add candidates one edit away from the misspelling
    known_words = set(frequencies) | set(spelling_errors)
    possible.update(edits_one(wrong) & known_words)
    possible.update(edits_one(wrong) & edit_vocabulary)

    # If no candidate is found, preserve the input
    if not possible:
        return wrong

    total_frequency = sum(frequencies.values()) or 1
    total_errors = sum(len(v) for v in spelling_errors.values()) or 1

    best_word = wrong
    best_score = -math.inf

    for candidate in possible:
        if not candidate:
            continue

        # Language model probability P(correct word)
        word_count = frequencies.get(candidate, 0)
        p_word = (word_count + 1) / (
            total_frequency + len(frequencies) + 1
        )

        # Error channel probability P(wrong | correct)
        mistakes = spelling_errors.get(candidate, set())

        if wrong == candidate:
            error_count = 1
        elif wrong in mistakes:
            error_count = 1
        else:
            error_count = 0

        # Smoothing prevents zero probabilities
        p_error = (error_count + 1e-6) / (
            len(mistakes) + total_errors + 1
        )

        score = math.log(p_word) + math.log(p_error)

        if score > best_score:
            best_score = score
            best_word = candidate

    return best_word


# ------------------------------------------------------------
# 6. Evaluate accuracy using the known spelling-error pairs
# ------------------------------------------------------------
def main():
    spelling_errors = load_spelling_errors(ERROR_FILE)
    frequencies = load_unigrams(UNIGRAM_FILE)
    edit_candidates, edit_vocabulary = load_edit_file(EDIT_FILE)

    print("=" * 65)
    print("SPELL CORRECTION USING THE NOISY CHANNEL MODEL")
    print("=" * 65)

    print("\nDataset statistics")
    print("-" * 65)
    print("Correct words:", len(spelling_errors))
    print("Unigram entries:", len(frequencies))
    print("One-edit vocabulary entries:", len(edit_vocabulary))

    total = 0
    correct_predictions = 0

    print("\nSPELL CORRECTION RESULTS")
    print("-" * 65)
    print(f"{'Incorrect word':<20}{'Expected word':<20}{'Prediction':<20}")
    print("-" * 65)

    for expected, mistakes in spelling_errors.items():
        for wrong in sorted(mistakes):
            prediction = correct_word(
                wrong,
                spelling_errors,
                frequencies,
                edit_candidates,
                edit_vocabulary
            )

            total += 1
            if prediction == expected:
                correct_predictions += 1

            print(f"{wrong:<20}{expected:<20}{prediction:<20}")

    accuracy = (
        100 * correct_predictions / total if total else 0
    )

    print("-" * 65)
    print("Total spelling-error cases:", total)
    print("Correct predictions:", correct_predictions)
    print("Incorrect predictions:", total - correct_predictions)
    print(f"Spell checker accuracy: {accuracy:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
