from collections import Counter

corpus = [
    "low", "lower", "low", "lower", "lowest",
    "low", "lowest", "low", "lowest",
    "low", "new",
    "new", "new", "newer", "newer",
    "new"
]

# Represent every word as a list of characters.
words = [list(word) for word in corpus]

for merge_number in range(1, 3):
    pair_counts = Counter()

    # Count adjacent pairs, including repeated words.
    for tokens in words:
        for i in range(len(tokens) - 1):
            pair_counts[(tokens[i], tokens[i + 1])] += 1

    best_pair, frequency = pair_counts.most_common(1)[0]

    print(f"Merge {merge_number}: {best_pair}, count={frequency}")

    # Merge the selected pair throughout the corpus.
    merged_words = []

    for tokens in words:
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == best_pair
            ):
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        merged_words.append(new_tokens)

    words = merged_words

# Tokenize newest using the learned vocabulary.
# Greedy longest-match tokenization is used.
vocab = set(token for word in words for token in word)

def tokenize(word, vocab):
    result = []
    i = 0

    while i < len(word):
        match = None

        # Search for the longest vocabulary token at this position.
        for end in range(len(word), i, -1):
            candidate = word[i:end]
            if candidate in vocab:
                match = candidate
                break

        if match is None:
            return ["[UNK]"]

        result.append(match)
        i += len(match)

    return result

print("Vocabulary:", sorted(vocab))
print("Tokenization of newest:", tokenize("newest", vocab))
