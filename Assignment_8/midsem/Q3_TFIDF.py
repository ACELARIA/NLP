import math
from collections import Counter
from itertools import combinations

sentences = [
    "cat dog animal",
    "cat cat pet",
    "dog pet animal",
    "cat dog pet"
]

# Tokenize sentences.
documents = [sentence.split() for sentence in sentences]

# Create vocabulary.
vocabulary = sorted(set(
    word for document in documents for word in document
))

N = len(documents)

# Calculate document frequency.
df = {
    word: sum(word in document for document in documents)
    for word in vocabulary
}

# Calculate unsmoothed IDF.
idf = {
    word: math.log(N / df[word])
    for word in vocabulary
}

print("Vocabulary:", vocabulary)
print("\nDocument frequency and IDF:")

for word in vocabulary:
    print(f"{word:8} df={df[word]} IDF={idf[word]:.4f}")

# Calculate TF-IDF vectors using raw term counts.
vectors = []

for document in documents:
    counts = Counter(document)
    vector = [
        counts[word] * idf[word]
        for word in vocabulary
    ]
    vectors.append(vector)

print("\nTF-IDF vectors:")

for i, vector in enumerate(vectors, start=1):
    print(f"S{i}: {[round(x, 4) for x in vector]}")

# Cosine similarity.
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)

print("\nPairwise cosine similarities:")

best_pair = None
best_similarity = -1

for i, j in combinations(range(N), 2):
    similarity = cosine_similarity(vectors[i], vectors[j])

    print(f"S{i+1} vs S{j+1}: {similarity:.4f}")

    if similarity > best_similarity:
        best_similarity = similarity
        best_pair = (i + 1, j + 1)

print("\nClosest sentences:", best_pair)
print(f"Maximum similarity: {best_similarity:.4f}")
