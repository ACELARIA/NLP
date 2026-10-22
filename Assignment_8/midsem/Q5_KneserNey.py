from collections import Counter, defaultdict

sentences = [
    "I like NLP",
    "I like AI",
    "You like NLP",
    "You love AI",
    "We love NLP"
]

d = 0.5

# Create bigrams without adding start/end markers.
bigram_counts = Counter()
history_counts = Counter()
continuations = defaultdict(set)
distinct_bigrams = set()

for sentence in sentences:
    words = sentence.split()

    for i in range(len(words) - 1):
        h = words[i]
        w = words[i + 1]

        bigram_counts[(h, w)] += 1
        history_counts[h] += 1
        continuations[h].add(w)
        distinct_bigrams.add((h, w))

# Count distinct predecessor words for each target.
predecessors = defaultdict(set)

for h, w in distinct_bigrams:
    predecessors[w].add(h)

total_distinct_bigrams = len(distinct_bigrams)

def p_cont(word):
    return len(predecessors[word]) / total_distinct_bigrams

def p_kn(word, history):
    count_bigram = bigram_counts[(history, word)]
    count_history = history_counts[history]

    # Number of distinct words following the history.
    distinct_followers = len(continuations[history])

    # Interpolation weight.
    lam = d * distinct_followers / count_history

    # Discounted maximum-likelihood component.
    first_term = max(count_bigram - d, 0) / count_history

    # Continuation probability component.
    second_term = lam * p_cont(word)

    print(f"\nP({word} | {history})")
    print(f"Bigram count = {count_bigram}")
    print(f"History count = {count_history}")
    print(f"Distinct followers = {distinct_followers}")
    print(f"P_cont({word}) = {p_cont(word):.4f}")
    print(f"Lambda({history}) = {lam:.4f}")
    print(f"Discounted term = {first_term:.4f}")
    print(f"Interpolation term = {second_term:.4f}")

    return first_term + second_term

print("Distinct bigram types:", total_distinct_bigrams)

p1 = p_kn("AI", "love")
p2 = p_kn("NLP", "like")

print("\nFinal answers:")
print(f"P_KN(AI | love) = {p1:.4f}")
print(f"P_KN(NLP | like) = {p2:.4f}")
