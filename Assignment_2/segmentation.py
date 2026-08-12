import json
import math


# =========================================================
# 1. LOAD DATASET
# =========================================================

with open("text_segmentation_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

word_counts = data["word_counts"]
test_cases = data["test_cases"]

vocabulary = set(word_counts.keys())

print("Dataset loaded successfully")
print("Vocabulary size:", len(vocabulary))
print("Number of test cases:", len(test_cases))


# =========================================================
# 2. GREEDY LONGEST-MATCH APPROACH
# =========================================================

max_word_length = max(len(word) for word in vocabulary)


def greedy_segment(text):
    result = []
    i = 0

    while i < len(text):

        match = None

        # Try longest word first
        for length in range(
            min(max_word_length, len(text) - i),
            0,
            -1
        ):

            candidate = text[i:i + length]

            if candidate in vocabulary:
                match = candidate
                break

        # If no word is found
        if match is None:
            match = text[i]

        result.append(match)
        i += len(match)

    return result


# =========================================================
# 3. DYNAMIC PROGRAMMING APPROACH
# =========================================================

total_frequency = sum(word_counts.values())

# Log probability of every word
log_probability = {}

for word, frequency in word_counts.items():
    probability = frequency / total_frequency
    log_probability[word] = math.log(probability)


def dp_segment(text):

    n = len(text)

    # dp[i] = best log probability for text[0:i]
    dp = [-float("inf")] * (n + 1)

    # previous[i] = word used to reach position i
    previous = [None] * (n + 1)

    dp[0] = 0

    # Build solution from left to right
    for i in range(1, n + 1):

        for length in range(
            1,
            min(max_word_length, i) + 1
        ):

            word = text[i - length:i]

            if word in log_probability:

                score = (
                    dp[i - length]
                    + log_probability[word]
                )

                if score > dp[i]:
                    dp[i] = score
                    previous[i] = word

    # -----------------------------------------------------
    # Backtracking
    # -----------------------------------------------------

    if previous[n] is None:
        return []

    result = []
    position = n

    while position > 0:

        word = previous[position]

        result.append(word)

        position -= len(word)

    result.reverse()

    return result


# =========================================================
# 4. ACCURACY
# =========================================================

def calculate_accuracy(predicted, actual):

    correct = 0

    for p, a in zip(predicted, actual):

        if p == a:
            correct += 1

    return correct / len(actual)


# =========================================================
# 5. EDIT DISTANCE
# =========================================================

def edit_distance(predicted, actual):

    # Word-level Levenshtein distance

    m = len(predicted)
    n = len(actual)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Empty actual sequence
    for i in range(m + 1):
        dp[i][0] = i

    # Empty predicted sequence
    for j in range(n + 1):
        dp[0][j] = j

    # Calculate distance
    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if predicted[i - 1] == actual[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[m][n]


# =========================================================
# 6. EVALUATION
# =========================================================

greedy_accuracy = []
greedy_edit_distance = []

dp_accuracy = []
dp_edit_distance = []


# Store examples for demonstration
examples = []


for test_case in test_cases:

    text = test_case["input"]

    actual = test_case["ground_truth"].split()


    # -----------------------------------------------------
    # GREEDY
    # -----------------------------------------------------

    greedy_result = greedy_segment(text)

    g_accuracy = calculate_accuracy(
        greedy_result,
        actual
    )

    g_edit = edit_distance(
        greedy_result,
        actual
    )

    greedy_accuracy.append(g_accuracy)
    greedy_edit_distance.append(g_edit)


    # -----------------------------------------------------
    # DYNAMIC PROGRAMMING
    # -----------------------------------------------------

    dp_result = dp_segment(text)

    d_accuracy = calculate_accuracy(
        dp_result,
        actual
    )

    d_edit = edit_distance(
        dp_result,
        actual
    )

    dp_accuracy.append(d_accuracy)
    dp_edit_distance.append(d_edit)


    # Save first few examples
    if len(examples) < 5:
        examples.append(
            (
                text,
                actual,
                greedy_result,
                dp_result
            )
        )


# =========================================================
# 7. FINAL RESULTS
# =========================================================

avg_greedy_accuracy = (
    sum(greedy_accuracy)
    / len(greedy_accuracy)
) * 100

avg_greedy_edit = (
    sum(greedy_edit_distance)
    / len(greedy_edit_distance)
)

avg_dp_accuracy = (
    sum(dp_accuracy)
    / len(dp_accuracy)
) * 100

avg_dp_edit = (
    sum(dp_edit_distance)
    / len(dp_edit_distance)
)


# =========================================================
# 8. DISPLAY RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("TEXT SEGMENTATION RESULTS")
print("=" * 60)

print("\nGreedy Longest-Match Approach")
print("-" * 60)
print(f"Accuracy      : {avg_greedy_accuracy:.2f}%")
print(f"Edit Distance : {avg_greedy_edit:.3f}")

print("\nDynamic Programming Approach")
print("-" * 60)
print(f"Accuracy      : {avg_dp_accuracy:.2f}%")
print(f"Edit Distance : {avg_dp_edit:.3f}")


# =========================================================
# 9. SHOW EXAMPLES
# =========================================================

print("\n")
print("=" * 60)
print("SAMPLE RESULTS")
print("=" * 60)

for text, actual, greedy, dp in examples:

    print("\nInput:")
    print(text)

    print("\nGround Truth:")
    print(" ".join(actual))

    print("\nGreedy:")
    print(" ".join(greedy))

    print("\nDynamic Programming:")
    print(" ".join(dp))

    print("-" * 60)