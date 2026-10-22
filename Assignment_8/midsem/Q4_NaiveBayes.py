from fractions import Fraction

# Rows contain free, win, offer, class.
data = [
    (1, 1, 1, "Spam"),
    (1, 1, 0, "Spam"),
    (1, 0, 1, "Spam"),
    (0, 1, 1, "Spam"),
    (0, 0, 1, "NotSpam"),
    (0, 1, 0, "NotSpam"),
    (1, 0, 0, "NotSpam")
]

features = ["free", "win", "offer"]
classes = ["Spam", "NotSpam"]

total = len(data)

for class_name in classes:
    class_rows = [row for row in data if row[3] == class_name]
    class_count = len(class_rows)

    prior = Fraction(class_count, total)
    score = prior

    print(f"\nClass: {class_name}")
    print(f"Prior P({class_name}) = {prior}")

    # E8 has features (0, 0, 0).
    for i, feature in enumerate(features):
        count_zero = sum(row[i] == 0 for row in class_rows)
        probability = Fraction(count_zero, class_count)

        print(
            f"P({feature}=0 | {class_name}) "
            f"= {count_zero}/{class_count} = {probability}"
        )

        score *= probability

    print(f"Unnormalized score = {score} = {float(score):.6f}")

    if class_name == "Spam":
        spam_score = score
    else:
        notspam_score = score

print("\nFinal prediction:")

if spam_score > notspam_score:
    print("Spam")
else:
    print("NotSpam")
