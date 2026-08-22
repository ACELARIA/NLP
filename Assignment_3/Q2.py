# ============================================================
# FST Morphological Analyzer for Brown Noun Corpus
#
# Rules:
#   1. S addition:
#        bag -> bags
#
#   2. E insertion:
#        fox -> foxes
#        watch -> watches
#
#   3. Y replacement:
#        try -> tries
#
# Output:
#   fox      = fox+N+SG
#   foxes    = fox+N+PL
#   bags     = bag+N+PL
#
# Invalid:
#   foxs     = Invalid Word
# ============================================================
INPUT_FILE = "brown_nouns.txt"
OUTPUT_FILE = "brown_fst_output.txt"


# Load noun lexicon
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    words = {line.strip().lower() for line in f if line.strip()}

print("Number of nouns in corpus:", len(words))


class NounFST:

    def __init__(self, lexicon):
        self.lexicon = lexicon

    def analyze(self, word):
        word = word.lower()

        if not word or not word.isalpha():
            return "Invalid Word"

        # Rule 1: y -> ies
        if word.endswith("ies"):
            root = word[:-3] + "y"
            if root in self.lexicon:
                return f"{root}+N+PL"

        # Rule 2: add es
        if word.endswith("es"):
            root = word[:-2]
            if root in self.lexicon and root.endswith(
                ("s", "z", "x", "ch", "sh")
            ):
                return f"{root}+N+PL"

        # Rule 3: add s
        if word.endswith("s"):
            root = word[:-1]

            if root in self.lexicon:
                if root.endswith(("s", "z", "x", "ch", "sh", "y")):
                    return "Invalid Word"

                return f"{root}+N+PL"

        # Singular
        if word in self.lexicon:
            return f"{word}+N+SG"

        return "Invalid Word"


# Create FST
fst = NounFST(words)

# Analyze all words
results = [
    f"{word} = {fst.analyze(word)}"
    for word in sorted(words)
]

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results) + "\n")


# Display first 50
print("\nFirst 50 FST results:")
print("----------------------------------------")

for result in results[:50]:
    print(result)

print("\n----------------------------------------")
print("Processing complete.")
print("Output saved to:", OUTPUT_FILE)