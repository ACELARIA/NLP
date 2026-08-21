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


# ------------------------------------------------------------
# Read Brown noun corpus
# ------------------------------------------------------------

INPUT_FILE = "brown_nouns.txt"
OUTPUT_FILE = "brown_fst_output.txt"


with open(INPUT_FILE, "r", encoding="utf-8") as file:
    words = {
        line.strip().lower()
        for line in file
        if line.strip()
    }


print("Number of nouns in corpus:", len(words))


# ------------------------------------------------------------
# FST
# ------------------------------------------------------------

class NounFST:

    def __init__(self, lexicon):

        self.lexicon = lexicon

        # FST states
        self.states = {
            "q0",          # start
            "qROOT",       # reading root
            "qNORMAL",     # normal ending
            "qES",         # s, z, x, ch, sh ending
            "qY",          # y ending
            "qSG",         # singular
            "qPL",         # plural
            "qINVALID"     # invalid
        }

        self.start_state = "q0"

    # --------------------------------------------------------
    # Determine root class
    # --------------------------------------------------------

    def root_class(self, root):

        if root.endswith(("s", "z", "x", "ch", "sh")):
            return "ES"

        if root.endswith("y"):
            return "Y"

        return "NORMAL"

    # --------------------------------------------------------
    # Analyze one noun
    # --------------------------------------------------------

    def analyze(self, word):

        word = word.lower()

        # Ignore empty strings
        if not word:
            return "Invalid Word"

        # ----------------------------------------------------
        # q0 -> qROOT
        # ----------------------------------------------------

        state = "q0"

        if not word.isalpha():
            return "Invalid Word"

        state = "qROOT"

        # ====================================================
        # RULE 1: Y replacement
        #
        # try -> tries
        # city -> cities
        # ====================================================

        if word.endswith("ies"):

            root = word[:-3] + "y"

            if root in self.lexicon and root.endswith("y"):

                state = "qY"
                state = "qPL"

                return f"{root}+N+PL"

        # ====================================================
        # RULE 2: E insertion
        #
        # fox -> foxes
        # box -> boxes
        # watch -> watches
        # church -> churches
        # ====================================================

        if word.endswith("es"):

            root = word[:-2]

            if root in self.lexicon:

                if root.endswith(("s", "z", "x", "ch", "sh")):

                    state = "qES"
                    state = "qPL"

                    return f"{root}+N+PL"

        # ====================================================
        # RULE 3: S addition
        #
        # bag -> bags
        # dog -> dogs
        # book -> books
        # ====================================================

        if word.endswith("s"):

            root = word[:-1]

            if root in self.lexicon:

                # -s must NOT be used with roots requiring -es
                if root.endswith(("s", "z", "x", "ch", "sh")):
                    return "Invalid Word"

                # -y nouns use -ies
                if root.endswith("y"):
                    return "Invalid Word"

                state = "qNORMAL"
                state = "qPL"

                return f"{root}+N+PL"

        # ====================================================
        # RULE 4: Singular
        #
        # If the word itself is in the noun lexicon,
        # it can be analyzed as a singular noun.
        # ====================================================

        if word in self.lexicon:

            state = "qSG"

            return f"{word}+N+SG"

        # ====================================================
        # No valid transition
        # ====================================================

        state = "qINVALID"

        return "Invalid Word"


# ------------------------------------------------------------
# Create FST
# ------------------------------------------------------------

fst = NounFST(words)


# ------------------------------------------------------------
# Process complete Brown corpus
# ------------------------------------------------------------

results = []

for word in sorted(words):

    analysis = fst.analyze(word)

    results.append(
        f"{word} = {analysis}"
    )


# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    for result in results:
        file.write(result + "\n")


# ------------------------------------------------------------
# Display first 50 results
# ------------------------------------------------------------

print("\nFirst 50 FST results:")
print("----------------------------------------")

for result in results[:50]:
    print(result)


print("\n----------------------------------------")
print("Processing complete.")
print("Output saved to:", OUTPUT_FILE)