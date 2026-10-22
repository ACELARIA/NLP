
# Q1_FST.py
# Finite State Transducer for English noun plurals

SPECIAL_ENDINGS = ("s", "x", "z", "ch", "sh")

def fst(noun, number):
    noun = noun.strip().lower()

    if not noun:
        return "Error: noun cannot be empty"

    if number == "SG":
        return f"{noun}+SG+N"

    if number == "PL":
        if noun.endswith(SPECIAL_ENDINGS):
            return f"{noun}es+PL+N"
        return f"{noun}s+PL+N"

    return "Error: number must be SG or PL"


nouns = ["cat", "bus", "box", "buzz", "church", "dish"]

for noun in nouns:
    print(f"{noun}:")
    print("  Singular:", fst(noun, "SG"))
    print("  Plural:  ", fst(noun, "PL"))
