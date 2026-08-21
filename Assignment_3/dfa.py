def dfa(word):
    state = "q0"

    for ch in word:
        if state == "q0":
            if 'a' <= ch <= 'z':
                state = "q1"
            else:
                return "Not Accepted"

        elif state == "q1":
            if 'a' <= ch <= 'z':
                state = "q1"
            else:
                return "Not Accepted"

    return "Accepted" if state == "q1" else "Not Accepted"


word = input("Enter a word: ")
print(dfa(word))