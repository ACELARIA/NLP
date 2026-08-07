"""
tokenizer.py

Custom Sentence and Word Tokenizer

Handles:
- URLs
- Email IDs
- Dates
- Decimal Numbers
- Integers
- Punctuation
"""

import re

# -----------------------------
# Regular Expressions
# -----------------------------

URL = r'https?://[^\s]+|www\.[^\s]+'

EMAIL = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'

DATE = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'

DECIMAL = r'\b\d+\.\d+\b'

INTEGER = r'\b\d+\b'

WORD = r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u0900-\u097F]+"

PUNCT = r"[.,!?;:\"'()\[\]{}<>/%&*@#$+=|`~^_-]"

TOKEN_PATTERN = re.compile(
    f"{URL}"
    f"|{EMAIL}"
    f"|{DATE}"
    f"|{DECIMAL}"
    f"|{INTEGER}"
    f"|{WORD}"
    f"|{PUNCT}"
)

# Sentence split symbols
SENTENCE_END = re.compile(r'(?<=[.!?।])\s+')


# -------------------------------------------------
# Sentence Tokenizer
# -------------------------------------------------

def sentence_tokenize(text):

    text = text.strip()

    if not text:
        return []

    sentences = SENTENCE_END.split(text)

    output = []

    for s in sentences:

        s = s.strip()

        if s:
            output.append(s)

    return output


# -------------------------------------------------
# Word Tokenizer
# -------------------------------------------------

def word_tokenize(sentence):

    return TOKEN_PATTERN.findall(sentence)


# -------------------------------------------------
# Tokenize Paragraph
# -------------------------------------------------

def tokenize_paragraph(paragraph):

    tokenized_sentences = []

    sentences = sentence_tokenize(paragraph)

    for sentence in sentences:

        words = word_tokenize(sentence)

        tokenized_sentences.append(words)

    return tokenized_sentences


# -------------------------------------------------
# Test
# -------------------------------------------------

if __name__ == "__main__":

    sample = """
    Hello! Visit https://openai.com today.

    Email me at abc@gmail.com.

    Today is 12/08/2026.

    Price is 123.45 dollars.

    यह एक हिन्दी वाक्य है।
    """

    print("=" * 60)

    print("Sentence Tokenizer")

    print("=" * 60)

    sentences = sentence_tokenize(sample)

    for s in sentences:
        print(s)

    print()

    print("=" * 60)

    print("Word Tokenizer")

    print("=" * 60)

    for s in sentences:

        print(word_tokenize(s))