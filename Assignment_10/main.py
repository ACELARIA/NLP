"""
Assignment 10: Pre-trained Word Embeddings, Word Analogies and Neural Networks
Python 3.14 compatible; does NOT require Gensim.

Expected folder layout:
Assignment_10/
  main.py
  train.csv
  test.csv
  embeddings/
    word2vec-google-news-300.model.vectors.npy
    word2vec-vocabulary.txt
    word2vec-google-news-300.model   (kept for reference; not loaded)
  Glove Pre-Trained Word Embeddings/
    glove.6B.300d.txt
  wiki-news-300d-1M-subword.vec

Install dependencies:
python3 -m pip install numpy pandas scikit-learn tqdm matplotlib --break-system-packages

Notes:
- Text embeddings are loaded only for words occurring in the sampled reviews and analogy
  questions, keeping memory use manageable.
- Analogy evaluation uses the same candidate vocabulary for all three embeddings:
  words appearing in the analogy benchmark plus the two required custom examples.
  This keeps evaluation fast and makes the candidate scope comparable across models.
  The reported accuracy is candidate-restricted, not full-vocabulary Google analogy accuracy.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
EMBED_DIR = BASE_DIR / "embeddings"
RESULT_DIR = BASE_DIR / "results"

TRAIN_FILE = BASE_DIR / "train.csv"
TEST_FILE = BASE_DIR / "test.csv"

WORD2VEC_VECTORS = EMBED_DIR / "word2vec-google-news-300.model.vectors.npy"
WORD2VEC_VOCAB = EMBED_DIR / "word2vec-vocabulary.txt"

GLOVE_DIR = BASE_DIR / "Glove Pre-Trained Word Embeddings"
GLOVE_FILE = GLOVE_DIR / "glove.6B.300d.txt"
FASTTEXT_FILE = BASE_DIR / "wiki-news-300d-1M-subword.vec"

TRAIN_SIZE = 10000
TEST_SIZE = 1000
RANDOM_STATE = 42
EMBED_DIM = 300

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# EMBEDDING INTERFACE
# ============================================================

class SelectedTextVectors:
    """Stores only requested vectors from a large text embedding file."""

    def __init__(self, vectors, vector_size=300):
        self.vectors = vectors
        self.vector_size = vector_size

    def __contains__(self, word):
        return word in self.vectors

    def __getitem__(self, word):
        return self.vectors[word]

    def __len__(self):
        return len(self.vectors)

    def prepare_analogy_candidates(self, candidate_words):
        """Cache a compact matrix used for every analogy query."""
        words = [w for w in candidate_words if w in self.vectors]
        self._analogy_words = words
        self._analogy_matrix = (
            np.asarray([self.vectors[w] for w in words], dtype=np.float32)
            if words else np.empty((0, self.vector_size), dtype=np.float32)
        )
        self._analogy_norms = (
            np.linalg.norm(self._analogy_matrix, axis=1)
            if words else np.empty(0, dtype=np.float32)
        )

    def most_similar_vector(self, query, topn=1, exclude=None):
        """Search the cached, benchmark-wide candidate vocabulary."""
        q = np.asarray(query, dtype=np.float32)
        qnorm = float(np.linalg.norm(q))
        if qnorm == 0 or not hasattr(self, "_analogy_matrix"):
            return []
        scores = (self._analogy_matrix @ q) / np.maximum(
            self._analogy_norms * qnorm, 1e-12
        )
        excluded = set(exclude or [])
        if excluded:
            for i, word in enumerate(self._analogy_words):
                if word in excluded:
                    scores[i] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        if not len(valid):
            return []
        k = min(topn, len(valid))
        idx = valid[np.argpartition(scores[valid], -k)[-k:]]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(self._analogy_words[i], float(scores[i])) for i in idx]


class CustomWord2Vec:
    """Google News vectors backed by NumPy mmap and an extracted vocabulary."""

    def __init__(self, vectors_path, vocab_path):
        print("Opening Word2Vec vector matrix with memory mapping...")
        self.vectors = np.load(vectors_path, mmap_mode="r")
        self.vector_size = int(self.vectors.shape[1])
        self.word_to_index = {}
        print("Reading Word2Vec vocabulary...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            for index, word in enumerate(f):
                self.word_to_index[word.rstrip("\n")] = index
        if len(self.word_to_index) != self.vectors.shape[0]:
            raise ValueError(
                f"Vocabulary/vector mismatch: {len(self.word_to_index)} words, "
                f"{self.vectors.shape[0]} vectors"
            )
        print(f"Word2Vec ready: {len(self.word_to_index):,} words, "
              f"{self.vector_size} dimensions")

    def __contains__(self, word):
        return word in self.word_to_index

    def __getitem__(self, word):
        return self.vectors[self.word_to_index[word]]

    def __len__(self):
        return len(self.word_to_index)

    def prepare_analogy_candidates(self, candidate_words):
        """Cache only analogy-benchmark vectors, not all 3 million vectors."""
        words = [w for w in candidate_words if w in self.word_to_index]
        indices = np.asarray(
            [self.word_to_index[w] for w in words], dtype=np.int64
        )
        self._analogy_words = words
        self._analogy_matrix = np.asarray(self.vectors[indices], dtype=np.float32)
        self._analogy_norms = np.linalg.norm(self._analogy_matrix, axis=1)

    def most_similar_vector(self, query, topn=1, exclude=None):
        """Fast cosine search over the cached analogy candidate vocabulary."""
        q = np.asarray(query, dtype=np.float32)
        qnorm = float(np.linalg.norm(q))
        if qnorm == 0 or not hasattr(self, "_analogy_matrix"):
            return []
        scores = (self._analogy_matrix @ q) / np.maximum(
            self._analogy_norms * qnorm, 1e-12
        )
        excluded = set(exclude or [])
        if excluded:
            for i, word in enumerate(self._analogy_words):
                if word in excluded:
                    scores[i] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        if not len(valid):
            return []
        k = min(topn, len(valid))
        idx = valid[np.argpartition(scores[valid], -k)[-k:]]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(self._analogy_words[i], float(scores[i])) for i in idx]

    @property
    def index_to_word(self):
        # Created lazily; list is ~3 million string references.
        if not hasattr(self, "_index_to_word"):
            self._index_to_word = [None] * len(self.word_to_index)
            for word, index in self.word_to_index.items():
                self._index_to_word[index] = word
        return self._index_to_word


def load_selected_text_vectors(path, wanted_words, expected_dim=300,
                               has_header=None, label="embeddings"):
    """
    Stream a GloVe/FastText text file and retain only requested words.
    This avoids keeping the full 1M+ vocabulary in RAM.
    """
    wanted = set(wanted_words)
    found = {}
    print(f"Scanning {label}: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        parts = first.rstrip().split()

        # FastText .vec usually starts with: vocabulary_size dimension.
        first_is_header = (
            len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
        )
        if has_header is None:
            has_header = first_is_header

        if not has_header:
            f.seek(0)

        for line in tqdm(f, desc=f"Reading {label}"):
            pieces = line.rstrip().split()
            if len(pieces) != expected_dim + 1:
                continue
            word = pieces[0]
            if word not in wanted or word in found:
                continue
            try:
                found[word] = np.asarray(pieces[1:], dtype=np.float32)
            except ValueError:
                continue

    print(f"{label}: found {len(found):,} / {len(wanted):,} requested words")
    return SelectedTextVectors(found, expected_dim)


# ============================================================
# Q1: DATA LOADING AND PREPROCESSING
# ============================================================

print("\n========== Q1: DATA PREPROCESSING ==========")


def load_dataset(filepath):
    # Amazon Reviews Polarity CSV is normally headerless:
    # label, title, review. Also accept files with headers.
    try:
        df = pd.read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath, header=None)

    normalized = [str(c).strip().lower() for c in df.columns]
    header_names = {"label", "class", "polarity", "sentiment"}
    if any(c in header_names for c in normalized):
        df.columns = normalized
    else:
        df = pd.read_csv(
            filepath, header=None,
            names=["label", "title", "review"]
        )

    review_col = next(
        (c for c in ("review", "text", "description") if c in df.columns),
        None
    )
    label_col = next(
        (c for c in ("label", "class", "polarity", "sentiment")
         if c in df.columns),
        None
    )
    if review_col is None or label_col is None:
        raise ValueError(f"Could not identify label/review columns: {list(df.columns)}")

    result = df[[label_col, review_col]].copy()
    result.columns = ["label", "review"]
    result = result.dropna()
    result["review"] = result["review"].astype(str)
    return result


train_df = load_dataset(TRAIN_FILE)
test_df = load_dataset(TEST_FILE)

print("Original training dataset:", train_df.shape)
print("Original testing dataset:", test_df.shape)

if len(train_df) < TRAIN_SIZE or len(test_df) < TEST_SIZE:
    raise ValueError("The CSV files contain fewer rows than requested sample sizes.")

train_df, _ = train_test_split(
    train_df, train_size=TRAIN_SIZE, stratify=train_df["label"],
    random_state=RANDOM_STATE
)
test_df, _ = train_test_split(
    test_df, train_size=TEST_SIZE, stratify=test_df["label"],
    random_state=RANDOM_STATE
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))
print("\nTraining class distribution:")
print(train_df["label"].value_counts(normalize=True))
print("\nTesting class distribution:")
print(test_df["label"].value_counts(normalize=True))

stop_words = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if w not in stop_words]


print("\nPreprocessing reviews...")
train_tokens = [preprocess_text(x) for x in tqdm(train_df["review"], desc="Train")]
test_tokens = [preprocess_text(x) for x in tqdm(test_df["review"], desc="Test")]

train_df["tokens"] = [" ".join(x) for x in train_tokens]
test_df["tokens"] = [" ".join(x) for x in test_tokens]
train_df.to_csv(RESULT_DIR / "processed_train.csv", index=False)
test_df.to_csv(RESULT_DIR / "processed_test.csv", index=False)

# Load analogy questions before embedding loading so their words are included.
ANALOGY_URL = (
    "https://raw.githubusercontent.com/nicholas-leonard/"
    "word2vec/master/questions-words.txt"
)
ANALOGY_FILE = BASE_DIR / "questions-words.txt"


def load_analogy_questions(filepath):
    questions = []
    category = "unknown"
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                category = line[1:].strip()
                continue
            words = line.lower().split()
            if len(words) == 4:
                questions.append((category, *words))
    return questions


if not ANALOGY_FILE.exists():
    import urllib.request
    print("\nDownloading analogy dataset...")
    urllib.request.urlretrieve(ANALOGY_URL, ANALOGY_FILE)

analogy_questions = load_analogy_questions(ANALOGY_FILE)
print("\nAnalogy questions loaded:", len(analogy_questions))

# The selected vocabulary is all words present in sampled reviews plus analogy words.
wanted_words = set()
for tokens in train_tokens + test_tokens:
    wanted_words.update(tokens)
for _, a, b, c, d in analogy_questions:
    wanted_words.update((a, b, c, d))
wanted_words.update({
    "bad", "good", "excellent",
    "late", "delivery", "fast"
})
# Common capitalization variants help with case-sensitive pretrained vocabularies.
wanted_words |= {w.capitalize() for w in list(wanted_words)}

# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\n========== LOADING EMBEDDINGS ==========")

if not WORD2VEC_VECTORS.exists() or not WORD2VEC_VOCAB.exists():
    raise FileNotFoundError(
        "Word2Vec files missing. Expected the .npy vector matrix and "
        "word2vec-vocabulary.txt inside embeddings/."
    )

word2vec = CustomWord2Vec(WORD2VEC_VECTORS, WORD2VEC_VOCAB)

if not GLOVE_FILE.exists():
    candidates = list(GLOVE_DIR.glob("glove.6B.300d.txt"))
    if candidates:
        GLOVE_FILE = candidates[0]
    else:
        raise FileNotFoundError(
            f"GloVe file not found at {GLOVE_FILE}. Check the folder and filename."
        )

if not FASTTEXT_FILE.exists():
    raise FileNotFoundError(f"FastText file not found: {FASTTEXT_FILE}")

glove = load_selected_text_vectors(
    GLOVE_FILE, wanted_words, expected_dim=EMBED_DIM,
    has_header=False, label="GloVe"
)
fasttext = load_selected_text_vectors(
    FASTTEXT_FILE, wanted_words, expected_dim=EMBED_DIM,
    has_header=True, label="FastText"
)

models = {
    "Word2Vec": word2vec,
    "GloVe": glove,
    "FastText": fasttext
}
def resolve_word(model, word):
    for candidate in (word, word.lower(), word.capitalize(), word.upper()):
        if candidate in model:
            return candidate
    return None

# ============================================================
# PREPARE A FAIR, FAST ANALOGY CANDIDATE VOCABULARY
# ============================================================

# Candidate answers are restricted to words occurring in the benchmark.
# This is intentionally identical in scope for Word2Vec, GloVe and FastText.
custom_analogy_words = {"bad", "good", "excellent", "late", "delivery", "fast"}
analogy_candidate_source = set(custom_analogy_words)
for _, a, b, c, d in analogy_questions:
    analogy_candidate_source.update((a, b, c, d))

for model_name, model in models.items():
    resolved_candidates = []
    seen_candidates = set()

    for word in sorted(analogy_candidate_source):
        resolved = resolve_word(model, word)

        if resolved is not None and resolved not in seen_candidates:
            seen_candidates.add(resolved)
            resolved_candidates.append(resolved)

    model.prepare_analogy_candidates(resolved_candidates)

    print(
        f"{model_name} analogy candidates cached: "
        f"{len(resolved_candidates):,}"
    )
# ============================================================
# Q1: SENTENCE REPRESENTATIONS
# ============================================================

print("\n========== Q1: SENTENCE REPRESENTATIONS ==========")


def average_sentence_vector(tokens, model):
    vectors = [model[w] for w in tokens if w in model]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(np.asarray(vectors, dtype=np.float32), axis=0)


vectorizer = TfidfVectorizer(
    tokenizer=str.split,
    preprocessor=None,
    token_pattern=None,
    lowercase=False,
    max_features=100000
)
vectorizer.fit(train_df["tokens"])
feature_names = vectorizer.get_feature_names_out()
feature_to_index = {w: i for i, w in enumerate(feature_names)}
print("TF-IDF vocabulary size:", len(feature_names))


def tfidf_sentence_vectors(tokens_list, model, fitted_vectorizer):
    matrix = fitted_vectorizer.transform([" ".join(x) for x in tokens_list])
    output = np.zeros((len(tokens_list), model.vector_size), dtype=np.float32)

    for i, tokens in enumerate(tqdm(tokens_list, desc="TF-IDF vectors")):
        row = matrix.getrow(i)
        weights = dict(zip(row.indices, row.data))
        total_weight = 0.0
        accum = np.zeros(model.vector_size, dtype=np.float32)

        for word in set(tokens):
            if word not in model:
                continue
            col = feature_to_index.get(word)
            if col is None:
                continue
            weight = weights.get(col, 0.0)
            if weight:
                accum += np.asarray(model[word], dtype=np.float32) * weight
                total_weight += weight

        if total_weight:
            output[i] = accum / total_weight

    return output


representations = {}
for model_name, model in models.items():
    print(f"\nGenerating representations: {model_name}")

    X_train_avg = np.vstack([
        average_sentence_vector(tokens, model) for tokens in tqdm(train_tokens)
    ]).astype(np.float32)
    X_test_avg = np.vstack([
        average_sentence_vector(tokens, model) for tokens in tqdm(test_tokens)
    ]).astype(np.float32)

    X_train_tfidf = tfidf_sentence_vectors(train_tokens, model, vectorizer)
    X_test_tfidf = tfidf_sentence_vectors(test_tokens, model, vectorizer)

    representations[model_name] = {
        "average": (X_train_avg, X_test_avg),
        "tfidf": (X_train_tfidf, X_test_tfidf)
    }

    np.savez_compressed(
        RESULT_DIR / f"{model_name}_representations.npz",
        X_train_avg=X_train_avg, X_test_avg=X_test_avg,
        X_train_tfidf=X_train_tfidf, X_test_tfidf=X_test_tfidf
    )
    print("Average shape:", X_train_avg.shape)
    print("TF-IDF shape:", X_train_tfidf.shape)

print("\nQ1 completed.")

# ============================================================
# Q2: WORD ANALOGY EVALUATION
# ============================================================

print("\n========== Q2: WORD ANALOGY ==========")

def predict_analogy(model, a, b, c, topn=1):
    """
    Standard analogy: a is to b as c is to ?.
    Query vector = b - a + c.
    """
    wa, wb, wc = (resolve_word(model, x) for x in (a, b, c))
    if None in (wa, wb, wc):
        return None

    query = (
        np.asarray(model[wb], dtype=np.float32)
        - np.asarray(model[wa], dtype=np.float32)
        + np.asarray(model[wc], dtype=np.float32)
    )
    results = model.most_similar_vector(
        query, topn=topn, exclude=[wa, wb, wc]
    )
    return results[0][0] if results else None


def evaluate_analogies(model, questions, model_name):
    correct = 0
    evaluated = 0
    skipped = 0
    category = {}

    for cat, a, b, c, expected in tqdm(
        questions, desc=f"{model_name} analogies"
    ):
        target = resolve_word(model, expected)
        prediction = predict_analogy(model, a, b, c)

        if target is None or prediction is None:
            skipped += 1
            continue

        hit = prediction.lower() == target.lower()
        evaluated += 1
        correct += int(hit)
        stats = category.setdefault(cat, [0, 0])
        stats[0] += int(hit)
        stats[1] += 1

    accuracy = 100.0 * correct / evaluated if evaluated else 0.0
    return accuracy, correct, evaluated, skipped, category


analogy_results = []
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    accuracy, correct, total, skipped, category = evaluate_analogies(
        model, analogy_questions, name
    )
    print(f"Correct: {correct}; evaluated: {total}; skipped: {skipped}")
    print(f"Accuracy: {accuracy:.2f}%")
    analogy_results.append({
        "Embedding": name,
        "Correct": correct,
        "Total Evaluated": total,
        "Skipped": skipped,
        "Accuracy (%)": accuracy,
        "Candidate Scope": "Analogy benchmark vocabulary (restricted)"
    })
    category_df = pd.DataFrame([
        {"Category": cat, "Correct": vals[0], "Total": vals[1],
         "Accuracy (%)": 100 * vals[0] / vals[1]}
        for cat, vals in category.items()
    ])
    category_df.to_csv(RESULT_DIR / f"{name}_analogy_categories.csv", index=False)

analogy_df = pd.DataFrame(analogy_results)
analogy_df.to_csv(RESULT_DIR / "analogy_comparison.csv", index=False)
print("\nAnalogy comparison:")
print(analogy_df.to_string(index=False))

print("\n========== REQUIRED CUSTOM ANALOGIES ==========")
for name, model in models.items():
    result1 = predict_analogy(model, "bad", "good", "excellent")
    result2 = predict_analogy(model, "late", "delivery", "fast")
    print(f"\n{name}")
    print("good - bad + excellent =", result1)
    print("delivery - late + fast =", result2)

# ============================================================
# Q3: SIX SINGLE-HIDDEN-LAYER NEURAL NETWORKS
# ============================================================

print("\n========== Q3: NEURAL NETWORK CLASSIFICATION ==========")

y_train = train_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()


def create_neural_network():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=40,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=RANDOM_STATE,
            batch_size=64
        )
    )


nn_results = []
for embedding_name, methods in representations.items():
    for method_name, (X_train, X_test) in methods.items():
        print("\n----------------------------------")
        print("Embedding:", embedding_name)
        print("Representation:", method_name)
        print("----------------------------------")

        classifier = create_neural_network()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")

        nn_results.append({
            "Embedding": embedding_name,
            "Representation": method_name,
            "Accuracy": accuracy,
            "Macro F1-score": macro_f1
        })

nn_df = pd.DataFrame(nn_results)
nn_df.to_csv(RESULT_DIR / "neural_network_comparison.csv", index=False)

print("\n========== FINAL NEURAL NETWORK RESULTS ==========")
print(nn_df.sort_values("Macro F1-score", ascending=False).to_string(index=False))

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========== ASSIGNMENT COMPLETED ==========")
print("\nQ2: Analogy comparison")
print(analogy_df.to_string(index=False))
print("\nQ3: Neural-network comparison")
print(nn_df.to_string(index=False))
print("\nResults saved in:", RESULT_DIR)
for output_file in sorted(RESULT_DIR.iterdir()):
    if output_file.is_file():
        print("-", output_file.name)
