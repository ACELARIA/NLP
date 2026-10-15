"""
Hindi POS Tagger
Algorithms:
1. Greedy Naive Bayes with features: length, word, previous word, next word, previous predicted tag
2. Forward Viterbi HMM
3. Backward Viterbi HMM
4. Forward-Backward HMM (bidirectional contextual tag scoring)

Expected input:
One token per line, with a word and POS tag separated by whitespace or a delimiter.
Blank lines separate sentences.

Examples supported:
राम NP
राम<TAB>NP
राम/NP
Word=राम Tag=NP

Edit TRAIN_FILE, DEV_FILE, TEST_FILE and, if needed, parse_token_line().
"""

from collections import Counter, defaultdict
from pathlib import Path
import math
import re
import csv
import json
import matplotlib.pyplot as plt

# ------------------------- CONFIGURATION -------------------------
TRAIN_FILE = "hindi-train-pos.txt"
DEV_FILE = "hindi-dev-pos.txt"
TEST_FILE = "hindi-test-pos.txt"
OUTPUT_DIR = "pos_tagger_results"
HMM_ALPHA = 0.1                 # Additive smoothing for HMM probabilities
NB_ALPHA = 1.0              # Additive smoothing for Naive Bayes feature likelihoods
UNKNOWN_WORD = "<UNK>"
# -----------------------------------------------------------------

def parse_token_line(line):
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^Word=(.*?)\s+Tag=(\S+)$", line)
    if m:
        return m.group(1), m.group(2)
    if "\t" in line:
        fields = line.split("\t")
        if len(fields) >= 2:
            return fields[0].strip(), fields[1].strip()
    if len(line.split()) == 1 and "/" in line:
        word, tag = line.rsplit("/", 1)
        if word and tag:
            return word, tag
    fields = line.split()
    if len(fields) >= 2:
        return fields[0], fields[-1]
    raise ValueError(f"Cannot parse line {line!r}; expected 'word tag'.")


def load_tagged_sentences(filename):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path.resolve()}; edit the file paths at the top.")
    sentences, current = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
            else:
                item = parse_token_line(line)
                if item:
                    current.append(item)
    if current:
        sentences.append(current)
    if not sentences:
        raise ValueError(f"No tagged sentences found in {path.resolve()}")
    return sentences


def logsumexp(values):
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


# ----------------------------- NAIVE BAYES -----------------------------
class NaiveBayesPOSTagger:
    FEATURES = ("length", "word", "previous_word", "next_word", "previous_tag")

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    @staticmethod
    def feature_values(words, i, previous_tag):
        return {
            "length": "MORE" if len(words[i]) > 4 else "LESS",
            "word": words[i],
            "previous_word": words[i - 1] if i > 0 else "<BOS>",
            "next_word": words[i + 1] if i + 1 < len(words) else "<EOS>",
            "previous_tag": previous_tag if previous_tag is not None else "<BOS_TAG>",
        }

    def fit(self, sentences):
        self.tag_counts = Counter()
        self.counts = {f: defaultdict(Counter) for f in self.FEATURES}
        self.vocabs = {f: set() for f in self.FEATURES}
        for sentence in sentences:
            words = [w for w, _ in sentence]
            previous_gold_tag = None
            for i, (_, tag) in enumerate(sentence):
                self.tag_counts[tag] += 1
                for feature, value in self.feature_values(words, i, previous_gold_tag).items():
                    self.counts[feature][tag][value] += 1
                    self.vocabs[feature].add(value)
                previous_gold_tag = tag
        self.tags = sorted(self.tag_counts)
        self.total = sum(self.tag_counts.values())
        return self

    def _score(self, tag, features):
        a = self.alpha
        score = math.log((self.tag_counts[tag] + a) / (self.total + a * len(self.tags)))
        for feature, value in features.items():
            vocab_size = len(self.vocabs[feature]) + 1
            score += math.log(
                (self.counts[feature][tag][value] + a)
                / (self.tag_counts[tag] + a * vocab_size)
            )
        return score

    def predict_sentence(self, words):
        result, previous_predicted_tag = [], None
        for i in range(len(words)):
            features = self.feature_values(words, i, previous_predicted_tag)
            tag = max(self.tags, key=lambda t: self._score(t, features))
            result.append(tag)
            previous_predicted_tag = tag
        return result

    def predict(self, sentences):
        return [self.predict_sentence([w for w, _ in s]) for s in sentences]


# -------------------------------- HMM ----------------------------------
class HMMPOSTagger:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, sentences):
        self.tag_counts = Counter()
        self.start_counts = Counter()
        self.transitions = defaultdict(Counter)
        self.emissions = defaultdict(Counter)
        self.vocab = set()
        self.num_sentences = sum(bool(s) for s in sentences)
        for sentence in sentences:
            if not sentence:
                continue
            tags = [t for _, t in sentence]
            self.start_counts[tags[0]] += 1
            for word, tag in sentence:
                self.tag_counts[tag] += 1
                self.emissions[tag][word] += 1
                self.vocab.add(word)
            for a, b in zip(tags, tags[1:]):
                self.transitions[a][b] += 1
        self.tags = sorted(self.tag_counts)
        return self

    def log_start(self, tag):
        return math.log((self.start_counts[tag] + self.alpha) /
                        (self.num_sentences + self.alpha * len(self.tags)))

    def log_trans(self, a, b):
        total = sum(self.transitions[a].values())
        return math.log((self.transitions[a][b] + self.alpha) /
                        (total + self.alpha * len(self.tags)))

    def log_emit(self, tag, word):
        return math.log((self.emissions[tag][word] + self.alpha) /
                        (self.tag_counts[tag] + self.alpha * (len(self.vocab) + 1)))

    def forward_viterbi_sentence(self, words):
        """Left-to-right max-product DP with backpointers."""
        if not words:
            return []
        n, dp, back = len(words), [], []
        dp = [{} for _ in range(n)]
        back = [{} for _ in range(n)]
        for t in self.tags:
            dp[0][t] = self.log_start(t) + self.log_emit(t, words[0])
            back[0][t] = None
        for i in range(1, n):
            for t in self.tags:
                prev = max(self.tags, key=lambda p: dp[i-1][p] + self.log_trans(p, t))
                dp[i][t] = dp[i-1][prev] + self.log_trans(prev, t) + self.log_emit(t, words[i])
                back[i][t] = prev
        last = max(self.tags, key=lambda t: dp[-1][t])
        path = [last]
        for i in range(n - 1, 0, -1):
            path.append(back[i][path[-1]])
        return path[::-1]

    def backward_viterbi_sentence(self, words):
        """Right-to-left max-product DP; each word emission is counted once."""
        if not words:
            return []
        n = len(words)
        dp, next_ptr = [{} for _ in range(n)], [{} for _ in range(n)]
        for t in self.tags:
            dp[n-1][t] = self.log_emit(t, words[n-1])
            next_ptr[n-1][t] = None
        for i in range(n-2, -1, -1):
            for t in self.tags:
                nxt = max(self.tags, key=lambda q: self.log_trans(t, q) + dp[i+1][q])
                dp[i][t] = self.log_emit(t, words[i]) + self.log_trans(t, nxt) + dp[i+1][nxt]
                next_ptr[i][t] = nxt
        first = max(self.tags, key=lambda t: self.log_start(t) + dp[0][t])
        path = [first]
        for i in range(n-1):
            path.append(next_ptr[i][path[-1]])
        return path

    def forward_backward_sentence(self, words):
        """
        Forward-backward posterior decoding: sum over paths in the forward and
        backward recurrences, then choose the most probable tag at each token.
        This is not the same objective as Viterbi's best complete sequence.
        """
        if not words:
            return []
        n = len(words)
        fwd, bwd = [{} for _ in range(n)], [{} for _ in range(n)]
        for t in self.tags:
            fwd[0][t] = self.log_start(t) + self.log_emit(t, words[0])
        for i in range(1, n):
            for t in self.tags:
                fwd[i][t] = self.log_emit(t, words[i]) + logsumexp([
                    fwd[i-1][p] + self.log_trans(p, t) for p in self.tags
                ])
        for t in self.tags:
            bwd[n-1][t] = 0.0
        for i in range(n-2, -1, -1):
            for t in self.tags:
                bwd[i][t] = logsumexp([
                    self.log_trans(t, q) + self.log_emit(q, words[i+1]) + bwd[i+1][q]
                    for q in self.tags
                ])
        return [
            max(self.tags, key=lambda t: fwd[i][t] + bwd[i][t])
            for i in range(n)
        ]

    def predict(self, sentences, algorithm):
        method = {
            "forward": self.forward_viterbi_sentence,
            "backward": self.backward_viterbi_sentence,
            "forward_backward": self.forward_backward_sentence,
        }[algorithm]
        return [method([w for w, _ in s]) for s in sentences]


# ------------------------------ EVALUATION ------------------------------
def evaluate(gold_sentences, predicted_sentences, split, model):
    gold = [tag for s in gold_sentences for _, tag in s]
    pred = [tag for s in predicted_sentences for tag in s]
    if len(gold) != len(pred):
        raise ValueError(f"Token count mismatch: {split}/{model}")
    tags = sorted(set(gold) | set(pred))
    rows = []
    for tag in tags:
        tp = sum(g == tag and p == tag for g, p in zip(gold, pred))
        fp = sum(g != tag and p == tag for g, p in zip(gold, pred))
        fn = sum(g == tag and p != tag for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(g == tag for g in gold)
        rows.append(dict(split=split, model=model, tag=tag, precision=precision,
                         recall=recall, f1=f1, support=support))
    total = len(gold)
    accuracy = sum(g == p for g, p in zip(gold, pred)) / total if total else 0.0
    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    weighted_f1 = sum(r["f1"] * r["support"] for r in rows) / total if total else 0.0
    summary = dict(split=split, model=model, tokens=total, accuracy=accuracy,
                   macro_f1=macro_f1, weighted_f1=weighted_f1)
    return rows, summary


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def make_plots(per_tag_rows, summaries, output_dir):
    out = Path(output_dir)
    for split in ("dev", "test"):
        rows = [r for r in per_tag_rows if r["split"] == split]
        tags = sorted({r["tag"] for r in rows})
        models = list(dict.fromkeys(r["model"] for r in rows))
        if tags and models:
            width, x = 0.8 / len(models), list(range(len(tags)))
            plt.figure(figsize=(max(12, len(tags) * 0.55), 6))
            for j, model in enumerate(models):
                values = {r["tag"]: r["f1"] for r in rows if r["model"] == model}
                offsets = [i + (j - (len(models)-1)/2) * width for i in x]
                plt.bar(offsets, [values.get(t, 0.0) for t in tags], width=width, label=model)
            plt.xticks(x, tags, rotation=65, ha="right")
            plt.xlabel("POS tag"); plt.ylabel("F1-score"); plt.ylim(0, 1)
            plt.title(f"Per-tag F1-score comparison ({split})")
            plt.legend(); plt.tight_layout()
            plt.savefig(out / f"tag_f1_{split}.png", dpi=200); plt.close()

        sm = [r for r in summaries if r["split"] == split]
        if sm:
            metrics = ("accuracy", "macro_f1", "weighted_f1")
            x, width = list(range(len(sm))), 0.25
            plt.figure(figsize=(11, 6))
            for j, metric in enumerate(metrics):
                plt.bar([i + (j-1)*width for i in x], [r[metric] for r in sm],
                        width=width, label=metric.replace("_", " ").title())
            plt.xticks(x, [r["model"] for r in sm], rotation=20, ha="right")
            plt.ylabel("Score"); plt.ylim(0, 1)
            plt.title(f"Overall performance comparison ({split})")
            plt.legend(); plt.tight_layout()
            plt.savefig(out / f"overall_metrics_{split}.png", dpi=200); plt.close()


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    train, dev, test = (load_tagged_sentences(p) for p in
                        (TRAIN_FILE, DEV_FILE, TEST_FILE))
    for name, data in (("train", train), ("dev", dev), ("test", test)):
        words = [w for s in data for w, _ in s]
        tags = {t for s in data for _, t in s}
        print(f"{name}: {len(data)} sentences, {len(words)} tokens, {len(tags)} tags")

    print("\nTraining Naive Bayes...")
    nb = NaiveBayesPOSTagger(NB_ALPHA).fit(train)
    print("Training HMM...")
    hmm = HMMPOSTagger(HMM_ALPHA).fit(train)

    models = [
        ("Naive Bayes", lambda data: nb.predict(data)),
        ("Forward Viterbi", lambda data: hmm.predict(data, "forward")),
        ("Backward Viterbi", lambda data: hmm.predict(data, "backward")),
        ("Forward-Backward (posterior)", lambda data: hmm.predict(data, "forward_backward")),
    ]
    all_rows, summaries = [], []
    for split, dataset in (("dev", dev), ("test", test)):
        print(f"\nEvaluating {split} split...")
        for name, predict in models:
            predictions = predict(dataset)
            rows, summary = evaluate(dataset, predictions, split, name)
            all_rows.extend(rows); summaries.append(summary)
            print(f"{name:31s} | accuracy={summary['accuracy']:.4f} "
                  f"| macro-F1={summary['macro_f1']:.4f} "
                  f"| weighted-F1={summary['weighted_f1']:.4f}")

    write_csv(out / "per_tag_metrics.csv", all_rows)
    write_csv(out / "summary_metrics.csv", summaries)
    with (out / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    make_plots(all_rows, summaries, out)
    print(f"\nDone. Results saved in: {out.resolve()}")
    print("Files: per_tag_metrics.csv, summary_metrics.csv, summary_metrics.json")
    print("Graphs: tag_f1_dev.png, tag_f1_test.png, overall_metrics_dev.png, overall_metrics_test.png")
    print("Note: Forward-backward uses posterior decoding, not Viterbi sequence decoding.")


if __name__ == "__main__":
    main()
