# NLP Assignment 1

## Objective

Corpus preprocessing using AI4Bharat IndicCorpV2.

## Dataset

IndicCorpV2

Language Used

Hindi

## Tasks Performed

- Downloaded IndicCorpV2 corpus
- Used subset of corpus
- Custom Sentence Tokenizer
- Custom Word Tokenizer
- Handles
  - URLs
  - Emails
  - Dates
  - Decimal Numbers
  - Punctuation
- Saved tokenized sentences
- Saved compressed Parquet file
- Computed corpus statistics

## Output Files

```
data/output/indic_tokenized.txt
data/output/indic_statistics.txt
data/parquet/indic.parquet
```

## Run

```
python3 src/main.py
```