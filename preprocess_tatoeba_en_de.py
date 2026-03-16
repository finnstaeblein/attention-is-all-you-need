#!/usr/bin/env python3
"""
Preprocess Tatoeba English-German sentence pairs for a toy Transformer experiment.

Reads deu.txt.zip (ManyThings/Tatoeba deu-eng file), cleans and filters sentence
pairs, splits into train/val/test, and produces tokenized versions plus a summary
report.

Usage:
    python preprocess_tatoeba_en_de.py
"""

import csv
import os
import random
import re
import unicodedata
import zipfile
from collections import Counter

import pandas as pd

# =============================================================================
# Configuration constants — adjust as needed
# =============================================================================

ZIP_PATH = "deu.txt.zip"          # Path to the zipped dataset
TXT_NAME = "deu.txt"              # Name of the text file inside the zip
OUTPUT_DIR = "."                   # Where to write output files

MAX_PAIRS = 20000                  # Max sentence pairs to keep (after filtering)
SEED = 42                          # Random seed for reproducibility

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

EN_MIN_WORDS = 2                   # Min whitespace-tokenized words for English
EN_MAX_WORDS = 12                  # Max whitespace-tokenized words for English
DE_MIN_WORDS = 2                   # Min whitespace-tokenized words for German
DE_MAX_WORDS = 15                  # Max whitespace-tokenized words for German
LENGTH_RATIO_MAX = 2.5             # Max ratio: de_words / en_words

CLEAN_CHAR_THRESHOLD = 0.85        # Min fraction of "clean" characters per sentence
NUMERIC_THRESHOLD = 0.30           # Max fraction of digit characters per sentence

# =============================================================================
# Helper functions
# =============================================================================

def clean_text(text):
    """Clean a single sentence string."""
    # Unicode NFC normalization (composes umlauts correctly)
    text = unicodedata.normalize("NFC", text)

    # Strip whitespace
    text = text.strip()

    # Remove control characters and zero-width chars
    text = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\ufeff]", "", text)

    # Lowercase
    text = text.lower()

    # Remove surrounding quotes if they look like formatting artifacts
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()

    # Normalize multiple spaces to one
    text = re.sub(r"\s+", " ", text).strip()

    # Keep only simple sentence-final punctuation (. ? !)
    # Remove other trailing punctuation like ; : , etc.
    text = re.sub(r"[;:,]+$", "", text).strip()

    return text


def has_url(text):
    """Check if text contains a URL."""
    return bool(re.search(r"https?://|www\.", text))


def has_email(text):
    """Check if text contains an email address."""
    return bool(re.search(r"\S+@\S+\.\S+", text))


def is_heavy_numeric(text):
    """Check if text has too many digit characters."""
    if len(text) == 0:
        return True
    digit_count = sum(1 for c in text if c.isdigit())
    return (digit_count / len(text)) > NUMERIC_THRESHOLD


def clean_char_fraction(text):
    """Compute fraction of characters that are 'clean' (letters, spaces, simple punct)."""
    if len(text) == 0:
        return 0.0
    clean_count = sum(1 for c in text if re.match(r"[a-zäöüß \.,\?!'\-]", c))
    return clean_count / len(text)


def simple_tokenize(text):
    """
    Simple whitespace + punctuation tokenizer.
    Splits words and separates punctuation as individual tokens.
    Preserves German umlauts and ß.
    """
    tokens = re.findall(r"[a-zäöüß]+|[.,?!'\-]", text)
    return " ".join(tokens)


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    stats = {}  # Track counts at each filtering step

    # ----- Step 1: Load dataset from zip -----
    print("Loading dataset from zip...")
    pairs = []
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        with zf.open(TXT_NAME) as f:
            for line in f:
                line = line.decode("utf-8")
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))

    df = pd.DataFrame(pairs, columns=["en", "de"])
    stats["01_loaded"] = len(df)
    print(f"  Loaded {len(df)} sentence pairs")

    # ----- Step 2: Clean text -----
    print("Cleaning text...")
    df["en"] = df["en"].apply(clean_text)
    df["de"] = df["de"].apply(clean_text)

    # ----- Step 3: Sentence filtering -----
    print("Filtering sentences...")

    # Remove empty pairs
    df = df[(df["en"].str.len() > 0) & (df["de"].str.len() > 0)]
    stats["02_non_empty"] = len(df)
    print(f"  After removing empty: {len(df)}")

    # Word count filters
    df["en_wc"] = df["en"].apply(lambda x: len(x.split()))
    df["de_wc"] = df["de"].apply(lambda x: len(x.split()))

    df = df[(df["en_wc"] >= EN_MIN_WORDS) & (df["en_wc"] <= EN_MAX_WORDS)]
    stats["03_en_length"] = len(df)
    print(f"  After English length filter ({EN_MIN_WORDS}-{EN_MAX_WORDS} words): {len(df)}")

    df = df[(df["de_wc"] >= DE_MIN_WORDS) & (df["de_wc"] <= DE_MAX_WORDS)]
    stats["04_de_length"] = len(df)
    print(f"  After German length filter ({DE_MIN_WORDS}-{DE_MAX_WORDS} words): {len(df)}")

    # No URLs
    mask_url = df["en"].apply(has_url) | df["de"].apply(has_url)
    df = df[~mask_url]
    stats["05_no_urls"] = len(df)
    print(f"  After removing URLs: {len(df)}")

    # No emails
    mask_email = df["en"].apply(has_email) | df["de"].apply(has_email)
    df = df[~mask_email]
    stats["06_no_emails"] = len(df)
    print(f"  After removing emails: {len(df)}")

    # No heavy numeric content
    mask_numeric = df["en"].apply(is_heavy_numeric) | df["de"].apply(is_heavy_numeric)
    df = df[~mask_numeric]
    stats["07_no_heavy_numeric"] = len(df)
    print(f"  After removing heavy numeric: {len(df)}")

    # Length ratio filter
    df = df[df["de_wc"] <= LENGTH_RATIO_MAX * df["en_wc"]]
    stats["08_length_ratio"] = len(df)
    print(f"  After length ratio filter (de ≤ {LENGTH_RATIO_MAX}x en): {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates(subset=["en", "de"])
    stats["09_no_duplicates"] = len(df)
    print(f"  After removing duplicates: {len(df)}")

    # ----- Step 4: Character filtering -----
    print("Character filtering...")
    mask_chars = (
        df["en"].apply(clean_char_fraction).ge(CLEAN_CHAR_THRESHOLD)
        & df["de"].apply(clean_char_fraction).ge(CLEAN_CHAR_THRESHOLD)
    )
    df = df[mask_chars]
    stats["10_char_filter"] = len(df)
    print(f"  After character filtering (≥{CLEAN_CHAR_THRESHOLD:.0%} clean): {len(df)}")

    # ----- Step 5: Vocabulary sanity filtering -----
    print("Vocabulary sanity filtering...")

    # Tokenize for frequency analysis
    all_en_tokens = []
    all_de_tokens = []
    for _, row in df.iterrows():
        all_en_tokens.extend(row["en"].split())
        all_de_tokens.extend(row["de"].split())

    en_freq = Counter(all_en_tokens)
    de_freq = Counter(all_de_tokens)

    # Find hapax legomena (tokens appearing exactly once)
    en_hapax = {t for t, c in en_freq.items() if c == 1}
    de_hapax = {t for t, c in de_freq.items() if c == 1}

    def has_suspicious_hapax(text, hapax_set):
        """Check if a sentence has ≥2 hapax tokens that contain non-letter chars."""
        words = text.split()
        suspicious = [w for w in words if w in hapax_set and not re.match(r"^[a-zäöüß]+$", w)]
        return len(suspicious) >= 2

    mask_hapax = (
        df["en"].apply(lambda x: has_suspicious_hapax(x, en_hapax))
        | df["de"].apply(lambda x: has_suspicious_hapax(x, de_hapax))
    )
    df = df[~mask_hapax]
    stats["11_vocab_sanity"] = len(df)
    print(f"  After vocabulary sanity filtering: {len(df)}")

    # Drop helper columns
    df = df[["en", "de"]].reset_index(drop=True)

    # ----- Step 6: Shuffle, cap, and split -----
    print("Shuffling and splitting...")

    # Shuffle with fixed seed
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Cap at MAX_PAIRS
    if len(df) > MAX_PAIRS:
        df = df.iloc[:MAX_PAIRS].reset_index(drop=True)
    stats["12_capped"] = len(df)
    print(f"  After capping at {MAX_PAIRS}: {len(df)}")

    # Split
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    stats["train_size"] = len(train_df)
    stats["val_size"] = len(val_df)
    stats["test_size"] = len(test_df)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ----- Save clean CSVs -----
    print("Saving clean CSVs...")
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, encoding="utf-8")

    # ----- Step 7: Tokenize and save -----
    print("Tokenizing and saving tokenized CSVs...")

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tok_df = pd.DataFrame({
            "en_tok": split_df["en"].apply(simple_tokenize),
            "de_tok": split_df["de"].apply(simple_tokenize),
        })
        tok_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{split_name}_tok.csv"),
            index=False, encoding="utf-8"
        )

    # ----- Step 8: Generate report -----
    print("Generating dataset report...")

    # Compute stats on final data
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_data["en_wc"] = all_data["en"].apply(lambda x: len(x.split()))
    all_data["de_wc"] = all_data["de"].apply(lambda x: len(x.split()))

    # Token frequencies from tokenized train set
    en_tok_counter = Counter()
    de_tok_counter = Counter()
    for _, row in train_df.iterrows():
        en_tok_counter.update(simple_tokenize(row["en"]).split())
        de_tok_counter.update(simple_tokenize(row["de"]).split())

    # Random examples
    random.seed(SEED)
    sample_indices = random.sample(range(len(train_df)), min(20, len(train_df)))
    examples = train_df.iloc[sample_indices]

    # Write report
    report_path = os.path.join(OUTPUT_DIR, "dataset_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Preprocessing Report\n\n")
        f.write("## Filtering Pipeline\n\n")
        f.write("| Step | Description | Pairs Remaining |\n")
        f.write("|------|-------------|----------------:|\n")

        step_labels = {
            "01_loaded": "Loaded from zip",
            "02_non_empty": "Non-empty pairs",
            "03_en_length": f"English length ({EN_MIN_WORDS}-{EN_MAX_WORDS} words)",
            "04_de_length": f"German length ({DE_MIN_WORDS}-{DE_MAX_WORDS} words)",
            "05_no_urls": "No URLs",
            "06_no_emails": "No emails",
            "07_no_heavy_numeric": f"No heavy numeric (>{NUMERIC_THRESHOLD:.0%} digits)",
            "08_length_ratio": f"Length ratio (de ≤ {LENGTH_RATIO_MAX}x en)",
            "09_no_duplicates": "Remove duplicates",
            "10_char_filter": f"Character filter (≥{CLEAN_CHAR_THRESHOLD:.0%} clean)",
            "11_vocab_sanity": "Vocabulary sanity",
            "12_capped": f"Capped at {MAX_PAIRS}",
        }

        for key, label in step_labels.items():
            f.write(f"| {key.split('_')[0]} | {label} | {stats[key]:,} |\n")

        f.write(f"\n## Final Split Sizes\n\n")
        f.write(f"| Split | Size |\n")
        f.write(f"|-------|-----:|\n")
        f.write(f"| Train | {stats['train_size']:,} |\n")
        f.write(f"| Val   | {stats['val_size']:,} |\n")
        f.write(f"| Test  | {stats['test_size']:,} |\n")
        f.write(f"| **Total** | **{stats['train_size'] + stats['val_size'] + stats['test_size']:,}** |\n")

        f.write(f"\n## Sentence Length Statistics\n\n")
        f.write(f"| Metric | English | German |\n")
        f.write(f"|--------|--------:|-------:|\n")
        f.write(f"| Average words | {all_data['en_wc'].mean():.1f} | {all_data['de_wc'].mean():.1f} |\n")
        f.write(f"| Max words     | {all_data['en_wc'].max()} | {all_data['de_wc'].max()} |\n")
        f.write(f"| Min words     | {all_data['en_wc'].min()} | {all_data['de_wc'].min()} |\n")

        f.write(f"\n## Top 50 English Tokens\n\n")
        f.write("| Rank | Token | Count |\n")
        f.write("|-----:|-------|------:|\n")
        for i, (token, count) in enumerate(en_tok_counter.most_common(50), 1):
            f.write(f"| {i} | `{token}` | {count:,} |\n")

        f.write(f"\n## Top 50 German Tokens\n\n")
        f.write("| Rank | Token | Count |\n")
        f.write("|-----:|-------|------:|\n")
        for i, (token, count) in enumerate(de_tok_counter.most_common(50), 1):
            f.write(f"| {i} | `{token}` | {count:,} |\n")

        f.write(f"\n## 20 Random Cleaned Example Pairs\n\n")
        f.write("| # | English | German |\n")
        f.write("|--:|---------|--------|\n")
        for i, (_, row) in enumerate(examples.iterrows(), 1):
            # Escape pipes in text for markdown table
            en_text = row["en"].replace("|", "\\|")
            de_text = row["de"].replace("|", "\\|")
            f.write(f"| {i} | {en_text} | {de_text} |\n")

        f.write(f"\n---\n")
        f.write(f"*Generated by `preprocess_tatoeba_en_de.py` with seed={SEED}*\n")

    print(f"  Report saved to {report_path}")
    print("\nDone! All files saved.")

    # Print summary for user
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {stats['train_size']:,} pairs")
    print(f"  Val:   {stats['val_size']:,} pairs")
    print(f"  Test:  {stats['test_size']:,} pairs")
    print(f"  Total: {stats['train_size'] + stats['val_size'] + stats['test_size']:,} pairs")
    print(f"\n  10 sample pairs:")
    sample10 = train_df.sample(10, random_state=SEED)
    for _, row in sample10.iterrows():
        print(f"    EN: {row['en']}")
        print(f"    DE: {row['de']}")
        print()


if __name__ == "__main__":
    main()
