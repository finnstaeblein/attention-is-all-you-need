"""
Evaluation, translation, BLEU scoring, and attention visualization.

Loads the best trained model, generates translations on test data,
computes BLEU scores, and produces attention heatmap figures.
"""

import csv
import json
import math
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch

from dataset import Vocabulary, TranslationDataset, collate_fn, build_vocabs
from transformer import Transformer

# ── Config ────────────────────────────────────────────────────────────────────

BEST_MODEL_PATH = "best_model.pt"
HISTORY_PATH = "training_history.json"
FIGURES_DIR = "writeup/figures"
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load model and vocabs ─────────────────────────────────────────────────────

def load_model_and_vocabs():
    src_vocab = Vocabulary.load("src_vocab.json")
    tgt_vocab = Vocabulary.load("tgt_vocab.json")

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=0.0,  # no dropout at inference
    ).to(device)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model, src_vocab, tgt_vocab


# ── Greedy translation ────────────────────────────────────────────────────────

@torch.no_grad()
def translate(model, src_tokens, src_vocab, tgt_vocab, max_len=50):
    """Greedily translate English tokens to German.

    Returns:
        generated_tokens: list of German token strings
        cross_attentions: list (per layer) of (n_heads, tgt_len, src_len) tensors
    """
    # Encode source
    src_ids = src_vocab.encode(src_tokens)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src_tensor == 0)

    # Start with <sos>
    tgt_ids = [Vocabulary.SOS_IDX]

    cross_attentions_all = []

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        tgt_mask = (tgt_tensor == 0)

        logits, _, cross_attns = model(src_tensor, tgt_tensor, src_mask, tgt_mask)

        # Take the last token's prediction
        next_token_id = logits[0, -1].argmax().item()
        tgt_ids.append(next_token_id)
        cross_attentions_all = cross_attns  # keep latest full set

        if next_token_id == Vocabulary.EOS_IDX:
            break

    generated_tokens = tgt_vocab.decode(tgt_ids)

    # Extract attention weights: list of (n_heads, tgt_len, src_len)
    cross_attn_weights = [attn[0].cpu() for attn in cross_attentions_all]

    return generated_tokens, cross_attn_weights


# ── BLEU score ────────────────────────────────────────────────────────────────

def compute_ngrams(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference, hypothesis, max_n=4):
    """Compute sentence-level BLEU with brevity penalty."""
    if len(hypothesis) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(reference, n)
        hyp_ngrams = compute_ngrams(hypothesis, n)

        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    # Avoid log(0)
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    bp = 1.0
    if len(hypothesis) < len(reference):
        bp = math.exp(1 - len(reference) / len(hypothesis))

    return bp * math.exp(log_avg)


def compute_corpus_bleu(references, hypotheses):
    """Average sentence-level BLEU over a corpus."""
    scores = [sentence_bleu(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return sum(scores) / len(scores) if scores else 0.0


# ── Read test data ────────────────────────────────────────────────────────────

def read_test_pairs(csv_path="test_tok.csv"):
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            en_tokens = row[0].split()
            de_tokens = row[1].split()
            pairs.append((en_tokens, de_tokens))
    return pairs


# ── Visualization helpers ─────────────────────────────────────────────────────

def plot_training_curves(history_path, save_path):
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", label="Train loss", color="#2563eb")
    ax.plot(epochs, history["val_loss"], "s-", label="Val loss", color="#dc2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(epochs))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_attention_heatmap(attention, src_tokens, tgt_tokens, save_path, title="Cross-Attention"):
    """Plot a single attention heatmap.

    Args:
        attention: (tgt_len, src_len) tensor
        src_tokens: list of source token strings (x-axis)
        tgt_tokens: list of target token strings (y-axis)
    """
    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.7), max(4, len(tgt_tokens) * 0.5)))

    attn_np = attention.numpy()
    ax.imshow(attn_np, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=9)

    ax.set_xlabel("English (source)")
    ax.set_ylabel("German (target)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_attention_heads_comparison(attention_heads, src_tokens, tgt_tokens, save_path):
    """Plot all attention heads in a 2x2 grid.

    Args:
        attention_heads: (n_heads, tgt_len, src_len) tensor
    """
    n_heads = attention_heads.size(0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, ax in enumerate(axes.flat):
        if i < n_heads:
            attn_np = attention_heads[i].numpy()
            im = ax.imshow(attn_np, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(tgt_tokens)))
            ax.set_yticklabels(tgt_tokens, fontsize=8)
            ax.set_title(f"Head {i}")
        else:
            ax.axis("off")

    fig.suptitle("Attention Heads Comparison (Cross-Attention, Layer 1)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading model...")
    model, src_vocab, tgt_vocab = load_model_and_vocabs()

    print("Reading test data...")
    test_pairs = read_test_pairs()

    # ── 1. Training curves ────────────────────────────────────────────────
    plot_training_curves(HISTORY_PATH, os.path.join(FIGURES_DIR, "training_curves.png"))

    # ── 2. Generate translations + BLEU ───────────────────────────────────
    print("\nTranslating test set (first 200 pairs)...")
    references = []
    hypotheses = []
    translations = []  # (en_tokens, ref_de_tokens, gen_de_tokens, cross_attns)

    for i, (en_tokens, de_tokens) in enumerate(test_pairs[:200]):
        gen_tokens, cross_attns = translate(model, en_tokens, src_vocab, tgt_vocab)
        references.append(de_tokens)
        hypotheses.append(gen_tokens)
        translations.append((en_tokens, de_tokens, gen_tokens, cross_attns))

        if (i + 1) % 50 == 0:
            print(f"  Translated {i + 1}/200")

    bleu = compute_corpus_bleu(references, hypotheses)
    print(f"\nCorpus BLEU (200 test pairs): {bleu:.4f}")

    # ── 3. Save example translations ──────────────────────────────────────
    examples_path = os.path.join(FIGURES_DIR, "..", "example_translations.txt")
    with open(examples_path, "w", encoding="utf-8") as f:
        for i in range(min(20, len(translations))):
            en, ref, gen, _ = translations[i]
            f.write(f"EN:  {' '.join(en)}\n")
            f.write(f"REF: {' '.join(ref)}\n")
            f.write(f"GEN: {' '.join(gen)}\n")
            f.write("---\n")
    print(f"Saved {examples_path}")

    # ── 4. Attention heatmaps ─────────────────────────────────────────────
    # Pick 5 diverse examples by length
    # Sort by source length to find short/medium/long/question
    candidates = []
    for i, (en, de, gen, attns) in enumerate(translations):
        is_question = en[-1] == "?" if en else False
        candidates.append((i, len(en), is_question))

    # Select diverse examples
    short = [c for c in candidates if 3 <= c[1] <= 5 and not c[2]]
    medium = [c for c in candidates if 6 <= c[1] <= 8 and not c[2]]
    longer = [c for c in candidates if 9 <= c[1] <= 12 and not c[2]]
    questions = [c for c in candidates if c[2]]

    selected_indices = []
    for group in [short, medium, longer, questions]:
        if group:
            selected_indices.append(group[0][0])

    # Fill to 5 if needed
    for c in candidates:
        if len(selected_indices) >= 5:
            break
        if c[0] not in selected_indices:
            selected_indices.append(c[0])

    selected_indices = selected_indices[:5]

    print(f"\nGenerating attention heatmaps for {len(selected_indices)} examples...")
    for fig_idx, data_idx in enumerate(selected_indices, 1):
        en_tokens, de_ref, gen_tokens, cross_attns = translations[data_idx]

        if not cross_attns or not gen_tokens:
            continue

        # Use last decoder layer's cross-attention, averaged across heads
        last_layer_attn = cross_attns[-1]  # (n_heads, tgt_len, src_len)

        # Trim attention to match generated tokens (exclude <sos>, include up to <eos>)
        # gen_tokens are already decoded (no sos/eos), but attention includes sos position
        # Attention rows correspond to decoder positions: [sos, tok1, tok2, ..., eos]
        # We want rows 1 through len(gen_tokens) (skip sos row)
        n_gen = len(gen_tokens)
        # Source includes [sos, tok1, ..., tokN, eos] — attention columns match these
        src_display = ["<sos>"] + en_tokens + ["<eos>"]
        tgt_display = gen_tokens

        # Average across heads
        avg_attn = last_layer_attn.mean(dim=0)  # (tgt_len, src_len)

        # Trim: skip sos row (index 0), take next n_gen rows
        if avg_attn.size(0) > n_gen:
            avg_attn = avg_attn[1 : n_gen + 1]

        # Clamp to available src columns
        n_src = min(avg_attn.size(1), len(src_display))
        avg_attn = avg_attn[:, :n_src]
        src_display = src_display[:n_src]

        save_path = os.path.join(FIGURES_DIR, f"attention_heatmap_{fig_idx}.png")
        plot_attention_heatmap(
            avg_attn,
            src_display,
            tgt_display,
            save_path,
            title=f"Cross-Attention (avg heads) — Example {fig_idx}",
        )

    # ── 5. Heads comparison for one example ───────────────────────────────
    # Use the first selected example
    if selected_indices:
        idx = selected_indices[0]
        en_tokens, de_ref, gen_tokens, cross_attns = translations[idx]

        if cross_attns and gen_tokens:
            last_layer_attn = cross_attns[-1]  # (n_heads, tgt_len, src_len)
            n_gen = len(gen_tokens)
            src_display = ["<sos>"] + en_tokens + ["<eos>"]

            # Trim per-head attention
            trimmed = last_layer_attn[:, 1 : n_gen + 1]
            n_src = min(trimmed.size(2), len(src_display))
            trimmed = trimmed[:, :, :n_src]
            src_display = src_display[:n_src]

            save_path = os.path.join(FIGURES_DIR, "attention_heads_comparison.png")
            plot_attention_heads_comparison(trimmed, src_display, gen_tokens, save_path)

    # ── 6. Save BLEU score for report ─────────────────────────────────────
    results = {"bleu_200": bleu, "num_test_pairs": len(test_pairs)}
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone! All figures saved to writeup/figures/")
