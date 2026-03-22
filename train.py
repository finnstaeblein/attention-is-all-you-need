"""
Training script for the Transformer EN→DE translation model.

Loads data via dataset.py, builds the model from transformer.py,
trains with teacher forcing, and saves the best checkpoint.
"""

import json
import random
import time

import torch
import torch.nn as nn

from dataset import get_dataloaders
from transformer import Transformer

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 512
DROPOUT = 0.1

BEST_MODEL_PATH = "best_model.pt"
HISTORY_PATH = "training_history.json"

# ── Reproducibility ───────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────

train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
    data_dir=".", batch_size=BATCH_SIZE
)
print(f"Vocab sizes — EN: {len(src_vocab)}, DE: {len(tgt_vocab)}")
print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}")

# Save vocabs for later use in analyze.py
src_vocab.save("src_vocab.json")
tgt_vocab.save("tgt_vocab.json")

# ── Model ─────────────────────────────────────────────────────────────────────

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Optimizer and loss ────────────────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>


# ── Training loop ─────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for src_batch, tgt_batch, src_mask, tgt_mask in loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        # Teacher forcing: input = tgt[:, :-1], labels = tgt[:, 1:]
        tgt_input = tgt_batch[:, :-1]
        tgt_labels = tgt_batch[:, 1:]
        tgt_input_mask = tgt_mask[:, :-1]

        logits, _, _ = model(src_batch, tgt_input, src_mask, tgt_input_mask)

        # Flatten for cross-entropy: (B * T, vocab) vs (B * T)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for src_batch, tgt_batch, src_mask, tgt_mask in loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        tgt_input = tgt_batch[:, :-1]
        tgt_labels = tgt_batch[:, 1:]
        tgt_input_mask = tgt_mask[:, :-1]

        logits, _, _ = model(src_batch, tgt_input, src_mask, tgt_input_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# ── Main training loop ────────────────────────────────────────────────────────

history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")

print(f"\nStarting training for {EPOCHS} epochs...")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)

    elapsed = time.time() - t0
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    improved = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        improved = " ← saved"

    print(
        f"Epoch {epoch:2d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"time={elapsed:.1f}s{improved}"
    )

print("-" * 60)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to {BEST_MODEL_PATH}")

# Save history
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)
print(f"Training history saved to {HISTORY_PATH}")
