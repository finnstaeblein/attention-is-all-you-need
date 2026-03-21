"""
Vocabulary building, numericalization, and PyTorch Dataset/DataLoader pipeline
for the EN-DE translation dataset.

Reads tokenized CSVs (train_tok.csv, val_tok.csv, test_tok.csv) produced by
preprocess_tatoeba_en_de.py, builds integer vocabularies, and provides
batched, padded tensors ready for a Transformer model.
"""

import csv
import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class Vocabulary:
    """Maps tokens (strings) to integer indices and back."""

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build(self, token_lists):
        """Build vocabulary from a list of token sequences (training data only).

        Tokens are sorted alphabetically for deterministic index assignment.
        """
        unique_tokens = set()
        for tokens in token_lists:
            unique_tokens.update(tokens)

        # Remove any tokens that collide with special token names
        unique_tokens -= set(self.SPECIAL_TOKENS)

        # Sort for deterministic ordering across runs
        for token in sorted(unique_tokens):
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        return self

    def encode(self, tokens):
        """Convert token strings to integer indices, wrapped with <sos> and <eos>."""
        ids = [self.SOS_IDX]
        ids.extend(self.token2idx.get(t, self.UNK_IDX) for t in tokens)
        ids.append(self.EOS_IDX)
        return ids

    def decode(self, indices):
        """Convert integer indices back to token strings.

        Skips <pad> tokens and stops at the first <eos>.
        """
        tokens = []
        for idx in indices:
            if idx == self.EOS_IDX:
                break
            if idx == self.PAD_IDX or idx == self.SOS_IDX:
                continue
            tokens.append(self.idx2token.get(idx, "<unk>"))
        return tokens

    def __len__(self):
        return len(self.token2idx)

    def save(self, path):
        """Save vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """Load vocabulary from a JSON file."""
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.token2idx = json.load(f)
        vocab.idx2token = {int(idx): token for token, idx in vocab.token2idx.items()}
        return vocab


class TranslationDataset(Dataset):
    """PyTorch Dataset for tokenized EN-DE sentence pairs."""

    def __init__(self, csv_path, src_vocab, tgt_vocab):
        self.pairs = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header (en_tok, de_tok)
            for row in reader:
                src_tokens = row[0].split()
                tgt_tokens = row[1].split()
                src_ids = src_vocab.encode(src_tokens)
                tgt_ids = tgt_vocab.encode(tgt_tokens)
                self.pairs.append((src_ids, tgt_ids))

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)


def collate_fn(batch):
    """Pad source and target sequences to the max length within the batch.

    Returns:
        src_batch:        LongTensor  (B, S_src)
        tgt_batch:        LongTensor  (B, S_tgt)
        src_padding_mask: BoolTensor  (B, S_src) — True where padded
        tgt_padding_mask: BoolTensor  (B, S_tgt) — True where padded
    """
    src_seqs, tgt_seqs = zip(*batch)

    src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=Vocabulary.PAD_IDX)
    tgt_batch = pad_sequence(tgt_seqs, batch_first=True, padding_value=Vocabulary.PAD_IDX)

    src_padding_mask = src_batch == Vocabulary.PAD_IDX
    tgt_padding_mask = tgt_batch == Vocabulary.PAD_IDX

    return src_batch, tgt_batch, src_padding_mask, tgt_padding_mask


def build_vocabs(train_csv_path):
    """Build source (EN) and target (DE) vocabularies from training data."""
    en_token_lists = []
    de_token_lists = []

    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            en_token_lists.append(row[0].split())
            de_token_lists.append(row[1].split())

    src_vocab = Vocabulary().build(en_token_lists)
    tgt_vocab = Vocabulary().build(de_token_lists)

    return src_vocab, tgt_vocab


def get_dataloaders(data_dir=".", batch_size=32):
    """Build vocabs and create DataLoaders for train, val, and test splits.

    Returns:
        (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)
    """
    train_path = os.path.join(data_dir, "train_tok.csv")
    val_path = os.path.join(data_dir, "val_tok.csv")
    test_path = os.path.join(data_dir, "test_tok.csv")

    src_vocab, tgt_vocab = build_vocabs(train_path)

    train_ds = TranslationDataset(train_path, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_path, src_vocab, tgt_vocab)
    test_ds = TranslationDataset(test_path, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    data_dir = "."
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(data_dir)

    print(f"Source (EN) vocab size: {len(src_vocab)}")
    print(f"Target (DE) vocab size: {len(tgt_vocab)}")
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Grab one batch
    src_batch, tgt_batch, src_mask, tgt_mask = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  src: {src_batch.shape}  tgt: {tgt_batch.shape}")
    print(f"  src_mask: {src_mask.shape} ({src_mask.dtype})  tgt_mask: {tgt_mask.shape} ({tgt_mask.dtype})")

    # Round-trip decode the first sample
    src_tokens = src_vocab.decode(src_batch[0].tolist())
    tgt_tokens = tgt_vocab.decode(tgt_batch[0].tolist())
    print(f"\nSample pair:")
    print(f"  EN: {' '.join(src_tokens)}")
    print(f"  DE: {' '.join(tgt_tokens)}")

    # Verify SOS/EOS positions in raw indices
    raw_src = src_batch[0].tolist()
    print(f"\nRaw src indices (first 5): {raw_src[:5]}  (last non-pad): {[x for x in raw_src if x != 0][-3:]}")
    print(f"  SOS_IDX={Vocabulary.SOS_IDX}, EOS_IDX={Vocabulary.EOS_IDX}")

    # Test vocab save/load round-trip
    src_vocab.save("/tmp/test_en_vocab.json")
    loaded = Vocabulary.load("/tmp/test_en_vocab.json")
    assert len(loaded) == len(src_vocab), "Vocab size mismatch after load"
    assert loaded.encode(["hello", "world"]) == src_vocab.encode(["hello", "world"]), "Encode mismatch after load"
    print("\nVocab save/load round-trip: OK")
