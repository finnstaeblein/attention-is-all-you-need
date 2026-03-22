"""
Transformer model for sequence-to-sequence translation, implemented from scratch.

Based on "Attention Is All You Need" (Vaswani et al., 2017).
Uses only PyTorch primitives (nn.Linear, nn.LayerNorm, nn.Embedding, etc.)
— no pre-built transformer modules.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    Args:
        Q: (batch, heads, seq_len_q, d_k)
        K: (batch, heads, seq_len_k, d_k)
        V: (batch, heads, seq_len_k, d_k)
        mask: broadcastable boolean mask — True at positions to mask out

    Returns:
        output: (batch, heads, seq_len_q, d_k)
        weights: (batch, heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.

    Splits d_model into n_heads parallel attention heads, each with d_k = d_model // n_heads.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
            mask: 2D padding mask (batch, seq_len_k) or 3D causal mask (1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len_q, d_model)
            weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = Q.size(0)

        # Project and reshape to (batch, n_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Expand mask for multi-head: add head dimension
        if mask is not None:
            if mask.dim() == 2:
                # Padding mask: (batch, seq_len_k) → (batch, 1, 1, seq_len_k)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Causal mask: (1, seq_len, seq_len) → (1, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)

        output, weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads: (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the paper.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model) — registered as buffer, not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    """Position-wise feed-forward network: d_model → d_ff → d_model with ReLU."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class EncoderBlock(nn.Module):
    """Single encoder block: self-attention → add & norm → FFN → add & norm."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: source padding mask (batch, seq_len)
        """
        attn_out, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class DecoderBlock(nn.Module):
    """Single decoder block: masked self-attn → add & norm → cross-attn → add & norm → FFN → add & norm."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch, tgt_len, d_model)
            encoder_output: (batch, src_len, d_model)
            src_mask: source padding mask (batch, src_len)
            tgt_mask: combined causal + padding mask for target
        """
        # Masked self-attention
        self_attn_out, self_attn_weights = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention (Q from decoder, K/V from encoder)
        cross_attn_out, cross_attn_weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence translation."""

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings (padding_idx=0 ensures <pad> embeddings stay zero)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Output projection to target vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    @staticmethod
    def make_causal_mask(seq_len, device):
        """Create upper-triangular causal mask.

        Returns boolean tensor of shape (1, seq_len, seq_len) where True = masked.
        """
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool().unsqueeze(0)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch, src_len) — source token indices
            tgt: (batch, tgt_len) — target token indices
            src_mask: (batch, src_len) — source padding mask (True where padded)
            tgt_mask: (batch, tgt_len) — target padding mask (True where padded)

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
            encoder_self_attn: list of (batch, n_heads, src_len, src_len)
            decoder_cross_attn: list of (batch, n_heads, tgt_len, src_len)
        """
        # Scale embeddings as in the paper (multiply by sqrt(d_model))
        src_emb = self.dropout(self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        # Encode
        encoder_output = src_emb
        encoder_self_attns = []
        for layer in self.encoder_layers:
            encoder_output, attn_w = layer(encoder_output, src_mask)
            encoder_self_attns.append(attn_w)

        # Build causal mask for decoder and combine with target padding mask
        causal_mask = self.make_causal_mask(tgt.size(1), tgt.device)
        if tgt_mask is not None:
            # Expand padding mask: (batch, tgt_len) → (batch, 1, tgt_len)
            # Combine with causal: positions masked if EITHER causal or padding
            tgt_combined_mask = causal_mask | tgt_mask.unsqueeze(1)
        else:
            tgt_combined_mask = causal_mask

        # Decode
        decoder_output = tgt_emb
        decoder_cross_attns = []
        for layer in self.decoder_layers:
            decoder_output, _, cross_attn_w = layer(
                decoder_output, encoder_output, src_mask, tgt_combined_mask
            )
            decoder_cross_attns.append(cross_attn_w)

        logits = self.output_projection(decoder_output)
        return logits, encoder_self_attns, decoder_cross_attns


if __name__ == "__main__":
    # Quick sanity check with dummy data
    src_vocab_size = 5649
    tgt_vocab_size = 9064

    model = Transformer(src_vocab_size, tgt_vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created. Total parameters: {total_params:,}")

    # Dummy forward pass
    batch_size = 4
    src = torch.randint(1, src_vocab_size, (batch_size, 10))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, 8))
    src_mask = torch.zeros(batch_size, 10).bool()
    tgt_mask = torch.zeros(batch_size, 8).bool()

    logits, enc_attns, dec_attns = model(src, tgt, src_mask, tgt_mask)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, 8, {tgt_vocab_size})")
    assert logits.shape == (batch_size, 8, tgt_vocab_size), "Shape mismatch!"
    print("Forward pass: OK")
