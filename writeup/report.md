# Reconstructing the Transformer: Attention from Scratch on English–German Translation

## 1. Introduction

The Transformer architecture, introduced by Vaswani et al. [1], fundamentally changed how sequence-to-sequence models are built. Prior to the Transformer, recurrent neural networks (RNNs) and their variants (LSTMs, GRUs) dominated machine translation, but they suffered from sequential computation bottlenecks – since each hidden state depends on the previous one, the computation cannot be parallelized across time steps – and difficulty capturing long-range dependencies. The Transformer replaced recurrence entirely with an attention mechanism that allows every position in a sequence to directly attend to every other position, enabling massive parallelism and more effective modeling of long-range relationships.

Attention mechanisms were first introduced for machine translation by Bahdanau et al. [2], who showed that allowing a decoder to selectively focus on different parts of the source sentence dramatically improved translation quality compared to fixed-length encoded representations. The key innovation of Vaswani et al. [1] was demonstrating that attention alone – without any recurrence or convolution – is sufficient to build a state-of-the-art sequence model.

The goal of this project is threefold: (1) reconstruct the Transformer's core attention mechanism mathematically, (2) implement the full encoder-decoder architecture from scratch using only PyTorch primitives, and (3) train it on a real English–German translation dataset to analyze what the attention mechanism actually learns. This is not an attempt to achieve state-of-the-art translation quality – rather, it is an exercise in understanding the mechanism by building it, training it, and visualizing its internal representations. All code is available at https://github.com/finnstaeblein/attention-is-all-you-need and is runnable from start to finish.

## 2. Theory – The Attention Mechanism

### Scaled Dot-Product Attention

The fundamental operation in the Transformer is scaled dot-product attention. Given three matrices – Queries ($Q$), Keys ($K$), and Values ($V$) – attention computes a weighted sum of values where the weights are determined by the compatibility between queries and keys:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$$

where $Q \in \mathbb{R}^{n \times d_k}$ is the query matrix containing $n$ query vectors of dimension $d_k$, $K \in \mathbb{R}^{m \times d_k}$ is the key matrix containing $m$ key vectors, $V \in \mathbb{R}^{m \times d_v}$ is the value matrix, and $d_k$ is the key dimension used as a scaling factor. The intuition is straightforward: each query vector "asks a question," each key vector "advertises what information it holds," and the dot product between them measures compatibility. The softmax converts these raw compatibility scores into a probability distribution, and the result is a weighted combination of value vectors. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large in magnitude as the dimension increases, which would push the softmax into regions with extremely small gradients [1].

### Multi-Head Attention

Rather than performing a single attention computation, the Transformer uses multi-head attention. The input is linearly projected $h$ times into different subspaces of dimension $d_k = d_{\text{model}} / h$, attention is computed independently in each subspace (each "head"), and the results are concatenated and projected back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

where:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Here $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ are learned projection matrices for head $i$, $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ is the output projection, $h$ is the number of heads, and $d_{\text{model}}$ is the model embedding dimension.

This allows different heads to learn different types of relationships. As I will show in my analysis, some heads learn positional alignment while others capture semantic or syntactic patterns.

### Positional Encoding

Since the Transformer has no recurrence and no convolution, it has no inherent notion of token order. Positional encodings are added to the input embeddings to inject sequence position information. The original paper uses sinusoidal functions [1]:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the position index in the sequence, $i$ is the dimension index, and $d_{\text{model}}$ is the embedding dimension. Each dimension of the positional encoding oscillates at a different frequency, creating a unique "fingerprint" for each position. The sinusoidal form was chosen because it allows the model to learn relative positions through linear transformations.

## 3. Architecture and Methods

My implementation follows the original encoder-decoder structure from Vaswani et al. [1], scaled down for a toy experiment.

**Encoder.** Each encoder block consists of two sub-layers: (1) multi-head self-attention, where every source token attends to every other source token, and (2) a position-wise feed-forward network (two linear transformations with ReLU activation: $d_{\text{model}} \to d_{ff} \to d_{\text{model}}$, where $d_{ff}$ is the inner feed-forward dimension). Each sub-layer is wrapped with a residual connection and layer normalization: $\text{LayerNorm}(x + \text{Sublayer}(x))$.

**Decoder.** Each decoder block has three sub-layers: (1) masked multi-head self-attention over the target sequence (masking prevents attending to future positions during training), (2) multi-head cross-attention where decoder queries attend to encoder outputs (this is where the translation "alignment" happens), and (3) a position-wise feed-forward network. All three sub-layers use residual connections and layer normalization.

**Configuration.** The model uses: $d_{\text{model}} = 128$, $n_{\text{heads}} = 4$, $n_{\text{layers}} = 2$, $d_{ff} = 512$, dropout $= 0.1$. This gives approximately **3,978,216 total parameters**. For comparison, the original Transformer base model used $d_{\text{model}} = 512$, 8 heads, 6 layers, and had 65M parameters [1]. The model is intentionally small – the goal is to observe attention behavior, not to maximize translation quality.

Crucially, no pre-built transformer modules (such as `nn.Transformer` or `nn.TransformerEncoder`) were used. Every component – scaled dot-product attention, multi-head attention, positional encoding, encoder blocks, decoder blocks, and the full model – was implemented from scratch using only `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, and standard tensor operations.

**Evaluation.** Translation quality was evaluated using corpus-level BLEU [3], which measures n-gram overlap between generated translations and reference translations. I also extracted and visualized cross-attention weights from the decoder's last layer to analyze learned alignment patterns.

## 4. Data

I use the Tatoeba English–German parallel corpus [4], a community-contributed dataset of sentence pairs. The raw dataset contains 331,266 pairs. I applied the following preprocessing pipeline:

1. **Text cleaning**: Unicode NFC normalization, lowercasing, removal of control characters and formatting artifacts
2. **Length filtering**: English sentences 2–12 words, German sentences 2–15 words, with a maximum length ratio of 2.5
3. **Quality filtering**: removed sentences with URLs, emails, excessive numeric content, or low "clean character" fractions
4. **Deduplication**: exact duplicate pairs removed
5. **Sampling**: shuffled with seed 42 and capped at 20,000 pairs

The resulting dataset was split 80/10/10 into 16,000 training, 2,000 validation, and 2,000 test pairs.

**Tokenization** used a simple regex-based word-level tokenizer matching the pattern `[a-zäöüß]+|[.,?!'-]`, which splits on word boundaries and preserves common punctuation. This produced vocabulary sizes of **5,649 tokens for English** and **9,064 tokens for German**. German's larger vocabulary reflects its richer morphology – compound words, case inflections, and grammatical gender create more unique word forms.

Each token is mapped to a unique integer index. Four special tokens are reserved: `<pad>` (index 0) for batch padding, `<sos>` (index 1) to signal the start of a sequence, `<eos>` (index 2) to signal the end, and `<unk>` (index 3) for out-of-vocabulary tokens.

## 5. Training

The model was trained for 10 epochs using the Adam optimizer [5] with a constant learning rate of $\text{lr} = 10^{-4}$. The loss function was cross-entropy with `ignore_index=0` to exclude padding tokens from the loss computation. I used teacher forcing during training: the decoder receives the ground-truth target tokens (shifted right by one position) as input, and the loss is computed against the next token at each position.

Batch size was 32, giving 500 training batches per epoch. Training was performed on CPU and completed in approximately 6.5 minutes total (~40 seconds per epoch).

**Figure 1** shows the training and validation loss curves across all 10 epochs.

![Figure 1: Training and validation cross-entropy loss over 10 epochs. Both curves decrease steadily, with training loss dropping from 6.03 to 3.35 and validation loss from 5.02 to 3.62. The widening gap in later epochs indicates the onset of overfitting.](figures/training_curves.png)

Training loss decreased from 6.03 (epoch 1) to 3.35 (epoch 10), and validation loss decreased from 5.02 to 3.62. The gap between training and validation loss widened gradually in later epochs, indicating the onset of overfitting, though both losses were still decreasing. The best validation loss of **3.62** was achieved at epoch 10.

## 6. Results

### Translation Quality

With only 3.9M parameters trained on 16K sentence pairs for 10 epochs, translation quality is limited. The model learns basic German sentence structure – it correctly starts many translations with appropriate pronouns ("wir," "ich," "tom," "es") and produces grammatically plausible fragments – but it struggles with content words and often generates repetitive or generic phrases.

Table 1 shows representative examples from the test set.

**Table 1:** Sample translations from the test set, comparing reference German translations to model-generated output.

| English | Reference | Generated |
|---|---|---|
| we know your father . | wir kennen euren vater . | wir wissen uns auf uns . |
| what ' s your opinion ? | was meinen sie ? | was ist dein ? |
| that ' s really very interesting . | das ist ja wirklich sehr interessant . | das ist sehr gut . |
| can you turn on the tv ? | kannst du den fernseher einschalten ? | kannst du den ganzen tag ? |

The model captures the right sentence openings ("wir," "was ist," "das ist," "kannst du den") but fails to produce correct content words. The corpus BLEU score [3] over 200 test pairs was **0.021**, which is very low but not unexpected for this scale of experiment.

### Attention Analysis

The most valuable output from this project is not the translation quality but the attention patterns the model learned. I extracted cross-attention weights from the decoder's last layer, which reveal how each generated German token attends to the English source tokens.

Figure 2 shows the cross-attention heatmap for the sentence "we know your father."

![Figure 2: Cross-attention heatmap for "we know your father." Each row corresponds to a generated German token and each column to a source English token. Color intensity indicates attention weight (darker = stronger attention). The model shows structured alignment: "wir" attends to the sentence start and "know," while the period aligns to the source period and end-of-sequence token.](figures/attention_heatmap_1.png)

In this example, the attention pattern shows clear structure. The first generated token "wir" attends most strongly to "know" and the beginning of the source sentence. The model has learned that the German translation should begin by referencing the subject and verb of the English source. The period token at the end of the generated sequence attends primarily to the source period and `<eos>` token, showing that the model learned to align punctuation.

Figure 3 shows the attention pattern for a longer sentence.

![Figure 3: Cross-attention heatmap for a longer sentence about "fundamental difference." The token "es" attends strongly to "there," correctly mapping the English "there is" construction to German "es gibt/ist." Later tokens show more diffuse attention, reflecting the model's difficulty with content words in longer sequences.](figures/attention_heatmap_3.png)

Here the first token "es" attends strongly to "\<sos\>" and "there" – correctly identifying the English "there is" construction that maps to German "es gibt/ist." Later tokens show more diffuse attention across multiple source positions, which is expected for a model that hasn't fully learned content word mappings.

### Attention Head Specialization

Perhaps the most interesting finding is that different attention heads learn different patterns. Figure 4 shows all four cross-attention heads for the same example in a 2x2 grid.

![Figure 4: Comparison of all four cross-attention heads for the same input sentence. Head 0 shows a roughly diagonal pattern indicating positional alignment. Head 1 displays broader, more diffuse attention. Head 2 concentrates on source-initial positions. Head 3 focuses on middle and end positions. This specialization confirms that multi-head attention learns diverse, complementary representations.](figures/attention_heads_comparison.png)

Head 0 shows a roughly diagonal pattern, focusing on positional alignment. Head 1 shows broader, more diffuse attention. Head 2 concentrates heavily on the beginning of the source sequence. Head 3 shows a pattern focused on the middle and end positions. This differentiation demonstrates that multi-head attention is not redundant – each head captures different aspects of the source-target relationship, exactly as theorized in the original paper [1].

### Summary of Learned Behaviors

**Learned successfully:**
- German sentence structure (subject-verb ordering for simple sentences)
- Pronoun alignment (English "we" → German "wir," "I" → "ich")
- Punctuation alignment (periods, question marks)
- Basic function words ("ist," "hat," "du," "nicht")
- Positional correspondence through attention

**Struggles with:**
- Content/open-class vocabulary (nouns, adjectives, specific verbs)
- German compound words and morphological variants
- Longer sentences (attention becomes diffuse)
- Avoiding repetitive generic phrases ("ein paar tage," "ist sehr gut")

## 7. Discussion

### Critique and Limitations

This experiment has several deliberate limitations. The model is extremely small (3.9M parameters vs. 65M+ for production systems). The dataset is tiny (16K training pairs vs. millions used in real NMT). Word-level tokenization with a simple regex creates an artificially large vocabulary without the subword compositionality that BPE or SentencePiece [6] provide. And training for only 10 epochs on CPU leaves substantial room for improvement.

The model clearly has not learned to produce fluent translations. However, the primary goal was not translation quality but rather understanding the attention mechanism, and in that regard the experiment succeeded: the attention heatmaps (Figures 2–4) demonstrate that meaningful alignment emerges from the training signal alone.

### Broader Implications and Next Steps

The Transformer architecture has rapidly become foundational not only in NLP but across scientific domains. In computational biology, Transformer-based models have been applied to protein structure prediction (AlphaFold2 [7]), genomic sequence modeling, and drug discovery. The attention mechanism's ability to capture long-range dependencies without sequential processing makes it particularly well-suited for biological sequences, where distant residues or nucleotides can have strong functional relationships.

Several concrete changes would improve this project's translation results:

1. **Subword tokenization** (BPE [8] or SentencePiece [6]) would reduce vocabulary size, handle unseen words through subword composition, and improve generalization.
2. **A larger model** ($d_{\text{model}} = 256$+, 4–6 layers) would increase capacity for learning complex lexical mappings.
3. **More training data** from the full 331K-pair Tatoeba corpus would expose the model to far more vocabulary and syntactic patterns.
4. **Learning rate warmup** and **label smoothing** ($\epsilon = 0.1$), as described in the original paper [1], would stabilize training and improve generalization.

Even a minimal Transformer with under 4 million parameters, trained on 16,000 sentence pairs for under 7 minutes, learns meaningful attention patterns that reflect real linguistic structure. The cross-attention heatmaps show that the model discovers source-target alignment without any explicit alignment supervision – this emergent alignment is one of the most powerful properties of the attention mechanism. The head specialization I observed (Figure 4) confirms that multi-head attention provides genuine representational diversity, not mere redundancy.

## 8. References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

[2] Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *Proceedings of ICLR 2015*.

[3] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. *Proceedings of ACL 2002*, 311–318.

[4] Tatoeba Project. https://tatoeba.org. Creative Commons Attribution 2.0 license.

[5] Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *Proceedings of ICLR 2015*.

[6] Kudo, T., & Richardson, J. (2018). SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing. *Proceedings of EMNLP 2018*, 66–71.

[7] Jumper, J., Evans, R., Pritzel, A., et al. (2021). Highly Accurate Protein Structure Prediction with AlphaFold. *Nature*, 596, 583–589.

[8] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of ACL 2016*, 1715–1725.
