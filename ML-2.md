# ML Fundamentals - Part 3: RNNs, Loss Functions, Evaluation Metrics & Advanced Topics

**Version 1.0 - Complete Interview-Ready Notes**

Sources: Ian Goodfellow's Deep Learning, Stanford CS224n, Andrej Karpathy, Colah's blog, Chip Huyen, Hao Hoang, Production systems from Google/Meta/OpenAI.

---

## 14.4 RNN AND SEQUENTIAL MODELS

### Why RNNs?

**Sequential Data Challenges**:
```
Fixed-size input (MLPs, CNNs):
- Image: 224×224×3 (fixed)
- Can't handle variable-length sequences

Sequential data:
- Text: "I love ML" (3 words) vs "Deep learning is amazing" (4 words)
- Speech: Variable duration
- Time series: Different lengths

Need: Model that processes sequences of any length
```

**Applications**:
```
Sequence-to-sequence:
- Machine translation: English → French
- Speech recognition: Audio → Text
- Summarization: Long document → Short summary

Sequence-to-one:
- Sentiment analysis: Review → Positive/Negative
- Video classification: Frames → Action label

One-to-sequence:
- Image captioning: Image → Description
- Music generation: Seed → Melody

Sequence labeling:
- Named Entity Recognition: Words → Tags (PER, ORG, LOC)
- Part-of-speech tagging: Words → POS tags
```

---

### Vanilla RNN

**Architecture**:
```
At each timestep t:
hₜ = tanh(Wₕₕhₜ₋₁ + Wₓₕxₜ + bₕ)  [hidden state update]
yₜ = Wₕᵧhₜ + bᵧ                   [output]

Where:
- hₜ: Hidden state at time t (memory)
- xₜ: Input at time t
- yₜ: Output at time t
- Wₕₕ: Hidden-to-hidden weights
- Wₓₕ: Input-to-hidden weights
- Wₕᵧ: Hidden-to-output weights
```

**Unrolled View** (3 timesteps):
```
x₁ → [RNN] → h₁ → y₁
       ↓
x₂ → [RNN] → h₂ → y₂
       ↓
x₃ → [RNN] → h₃ → y₃

Key: Same weights (Wₕₕ, Wₓₕ) used at each timestep (weight sharing)
```

**Backpropagation Through Time (BPTT)**:
```
Forward pass: Compute h₁, h₂, ..., hₜ
Backward pass: Compute gradients by unrolling

∂L/∂Wₕₕ = Σₜ ∂L/∂hₜ × ∂hₜ/∂Wₕₕ

Chain rule across time:
∂hₜ/∂hₜ₋₁ = tanh'(...)×Wₕₕ

Problem: Product of many terms
∂hₜ/∂h₁ = ∂hₜ/∂hₜ₋₁ × ∂hₜ₋₁/∂hₜ₋₂ × ... × ∂h₂/∂h₁
        = (tanh'×Wₕₕ)^(t-1)
```

---

### Vanishing Gradient Problem

**The Core Problem**:
```
Gradient: ∂L/∂h₁ = ∂L/∂hₜ × (∂hₜ/∂hₜ₋₁)^(t-1)
                  = ∂L/∂hₜ × (tanh'×Wₕₕ)^(t-1)

tanh': Range [0, 1] (typically < 1)
If tanh' = 0.25 and t = 100:
  Gradient ∝ (0.25)^99 ≈ 10⁻⁶⁰ → Effectively 0!

Result: RNN can't learn long-term dependencies (>10 timesteps)
```

**Why This Matters**:
```
Example sentence: "The cat, which was very fluffy, ... was sleeping."
- "cat" (position 1) and "was" (position 10) are related
- Gradient from "was" to "cat" ≈ 0 (vanished)
- RNN can't learn this dependency
```

**Symptoms**:
```
1. Loss plateaus early (not decreasing)
2. Model only remembers last 5-10 tokens
3. Gradients near zero in early layers (check grad norms)
4. Performance worse than simple baseline (bag-of-words)
```

**Solutions**:
```
1. LSTM/GRU (best solution) ✓
2. Gradient clipping (clips to [-1, 1]) - bandaid
3. Better initialization (orthogonal matrix) - helps slightly
4. Residual connections (skip connections) - helps slightly
5. Switch to Transformers (best for very long sequences) ✓
```

---

### LSTM (Long Short-Term Memory)

**Motivation** (Hochreiter & Schmidhuber, 1997):
```
Problem: Vanilla RNN can't remember long-term
Solution: Explicit memory cell with gating mechanisms
```

**Architecture**:
```
At each timestep t:

Gates:
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)  [Forget gate: What to forget from cell]
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)  [Input gate: What to add to cell]
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)  [Output gate: What to output from cell]

Cell update:
c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)     [Candidate cell state]
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ          [New cell state]
hₜ = oₜ ⊙ tanh(cₜ)                [Hidden state]

Where ⊙ = element-wise multiplication
```

**Intuition**:
```
Cell state cₜ: Information highway (carries info across time)
Forget gate fₜ: Decides what to remove from cell
Input gate iₜ: Decides what to add to cell
Output gate oₜ: Decides what to output

Example (remembering gender):
Input: "The cat, which was very fluffy and cute, was sleeping."
- c₀: Empty
- c₁: After "cat" → Store "singular, feminine"
- c₂-c₉: Keep "singular, feminine" (forget gate = 1)
- c₁₀: At "was" → Output "singular" (output gate = 1)
- Result: Correct agreement ("was" not "were")
```

**Why LSTM Solves Vanishing Gradient**:
```
Gradient flow through cell state:
∂cₜ/∂cₜ₋₁ = fₜ (forget gate, typically close to 1)

Uninterrupted gradient highway:
∂c₁₀₀/∂c₁ = f₁₀₀ × f₉₉ × ... × f₁
          ≈ 1 × 1 × ... × 1 = 1 (doesn't vanish!)

Result: Can learn dependencies across 100+ timesteps
```

**Parameters**:
```
Vanilla RNN: 2 weight matrices (Wₕₕ, Wₓₕ)
LSTM: 8 weight matrices (4 gates × 2 matrices each)

If hidden size = 512, input size = 256:
Vanilla RNN: (512×512 + 256×512) × 2 = 1M params
LSTM: (512×512 + 256×512) × 8 = 4M params

Trade-off: 4x more parameters but can learn long-term dependencies
```

---

### GRU (Gated Recurrent Unit)

**Motivation** (Cho et al., 2014):
```
Problem: LSTM complex (4 gates), many parameters
Solution: Simplified variant with 2 gates (fewer parameters)
```

**Architecture**:
```
At each timestep t:

Gates:
rₜ = σ(Wr·[hₜ₋₁, xₜ] + br)  [Reset gate: What to forget from hₜ₋₁]
zₜ = σ(Wz·[hₜ₋₁, xₜ] + bz)  [Update gate: How much to update]

Hidden update:
h̃ₜ = tanh(Wh·[rₜ ⊙ hₜ₋₁, xₜ] + bh)  [Candidate hidden state]
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ      [New hidden state]
```

**Key Differences from LSTM**:
```
LSTM: Separate cell state (cₜ) and hidden state (hₜ)
GRU: Only hidden state (hₜ)

LSTM: 4 gates (forget, input, output, cell)
GRU: 2 gates (reset, update)

LSTM: 4M params (example)
GRU: 3M params (25% fewer)

Performance:
- LSTM: Slightly better on tasks requiring long memory (>100 steps)
- GRU: Faster training (fewer params), similar performance (50-100 steps)
```

**When to Use**:
```
LSTM:
✓ Long sequences (>100 timesteps)
✓ Need explicit memory cell
✓ Have enough data and compute

GRU:
✓ Medium sequences (20-100 timesteps)
✓ Faster training desired
✓ Limited data (fewer params → less overfitting)

Production:
- Speech recognition: LSTM (long audio sequences)
- Machine translation: GRU or LSTM (similar performance)
- Sentiment analysis: GRU (short sequences, faster)
```

---

### Bidirectional RNNs

**Motivation**:
```
Problem: Forward RNN only sees past context
Example: "I love this __"
- Forward: Sees "I love this"
- Can't see: "movie" (comes after blank)

Solution: Process sequence in both directions
```

**Architecture**:
```
Forward RNN: h⃗ₜ = RNN(h⃗ₜ₋₁, xₜ)
Backward RNN: h⃖ₜ = RNN(h⃖ₜ₊₁, xₜ)
Output: yₜ = f([h⃗ₜ, h⃖ₜ])

At position t:
- h⃗ₜ: Context from positions 1...t (past)
- h⃖ₜ: Context from positions t...n (future)
- yₜ: Combines past and future
```

**Use Cases**:
```
✓ Sequence labeling:
  - Named Entity Recognition (NER)
  - Part-of-speech tagging
  - Protein structure prediction
  
✓ Fill-in-the-blank:
  - BERT-style masked language modeling
  
✗ Real-time generation:
  - Language modeling (can't see future)
  - Real-time speech recognition (must process online)
  
✗ Autoregressive tasks:
  - Text generation (GPT-style)
  - Machine translation decoder
```

**Production Note**:
```
BERT uses bidirectional LSTM:
- Encoder: Bidirectional (sees both directions)
- MLM objective: Predict masked words using context

GPT uses unidirectional LSTM:
- Decoder: Forward only (autoregressive)
- LM objective: Predict next word
```

---

### Sequence-to-Sequence Models

**Architecture** (Sutskever et al., 2014):
```
Encoder → Context Vector → Decoder

Encoder (LSTM):
x₁, x₂, ..., xₙ → h₁, h₂, ..., hₙ
Context vector: c = hₙ (final hidden state)

Decoder (LSTM):
c → h'₁ → y₁ → h'₂ → y₂ → ... → y'ₘ

Example (Translation):
Encoder: "I love cats" → c
Decoder: c → "J'aime les chats"
```

**Problem: Bottleneck**:
```
Issue: Single vector c must encode entire input
- Short input: OK (c = 512 dims)
- Long input: Information loss (c still 512 dims!)

Example:
Input: 100-word sentence
Context: 512 dimensions
Compression ratio: 100 words / 512 dims = 0.2 words/dim

Result: Poor translation quality for long sentences
```

---

### Attention Mechanisms

**Motivation** (Bahdanau et al., 2015):
```
Problem: Fixed-size context vector bottleneck
Solution: Decoder attends to all encoder hidden states (not just last)
```

**Bahdanau Attention (Additive)**:
```
At each decoder timestep t:

1. Compute attention scores:
   eₜⱼ = vᵀ tanh(Ws·sₜ₋₁ + Wh·hⱼ)
   where sₜ₋₁ = decoder state, hⱼ = encoder state j

2. Normalize (softmax):
   αₜⱼ = exp(eₜⱼ) / Σₖ exp(eₜₖ)

3. Context vector (weighted sum):
   cₜ = Σⱼ αₜⱼ·hⱼ

4. Decoder update:
   sₜ = LSTM(sₜ₋₁, [yₜ₋₁, cₜ])
```

**Intuition**:
```
Translation: "I love cats" → "J'aime les chats"

Decoder at t=2 (generating "les"):
- αₜ,₁ = 0.1 ("I")     → Low attention
- αₜ,₂ = 0.2 ("love")  → Low attention
- αₜ,₃ = 0.7 ("cats")  → High attention! ✓

Result: "les" focuses on "cats" (learns alignment)
```

**Luong Attention (Multiplicative)**:
```
1. Score: eₜⱼ = sₜᵀWₐhⱼ (simpler than Bahdanau)
2. Normalize: αₜⱼ = softmax(eₜⱼ)
3. Context: cₜ = Σⱼ αₜⱼ·hⱼ
4. Output: ỹₜ = tanh(Wc[sₜ, cₜ])

Variants:
- Dot: eₜⱼ = sₜᵀhⱼ (if dim(sₜ) = dim(hⱼ))
- General: eₜⱼ = sₜᵀWₐhⱼ (learnable Wₐ)
- Concat: eₜⱼ = vᵀ tanh(W[sₜ, hⱼ]) (Bahdanau-style)
```

**Impact**:
```
Before Attention:
- WMT'14 EN→FR: BLEU 27 (Sutskever et al., 2014)

After Attention:
- WMT'14 EN→FR: BLEU 34 (Bahdanau et al., 2015)
- +7 BLEU points (huge improvement!)

Why:
- No bottleneck (decoder accesses all encoder states)
- Learns soft alignment (which input words → which output words)
- Interpretable (can visualize attention weights)
```

**Attention Visualization**:
```
Input: "The cat sat on the mat"
Output: "Le chat s'est assis sur le tapis"

Attention matrix (7×7):
          The  cat  sat  on  the  mat
Le        0.8  0.1  0.0  0.0  0.1  0.0
chat      0.1  0.8  0.0  0.0  0.1  0.0
s'est     0.0  0.0  0.8  0.0  0.0  0.2
assis     0.0  0.0  0.8  0.0  0.0  0.2
sur       0.0  0.0  0.0  0.8  0.0  0.2
le        0.0  0.0  0.0  0.2  0.6  0.2
tapis     0.0  0.0  0.0  0.0  0.2  0.8

Observations:
- Diagonal pattern (mostly monotonic alignment)
- "Le" attends to "The" (0.8) → Correct article translation
- "tapis" attends to "mat" (0.8) → Correct word translation
```

---

### Self-Attention (Transformer Precursor)

**Motivation**:
```
Problem: RNN processes sequentially (can't parallelize)
Solution: Attend to all positions simultaneously (parallelizable)
```

**Operation**:
```
For each position i:
1. Compute attention to all positions j:
   eᵢⱼ = (Wqxᵢ)ᵀ(Wkxⱼ) / √dk

2. Normalize:
   αᵢⱼ = softmax(eᵢⱼ)

3. Output:
   yᵢ = Σⱼ αᵢⱼ(Wvxⱼ)

Intuition: Each word attends to all other words (including itself)
```

**Why This Matters**:
```
RNN: Sequential (can't parallelize)
Self-Attention: Parallel (compute all positions simultaneously)

Speed: Self-attention 10-100x faster (on GPU)

This led to: Transformers (Vaswani et al., 2017)
- "Attention is All You Need"
- Self-attention only (no RNN!)
- State-of-the-art for NLP (BERT, GPT, T5, etc.)
```

---

## 14.5 LOSS FUNCTIONS

### Cross-Entropy Loss

**Binary Cross-Entropy (BCE)**:
```
For binary classification:
L = -(y log(ŷ) + (1-y) log(1-ŷ))

Where:
- y ∈ {0, 1}: True label
- ŷ ∈ (0, 1): Predicted probability

Example:
y=1 (positive class), ŷ=0.9: L = -log(0.9) = 0.11
y=1, ŷ=0.1: L = -log(0.1) = 2.30 (high penalty!)
y=0, ŷ=0.1: L = -log(0.9) = 0.11
```

**Why Cross-Entropy (Not MSE)?**:
```
MSE: L = (y - ŷ)²
- For y=1, ŷ=0.1: L = 0.81
- For y=1, ŷ=0.01: L = 0.98 (only slightly worse!)

BCE: L = -y log(ŷ)
- For y=1, ŷ=0.1: L = 2.30
- For y=1, ŷ=0.01: L = 4.61 (2x worse!)

BCE penalizes confident wrong predictions more heavily
```

**Categorical Cross-Entropy (Multi-class)**:
```
For K classes:
L = -Σₖ yₖ log(ŷₖ)

Where:
- y: One-hot vector (e.g., [0, 1, 0, 0] for class 2)
- ŷ: Predicted probabilities (softmax output)

Example (4 classes):
y = [0, 1, 0, 0] (true class = 2)
ŷ = [0.1, 0.7, 0.1, 0.1] (predicted)
L = -log(0.7) = 0.36

Good prediction: ŷ = [0.05, 0.85, 0.05, 0.05]
L = -log(0.85) = 0.16 (lower loss ✓)

Bad prediction: ŷ = [0.4, 0.3, 0.2, 0.1]
L = -log(0.3) = 1.20 (higher loss ✓)
```

**Numerical Stability**:
```
Problem: log(0) = -∞ (if ŷ=0, loss = inf)

Solution: Clip predictions
ŷ = clip(ŷ, ε, 1-ε)
where ε = 1e-7

In code:
loss = -np.log(np.clip(y_pred, 1e-7, 1-1e-7))

PyTorch/TensorFlow handle this automatically in BCELoss/CrossEntropyLoss
```

**Production Notes**:
```
PyTorch:
- nn.BCELoss(): For binary (expects sigmoid output)
- nn.BCEWithLogitsLoss(): For binary (combines sigmoid + BCE, more stable)
- nn.CrossEntropyLoss(): For multi-class (combines softmax + CE, more stable)

Always use WithLogits version (more numerically stable)
```

---

### Mean Squared Error (MSE)

**Definition**:
```
L = (1/N) Σᵢ (yᵢ - ŷᵢ)²

Where:
- yᵢ: True value
- ŷᵢ: Predicted value
- N: Number of samples
```

**Use Cases**:
```
✓ Regression (continuous output):
  - House price prediction
  - Temperature forecasting
  - Stock price prediction

✗ Classification:
  - Poor gradient properties (flat gradients for wrong predictions)
  - Use cross-entropy instead
```

**Properties**:
```
+ Smooth, differentiable everywhere
+ Penalizes large errors heavily (squared)
+ Easy to optimize (convex for linear models)

- Sensitive to outliers (squared term amplifies)
- Units: Squared units (hard to interpret)
```

**Gradient**:
```
∂L/∂ŷ = (2/N) Σᵢ (ŷᵢ - yᵢ)

For neural network:
∂L/∂θ = (2/N) Σᵢ (ŷᵢ - yᵢ) × ∂ŷᵢ/∂θ
```

---

### Mean Absolute Error (MAE)

**Definition**:
```
L = (1/N) Σᵢ |yᵢ - ŷᵢ|
```

**MSE vs MAE**:
```
| Metric | Formula     | Gradient   | Outlier Sensitivity | Units      |
|--------|-------------|------------|---------------------|------------|
| MSE    | (y - ŷ)²    | 2(ŷ - y)   | High (squared)      | Squared    |
| MAE    | |y - ŷ|     | sign(ŷ-y)  | Low (linear)        | Original   |

MSE: Penalizes large errors heavily (good if outliers are errors)
MAE: Treats all errors equally (good if outliers are valid)
```

**When to Use**:
```
MSE:
✓ Outliers are errors (should be penalized)
✓ Want smooth gradients
✓ Standard regression problems

MAE:
✓ Outliers are valid (don't want to over-penalize)
✓ Want robust estimator (median vs mean)
✓ Errors are interpretable (same units as target)

Production:
- Most regression: MSE (default)
- Robust regression (outliers): MAE or Huber
```

---

### Contrastive Loss

**Definition** (Hadsell et al., 2006):
```
L = (1/N) Σᵢ [ y·d² + (1-y)·max(0, m - d)² ]

Where:
- d: Distance between embeddings (e.g., Euclidean)
- y: 1 if similar pair, 0 if dissimilar
- m: Margin (typically 1.0)

Intuition:
- Similar pairs (y=1): Minimize distance d
- Dissimilar pairs (y=0): Push apart by at least margin m
```

**Use Cases**:
```
✓ Face verification (same person vs different person)
✓ Signature verification (genuine vs forged)
✓ Siamese networks (similarity learning)
✓ Metric learning
```

**Example**:
```
Face recognition:
- Input: Two face images
- Embedding: f(img) ∈ ℝ¹²⁸ (e.g., FaceNet)
- Distance: d = ||f(img1) - f(img2)||₂

Training pairs:
- Same person: (img1, img2, y=1) → Minimize d
- Different people: (img1, img3, y=0) → Maximize d (push to m)

Inference:
- If d < threshold (e.g., 0.5): Same person
- If d ≥ threshold: Different people
```

---

### Triplet Loss

**Definition** (Schroff et al., 2015):
```
L = max(0, d(a, p) - d(a, n) + m)

Where:
- a: Anchor (reference sample)
- p: Positive (same class as anchor)
- n: Negative (different class)
- d: Distance function (Euclidean, cosine)
- m: Margin (typically 0.2 to 1.0)

Intuition:
- d(a, p): Should be small (anchor close to positive)
- d(a, n): Should be large (anchor far from negative)
- Margin: d(a, n) - d(a, p) > m
```

**Why Triplet Loss?**:
```
Contrastive: Pairs (a, p) and (a, n) trained separately
Triplet: Relative comparison (a, p, n) in one loss

Advantage: More efficient learning
- Directly learns relative distances
- Fewer training iterations
```

**Hard Negative Mining**:
```
Problem: Random negatives are too easy (d(a, n) >> d(a, p))
- Loss = 0 (no learning!)

Solution: Hard negative mining
- Hard negative: n such that d(a, n) ≈ d(a, p) (difficult to distinguish)
- Mine hard negatives within mini-batch (online mining)

Example:
Anchor: "Dog" image
Easy negative: "Car" image (clearly different)
Hard negative: "Cat" image (similar to dog)
→ Use hard negative for effective learning
```

**Production (FaceNet)**:
```
FaceNet (Google, 2015):
- 128-dim face embeddings
- Trained with triplet loss
- Result: 99.6% accuracy on LFW dataset

Training:
- 200M face images
- Hard negative mining (crucial for convergence)
- Batch size: 1800 (many triplets per batch)
```

---

### Focal Loss

**Motivation** (Lin et al., 2017):
```
Problem: Class imbalance (e.g., 99% background, 1% object)
- Standard CE: Focuses equally on easy and hard examples
- Easy examples (correct predictions) dominate loss
- Hard examples (incorrect) contribute little

Solution: Down-weight easy examples, focus on hard examples
```

**Definition**:
```
FL = -αₜ(1 - pₜ)^γ log(pₜ)

Where:
- pₜ = p if y=1, else 1-p (predicted probability for true class)
- αₜ: Class balance weight (typically 0.25 for rare class)
- γ: Focusing parameter (typically 2)

Compared to CE:
CE = -αₜ log(pₜ)
```

**How It Works**:
```
Easy example (pₜ = 0.9):
- CE: -log(0.9) = 0.11
- FL: -(1-0.9)² × log(0.9) = -0.01 × 0.11 = 0.001 (100x lower!)

Hard example (pₜ = 0.5):
- CE: -log(0.5) = 0.69
- FL: -(1-0.5)² × log(0.5) = -0.25 × 0.69 = 0.17 (only 4x lower)

Result: Model focuses on hard examples
```

**Use Cases**:
```
✓ Object detection (RetinaNet):
  - 99% background, 1% object
  - Focal loss: 1-2% mAP improvement

✓ Any class imbalance problem:
  - Fraud detection (99% legit, 1% fraud)
  - Anomaly detection
  - Rare disease diagnosis

✗ Balanced datasets (no benefit, use CE)
```

---

### Huber Loss

**Motivation**:
```
MSE: Sensitive to outliers (squared term)
MAE: Not smooth at 0 (gradient discontinuity)
Huber: Best of both (smooth + robust)
```

**Definition**:
```
L = { 0.5(y - ŷ)²         if |y - ŷ| ≤ δ
    { δ|y - ŷ| - 0.5δ²    otherwise

Where δ = threshold (typically 1.0)

Intuition:
- Small errors (|y - ŷ| < δ): Quadratic (MSE-like)
- Large errors (|y - ŷ| > δ): Linear (MAE-like)
```

**Properties**:
```
+ Smooth everywhere (differentiable)
+ Robust to outliers (linear for large errors)
+ Combines MSE and MAE benefits

Use when:
✓ Have outliers (but want smooth gradients)
✓ Regression with noisy data
✓ Reinforcement learning (Q-learning)
```

---

### Custom Loss Design

**Multi-Task Learning**:
```
L_total = λ₁L₁ + λ₂L₂ + λ₃L₃

Example (Autonomous Driving):
L = λ₁×L_detection + λ₂×L_segmentation + λ₃×L_depth

Challenges:
- Balancing weights (λ₁, λ₂, λ₃)
- Scales differ (L₁ ∈ [0, 10], L₂ ∈ [0, 1000])

Solutions:
1. Normalize losses (divide by initial loss)
2. Uncertainty weighting (Kendall et al., 2018)
3. GradNorm (Chen et al., 2018)
```

**Weighted Loss**:
```
L = Σᵢ wᵢ × loss(yᵢ, ŷᵢ)

Example (Class Imbalance):
- Class 0 (99%): w₀ = 0.01
- Class 1 (1%): w₁ = 0.99

Or inversely proportional:
wᵢ = 1 / (frequency of class i)
```

**Regularized Loss**:
```
L = L_data + λ×L_reg

L_data: Data fitting term (CE, MSE, etc.)
L_reg: Regularization term (L1, L2, etc.)

Example:
L = CrossEntropy(y, ŷ) + λ×||θ||₂²
```

---

## 14.6 EVALUATION METRICS

### Classification Metrics

**Confusion Matrix**:
```
                Predicted
                Pos    Neg
Actual  Pos    TP     FN
        Neg    FP     TN

TP: True Positive (correctly predicted positive)
FP: False Positive (incorrectly predicted positive)
TN: True Negative (correctly predicted negative)
FN: False Negative (incorrectly predicted negative)
```

**Accuracy**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Use when:
✓ Balanced classes (50-50 split)
✓ All errors equally important

Don't use when:
✗ Imbalanced classes (e.g., 99-1 split)

Example:
Dataset: 990 negative, 10 positive
Predict all negative: Accuracy = 990/1000 = 99% (but useless!)
```

**Precision**:
```
Precision = TP / (TP + FP)
"Of all predicted positives, how many are actually positive?"

High precision: Few false alarms
Example: Spam filter (don't want to flag legit emails)
```

**Recall (Sensitivity)**:
```
Recall = TP / (TP + FN)
"Of all actual positives, how many did we find?"

High recall: Don't miss positives
Example: Cancer screening (don't want to miss cancer)
```

**F1-Score**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Harmonic mean of precision and recall

Use when:
✓ Need balance between precision and recall
✓ Class imbalance (better than accuracy)

Properties:
- F1 ∈ [0, 1] (higher is better)
- Harmonic mean emphasizes lower value
  - P=0.9, R=0.9: F1 = 0.9
  - P=0.9, R=0.1: F1 = 0.18 (penalizes imbalance)
```

**F-Beta Score**:
```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)

β < 1: Emphasize precision (e.g., β=0.5)
β = 1: F1 score (equal weight)
β > 1: Emphasize recall (e.g., β=2)

Example:
Cancer screening: F2 (emphasize recall, don't miss cancer)
Spam filter: F0.5 (emphasize precision, don't block legit emails)
```

---

### ROC and AUC

**ROC Curve** (Receiver Operating Characteristic):
```
Plot: FPR (x-axis) vs TPR (y-axis)

TPR (True Positive Rate) = Recall = TP / (TP + FN)
FPR (False Positive Rate) = FP / (FP + TN)

For each threshold t:
1. Predict positive if p(x) > t
2. Compute TPR and FPR
3. Plot point (FPR, TPR)

Perfect classifier: (0, 1) [top-left corner]
Random classifier: Diagonal line (TPR = FPR)
```

**AUC (Area Under Curve)**:
```
AUC ∈ [0, 1]

Interpretation:
- AUC = 0.5: Random (coin flip)
- AUC = 0.7: Acceptable
- AUC = 0.8: Good
- AUC = 0.9: Excellent
- AUC = 1.0: Perfect

Probabilistic interpretation:
AUC = P(score(positive) > score(negative))
```

**When to Use**:
```
✓ Comparing models (higher AUC = better)
✓ Threshold-independent metric
✓ Imbalanced classes (better than accuracy)

Limitations:
- Doesn't tell optimal threshold
- Optimistic for imbalanced data (use PR-AUC instead)
```

---

### Precision-Recall Curve

**PR Curve**:
```
Plot: Recall (x-axis) vs Precision (y-axis)

For each threshold t:
1. Predict positive if p(x) > t
2. Compute precision and recall
3. Plot point (Recall, Precision)

Perfect classifier: (1, 1) [top-right corner]
Baseline: Horizontal line at y = (# positives) / (# total)
```

**PR-AUC vs ROC-AUC**:
```
Balanced data (50-50):
- ROC-AUC and PR-AUC similar

Imbalanced data (99-1):
- ROC-AUC: Optimistic (looks better than it is)
  - TN dominates, high TNR → high AUC
- PR-AUC: Realistic (focuses on positives)
  - Precision directly affected by FP

Recommendation:
✓ Imbalanced data: Use PR-AUC
✓ Balanced data: Use ROC-AUC (more common)
```

---

### Regression Metrics

**MAE (Mean Absolute Error)**:
```
MAE = (1/N) Σ |yᵢ - ŷᵢ|

Properties:
+ Same units as target (interpretable)
+ Robust to outliers
- Doesn't penalize large errors heavily

Example:
Actual: [10, 20, 30]
Predicted: [12, 18, 35]
MAE = (|10-12| + |20-18| + |30-35|) / 3 = (2 + 2 + 5) / 3 = 3.0
```

**RMSE (Root Mean Squared Error)**:
```
RMSE = √( (1/N) Σ (yᵢ - ŷᵢ)² )

Properties:
+ Penalizes large errors heavily (squared)
+ Differentiable (smooth gradients)
- Sensitive to outliers
- Units: Same as target (because of square root)

Example:
Actual: [10, 20, 30]
Predicted: [12, 18, 35]
MSE = ((10-12)² + (20-18)² + (30-35)²) / 3 = (4 + 4 + 25) / 3 = 11.0
RMSE = √11.0 = 3.32
```

**R² (Coefficient of Determination)**:
```
R² = 1 - (SS_res / SS_tot)

Where:
SS_res = Σ (yᵢ - ŷᵢ)² (residual sum of squares)
SS_tot = Σ (yᵢ - ȳ)²  (total sum of squares)

Interpretation:
- R² = 1: Perfect predictions
- R² = 0: Model no better than predicting mean
- R² < 0: Model worse than predicting mean!

Range: (-∞, 1]
Typical: R² > 0.7 is good
```

**MAPE (Mean Absolute Percentage Error)**:
```
MAPE = (100/N) Σ |yᵢ - ŷᵢ| / |yᵢ|

Properties:
+ Scale-independent (percentage)
+ Interpretable (e.g., "10% error")
- Undefined if yᵢ = 0
- Asymmetric (penalizes over-predictions more)

Example:
Actual: [10, 20, 30]
Predicted: [12, 18, 35]
MAPE = (|10-12|/10 + |20-18|/20 + |30-35|/30) × 100 / 3
     = (0.2 + 0.1 + 0.167) × 100 / 3 = 15.6%
```

---

### Ranking Metrics

**NDCG (Normalized Discounted Cumulative Gain)**:
```
DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)

NDCG@k = DCG@k / IDCG@k

Where:
- relᵢ: Relevance score at position i (e.g., 0-5)
- IDCG: DCG of ideal ranking (sorted by relevance)

Properties:
+ Position-aware (higher positions more important)
+ Graded relevance (not binary)
+ Normalized (comparable across queries)

Example:
Ranking: [5, 3, 2, 0, 1] (relevances)
DCG@5 = (2^5-1)/log₂(2) + (2^3-1)/log₂(3) + ... = 31 + 4.42 + 1.5 + 0 + 0.39 = 37.31

Ideal: [5, 3, 2, 1, 0]
IDCG@5 = 31 + 4.42 + 1.5 + 0.43 + 0 = 37.35

NDCG@5 = 37.31 / 37.35 = 0.999 (near-perfect!)
```

**MAP (Mean Average Precision)**:
```
AP = (1/R) Σᵣ₌₁ᴿ Precision@r

Where:
- R: Total number of relevant items
- r: Position of each relevant item

MAP = (1/Q) Σ APq (average over all queries)

Example:
Results: [R, N, R, N, R, N] (R=relevant, N=not)
Relevant positions: 1, 3, 5
Precision@1 = 1/1 = 1.0
Precision@3 = 2/3 = 0.67
Precision@5 = 3/5 = 0.60
AP = (1.0 + 0.67 + 0.60) / 3 = 0.76
```

**MRR (Mean Reciprocal Rank)**:
```
RR = 1 / rank of first relevant item

MRR = (1/Q) Σ RRq

Example:
Query 1: First relevant at position 1 → RR = 1.0
Query 2: First relevant at position 3 → RR = 0.33
Query 3: First relevant at position 2 → RR = 0.50
MRR = (1.0 + 0.33 + 0.50) / 3 = 0.61

Use case:
✓ Question answering (only care about first answer)
✓ Search (user rarely goes past first page)
```

---

## 14.7 OVERFITTING AND REGULARIZATION

### Detection and Diagnosis

**Learning Curves**:
```
Plot: Training loss and validation loss vs epochs

Underfitting:
- Train loss: High (not decreasing)
- Val loss: High (similar to train)
- Solution: Increase model complexity

Overfitting:
- Train loss: Low (decreasing)
- Val loss: High (increasing or plateau)
- Gap: Train-val > 10%
- Solution: Regularization, more data

Good fit:
- Train loss: Low
- Val loss: Low (close to train, gap < 5%)
- Both decreasing until plateau
```

**Validation Strategies**:
```
Train/Val/Test Split:
- Train: 60-80%
- Val: 10-20% (hyperparameter tuning)
- Test: 10-20% (final evaluation, never touched during development)

K-Fold Cross-Validation:
1. Split data into K folds (typically K=5 or 10)
2. For each fold k:
   - Train on K-1 folds
   - Validate on fold k
3. Average results

Stratified K-Fold:
- Maintain class distribution in each fold
- Critical for imbalanced data

Time Series CV:
- Don't shuffle (preserves temporal order)
- Rolling window or expanding window
```

---

### Regularization Techniques (Summary)

**L1 Regularization**:
```
Loss = L_data + λ Σ |θᵢ|

Effect:
- Sparse weights (many exactly 0)
- Feature selection
- Less smooth (non-differentiable at 0)

Use when:
✓ Want interpretability (few important features)
✓ High-dimensional data (feature selection)
```

**L2 Regularization**:
```
Loss = L_data + λ Σ θᵢ²

Effect:
- Small weights (close to 0)
- Smooth (differentiable)
- All features retained

Use when:
✓ Standard regularization (default)
✓ All features potentially useful
```

**Elastic Net (L1 + L2)**:
```
Loss = L_data + λ₁ Σ |θᵢ| + λ₂ Σ θᵢ²

Effect:
- Combines L1 (sparsity) and L2 (smoothness)

Use when:
✓ Want feature selection + stability
✓ Correlated features
```

**Early Stopping**:
```
Algorithm:
1. Monitor validation loss
2. If no improvement for N epochs (patience):
   - Stop training
   - Restore best checkpoint
3. Typical patience: 5-10 epochs

Benefits:
+ Simple, effective
+ No hyperparameters (except patience)
+ Always use in production
```

**Dropout**:
```
Training: Randomly set neurons to 0 (p=0.5 typical)
Inference: Use all neurons (no dropout)

Benefits:
+ Reduces overfitting (20-50% improvement)
+ Approximate ensemble

Limitations:
- Slows training (~2x longer)
- Adds hyperparameter (dropout rate)

Production:
- Use in FC layers (MLP, LSTM)
- Less common in Conv layers
```

**Data Augmentation**:
```
Image:
- Geometric: Flip, rotate, crop, zoom
- Color: Brightness, contrast, saturation
- Advanced: Cutout, mixup, cutmix

Text:
- Synonym replacement
- Back-translation
- Random insertion/deletion

Benefits:
+ Artificially increases dataset size
+ Reduces overfitting significantly
+ No additional compute at inference
```

**Ensemble Methods**:
```
Train multiple models, average predictions:

1. Bagging (Bootstrap Aggregating):
   - Train on different data subsets
   - Random Forest (decision trees)

2. Boosting:
   - Sequential training (correct previous errors)
   - XGBoost, LightGBM

3. Model averaging:
   - Train same architecture with different initializations
   - Average predictions (regression) or vote (classification)

Benefits:
+ Reduces variance (more stable predictions)
+ Often 1-3% accuracy improvement

Limitations:
- N models → N× inference time
- N× memory
```

---

### Bias-Variance Trade-off

**Decomposition**:
```
Error = Bias² + Variance + Irreducible Error

Bias: How far predictions are from truth (on average)
Variance: How much predictions vary (for different training sets)
Irreducible error: Noise in data (can't reduce)
```

**High Bias (Underfitting)**:
```
Symptoms:
- Low training accuracy (<80%)
- Low validation accuracy (similar to train)
- Simple model (too few parameters)

Solutions:
1. Increase model complexity (more layers, more neurons)
2. Add features
3. Reduce regularization (lower λ)
4. Train longer (more epochs)
```

**High Variance (Overfitting)**:
```
Symptoms:
- High training accuracy (>95%)
- Low validation accuracy (<80%)
- Large train-val gap (>10%)

Solutions:
1. Get more data (best solution)
2. Regularization (L2, dropout)
3. Early stopping
4. Reduce model complexity
5. Data augmentation
6. Ensemble methods
```

**Optimal Trade-off**:
```
Goal: Low bias AND low variance

Approach:
1. Start simple (high bias)
2. Increase complexity until training error low (reduce bias)
3. Add regularization until validation error low (reduce variance)
4. Iterate

Production:
- Bias-variance decomposition not commonly computed
- Use validation curves (practical proxy)
```

---

## Interview Questions (Hao Hoang Style)

### Question 11: RNN Vanishing Gradient (CRITICAL)
**Q**: "You're training an RNN on sequences of length 100. After 50 epochs, validation loss plateaus at 2.5 (not improving). Training loss is also 2.5. What's likely wrong?"

**Expected Answer**:
- **Diagnosis**: Not overfitting (train = val). Likely **vanishing gradient** (underfitting).
  - Gradient: (tanh')^100 ≈ 0 (can't learn long dependencies)
  - Model stuck, can't learn beyond first 5-10 timesteps

- **Solutions**:
  1. **Switch to LSTM/GRU** (best fix):
     - Gradient highway through cell state
     - Can learn 100+ step dependencies
     - Expected: Loss → 1.5 (significant improvement)
  2. **Gradient clipping** (bandaid):
     - Clip gradients to [-1, 1]
     - Helps exploding but not vanishing
  3. **Better initialization** (minor help):
     - Orthogonal initialization
     - Slightly helps gradient flow

**Red flag**: Suggesting learning rate changes (won't fix vanishing gradient)

**Follow-up**: "After switching to LSTM, training loss is 0.5 but validation is 2.0. Now what?"
- **New problem**: Overfitting (train-val gap = 1.5)
- **Solutions**:
  - Dropout (p=0.3-0.5) in LSTM layers
  - L2 regularization
  - More data or data augmentation
  - Early stopping

---

### Question 12: Loss Function Selection
**Q**: "You're building a face recognition system. Input: Two face images. Output: Same person or different? Which loss function do you use?"

**Expected Answer**:
- **Not binary cross-entropy** (doesn't learn embeddings)
- **Use Triplet Loss or Contrastive Loss**

**Best choice: Triplet Loss**:
```
Training:
- Anchor: Person A image 1
- Positive: Person A image 2
- Negative: Person B image 1
- Loss: max(0, d(a,p) - d(a,n) + margin)

Inference:
- Compute embeddings: f(img1), f(img2)
- Distance: d = ||f(img1) - f(img2)||
- If d < threshold: Same person
```

**Why not BCE?**:
- BCE trains binary classifier (same/different)
- Doesn't generalize to new people (not seen in training)
- Triplet loss learns embeddings (generalizes to unseen people)

**Production (FaceNet)**:
- 128-dim embeddings
- Triplet loss with hard negative mining
- 99.6% accuracy on LFW

**Follow-up**: "How do you select the margin?"
- Typical: margin = 0.2 to 1.0
- Grid search on validation set
- Too small: Underfitting (all embeddings collapse)
- Too large: Slow convergence (hard to satisfy)

---

### Question 13: Evaluation Metric Choice
**Q**: "Your fraud detection model gets 99% accuracy but management says it's useless. Dataset: 99% legitimate, 1% fraud. What's wrong and what metric should you use?"

**Expected Answer**:
- **Problem**: Accuracy misleading with imbalance
  - Predict all "legitimate" → 99% accuracy but 0% fraud caught!

- **Better metrics**:
  1. **Precision-Recall**:
     - Precision: Of flagged fraud, how many are real?
     - Recall: Of real fraud, how many did we catch?
  2. **F1-Score**: Balance precision and recall
  3. **PR-AUC**: Area under precision-recall curve (threshold-independent)
  4. **Cost-based**:
     - False positive cost (block legit transaction): $0
     - False negative cost (miss fraud): $100
     - Minimize expected cost

- **Never use**:
  - Accuracy (misleading)
  - ROC-AUC (too optimistic for imbalanced data)

**Business metric**:
- Fraud loss prevented: $X saved
- False positive cost: $Y lost (customer friction)
- Net benefit: $X - $Y

**Follow-up**: "After fixing, precision is 80% and recall is 60%. Business wants recall > 90%. What do you do?"
- **Lower decision threshold**:
  - Current: Flag if P(fraud) > 0.8
  - New: Flag if P(fraud) > 0.4
  - Result: Higher recall (90%), lower precision (50%)
- **Trade-off**: More false positives (acceptable if fraud cost high)

---

### Question 14: Overfitting Diagnosis
**Q**: "Your model: Training accuracy 98%, Validation accuracy 85%, Test accuracy 82%. What's happening and what do you do?"

**Expected Answer**:
- **Analysis**:
  - Train-Val gap: 13% → Overfitting (to training set)
  - Val-Test gap: 3% → Slight validation set overfitting (hyperparameter tuning)
  
- **Primary issue**: Overfitting to training set (13% gap)

- **Solutions (in order)**:
  1. **Regularization**:
     - L2 (weight decay): λ = 1e-4
     - Dropout: p = 0.3-0.5
     - Expected: Val → 90%, gap → 8%
  2. **Early stopping**:
     - Stop when val loss plateaus (patience=5)
     - Prevents over-training
  3. **More data**:
     - Double dataset (if possible)
     - Expected: Val → 92%, Test → 90%
  4. **Data augmentation**:
     - Artificially increase data
     - Expected: Val → 93%

- **Secondary issue**: Val-Test gap (3%)
  - Caused by: Hyperparameter tuning on validation set
  - Solution: Use cross-validation (K-fold) for hyperparameter tuning
  - Or: Keep test set truly held-out (no decisions based on test)

**Red flag**: Not distinguishing between overfitting (train-val) and validation leakage (val-test)

**Follow-up**: "After regularization, all three are 88%. Good enough?"
- **Depends on**:
  - Application: Medical (need >95%), Recommendation (88% OK)
  - Baseline: Random (50%), Previous model (85%) → 88% is improvement
  - Business impact: Does 88% meet requirements?
- **Don't over-optimize**: Diminishing returns after 90%

---

### Question 15: Loss Function Computation
**Q**: "Your multi-class classification model (10 classes) on 1000 samples. Estimate FLOPs for loss computation (forward pass of loss function only)."

**Expected Answer**:
- **Softmax**: O(K) per sample
  - K=10 classes
  - 1000 samples
  - FLOPs: 10K (negligible compared to model forward pass)

- **Cross-entropy**: O(K) per sample
  - FLOPs: 10K

- **Total loss computation**: ~20K FLOPs
  - Compare to model: 1M-1B FLOPs
  - Loss << 1% of total computation

**Insight**: Loss computation is negligible (focus optimization on model, not loss)

**Follow-up**: "What if you use triplet loss instead?"
- **Triplet loss**: O(N²) for hard negative mining
  - N=1000 samples → 1M comparisons (to find hard negatives)
  - Much more expensive than CE!
- **Solution**: Online hard negative mining (within mini-batch)
  - Batch size = 64 → 64² = 4K comparisons (manageable)

---

### Question 16: Regularization Comparison
**Q**: "You can use either L1 or L2 regularization. Your dataset: 1000 features, 10K samples. 900 features are irrelevant. Which regularization do you choose?"

**Expected Answer**:
- **Choose L1** (Lasso)

**Reasoning**:
- L1 → Sparse weights (many exactly 0)
  - 900 irrelevant features → weights = 0 (automatic feature selection)
  - 100 relevant features → non-zero weights
- L2 → All weights small (but not 0)
  - All 1000 features retained (including 900 irrelevant)
  - Worse generalization (noisy features interfere)

**Expected improvement**:
- L2: 75% validation accuracy (all features)
- L1: 85% validation accuracy (100 features) + faster inference

**Follow-up**: "What if only 100 features are irrelevant (not 900)?"
- **L2 may be better**:
  - 900 relevant features benefit from L2 (shrinks weights smoothly)
  - L1 may remove useful features (too aggressive)
- **Try both**: Cross-validate to choose

---

### Question 17: LSTM vs GRU
**Q**: "You're building a machine translation system (English → French). Average sentence length: 50 words. Training data: 1M sentence pairs. Should you use LSTM or GRU?"

**Expected Answer**:
- **Either works**, but **GRU slightly better choice**

**Reasoning**:
- **Sequence length**: 50 words (medium) → Both handle well
- **Data size**: 1M pairs (sufficient) → Both have enough data
- **GRU advantages**:
  - 25% fewer parameters → Faster training (matters for 1M samples)
  - Comparable accuracy to LSTM (for 50-word sequences)
  - Simpler (fewer hyperparameters)

**LSTM advantages**:
- Slightly better for very long sequences (>100 words)
- Explicit memory cell (better for complex dependencies)

**Production choice**: GRU (faster, similar quality)

**Modern answer**: "Use **Transformer** (attention-based)"
- Better quality than LSTM/GRU
- Parallelizable (10x faster training)
- State-of-the-art for translation

**Follow-up**: "Sequence length increases to 500 words. Change your answer?"
- **Still Transformer** (best choice)
- If must use RNN: **LSTM** (handles longer sequences better than GRU)
- Add: Bidirectional LSTM (see future context)

---

### Question 18: Attention Mechanism
**Q**: "Explain why attention mechanism improved machine translation. Use specific BLEU scores if possible."

**Expected Answer**:
- **Problem (before attention)**:
  - Seq2seq with fixed-size context vector (bottleneck)
  - Long sentences: Information loss
  - BLEU: ~27 (WMT'14 EN→FR)

- **Attention solution**:
  - Decoder attends to all encoder states (not just last)
  - No bottleneck (variable-size context)
  - Learns soft alignment (which source word → which target word)

- **Results**:
  - Bahdanau attention (2015): BLEU 34 (+7 points!)
  - Luong attention (2015): BLEU 35
  - Transformer (2017): BLEU 41 (current state-of-the-art)

**Example**:
```
Input: "The cat sat on the mat"
Output: "Le chat s'est assis sur le tapis"

Without attention:
- "s'est assis" might translate poorly (loses "sat" in context vector)

With attention:
- "s'est assis" attends to "sat" (α=0.8) → Correct translation
```

**Follow-up**: "What's the computational cost of attention?"
- **Time complexity**: O(n²) where n = sequence length
  - Each of n decoder steps attends to n encoder steps
  - 50-word sentence: 2500 attention computations
- **Space complexity**: O(n²) to store attention weights
- **Trade-off**: Better quality but slower (vs fixed context vector)

---

### Question 19: Training-Serving Skew
**Q**: "Your sequence model works great on validation (95% accuracy) but poorly in production (70%). You suspect training-serving skew. How do you debug?"

**Expected Answer**:
1. **Log production inputs and predictions**:
   - Sample 100 production inputs
   - Run through validation pipeline (offline)
   - Compare predictions
   
2. **Common causes**:
   - **Sequence length mismatch**:
     - Training: Max length 50 (truncated)
     - Production: Full length 200 (model hasn't seen)
     - Fix: Train on full-length sequences
   - **Tokenization mismatch**:
     - Training: Lowercase + tokenized
     - Production: Case-sensitive (forgot lowercase)
     - Fix: Align preprocessing
   - **Feature computation**:
     - Training: Features from batch pipeline
     - Production: Features from real-time API (different logic)
     - Fix: Use same feature computation code
   - **Model state (LSTM)**:
     - Training: Hidden state initialized to 0
     - Production: Hidden state from previous batch (stateful)
     - Fix: Reset hidden state between sequences

3. **Debugging steps**:
   - Step 1: Reproduce production inputs in validation → If accuracy still 95%, preprocessing issue
   - Step 2: Export model, run in production environment → If accuracy 95%, deployment issue
   - Step 3: Log intermediate outputs (embeddings, attention) → Compare train vs prod

**Red flag**: Blaming "production data distribution" without checking feature computation

**Follow-up**: "After fixing, accuracy is 90% (better but not 95%). Acceptable?"
- **Expected**: Some degradation (production data noisier)
- **5% gap reasonable** if:
  - Production has more diverse inputs
  - Real-time processing adds noise
  - Validation was "clean" data

---

### Question 20: Production Optimization
**Q**: "Your LSTM model (500 timesteps, hidden size 512) is too slow. Inference: 200ms per sequence. Need: <50ms. What do you do?"

**Expected Answer**:
1. **Reduce sequence length** (biggest impact):
   - 500 → 250 timesteps (2x speedup)
   - Or: Only use last 250 timesteps
   - Expected: 100ms
   
2. **Reduce hidden size**:
   - 512 → 256 dimensions (4x fewer params)
   - Expected: 50ms (meets requirement!)
   - Trade-off: ~2-3% accuracy loss
   
3. **Switch to GRU**:
   - 25% fewer parameters than LSTM
   - Expected: 150ms
   - Trade-off: Minimal accuracy loss (<1%)
   
4. **Model quantization**:
   - FP32 → FP16 or INT8
   - Expected: 2-4x speedup (50-100ms)
   - Trade-off: <1% accuracy loss
   
5. **Switch to Transformer** (best long-term):
   - Self-attention (parallelizable)
   - With optimization (FlashAttention): 30-40ms
   - Better accuracy + faster inference

**Production recommendation**:
- Quick fix: Reduce hidden size (512→256) + quantization → 25ms
- Long-term: Switch to Transformer (better accuracy + speed)

**Follow-up**: "After optimization, accuracy dropped 5%. Business says unacceptable. What now?"
- **Distillation**:
  - Train smaller model (student) to mimic large model (teacher)
  - Keep teacher's knowledge, student's speed
  - Expected: 3% accuracy loss (vs 5% from simple reduction)
- **Ensemble**:
  - Run 2 smaller models (2×256 hidden size)
  - Average predictions
  - Expected: 40ms (2×20ms), better accuracy
- **Re-train with optimization constraints**:
  - Train smaller model (256 hidden) from scratch
  - Tune hyperparameters specifically for smaller model
  - Expected: Better than just shrinking large model

---

## Key Takeaways for Interviews

### RNNs and Sequential Models
- **Vanishing gradient**: Core problem with vanilla RNN (can't learn >10 timesteps)
- **LSTM**: Solves vanishing gradient via cell state (gradient highway)
- **GRU**: Simpler than LSTM, 25% fewer params, similar performance
- **Attention**: Breakthrough for long sequences (no fixed context bottleneck)
- **Modern choice**: Transformers (replaced RNNs for most NLP tasks)

### Loss Functions
- **Classification**: Cross-entropy (not MSE!)
- **Imbalanced classes**: Focal loss (down-weight easy examples)
- **Regression**: MSE (default), MAE (robust), Huber (best of both)
- **Metric learning**: Triplet loss or contrastive loss
- **Custom loss**: Multi-task, weighted, regularized

### Evaluation Metrics
- **Imbalanced data**: Never use accuracy (use precision, recall, F1, PR-AUC)
- **Balanced data**: Accuracy, ROC-AUC acceptable
- **Regression**: MAE (interpretable), RMSE (penalizes outliers), R² (explained variance)
- **Ranking**: NDCG (position-aware), MAP (average precision), MRR (first result)

### Overfitting and Regularization
- **Detection**: Train-val gap >10% (overfitting), both high (underfitting)
- **Solutions**: L2 regularization, dropout, early stopping, data augmentation
- **Bias-variance**: High bias (underfit) → Increase complexity; High variance (overfit) → Regularize
- **Production**: Always use early stopping + validation monitoring

### Production Lessons
- **Google Translate**: Attention mechanism (BLEU 27→34)
- **FaceNet**: Triplet loss for face recognition (99.6% accuracy)
- **Object Detection**: Focal loss (RetinaNet) for class imbalance
- **Fraud Detection**: PR-AUC (not accuracy) for evaluation

### Interview Red Flags to Avoid
1. Using accuracy for imbalanced data (use precision/recall)
2. Not knowing LSTM solves vanishing gradient (cell state is key)
3. Using cross-entropy for regression (use MSE/MAE)
4. Not checking training-serving skew (common production issue)
5. Optimizing wrong metric (optimize business metric, not just accuracy)

---

## Additional Advanced Topics

### Sequence Labeling (NER, POS Tagging)

**Problem**: Tag each word in a sequence.

**Example (Named Entity Recognition)**:
```
Input:  "Apple is looking at buying U.K. startup"
Output: [ORG]  O  O       O  O      [LOC] O

Tags: O (Outside), ORG (Organization), LOC (Location), PER (Person)
```

**Approaches**:
1. **CRF (Conditional Random Fields)**:
   - Models transition probabilities: P(tag_t | tag_t-1)
   - Example: "PER" often followed by "O" (person name ends)
   - Classical baseline

2. **BiLSTM-CRF**:
   - BiLSTM: Contextual embeddings
   - CRF: Structured prediction (tag dependencies)
   - State-of-the-art (pre-BERT): F1 ~92% on CoNLL 2003

3. **BERT + Linear**:
   - BERT: Contextual word embeddings
   - Linear layer: Per-token classification
   - State-of-the-art (post-2018): F1 ~94% on CoNLL 2003

**Production (spaCy)**:
- Uses BiLSTM-CRF (pre-BERT models)
- Transformer models available (BERT-based)
- Accuracy: ~90% F1 (English NER)

---

### Attention Variants Summary

**Bahdanau (Additive)**:
```
score = v^T tanh(W_s s_t + W_h h_j)
+ Learnable alignment
- Slower (tanh computation)
```

**Luong (Multiplicative)**:
```
score = s_t^T W_a h_j
+ Faster (matrix multiplication)
- Requires same dimensions (or learned projection)
```

**Dot Product**:
```
score = s_t^T h_j
+ Fastest (no parameters)
- Only works if dim(s_t) = dim(h_j)
+ Used in Transformers (scaled dot-product)
```

**Self-Attention (Transformer)**:
```
score = (Q_i)^T K_j / sqrt(d_k)
+ Parallelizable (no sequential dependency)
+ Global context (all positions)
- O(n^2) complexity (quadratic in sequence length)
+ State-of-the-art for NLP (BERT, GPT, T5)
```

---

### Gradient Clipping (Production Detail)

**Why Needed**:
```
Problem: Exploding gradients (especially RNNs)
- Gradient norm: ||∇θ|| → ∞
- Loss: NaN (not a number)
- Training: Diverges

Causes:
- Deep networks (many layers)
- RNNs (long sequences, gradient chains)
- Large learning rate
```

**Gradient Clipping by Norm**:
```python
# Clip if gradient norm exceeds threshold
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

Effect:
if ||∇θ|| > max_norm:
    ∇θ = (max_norm / ||∇θ||) × ∇θ  # Scale down
```

**Gradient Clipping by Value**:
```python
# Clip each gradient component to [-clip_value, clip_value]
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

Effect: Each gradient clipped independently
```

**Production Settings**:
```
RNNs/LSTMs: max_norm = 1.0 to 5.0 (almost always needed)
CNNs: Usually not needed (but monitor grad norms)
Transformers: max_norm = 1.0 (common in BERT, GPT training)

Signs you need clipping:
- Loss spikes to NaN
- Validation loss jumps suddenly
- Gradient norms >100 (check with gradient monitoring)
```

---

### Batch Normalization vs Layer Normalization (Deep Dive)

**Batch Normalization** (2015):
```
Normalize across batch dimension:
For each feature j:
  μ_j = (1/B) Σ_b x_bj  (mean over batch)
  σ²_j = (1/B) Σ_b (x_bj - μ_j)²  (variance over batch)
  x̂_bj = (x_bj - μ_j) / sqrt(σ²_j + ε)
  y_bj = γ_j x̂_bj + β_j  (scale and shift)

Where:
- B = batch size
- γ, β = learnable parameters (per feature)

Pros:
+ Faster training (2-10x speedup)
+ Higher learning rates possible
+ Acts as regularization

Cons:
- Requires large batch (B ≥ 16, ideally 32+)
- Different behavior train vs inference
- Doesn't work with batch size 1
```

**Layer Normalization** (2016):
```
Normalize across feature dimension:
For each sample b:
  μ_b = (1/D) Σ_j x_bj  (mean over features)
  σ²_b = (1/D) Σ_j (x_bj - μ_b)²  (variance over features)
  x̂_bj = (x_bj - μ_b) / sqrt(σ²_b + ε)
  y_bj = γ_j x̂_bj + β_j  (scale and shift)

Where:
- D = feature dimension
- γ, β = learnable parameters (per feature)

Pros:
+ Works with any batch size (including 1)
+ Same behavior train and inference
+ Better for sequences (RNNs, Transformers)

Cons:
- Slightly slower than BatchNorm (less parallelizable)
```

**When to Use**:
```
BatchNorm:
✓ CNNs (almost always)
✓ Large batch sizes available (B ≥ 16)
✓ Images, spatial data

LayerNorm:
✓ Transformers (standard choice)
✓ RNNs, LSTMs
✓ Batch size 1 or varying
✓ Sequences (NLP)

Production:
- CNNs: BatchNorm (after Conv, before activation)
- Transformers: LayerNorm (after attention, after MLP)
- RNNs: LayerNorm (inside LSTM cells)
```

---

### Weight Initialization (Extended)

**Why Initialization Matters (Mathematical)**:
```
Forward pass variance:
Var(y) = n × Var(W) × Var(x)

For constant variance across layers:
Var(y) = Var(x) → n × Var(W) = 1 → Var(W) = 1/n

Backward pass variance:
Var(∂L/∂x) = m × Var(W) × Var(∂L/∂y)

For constant variance:
Var(∂L/∂x) = Var(∂L/∂y) → m × Var(W) = 1 → Var(W) = 1/m

Symmetric (Xavier):
Var(W) = 2/(n + m)  (harmonic mean)
```

**He Initialization Derivation**:
```
ReLU activation:
- Zeros out half the neurons (negative → 0)
- Effective variance: Var(y) = n × Var(W) × Var(x) / 2

For constant variance:
Var(y) = Var(x) → n × Var(W) / 2 = 1 → Var(W) = 2/n

Result: He initialization = Xavier with factor of 2
```

**Practical Implementation**:
```python
import torch.nn as nn

# He initialization (default for ReLU)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Xavier initialization (for tanh)
nn.init.xavier_normal_(layer.weight)

# Custom initialization
nn.init.normal_(layer.weight, mean=0, std=0.01)

# Bias initialization (usually zero)
nn.init.zeros_(layer.bias)
```

---

### Dealing with Class Imbalance (Production Strategies)

**Problem Severity**:
```
1:10 imbalance (90-10): Minor (class weights sufficient)
1:100 imbalance (99-1): Moderate (resampling + class weights)
1:1000 imbalance (99.9-0.1): Severe (focal loss + careful sampling)
```

**Strategy 1: Resampling**:
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE: Synthetic minority oversampling
smote = SMOTE(sampling_strategy=0.5)  # Minority:Majority = 1:2
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random undersampling
rus = RandomUnderSampler(sampling_strategy=0.5)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Combination (SMOTE + Undersampling)
from imblearn.combine import SMOTEENN
sme = SMOTEENN()
X_resampled, y_resampled = sme.fit_resample(X, y)
```

**Strategy 2: Class Weights**:
```python
from sklearn.utils.class_weight import compute_class_weight

# Compute balanced weights
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y), 
    y=y
)

# Example: 99% class 0, 1% class 1
# weights = [1.01, 99.0]  (inversely proportional)

# PyTorch
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

# Scikit-learn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

**Strategy 3: Threshold Tuning**:
```python
from sklearn.metrics import precision_recall_curve

# Get optimal threshold
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Choose threshold based on business metric
# Option 1: Maximize F1
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Option 2: Minimize cost
fp_cost = 1  # Cost of false positive
fn_cost = 100  # Cost of false negative
costs = fp_cost * (1 - precision) + fn_cost * (1 - recall)
optimal_threshold = thresholds[np.argmin(costs)]
```

**Strategy 4: Ensemble with Different Sampling**:
```python
# Train multiple models with different sampling ratios
models = []
for ratio in [0.1, 0.2, 0.5, 1.0]:
    # Sample data with ratio
    X_sampled, y_sampled = resample(X, y, ratio)
    
    # Train model
    model = train(X_sampled, y_sampled)
    models.append(model)

# Average predictions
predictions = np.mean([model.predict(X_test) for model in models], axis=0)
```

**Production Case Study - Credit Card Fraud (Stripe)**:
```
Problem: 99.9% legitimate, 0.1% fraud
Approach:
1. Undersample negatives: 1000 fraud + 10,000 legit (1:10 ratio)
2. Class weights: weight_fraud = 10
3. Focal loss: γ=2, α=0.25
4. Threshold tuning: Based on cost (FN=$100, FP=$0)

Result:
- Precision: 85% (of flagged fraud, 85% are real)
- Recall: 92% (of real fraud, 92% caught)
- False positive rate: 0.3% (acceptable)
- Fraud loss reduced: 90% (from $10M to $1M per year)
```

---

### Data Augmentation Techniques (Extended)

**Image Augmentation (Production)**:
```python
import torchvision.transforms as T

# Standard augmentation (ImageNet)
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Advanced augmentation (AutoAugment, RandAugment)
from timm.data import create_transform
train_transform = create_transform(
    input_size=224,
    is_training=True,
    auto_augment='rand-m9-mstd0.5',  # RandAugment
    interpolation='bicubic'
)

# Mixup (mix two images)
def mixup(images, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0))
    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_images, mixed_labels

# Cutmix (cut-and-paste)
def cutmix(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size(0))
    
    # Random box
    W = images.size(2)
    H = images.size(3)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Cut and mix
    images[:, :, cy:cy+cut_h, cx:cx+cut_w] = images[rand_index, :, cy:cy+cut_h, cx:cx+cut_w]
    labels = lam * labels + (1 - lam) * labels[rand_index]
    return images, labels
```

**Text Augmentation (NLP)**:
```python
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Synonym replacement
aug = naw.SynonymAug(aug_src='wordnet')
text = "The movie was great and fantastic"
augmented_text = aug.augment(text)
# Output: "The film was great and wonderful"

# Back-translation (translate to French and back)
aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en'
)
augmented_text = aug.augment(text)

# Random insertion/deletion
aug = naw.RandomWordAug(action="insert")
augmented_text = aug.augment(text)

# Contextual word embeddings (BERT)
aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute"
)
augmented_text = aug.augment(text)
```

**Time Series Augmentation**:
```python
# Jittering (add Gaussian noise)
def jitter(x, sigma=0.03):
    return x + np.random.normal(0, sigma, x.shape)

# Scaling (multiply by random scalar)
def scaling(x, sigma=0.1):
    factor = np.random.normal(1, sigma)
    return x * factor

# Magnitude warping (warp amplitude)
def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])
    random_warps = np.random.normal(1, sigma, size=(knot+2,))
    warp_steps = np.linspace(0, x.shape[0]-1, num=knot+2)
    warper = CubicSpline(warp_steps, random_warps)(orig_steps)
    return x * warper[:, None]

# Time warping (warp time axis)
def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
    warp_steps = (np.linspace(0, x.shape[0]-1, num=knot+2) * random_warps)
    warp_steps = np.clip(warp_steps, 0, x.shape[0]-1)
    warper = np.sort(warp_steps)
    warped = CubicSpline(warper, x)(orig_steps)
    return warped
```

---

## Final Production Checklist

### Before Training
- [ ] Data quality checks (missing values, outliers, label distribution)
- [ ] Train/val/test split (proper stratification)
- [ ] Baseline model (simple model for comparison)
- [ ] Compute class (GPU/TPU provisioned)
- [ ] Experiment tracking (MLflow, W&B)

### During Training
- [ ] Monitor train and val loss (detect overfitting)
- [ ] Monitor gradients (check for vanishing/exploding)
- [ ] Checkpoint saving (save best model)
- [ ] Early stopping (patience = 5-10)
- [ ] Learning rate scheduling (if applicable)

### After Training
- [ ] Evaluate on test set (final performance)
- [ ] Confusion matrix (understand errors)
- [ ] Feature importance (interpret model)
- [ ] Error analysis (sample failures)
- [ ] Compare to baseline (is improvement significant?)

### Before Deployment
- [ ] Model optimization (quantization, pruning)
- [ ] Latency profiling (measure inference time)
- [ ] Load testing (can handle production QPS?)
- [ ] A/B test infrastructure (gradual rollout)
- [ ] Monitoring dashboards (metrics, alerts)

### Production Monitoring
- [ ] Inference latency (P50, P95, P99)
- [ ] Prediction distribution (drift detection)
- [ ] Error rate (sudden spikes?)
- [ ] Business metrics (revenue, user satisfaction)
- [ ] Retraining triggers (performance degradation)

---

## Resources for Further Study

### Books
- "Deep Learning" (Goodfellow, Bengio, Courville) - Comprehensive theory
- "Dive into Deep Learning" (Aston Zhang et al.) - Interactive learning
- "Neural Networks and Deep Learning" (Michael Nielsen) - Intuitive explanations
- "Machine Learning Yearning" (Andrew Ng) - Practical advice

### Online Courses
- Stanford CS231n (CNNs) - Karpathy's legendary lectures
- Stanford CS224n (NLP) - Manning's RNN/Attention lectures
- Fast.ai - Practical deep learning (top-down approach)
- deeplearning.ai - Andrew Ng's specialization

### Blogs
- Colah's Blog (colah.github.io) - Beautiful visualizations
- Distill.pub - Interactive ML explanations
- Andrej Karpathy's blog - Practical insights
- Lil'Log (Lilian Weng) - Research summaries

### Papers (Must-Read)
- ImageNet Classification (AlexNet, 2012)
- Deep Residual Learning (ResNet, 2015)
- Attention Is All You Need (Transformer, 2017)
- BERT (2018), GPT-2 (2019) - Transformer applications
- EfficientNet (2019) - Model scaling

---

## Conclusion

This completes the ML Fundamentals notes covering:
- **Classical ML**: Decision Trees, Random Forests, Gradient Boosting
- **Neural Networks**: MLPs, Backpropagation, Activations, Optimizers
- **CNNs**: Convolution, Pooling, ResNet, VGG, Inception, EfficientNet, ViT
- **RNNs**: Vanilla RNN, LSTM, GRU, Attention, Seq2Seq
- **Loss Functions**: Cross-entropy, MSE, MAE, Triplet, Focal, Huber
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, NDCG
- **Regularization**: L1/L2, Dropout, BatchNorm, LayerNorm, Early Stopping

**Total Interview Questions**: 20 comprehensive questions covering all major topics with Hao Hoang-style depth and production context.

**Key Interview Success Factors**:
1. Know the fundamentals cold (backprop 6N FLOPs, vanishing gradient, etc.)
2. Understand production trade-offs (accuracy vs latency vs cost)
3. Connect theory to practice (why LSTM works, when to use which loss)
4. Quantify everything (model sizes, FLOPs, speedups, accuracy gains)
5. Learn from production systems (Google, Meta, OpenAI case studies)

**Good luck with your interviews! 🚀**
