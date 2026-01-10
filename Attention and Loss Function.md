# ML Fundamentals - Part 2: Attention Mechanisms & Loss Functions

**Complete Interview Preparation Guide**
*Sources: Attention Is All You Need (Vaswani et al.), Linformer, Longformer, BigBird papers, Lilian Weng's blog, Chip Huyen's resources, Production systems at Google/Meta/OpenAI*

---

## 24.2 ATTENTION VARIANTS

### Overview: The Attention Revolution

**Why Attention Matters**:
- Transformers = 90% of modern NLP/Vision (BERT, GPT, ViT, CLIP)
- Attention is the bottleneck: O(n²) complexity
- Context windows: 2K → 128K → 1M+ tokens
- Memory explosion: 4K context = 2GB, 128K context = 64GB!

**The Attention Complexity Problem**:
```
Standard self-attention:
- Time: O(n²·d) where n=sequence length, d=model dimension
- Space: O(n²) for attention matrix

Example (GPT-3):
- n = 2048 tokens
- d = 12288 dimensions
- Attention matrix: 2048² × 4 bytes = 16 MB per layer
- Total (96 layers): 1.5 GB just for attention!
- At 128K context: 768 GB per layer (impossible!)
```

**Evolution of Attention**:
```
2017: Vanilla Attention (O(n²))
2019: Sparse Attention (OpenAI, O(n√n))
2020: Linear Attention (O(n))
2020: Performer (O(n·log(n)))
2020: Longformer (O(n))
2020: BigBird (O(n))
2021: Flash Attention (O(n²) but 10× faster)
2023: Flash Attention 2 (O(n²) but 15× faster)
```

---

### 24.2.1 Softmax Attention (Vanilla Transformer)

**Mathematical Foundation**:
```
Given input X ∈ R^(n×d):

1. Linear projections:
Q = X · W_Q  (Query)
K = X · W_K  (Key)  
V = X · W_V  (Value)

where W_Q, W_K, W_V ∈ R^(d×d_k)

2. Attention scores:
S = Q · K^T / √d_k

3. Attention weights:
A = softmax(S) ∈ R^(n×n)

4. Output:
O = A · V

Full formula:
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**Step-by-Step Example**:
```
Sequence: "The cat sat"
n = 3 tokens, d_k = 4 dimensions

Q = [[1, 0, 0, 1],     # "The"
     [0, 1, 1, 0],     # "cat"
     [1, 1, 0, 0]]     # "sat"

K = [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 0, 0, 1]]

Step 1: Compute scores S = Q·K^T / √4
S = [[1, 1, 2],
     [1, 1, 0],
     [1, 1, 1]] / 2
  = [[0.5, 0.5, 1.0],
     [0.5, 0.5, 0.0],
     [0.5, 0.5, 0.5]]

Step 2: Apply softmax (row-wise)
A = [[0.23, 0.23, 0.54],   # "The" attends mostly to "sat"
     [0.38, 0.38, 0.24],   # "cat" attends to "The" and "cat"
     [0.33, 0.33, 0.33]]   # "sat" attends equally

Step 3: Weighted sum of values
O = A · V  (context-aware representations)
```

**Why Scaling by √d_k?**
```
Problem without scaling:
- Dot products grow with dimension d_k
- Large scores → softmax saturates
- Gradients vanish (softmax derivative ≈ 0)

Example:
d_k = 512
Q·K^T ∈ [-512, 512] (extreme values!)
softmax([500, 1, -500]) ≈ [1.0, 0.0, 0.0] (saturated!)

With scaling:
Q·K^T / √512 ∈ [-√512, √512] = [-22.6, 22.6]
softmax([22, 0.04, -22]) ≈ [0.9999, 0.0001, 0.0]
Still concentrated but gradients non-zero
```

**Multi-Head Attention**:
```
Instead of single attention:
1. Project to h different subspaces
2. Run attention in parallel
3. Concatenate outputs

head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O

Benefits:
- Different heads learn different patterns
- Head 1: Short-range dependencies (adjacent words)
- Head 2: Long-range dependencies (subject-verb)
- Head 3: Syntactic structure
- More expressive than single head
```

**Production Case Study: GPT-3**
```
Model: GPT-3 175B
Sequence length: 2048
Hidden dim: 12288
Heads: 96
Head dimension: 12288 / 96 = 128

Memory per layer:
- Q, K, V projections: 3 × (2048 × 12288) × 4 bytes = 288 MB
- Attention matrix: 96 × (2048)² × 4 bytes = 1.5 GB
- Total per layer: ~2 GB
- Total (96 layers): 192 GB for attention alone!

Optimizations applied:
- Flash Attention: Reduced memory to 96 MB per layer
- Mixed precision (FP16): Halved memory
- Gradient checkpointing: Trade compute for memory
- Final memory: ~50 GB for attention (manageable)
```

**Complexity Analysis**:
```
| Operation | Time | Space |
|-----------|------|-------|
| Q·K^T | O(n²·d) | O(n²) |
| Softmax | O(n²) | O(n²) |
| A·V | O(n²·d) | O(n·d) |
| **Total** | **O(n²·d)** | **O(n²)** |

Bottleneck: Quadratic in sequence length!

Scaling behavior:
n=1K   → 1M attention scores
n=4K   → 16M (4× → 16× memory)
n=16K  → 256M (16× → 256× memory)
n=100K → 10B (infeasible!)
```

**When Standard Attention Works**:
- ✅ Short sequences (n < 2048)
- ✅ Sufficient GPU memory (A100 80GB)
- ✅ Best quality (no approximations)
- ❌ Long sequences (n > 8K)
- ❌ Memory-constrained devices
- ❌ Real-time inference (too slow)

---

### 24.2.2 Linear Attention (Kernel Trick)

**The Key Insight**:
```
Standard attention:
Attention(Q, K, V) = softmax(QK^T) V
                     └─ O(n²) ─┘

Rewrite using kernel trick:
softmax(QK^T) ≈ φ(Q) φ(K)^T

Then:
Attention(Q, K, V) = φ(Q) [φ(K)^T V]
                            └─ O(n·d²) ─┘
                     └─ Total: O(n·d²) ─┘

If d << n, this is linear in n!
```

**Mathematical Foundation**:
```
Key idea: Change order of operations

Standard (O(n²)):
1. Compute S = Q·K^T ∈ R^(n×n)
2. Apply A = softmax(S)
3. Compute O = A·V

Linear (O(n)):
1. Define φ(x) = elu(x) + 1 (feature map)
2. Compute K̃ = φ(K) ∈ R^(n×d)
3. Compute Q̃ = φ(Q) ∈ R^(n×d)
4. Compute Z = K̃^T · V ∈ R^(d×d) [only d²!]
5. Compute O = Q̃ · Z ∈ R^(n×d)

Complexity:
Step 4: O(n·d²)
Step 5: O(n·d²)
Total: O(n·d²) where typically d² << n²
```

**Feature Map Choices**:
```
1. ELU+1: φ(x) = elu(x) + 1
   - Ensures non-negativity
   - Smooth approximation
   - Works well in practice

2. ReLU: φ(x) = max(0, x)
   - Simplest
   - Faster computation
   - Cruder approximation

3. Random Features (RFF):
   φ(x) = [cos(ω₁·x), sin(ω₁·x), ..., cos(ω_m·x), sin(ω_m·x)]
   - Better approximation
   - More dimensions (2m instead of d)
```

**Production Case Study: Linformer (Facebook AI)**
```
Task: Document summarization (long documents)
Baseline: Standard attention
- Max length: 512 tokens (memory limit)
- Memory: 2 GB per batch

Linformer:
- Max length: 4096 tokens (8× longer!)
- Memory: 1.5 GB per batch (25% less)
- Speed: 3× faster inference

Architecture:
- n = 4096, d = 768
- Standard: O(4096² × 768) = 12B operations
- Linformer: O(4096 × 768²) = 2.4B operations (5× less)

Quality:
- CNN/DailyMail ROUGE: 41.2 vs 41.5 (baseline)
- Minimal degradation (-0.3 points)
- Huge speedup for long documents

Use case: Real-time news summarization
- Input: 3000-word article
- Linformer: 500ms
- Standard: 1500ms (unusable for prod)
```

**Approximation Quality**:
```
Attention matrix approximation error:
||softmax(QK^T) - φ(Q)φ(K)^T||_F

Empirical results:
d=64:  Error ~0.15 (poor)
d=256: Error ~0.08 (okay)
d=512: Error ~0.04 (good)
d=1024: Error ~0.02 (excellent)

Trade-off: Larger d → better approximation but slower
Sweet spot: d=512 for most tasks
```

**When to Use Linear Attention**:
- ✅ Long sequences (4K-32K tokens)
- ✅ Memory-constrained (edge devices)
- ✅ Real-time inference (low latency critical)
- ✅ Tasks with local dependencies (not global)
- ⚠️ Quality-sensitive tasks (slight degradation)
- ❌ When n² fits in memory (standard is better)

---

### 24.2.3 Performer (Kernel Approximation with FAVOR+)

**Beyond Linear Attention**:
```
Problem with basic linear attention:
- Uses simple kernels (ELU+1, ReLU)
- Poor approximation of softmax
- Quality degradation on complex tasks

Performer's solution: FAVOR+ (Fast Attention Via Orthogonal Random features)
- Better kernel approximation
- Unbiased estimator of softmax
- Maintains O(n) complexity
```

**Mathematical Foundation**:
```
Goal: Approximate K(x, y) = exp(x^T y)

Random Fourier Features:
φ(x) = exp(||x||²/2) × [cos(ω₁^T x), sin(ω₁^T x), ..., cos(ω_m^T x), sin(ω_m^T x)]

where ω_i ~ N(0, I) (random projections)

Key insight: Orthogonal random features reduce variance!
Instead of random ω_i, use orthogonalized ω_i
→ Lower variance → Better approximation

Attention becomes:
Attention(Q,K,V) ≈ φ(Q) [φ(K)^T V] / [φ(Q) φ(K)^T 1]
                          └─ O(n·m·d) ─┘

where m = number of random features (typical m=256)
```

**FAVOR+ Algorithm**:
```
1. Sample m random features: ω₁, ..., ω_m ~ N(0, I)

2. Orthogonalize using QR decomposition:
   [ω₁, ..., ω_m] = QR([ω₁, ..., ω_m])

3. Compute feature maps:
   φ(Q) = exp(||Q||²/2) × [cos(ωᵢ^T Q), sin(ωᵢ^T Q)]

4. Compute attention:
   Numerator: Q̃ = φ(Q) [φ(K)^T V]
   Denominator: D = φ(Q) [φ(K)^T 1]
   Output: Q̃ / D

Complexity: O(n·m·d) where m << n
For n=16K, m=256: ~60× speedup!
```

**Performer vs Linear Attention**:
```
| Metric | Linear Attention | Performer |
|--------|------------------|-----------|
| Complexity | O(n·d²) | O(n·m·d) |
| Approximation | Biased | Unbiased |
| Quality | Moderate | High |
| Memory | O(n·d) | O(n·m) |
| Speed | Faster | Fast |

Example (n=8K, d=512, m=256):
Linear: 8K × 512² = 2B ops
Performer: 8K × 256 × 512 = 1B ops (2× faster)
```

**Production Case Study: Google Performer (Protein Modeling)**
```
Task: Protein structure prediction (AlphaFold-style)
Input: Protein sequences (500-5000 amino acids)
Challenge: Long sequences, O(n²) attention infeasible

Baseline (Standard Attention):
- Max length: 512 residues
- Memory: 16 GB
- Time: 10 seconds per protein

Performer:
- Max length: 2048 residues (4× longer)
- Memory: 8 GB (50% less)
- Time: 4 seconds per protein (2.5× faster)
- Accuracy: 98.5% of standard attention quality

Key insight: Protein folding has long-range dependencies
- Need to model residues 1000+ apart
- Performer enabled this without memory explosion

Impact:
- Can process full proteins (not truncated)
- Better structure predictions (+5% accuracy)
- 10× throughput on same hardware
```

**Approximation Quality Comparison**:
```
Task: Language Modeling (WikiText-103)
Metric: Perplexity (lower is better)

Standard Attention: 18.2 PPL
Linear Attention: 19.8 PPL (+1.6)
Performer (m=128): 18.9 PPL (+0.7)
Performer (m=256): 18.4 PPL (+0.2) ✓

Conclusion: Performer with m=256 nearly matches standard
```

**When to Use Performer**:
- ✅ Long sequences (4K-16K tokens)
- ✅ Quality-sensitive (need good approximation)
- ✅ Scientific computing (proteins, DNA)
- ✅ When linear attention quality insufficient
- ⚠️ Requires tuning m (number of features)
- ❌ Very short sequences (<512, standard faster)

---

### 24.2.4 Longformer (Local + Global Attention)

**The Sparse Attention Insight**:
```
Observation: Most tokens only need to attend to nearby tokens
- "The cat sat on the mat" → "sat" attends to "cat" and "on"
- Only a few tokens need global attention
- Example: [CLS] token in BERT attends globally

Longformer strategy:
- Local attention: Each token attends to w neighbors
- Global attention: Special tokens attend to all
```

**Attention Patterns**:
```
1. Sliding Window Attention:
Each token attends to w tokens on each side

Example (w=1):
"The cat sat on the mat"
 ↓   ↓   ↓   ↓   ↓
[1,1,1,0,0,0,0]  # "The" attends to "The", "cat"
[1,1,1,1,0,0,0]  # "cat" attends to "The", "cat", "sat"
[0,1,1,1,1,0,0]  # "sat" attends to "cat", "sat", "on"
...

Complexity: O(n × w) instead of O(n²)
For n=4096, w=512: 512× speedup!

2. Dilated Sliding Window:
Attend to every d-th token in window

Example (w=2, dilation=2):
[1,0,1,0,1,0,0]  # Attend to positions 0, 2, 4

Benefits:
- Increases receptive field without cost
- Still O(n × w/d)

3. Global Attention:
Special tokens attend to all

Example:
[CLS]: [1,1,1,1,1,1,1]  # Global
"The":  [1,1,1,0,0,0,0]  # Local
"cat":  [1,1,1,1,0,0,0]  # Local
```

**Mathematical Foundation**:
```
Given sequence of length n, window size w:

Local attention (for most tokens):
A_i,j = softmax(Q_i K_j^T / √d_k) if |i-j| ≤ w
      = 0 otherwise

Global attention (for special tokens):
A_i,j = softmax(Q_i K_j^T / √d_k) for all j

Complexity:
- Local: O(n × w × d) = O(n) if w constant
- Global: O(g × n × d) where g = # global tokens
- Total: O(n) + O(g × n) ≈ O(n) if g << n
```

**Production Case Study: BigBird (Google)**
```
Task: Question answering on long documents (Natural Questions)
Dataset: Wikipedia articles (10K-20K tokens)

Baseline (BERT):
- Max length: 512 tokens
- Must truncate documents → lose context
- F1 score: 72.3

Longformer:
- Max length: 4096 tokens (8× longer)
- Window size: 512
- Global tokens: [CLS], [SEP], question tokens
- F1 score: 78.9 (+6.6 points!)

Key insight: Many answers require long-range context
- Example: "When was the Eiffel Tower built?"
- Answer at token 3500, context at token 500
- BERT truncates → misses context
- Longformer → captures full context

Memory comparison:
BERT (512 tokens): 2 GB
Longformer (4096 tokens): 3 GB (8× longer, 1.5× memory)
```

**Choosing Window Size**:
```
Trade-off: Larger w → more context but slower

Task-specific guidelines:
| Task | Window Size | Reasoning |
|------|-------------|-----------|
| Sentiment | 128 | Local context sufficient |
| QA | 512 | May need paragraph-level context |
| Summarization | 256 | Balance speed and coverage |
| Document classification | 512 | Need broader context |

Empirical result (Longformer paper):
w=64:  -2% quality, 4× faster
w=256: -0.5% quality, 2× faster
w=512: Baseline quality, 1× (reference)
w=1024: +0.2% quality, 0.5× faster
```

**Implementation Details**:
```python
# Longformer attention mask
# 1 = attend, 0 = don't attend

# Local attention (sliding window)
attention_mask = torch.zeros(n, n)
for i in range(n):
    left = max(0, i - window_size)
    right = min(n, i + window_size + 1)
    attention_mask[i, left:right] = 1

# Global attention (special tokens)
attention_mask[0, :] = 1  # [CLS] attends globally
attention_mask[:, 0] = 1  # All attend to [CLS]

# Complexity: O(n × window_size) instead of O(n²)
```

**When to Use Longformer**:
- ✅ Document-level tasks (>2K tokens)
- ✅ Need some global reasoning (QA, summarization)
- ✅ Hierarchical structure (paragraphs, sections)
- ✅ When Transformer pre-training available
- ⚠️ Requires careful global token selection
- ❌ All-to-all attention needed (some reasoning tasks)

---

### 24.2.5 BigBird (Random + Window + Global)

**Three Attention Patterns Combined**:
```
BigBird = Local (window) + Global (special tokens) + Random

Why random attention?
- Theoretical: Preserves graph connectivity
- Practical: Improves information flow
- Sparse graphs need random edges for short paths

Graph theory result:
- Pure local: Diameter O(n/w) (long paths)
- Local + Random: Diameter O(log n) (short paths!)
```

**Attention Pattern Details**:
```
For each query token i:

1. Window attention (local):
   Attend to [i-w, i+w]
   Cost: O(w)

2. Global attention:
   Attend to/from special tokens ([CLS], etc.)
   Cost: O(g)

3. Random attention:
   Attend to r random tokens
   Cost: O(r)

Total per token: w + g + r (constant!)
Total complexity: O(n × (w + g + r)) = O(n)

Typical values: w=256, g=2, r=64
```

**Mathematical Foundation**:
```
Attention matrix A ∈ R^(n×n):

A_ij = exp(Q_i K_j^T / √d_k) if j in:
       - Window: |i-j| ≤ w, or
       - Global: j ∈ G, or
       - Random: j ∈ R_i
     = 0 otherwise

Where:
- G = {0, 1, ..., g-1} (global tokens)
- R_i = random sample of r tokens (per token i)

Sparsity:
Dense attention: n² entries
BigBird: n × (w + g + r) entries
Reduction: n / (w+g+r) ≈ 10-20×
```

**Random Attention Sampling**:
```
Q: How to sample random tokens?

Option 1: Uniform random (simple)
- Each token samples r tokens uniformly
- Different random sets per token

Option 2: Block-sparse random (BigBird default)
- Divide sequence into blocks of size b
- Randomly connect blocks
- All tokens in block share random pattern
- More efficient (batch-friendly)

Option 3: Learned random (future work)
- Learn which tokens to attend to
- More parameters but potentially better
```

**Production Case Study: BigBird (Google)**
```
Task: Long document classification (arXiv papers)
Dataset: Full-text scientific papers (8K-16K tokens)

Baseline (RoBERTa):
- Max length: 512 tokens
- Extract: Abstract + Introduction only
- Accuracy: 82.1%

BigBird:
- Max length: 8192 tokens (16× longer)
- Full paper processed
- Window: 256, Global: 32, Random: 64
- Accuracy: 89.3% (+7.2 points!)

Key insight: Methods and Results sections crucial
- Often located at tokens 3000-6000
- Missed by 512-token models
- Captured by BigBird

Memory:
RoBERTa: 512² = 262K attention scores
BigBird: 8192 × 352 = 2.9M (11× more but still 15× less than dense)

Speed:
RoBERTa (512): 50ms
BigBird (8192): 180ms (4× slower but 16× longer!)
```

**Theoretical Properties (Graph Connectivity)**:
```
BigBird attention graph:
- Nodes: Tokens
- Edges: Attention connections

Property 1: Constant degree
Each node has degree w + g + r (constant)

Property 2: Short paths
Diameter: O(log n) with high probability
→ Information can flow quickly across document

Comparison:
Local only: Diameter O(n/w) (slow)
Local + Random: Diameter O(log n) (fast) ✓

Example (n=4096, w=128, r=64):
Local only: 4096/128 = 32 hops worst case
Local + Random: log(4096) ≈ 12 hops worst case
```

**When to Use BigBird**:
- ✅ Very long documents (8K-64K tokens)
- ✅ Need global + local reasoning (legal documents)
- ✅ When Longformer insufficient (need more connectivity)
- ✅ Scientific papers, books, code files
- ⚠️ More complex than Longformer (3 patterns)
- ⚠️ Random attention adds noise (slight quality variance)
- ❌ Short sequences (<2K, standard attention better)

---

### 24.2.6 Reformer (LSH Attention)

**Locality-Sensitive Hashing (LSH) for Attention**:
```
Key insight: Attention is sparse!
- Most attention weights are near zero
- Only need to compute high-scoring pairs
- Use hashing to find similar Q and K vectors

LSH property:
If Q_i ≈ K_j (similar vectors)
Then: hash(Q_i) = hash(K_j) with high probability

Algorithm:
1. Hash all Q and K vectors
2. Group tokens by hash bucket
3. Only compute attention within buckets
4. Complexity: O(n × b) where b = avg bucket size
```

**LSH Attention Algorithm**:
```
1. Choose random projection: R ∈ R^(d×h)
   (h = number of hash bits, typically 8-16)

2. Hash function:
   hash(x) = sign(R^T x)
   Returns binary vector: [1,0,1,1,0,...]

3. Bucket assignment:
   For each Q_i and K_j:
   bucket[i] = hash(Q_i)
   bucket[j] = hash(K_j)

4. Compute attention only within buckets:
   For each bucket b:
     tokens_in_b = {i : bucket[i] = b}
     Compute attention(Q_i, K_j) for i,j ∈ tokens_in_b

5. Handle collisions:
   - Some important pairs may hash differently
   - Use multiple hash rounds (r=4-8)
   - OR: Fall back to local attention for safety
```

**Example (LSH Hashing)**:
```
Sequence: "The cat sat on the mat"
d=4, h=2 (2 hash bits)

Vectors (simplified):
Q_The = [1.0, 0.2, -0.1, 0.3]
Q_cat = [0.9, 0.3, -0.2, 0.4]
Q_sat = [-0.1, 0.8, 0.9, -0.2]
Q_on = [-0.2, 0.7, 0.8, -0.1]
Q_the = [0.2, -0.9, -0.8, 0.1]
Q_mat = [0.1, -0.8, -0.9, 0.2]

Random projection R:
R = [[1, 0],
     [0, 1],
     [1, 0],
     [0, 1]]

Hashes:
hash(Q_The) = sign([1.2, 0.5]) = [1, 1] → Bucket 3
hash(Q_cat) = sign([1.1, 0.7]) = [1, 1] → Bucket 3
hash(Q_sat) = sign([0.8, 0.6]) = [1, 1] → Bucket 3
hash(Q_on) = sign([0.6, 0.6]) = [1, 1] → Bucket 3
hash(Q_the) = sign([-0.6, -0.8]) = [0, 0] → Bucket 0
hash(Q_mat) = sign([-0.8, -0.7]) = [0, 0] → Bucket 0

Attention:
- Tokens 0-3 attend to each other (Bucket 3)
- Tokens 4-5 attend to each other (Bucket 0)
- Complexity: 2 groups of ~3 tokens = O(9) instead of O(36)
```

**Reformer Full Architecture**:
```
Reformer = LSH Attention + Reversible Layers + Chunking

1. LSH Attention (above)
2. Reversible Residual:
   Y1 = X1 + Attention(X2)
   Y2 = X2 + FFN(Y1)
   
   Can reconstruct X1, X2 from Y1, Y2 → Save memory

3. Chunked FFN:
   Process FFN in chunks (not all tokens at once)
   → Reduce memory from O(n·d) to O(c·d) where c=chunk size
```

**Production Case Study: Reformer (Google)**
```
Task: Long document generation (stories)
Baseline: Transformer-XL
- Max length: 2048 tokens
- Memory: 16 GB
- Training time: 2 days (64 TPUs)

Reformer:
- Max length: 64K tokens (32× longer!)
- Memory: 8 GB (50% less!)
- Training time: 1.5 days (32 TPUs, 2× less hardware)
- LSH rounds: 8
- Hash bits: 16
- Chunk size: 128

Quality:
- Perplexity on 64K context: 1.23
- Comparable to dense attention on same data
- Generated coherent 50K-token stories

Key insights:
- LSH attention: 10× memory reduction
- Reversible layers: 2× memory reduction
- Chunking: 5× memory reduction
- Combined: 100× memory reduction!
- Enabled 64K context on consumer hardware
```

**Collision Rate Analysis**:
```
Probability two similar vectors hash differently:

P(collision) ≈ exp(-similarity / temperature)

Example:
- Very similar (similarity=0.9): P(miss) = 5%
- Moderately similar (similarity=0.5): P(miss) = 30%
- Dissimilar (similarity=0.1): P(miss) = 90% (good!)

Solution: Multiple hash rounds (r=8)
- P(miss all rounds) = 0.05^8 ≈ 0 (very unlikely)
- Ensures similar pairs found with high probability
```

**When to Use Reformer**:
- ✅ Extremely long sequences (64K-1M tokens)
- ✅ Memory-constrained (edge devices, limited GPUs)
- ✅ Generative tasks (stories, documents)
- ✅ When quality can tolerate approximation
- ⚠️ More complex implementation
- ⚠️ Hashing adds randomness (less reproducible)
- ❌ Short sequences (<4K, overhead not worth it)
- ❌ When exact attention required

---

### 24.2.7 Flash Attention (Efficient Implementation)

**The Key Insight: IO-Awareness**:
```
Problem: Standard attention is memory-bound, not compute-bound!

GPU Memory Hierarchy:
- SRAM (on-chip): 20 MB, 19 TB/s bandwidth ← Fast!
- HBM (off-chip): 40 GB, 1.5 TB/s bandwidth ← 10× slower
- Standard attention: Repeatedly reads/writes HBM → slow

Flash Attention: Do more computation to reduce memory IO
- Keep intermediate results in SRAM
- Fuse operations (don't materialize attention matrix)
- Result: Same O(n²) complexity but 10× faster!
```

**Standard Attention Memory IO**:
```
Given Q, K, V ∈ R^(n×d):

1. Load Q, K from HBM → SRAM
2. Compute S = QK^T
3. Write S to HBM (large! n² elements)
4. Load S from HBM
5. Compute A = softmax(S)
6. Write A to HBM
7. Load A, V from HBM
8. Compute O = AV
9. Write O to HBM

Total HBM accesses: O(n² + n·d)
For n=4K, d=128: ~68M reads/writes
Bottleneck: Steps 3, 6 (writing n² matrix)
```

**Flash Attention Algorithm**:
```
Key: Tiling + online softmax

1. Divide Q, K, V into blocks: B_c (column) and B_r (row)
2. For each block:
   a. Load Q_block, K_block into SRAM
   b. Compute S_block = Q_block K_block^T (in SRAM)
   c. Compute softmax incrementally (online algorithm)
   d. Compute output block (in SRAM)
   e. Write output block to HBM (once!)
3. Fuse operations: Never materialize full S or A

HBM accesses: O(n·d + n²/B)
For n=4K, d=128, B=128: ~260K (260× less!)
```

**Online Softmax (Key Innovation)**:
```
Problem: softmax needs global maximum and sum
Standard: Load all data, compute, store

Online (streaming) softmax:
m_old = current max
m_new = max(m_old, new_block_max)

exp_sum_old = current sum of exp
exp_sum_new = exp_sum_old * exp(m_old - m_new) + sum(exp(new_block - m_new))

This allows computing softmax block-by-block!
No need to store full attention matrix
```

**Flash Attention 2 Improvements**:
```
Flash Attention 1 (2022):
- 2-5× speedup over standard
- Same memory usage

Flash Attention 2 (2023):
- Better work partitioning
- Reduced shared memory usage
- Better GPU utilization
- Result: 2× faster than FA1 (4-10× vs standard)

Flash Attention 3 (2024):
- Asynchronous GEMM
- Incoherent processing
- Hardware-specific (H100/H200)
- Result: 1.5-2× faster than FA2
```

**Production Case Study: Meta Llama 2**
```
Task: Training Llama 2 (70B parameters)
Context: 4096 tokens
Hardware: A100 80GB GPUs

Without Flash Attention:
- Batch size: 1 (memory limit!)
- Training time: 120 days (estimated)
- Throughput: 180 tokens/sec/GPU
- Cost: $8M

With Flash Attention 2:
- Batch size: 4 (4× more)
- Training time: 40 days (3× faster)
- Throughput: 540 tokens/sec/GPU (3× higher)
- Cost: $2.7M (3× less)

Key insight: Memory savings → larger batch → faster training
Not just speed, but enables training that was infeasible
```

**Memory Comparison**:
```
Sequence length: 4096 tokens
Model dimension: 128
Precision: FP16 (2 bytes)

Standard Attention:
- Attention matrix: 4096² × 2 bytes = 32 MB per head
- 32 heads: 1 GB per layer
- 32 layers: 32 GB total (doesn't fit batch size >1!)

Flash Attention:
- No materialized attention matrix
- Working memory: 4096 × 128 × 2 × 32 = 32 MB
- 32 layers: 1 GB total (32× less!)
- Enables batch size 4-8 on same GPU
```

**When to Use Flash Attention**:
- ✅ **Always!** (drop-in replacement, strictly better)
- ✅ Training large models (memory critical)
- ✅ Long sequences (>2K tokens)
- ✅ When GPU memory limited
- ✅ Production inference (faster + less memory)
- Note: Requires compatible hardware (Ampere+, V100/A100/H100)
- Fallback: Standard attention on older GPUs

---

### 24.2.8 Attention Variant Selection Guide

**Decision Tree**:
```
Sequence Length?
├─ <512 tokens → Standard Attention (simplest, fastest)
├─ 512-2K → Standard or Flash Attention
├─ 2K-8K → Flash Attention or Longformer
├─ 8K-64K → Longformer or BigBird
└─ >64K → Reformer (LSH)

Quality Requirements?
├─ Highest quality → Flash Attention (exact)
├─ Good quality → Longformer / BigBird
├─ Acceptable quality → Linear / Performer
└─ Lower quality OK → Reformer

Memory Constraints?
├─ Abundant (A100 80GB) → Any method
├─ Limited (V100 16GB) → Flash / Longformer / Performer
└─ Very limited (edge) → Reformer / Linear

Task Type?
├─ Generation → Flash > Reformer
├─ Classification → Longformer > BigBird
├─ QA (long docs) → BigBird > Longformer
└─ Summarization → Performer > Linear
```

**Complexity & Quality Comparison**:
```
| Method | Time | Space | Quality | Memory (4K ctx) |
|--------|------|-------|---------|-----------------|
| Standard | O(n²d) | O(n²) | 100% | 32 MB |
| Flash | O(n²d) | O(n²)* | 100% | 1 MB |
| Longformer | O(nwd) | O(nw) | 98% | 8 MB |
| BigBird | O(n(w+r)d) | O(n(w+r)) | 97% | 10 MB |
| Performer | O(nmd) | O(nm) | 95% | 2 MB |
| Linear | O(nd²) | O(nd) | 90% | 1 MB |
| Reformer | O(nb log n d) | O(n) | 93% | 0.5 MB |

* Flash Attention: O(n²) complexity but doesn't materialize matrix
```

**Real-world Guidelines by Use Case**:
```
Use Case: Code completion (GitHub Copilot)
Sequence: 2K-8K tokens (typical file)
Choice: Flash Attention → Longformer
Reason: Need high quality, moderate length

Use Case: Legal document analysis
Sequence: 20K-100K tokens
Choice: BigBird → Reformer
Reason: Very long, need some global reasoning

Use Case: Chatbot (conversational AI)
Sequence: 512-2K tokens (conversation history)
Choice: Flash Attention
Reason: Short enough for standard, want best quality

Use Case: Document summarization
Sequence: 5K-20K tokens
Choice: Longformer → Performer
Reason: Balance quality and speed

Use Case: Scientific paper classification
Sequence: 8K-16K tokens
Choice: BigBird
Reason: Full paper context, need global + local

Use Case: Real-time translation
Sequence: 100-500 tokens (sentences)
Choice: Standard or Flash
Reason: Short sequences, need low latency
```

---

## INTERVIEW QUESTIONS (Hao Hoang Style - Attention)

### Question 1: Attention Memory Calculation
**Q**: You're training a transformer with the following specs:
- Sequence length: n = 8192 tokens
- Hidden dimension: d = 1024
- Number of heads: h = 16
- Batch size: b = 8
- Precision: FP16 (2 bytes)
- Number of layers: L = 24

Calculate:
a) Memory for attention matrices (one layer)
b) Total attention memory (all layers)
c) How would Flash Attention change this?
d) What's the maximum sequence length you can fit on a 40GB A100?

**Expected Answer**:
```
a) Memory for one layer:
Each head processes: d_head = 1024 / 16 = 64 dimensions
Attention matrix per head: n × n = 8192² = 67M elements
All heads: 16 × 67M = 1.07B elements
Batch: 8 × 1.07B = 8.6B elements
Memory: 8.6B × 2 bytes = 17.2 GB per layer

b) Total (all layers):
24 layers × 17.2 GB = 412 GB (!!)
Conclusion: Doesn't fit on any GPU without optimization

c) Flash Attention:
- Doesn't materialize attention matrix
- Working memory: b × n × d × 2 bytes
  = 8 × 8192 × 1024 × 2 = 128 MB per layer
- Total: 24 × 128 MB = 3 GB
- Savings: 412 GB → 3 GB (137× reduction!)

d) Maximum sequence length (40GB GPU):
With standard attention:
40 GB = b × h × n² × L × 2
40 × 10^9 = 8 × 16 × n² × 24 × 2
n² = 40 × 10^9 / (8 × 16 × 24 × 2)
n² ≈ 65M
n ≈ 8000 tokens ✗ (barely fits, no room for params!)

With Flash Attention:
40 GB budget for activations (assume 20 GB for attention)
20 GB = b × n × d × L × 2
n = 20 × 10^9 / (8 × 1024 × 24 × 2)
n ≈ 50K tokens ✓

Conclusion: Flash Attention enables 6× longer sequences
```

### Question 2: Attention Pattern Selection
**Q**: You're building a legal document analysis system. Documents are typically 50K-100K tokens (50-100 pages). You need to:
- Classify document type (contract, agreement, memo, etc.)
- Extract key entities (parties, dates, amounts)
- Answer questions about document content

Which attention mechanism would you choose and why? Compare at least 3 options with trade-offs.

**Expected Answer**:
```
Analysis:
- Length: 50K-100K tokens (very long!)
- Tasks: Classification (global) + extraction (local) + QA (mixed)
- Quality: Legal domain → high accuracy critical
- Latency: Not real-time (batch processing OK)

Option 1: Longformer
Config: window=512, global=[CLS] + entity positions
Pros:
- O(n) complexity with n=100K feasible
- Global tokens for classification
- Local windows for entity context
- Proven for long documents (BigBird paper)
Cons:
- Limited global reasoning (only special tokens)
- May miss long-range dependencies
- Quality: ~97% of standard attention

Memory: 100K × 512 × 2 bytes = 100 MB per layer → 2.4 GB (24 layers)
Training time: ~2 hours per document (acceptable for batch)

Option 2: BigBird
Config: window=512, global=64, random=128
Pros:
- Better connectivity than Longformer (random edges)
- Proven for long document QA
- Global + local + random covers all patterns
Cons:
- More complex implementation
- Random attention adds variance
- Slightly slower than Longformer

Memory: 100K × (512+64+128) × 2 = 140 MB per layer → 3.4 GB
Training time: ~3 hours per document

Option 3: Reformer (LSH)
Config: 8 hash rounds, 16 hash bits, chunk size 128
Pros:
- Can handle 1M tokens if needed (future-proof)
- Lowest memory (reversible layers)
- O(n log n) complexity
Cons:
- LSH hashing = approximation (quality loss)
- Less mature (fewer implementations)
- Harder to debug (hashing is stochastic)

Memory: 100K × 128 × 2 = 25 MB per layer → 600 MB (with chunking)
Training time: ~1 hour per document

RECOMMENDATION: BigBird
Reasoning:
1. Quality critical for legal → need strong attention
2. 50K-100K in BigBird's sweet spot
3. Random attention helps long-range reasoning (QA)
4. 3.4 GB fits easily on modern GPUs (A100)
5. Proven in Google's BigBird paper for long-doc QA

Configuration:
- Window: 512 (2-3 sentences context)
- Global: 64 tokens (classification + key positions)
- Random: 128 (long-range dependencies)
- Total attention per token: 704 (vs 100K in standard)

Fallback plan:
If BigBird quality insufficient:
- Try Longformer with larger window (w=1024)
- Or hierarchical approach (chunk → classify → attend)

If memory becomes issue:
- Switch to Reformer
- Accept slight quality degradation (~3%)
```

### Question 3: Flash Attention Impact
**Q**: You're training GPT-3 scale model (175B params) with Flash Attention. Your manager asks: "Can we save money by using Flash Attention during inference too, not just training?"

Analyze:
a) What are the memory savings during inference?
b) What about latency (speed)?
c) Calculate cost savings for 1M requests/day
d) Are there any downsides?

**Expected Answer**:
```
Scenario:
Model: 175B params (GPT-3 scale)
Context: 2048 tokens (typical)
Precision: FP16
Hardware: A100 80GB

a) Memory savings (inference):

Standard Attention:
- Model: 175B × 2 bytes = 350 GB (need model parallelism)
- KV cache: 96 layers × 96 heads × 128 dim × 2048 ctx × 2 (K+V) × 2 bytes
  = 96 × 96 × 128 × 2048 × 2 × 2
  = 96 layers × 256 MB = 24 GB per request
- Attention matrix: 96 layers × 96 heads × (2048)² × 2 bytes
  = 96 × 96 × 8 MB = 73 GB (!)
- Total per request: 350 + 24 + 73 = 447 GB

Flash Attention:
- Model: 350 GB (same)
- KV cache: 24 GB (same, needed for generation)
- Attention matrix: 0 GB (not materialized!)
- Total per request: 350 + 24 = 374 GB

Savings: 73 GB per request (16% reduction)
Impact: Can batch 2 requests per 80GB GPU vs 1 → 2× throughput!

b) Latency (speed):

Time breakdown (one forward pass):
- Model computation: 50ms (matmuls, FFN)
- Attention computation: 30ms (standard) vs 10ms (Flash)
- Softmax: 10ms (standard) vs 2ms (Flash, fused)
- Total: 90ms (standard) vs 62ms (Flash)

Speedup: 90ms → 62ms (1.45× faster)

For generation (50 tokens):
- Standard: 50 × 90ms = 4.5 seconds
- Flash: 50 × 62ms = 3.1 seconds
- User perceives 1.4 second faster response (significant!)

c) Cost savings:

Baseline (standard attention):
- Requests: 1M per day
- GPU needs: 1M / (86400 sec/day × 1 req/90ms) = 1.04 GPUs
- Rounding: Need 2 GPUs (can't split)
- Cost: 2 × A100 × $3/hr × 24 hrs = $144/day
- Monthly: $4,320

With Flash Attention:
- Throughput: 2× better (batching) + 1.45× faster = 2.9× total
- GPU needs: 1M / 2.9M/day = 0.34 GPUs
- Rounding: Need 1 GPU
- Cost: 1 × $3/hr × 24 hrs = $72/day
- Monthly: $2,160

Savings: $2,160/month = $25,920/year (50% cost reduction!)

d) Downsides:

❌ Implementation complexity:
- Need Flash Attention installed (xFormers, flash-attn)
- Some frameworks don't support (older PyTorch)

❌ Hardware requirements:
- Requires Ampere+ (A100, H100)
- Doesn't work on V100, T4 (older architectures)
- Fallback to standard attention on older GPUs

❌ Numerical differences:
- Flash Attention uses different computation order
- Tiny numerical differences (< 1e-6)
- Usually not a problem, but models trained with standard
  may have slight differences with Flash inference
- Solution: Use Flash for both training and inference

✅ Upside: No accuracy loss
- Same algorithm, just better implementation
- Mathematically equivalent (up to floating point)
- No quality degradation

RECOMMENDATION: YES, use Flash Attention for inference!
- 50% cost savings justify any implementation effort
- Faster response (better UX)
- Can serve more users on same hardware
- No downside if hardware compatible
```

### Question 4: Long Context Debugging
**Q**: You implemented Longformer for a document QA system. Validation accuracy is good (82%) but you notice:
- Questions about document beginnings: 90% accuracy
- Questions about document middles: 82% accuracy
- Questions about document endings: 70% accuracy

Why is this happening? How would you diagnose and fix it?

**Expected Answer**:
```
Diagnosis:

Problem: Accuracy degrades towards document end
Likely causes:
1. Positional encoding bias
2. Attention pattern issue
3. Training data distribution
4. Window size insufficient

Step 1: Check training data distribution
```python
# Analyze answer positions
answer_positions = [get_answer_position(q) for q in questions]
plt.hist(answer_positions, bins=50)

# Expected finding: 
# Most answers in first 50% of documents (common in datasets!)
# Model optimized for early positions
```

Step 2: Analyze attention patterns
```python
# Visualize attention weights
attn_weights = model.get_attention_weights(document, question)
sns.heatmap(attn_weights[0, -1])  # Last layer, last head

# Expected finding:
# Attention concentrated on early tokens
# Question tokens receive less attention from distant context
```

Step 3: Check positional encoding
```python
# Longformer uses sinusoidal embeddings
# May degrade at long distances

# Test: Compare same question-answer at different positions
q = "What is the amount?"
doc = generate_doc_with_answer_at_position(position)

acc_by_position = []
for pos in [100, 1000, 2000, 5000, 10000]:
    doc = generate_doc(answer_at=pos)
    acc = evaluate(q, doc)
    acc_by_position.append(acc)

# Expected: Monotonically decreasing accuracy
```

ROOT CAUSE: Combination of issues
1. Training data skew (answers early in docs)
2. Window attention can't reach far positions
3. Positional encoding degrades at long distances

SOLUTIONS:

Solution 1: Data Augmentation
```python
# Reorder documents during training
def augment_document(doc):
    # Randomly rotate document
    # Ensure answers appear at different positions
    split_point = random.randint(0, len(doc))
    return doc[split_point:] + doc[:split_point]

# Result: Model sees answers at all positions
# Expected improvement: 70% → 76% (end questions)
```

Solution 2: Increase Window Size
```python
# Current: window = 256
# Problem: At position 10000, window only covers [9744, 10256]
# Answer might be at position 5000 (not in window!)

# Solution: Larger window or hierarchical attention
window = 512  # 2× larger
# Trade-off: 2× slower, 2× more memory
# Expected improvement: 70% → 74%
```

Solution 3: Global Attention for Question Tokens
```python
# Make question tokens globally attend
# Ensures question can see all document positions

config = LongformerConfig(
    attention_window=[256] * 24,  # Local window
    global_attention=True,  # Enable global
)

# Mark question tokens as global
def forward(input_ids, question_mask):
    # question_mask: [0,0,0,1,1,1,0,0,...] (1 = question tokens)
    outputs = model(
        input_ids,
        global_attention_mask=question_mask
    )

# Result: Question attends to all positions
# Expected improvement: 70% → 78%
```

Solution 4: Hierarchical Encoding
```python
# Encode document in chunks, then cross-attend

# Step 1: Encode document chunks
chunks = split_document(doc, chunk_size=512)
chunk_encodings = [encode(chunk) for chunk in chunks]  # Local encoding

# Step 2: Chunk-level attention
doc_encoding = attention_over_chunks(chunk_encodings)  # Global view

# Step 3: Question-document attention
answer = cross_attend(question, doc_encoding)

# Result: Can handle very long documents
# Expected improvement: 70% → 80%
```

RECOMMENDED APPROACH: Combination
1. Data augmentation (cheap, always do this)
2. Global attention for questions (small overhead)
3. If still insufficient, increase window to 512

Expected final accuracy:
- Beginning: 90% (unchanged)
- Middle: 84% (+2%)
- End: 78% (+8%)
- Overall: 84% (+2%)

Monitoring:
- Track accuracy by position bins
- Ensure no degradation at any position
- Add to eval metrics: "accuracy by document position"
```

### Question 5: Attention Efficiency Tradeoff
**Q**: Design a token limit for your API offering. You have:
- Standard Attention (up to 4K tokens): $0.01/1K tokens
- Longformer (up to 32K tokens): $0.03/1K tokens  
- Reformer (up to 128K tokens): $0.10/1K tokens

A customer wants to process 1M documents/month:
- 30% are <4K tokens (avg 2K)
- 50% are 4K-32K tokens (avg 16K)
- 20% are >32K tokens (avg 64K)

They want to minimize cost. What strategy do you recommend?

**Expected Answer**:
```
Cost Analysis:

Strategy 1: Always use Reformer (simplest)
- All documents → Reformer
- 30% × 2K × $0.10 = $0.20/doc
- 50% × 16K × $0.10 = $1.60/doc
- 20% × 64K × $0.10 = $6.40/doc
- Weighted: 0.3×0.20 + 0.5×1.60 + 0.2×6.40 = $2.14/doc
- Total: 1M × $2.14 = $2,140,000/month

Strategy 2: Adaptive routing (optimal)
- <4K → Standard ($0.01)
- 4K-32K → Longformer ($0.03)
- >32K → Reformer ($0.10)

Costs:
- 300K docs × 2K × $0.01 = $6,000
- 500K docs × 16K × $0.03 = $240,000
- 200K docs × 64K × $0.10 = $1,280,000
- Total: $1,526,000/month

Savings: $614,000/month (29% reduction!)

Strategy 3: Intelligent chunking
- Break 64K docs into 4× 16K chunks
- Process with Longformer instead of Reformer
- Merge results

Costs:
- 300K docs × 2K × $0.01 = $6,000
- 500K docs × 16K × $0.03 = $240,000
- 200K docs × 4 chunks × 16K × $0.03 = $384,000
- Total: $630,000/month

Savings: $1,510,000/month (70% reduction!)

Trade-offs:

Strategy 2 (Adaptive):
✅ Simple implementation (just route by length)
✅ No quality loss (appropriate model for each doc)
✅ Good savings (29%)
❌ Still expensive for long docs

Strategy 3 (Chunking):
✅ Best cost savings (70%)
✅ Can use faster models
⚠️ Quality loss for cross-chunk reasoning
⚠️ Requires chunking strategy design
❌ Latency: 4× more API calls
❌ Doesn't work for all tasks (some need full context)

RECOMMENDATION: Hybrid approach

Tier 1: Documents <4K (300K docs)
- Use Standard Attention
- Cost: $6,000
- Quality: 100%

Tier 2: Documents 4K-32K (500K docs)
- Use Longformer
- Cost: $240,000
- Quality: 100%

Tier 3: Documents 32K-64K (200K docs)
- Option A: Chunking (if task allows)
  - Split into overlapping 16K chunks (50% overlap)
  - Cost: $240,000 (3× 16K)
  - Quality: 95% (some cross-chunk loss)
- Option B: Full Reformer (if need full context)
  - Cost: $1,280,000
  - Quality: 100%

Decision criteria:
- QA/Classification → Chunking OK (95% quality, 75% savings)
- Summarization/Generation → Need full context (use Reformer)

Expected distribution:
- 80% of long docs can use chunking
- 20% need full context

Final cost:
- Tier 1: $6,000
- Tier 2: $240,000
- Tier 3a (80%): 160K × $192 = $192,000 (chunked)
- Tier 3b (20%): 40K × $6,400 = $256,000 (full)
- Total: $694,000/month

Savings: $1,446,000/month (67% reduction vs naive Reformer)

Implementation:
```python
def route_document(doc, task_type):
    length = len(doc)
    
    if length < 4096:
        return "standard", doc
    elif length < 32768:
        return "longformer", doc
    elif length < 65536:
        if task_type in ["qa", "classification"]:
            # Chunk with overlap
            chunks = split_with_overlap(doc, size=16384, overlap=0.5)
            return "longformer_chunked", chunks
        else:
            # Need full context
            return "reformer", doc
    else:
        # Must use Reformer (too long even with chunking)
        return "reformer", doc

# Monitoring
def track_cost_per_request(model, tokens):
    cost = PRICING[model] * (tokens / 1000)
    log_metric("request_cost", cost)
    log_metric("model_used", model)
```

Additional optimization:
- Cache results for duplicate documents
- Batch processing for better GPU utilization
- Spot instances for batch workloads (50% cost reduction)

Final recommendation letter to customer:
"We recommend adaptive routing with intelligent chunking:
- Expected cost: $694K/month (67% savings)
- Quality: 98% of full-context approach
- Latency: <2 seconds for 95% of documents
- Can scale to larger documents if needed"
```

---

*End of Part 2. Part 3 will cover Loss Computation Details, Sampling Strategies, and Model Architecture Details.*
