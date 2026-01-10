# ML Fundamentals - Part 1: Optimization Algorithms & Attention Mechanisms

**Complete Interview Preparation Guide**
*Sources: Research papers (Attention Is All You Need, Adam paper, LAMB paper), Hao Hoang's materials, Eugene Yan, Chip Huyen, Sebastian Raschka, Lilian Weng, OpenAI/Meta/Google production systems, Sunny Savita's resources*

---

## 24.1 OPTIMIZATION ALGORITHMS

### Overview: The Training Optimization Landscape

**Core Problem**: Given loss function L(θ), find parameters θ* that minimize L.

**Why This Matters (Production Reality)**:
- **Meta's Llama 2 (70B)**: Wrong optimizer choice = $2M extra in compute costs
- **OpenAI GPT-3**: Optimizer + LR schedule = 40% of final performance
- **Google BERT**: Switching SGD→Adam saved 30% training time

**Key Trade-offs**:
| Aspect | SGD+Momentum | Adam/AdamW | Adafactor | LAMB |
|--------|--------------|------------|-----------|------|
| Memory | 1x | 2x | 1x | 2x |
| Speed | Slow | Fast | Fast | Very Fast |
| Generalization | Best (CV) | Good (NLP) | Good | Good |
| Hyperparameter Sensitivity | High | Low | Very Low | Medium |
| Large Batch | Poor | Good | Good | Excellent |

---

### 24.1.1 SGD (Stochastic Gradient Descent)

**Mathematical Foundation**:
```
θ(t+1) = θ(t) - η · ∇L(θ(t))

Components:
- θ: model parameters (weights, biases)
- η: learning rate (0.001 to 0.1 typical range)
- ∇L: gradient of loss w.r.t. parameters
- t: iteration/step number
```

**Algorithm Intuition**:
1. Sample random mini-batch from training data
2. Compute loss on mini-batch: L(θ)
3. Compute gradient: ∇L = ∂L/∂θ
4. Update: θ_new = θ_old - η·∇L
5. Repeat for all batches (1 epoch), then repeat epochs

**Why "Stochastic"?**
- Uses mini-batches (not full dataset) → noisy gradient estimate
- Noise helps escape sharp minima (better generalization)
- Much faster than computing gradient on entire dataset

**Characteristics**:
- **Memory**: O(n) - only stores parameters and gradients
- **Compute**: 2N FLOPs per step (forward + backward)
- **Convergence**: O(1/t) for convex, O(1/√t) for non-convex

**Hyperparameters**:
1. **Learning Rate (η)**:
   - Too high → divergence (loss = NaN)
   - Too low → slow convergence, stuck in local minima
   - Sweet spot: 0.01 to 0.1 for CV, 0.001 to 0.01 for NLP
   
2. **Batch Size**:
   - Small (32-128): Noisy gradients, better generalization
   - Large (256-1024): Smooth gradients, faster convergence
   - Rule of thumb: √(batch_size) scaling for LR

**Production Case Study: Meta's ResNeXt-101**
```
Task: ImageNet classification (1000 classes, 1.2M images)
Model: ResNeXt-101 (44M parameters)
Optimizer: SGD with momentum (β=0.9)
Learning rate: 0.1, decay by 10× at epochs 30, 60, 90
Batch size: 256 (8 GPUs × 32 per GPU)
Weight decay: 1e-4
Training time: 3 days on 8× V100 GPUs

Results:
- Top-1 accuracy: 79.6%
- Better generalization than Adam (79.1%)
- Key insight: Large LR with schedule crucial for sharp→flat minima
```

**Common Pitfalls**:
```python
# ❌ WRONG: Fixed learning rate
optimizer = SGD(model.parameters(), lr=0.1)
# Loss plateaus around epoch 30

# ✅ CORRECT: Learning rate schedule
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

for epoch in range(100):
    train_one_epoch()
    scheduler.step()  # Decay LR at milestones
```

**When to Use SGD**:
- ✅ Computer Vision tasks (proven track record)
- ✅ When you have time to tune learning rate schedule
- ✅ Large batch training (>1K batch size)
- ✅ When memory is constrained (embedded devices)
- ❌ NLP/Transformers (Adam family dominates)
- ❌ Fast prototyping (requires careful tuning)

---

### 24.1.2 Momentum

**The Problem SGD Solves Poorly**:
Imagine a ball rolling in a ravine:
- High curvature across ravine → large gradients perpendicular
- Low curvature along ravine → small gradients parallel
- SGD oscillates wildly across ravine, slow progress along it

**Mathematical Foundation**:
```
v(t) = β · v(t-1) + ∇L(θ(t))
θ(t+1) = θ(t) - η · v(t)

Components:
- v: velocity (exponentially weighted moving average of gradients)
- β: momentum coefficient (0.9 standard, 0.99 for noisy gradients)
```

**Physical Intuition**:
- Ball rolling downhill accumulates velocity
- Accelerates in consistent directions (positive feedback)
- Dampens oscillations in inconsistent directions (negative feedback)
- Can "jump over" small bumps (local minima)

**Exponential Moving Average Math**:
```
v(t) = β·v(t-1) + (1-β)·∇L
     = (1-β) · [∇L(t) + β·∇L(t-1) + β²·∇L(t-2) + ...]

Effective window: 1/(1-β) steps
β=0.9 → averages last ~10 gradients
β=0.99 → averages last ~100 gradients
```

**Nesterov Momentum (Improved)**:
```
# Standard Momentum:
v(t) = β·v(t-1) + ∇L(θ(t))
θ(t+1) = θ(t) - η·v(t)

# Nesterov Momentum (look-ahead):
v(t) = β·v(t-1) + ∇L(θ(t) - η·β·v(t-1))
θ(t+1) = θ(t) - η·v(t)
```

**Why Nesterov is Better**:
- Compute gradient at "look-ahead" position
- More accurate gradient direction
- Better convergence in practice (5-10% faster)

**Hyperparameter Guidelines**:
| β Value | Use Case | Effective Window |
|---------|----------|------------------|
| 0.5 | Fast adaptation, online learning | ~2 steps |
| 0.9 | Standard (default choice) | ~10 steps |
| 0.95 | Noisy gradients | ~20 steps |
| 0.99 | Very noisy, small batches | ~100 steps |

**Production Case Study: Google's Inception-v3**
```
Task: ImageNet classification
Model: Inception-v3 (24M parameters)
Optimizer: RMSprop with momentum (β=0.9)
Learning rate: 0.045, decay 0.94 every 2 epochs
Batch size: 32 (limited by memory)
Auxiliary losses: Yes (for gradient flow)

Results:
- Without momentum: 76.3% Top-1, slow convergence
- With momentum (β=0.9): 78.8% Top-1, 2× faster
- Momentum helped overcome noisy gradients from small batches
```

**Memory & Compute**:
```
Memory: O(n) additional for velocity
Compute: Negligible (just EMA update)
Total overhead: ~33% memory, <1% compute
```

**When to Use Momentum**:
- ✅ **Always use with SGD** (strictly improves it)
- ✅ Noisy gradients (small batches, stochastic objectives)
- ✅ Oscillating loss curves
- ✅ When learning rate is tuned but convergence slow
- ❌ Already using Adam/AdamW (momentum built-in)

---

### 24.1.3 Adam (Adaptive Moment Estimation)

**Why Adam Revolutionized Deep Learning**:
Before Adam (2014), training deep networks required:
- Hand-tuned learning rates per layer
- Careful initialization schemes
- Gradient clipping thresholds
Adam made training "just work" with default hyperparameters.

**Mathematical Foundation**:
```
# First moment (mean of gradients)
m(t) = β₁ · m(t-1) + (1-β₁) · ∇L(θ)

# Second moment (variance of gradients)
v(t) = β₂ · v(t-1) + (1-β₂) · (∇L(θ))²

# Bias correction (early steps have low EMA)
m̂(t) = m(t) / (1 - β₁^t)
v̂(t) = v(t) / (1 - β₂^t)

# Parameter update
θ(t+1) = θ(t) - η · m̂(t) / (√v̂(t) + ε)

Default hyperparameters:
- η = 0.001 (learning rate)
- β₁ = 0.9 (first moment decay)
- β₂ = 0.999 (second moment decay)
- ε = 1e-8 (numerical stability)
```

**Intuition Breakdown**:
1. **First moment (m)**: Direction to move (like momentum)
2. **Second moment (v)**: Confidence in direction (like RMSprop)
3. **Division by √v**: Larger historical gradients → smaller current step
4. **Bias correction**: Prevents underestimation in early training

**Per-Parameter Adaptive Learning Rates**:
```
Consider two parameters: w₁ (embedding), w₂ (output layer)

If gradient history:
- w₁: [0.001, 0.001, 0.001] → small v → large effective LR
- w₂: [10.0, 12.0, 11.0] → large v → small effective LR

Adam automatically handles scale differences!
```

**Memory Cost Analysis**:
```
Parameters: θ ∈ R^n
SGD: stores θ, ∇L → O(n) memory
Adam: stores θ, m, v, ∇L → O(2n) memory

For GPT-3 (175B parameters):
- SGD: 700 GB (FP32) or 350 GB (FP16)
- Adam: 1.4 TB (FP32) or 700 GB (FP16)
```

**Production Case Study: OpenAI GPT-3**
```
Task: Autoregressive language modeling
Model: 175B parameters, 96 layers, 96 heads
Dataset: 300B tokens (filtered CommonCrawl, books, Wikipedia)

Optimizer: Adam
Learning rate: 6e-4
β₁ = 0.9, β₂ = 0.95 (NOT default 0.999!)
Batch size: 3.2M tokens
Sequence length: 2048
Warmup: 375M tokens (linear warmup)
Decay: Cosine to 10% of peak LR over 300B tokens

Why β₂ = 0.95 instead of 0.999?
- Suggested by Ilya Sutskever
- Faster adaptation to data distribution changes
- Better for large-scale (>10B params) training
- Empirically better final perplexity (2.20 vs 2.25)

Training time: ~34 days on 10,000 V100 GPUs
Estimated cost: $4.6M
```

**Bias Correction Example**:
```
Step 1: m(1) = 0.9·0 + 0.1·g = 0.1g
        Without correction: m̂(1) = 0.1g (underestimate!)
        With correction: m̂(1) = 0.1g / (1-0.9) = g ✓

Step 10: m(10) ≈ g
         Correction: m̂(10) = m(10) / (1-0.9^10) ≈ m(10)
         (correction becomes negligible)
```

**Common Pitfalls**:
```python
# ❌ WRONG: L2 regularization in loss
loss = mse_loss + 0.01 * sum(p.pow(2).sum() for p in model.parameters())
optimizer = Adam(model.parameters(), lr=1e-3)
# Problem: L2 penalty scaled by adaptive LR, weakens regularization

# ✅ CORRECT: Use AdamW (decoupled weight decay)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

**Convergence Properties**:
- **Theory**: Converges to critical points in non-convex problems
- **Practice**: Often faster than SGD in early training, may plateau
- **Caveat**: Can generalize worse on CV tasks (sharp minima)

**When to Use Adam**:
- ✅ NLP/Transformers (industry standard)
- ✅ Fast prototyping (less hyperparameter tuning)
- ✅ Sparse gradients (embedding layers, NLP)
- ✅ Small batches (<128)
- ✅ Noisy gradients (RL, adversarial training)
- ⚠️ Computer Vision (SGD+Momentum often better)
- ❌ When memory constrained (use AdaFactor instead)

---

### 24.1.4 AdamW (Adam with Decoupled Weight Decay)

**The Critical Bug in Original Adam**:
Original Adam paper proposed L2 regularization:
```
loss = original_loss + λ/2 · ||θ||²
```
Problem: In adaptive optimizers, this doesn't work as intended!

**L2 Regularization vs Weight Decay**:
```
# L2 Regularization (added to loss)
∇L_total = ∇L + λ·θ
In Adam: gets divided by √v (adaptive scaling)
Effective regularization varies per parameter!

# Weight Decay (decoupled from gradient)
θ(t+1) = θ(t) - η·gradient - η·λ·θ(t)
Applied uniformly to all parameters
```

**Mathematical Foundation (AdamW)**:
```
m(t) = β₁·m(t-1) + (1-β₁)·∇L(θ)
v(t) = β₂·v(t-1) + (1-β₂)·(∇L(θ))²

m̂(t) = m(t) / (1 - β₁^t)
v̂(t) = v(t) / (1 - β₂^t)

θ(t+1) = θ(t) - η·[m̂(t)/(√v̂(t) + ε) + λ·θ(t)]
                     └─ Adam update ─┘   └─ weight decay ─┘
```

**Why This Matters (Empirical Evidence)**:
```
Experiment: BERT fine-tuning on GLUE benchmark
Dataset: 8 NLP tasks (sentiment, QA, NLI, etc.)

Results:
Adam (L2 reg):     84.2% average score
AdamW (wd=0.01):   86.1% average score
Improvement:       +1.9 absolute points

Key insight: Weight decay improves generalization
across ALL tasks, especially small datasets (CoLA, RTE)
```

**Production Case Study: HuggingFace Transformers Default**
```
Task: Fine-tuning BERT/GPT/T5/Llama
Optimizer: AdamW (became default in transformers library)

Hyperparameters:
learning_rate: 5e-5 (start here)
weight_decay: 0.01 (critical for generalization)
β₁: 0.9
β₂: 0.999
ε: 1e-8
warmup_steps: 500 (linear warmup)
max_steps: 10000
scheduler: linear decay to 0

Why these values?
- 5e-5: Sweet spot for fine-tuning (10x smaller than pre-training)
- 0.01 wd: Prevents overfitting on small datasets
- Linear decay: Allows fine-grained convergence

Results (typical fine-tuning):
Training time: 30 minutes on 1× V100
Generalization: +2-3% over Adam
Reproducibility: High (less hyperparameter sensitive)
```

**Weight Decay Guidelines by Model Size**:
```
| Model Size | Weight Decay | Reasoning |
|------------|--------------|-----------|
| <100M      | 0.01 - 0.1   | Aggressive regularization |
| 100M-1B    | 0.01         | Standard (BERT, GPT-2) |
| 1B-10B     | 0.01 - 0.001 | Less regularization |
| >10B       | 0.0001-0.001 | Minimal (model self-regularizes) |
```

**Impact on Training Dynamics**:
```python
# Weight decay creates "implicit" learning rate schedule
effective_lr = base_lr * (1 - λ)^t

Example: lr=1e-3, λ=0.01, t=1000 steps
effective_lr ≈ 1e-3 * 0.99^1000 ≈ 4.3e-8
(automatic decay without explicit scheduler!)
```

**Comparison with Adam**:
| Aspect | Adam (L2) | AdamW (Weight Decay) |
|--------|-----------|----------------------|
| Regularization strength | Per-parameter (inconsistent) | Uniform (consistent) |
| Generalization | Worse | Better |
| Hyperparameter sensitivity | Higher | Lower |
| Memory | Same | Same |
| Compute | Same | Same |

**When to Use AdamW**:
- ✅ **Always prefer over Adam** (strictly better)
- ✅ Fine-tuning LLMs (BERT, GPT, T5, Llama)
- ✅ Training transformers from scratch
- ✅ Small datasets (< 100K examples)
- ✅ When generalization > training accuracy matters
- ✅ Production systems (stable, reproducible)

**Implementation Note**:
```python
# PyTorch: AdamW is separate optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # This is decoupled!
)

# HuggingFace: Default in Trainer
training_args = TrainingArguments(
    learning_rate=5e-5,
    weight_decay=0.01,  # Automatically uses AdamW
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8
)
```

---

### 24.1.5 Adafactor (Memory-Efficient Adaptive Optimizer)

**The Memory Crisis in Large-Scale Training**:
```
Problem: Adam stores 2 states per parameter (m and v)

For GPT-3 (175B params, FP32):
Parameters: 175B × 4 bytes = 700 GB
Adam states: 175B × 8 bytes = 1.4 TB
Total: 2.1 TB (requires 32× A100 80GB GPUs!)

For GPT-4 (rumored 1.76T params):
Parameters: 7 TB
Adam states: 14 TB
Total: 21 TB (infeasible even on supercomputers)
```

**Adafactor's Innovation: Factorized Second Moment**:
```
Instead of storing full second moment v ∈ R^(d₁×d₂):

v_full: d₁ × d₂ elements (e.g., 4096 × 4096 = 16M)

Factor into:
r ∈ R^d₁ (row statistics)
c ∈ R^d₂ (column statistics)

v ≈ r ⊗ c (outer product approximation)
Storage: d₁ + d₂ elements (e.g., 4096 + 4096 = 8K)

Memory reduction: (d₁×d₂)/(d₁+d₂) ≈ d/2 for square matrices
```

**Mathematical Foundation**:
```
# Row and column statistics
r(t) = β₂·r(t-1) + (1-β₂)·mean(g², axis=1)
c(t) = β₂·c(t-1) + (1-β₂)·mean(g², axis=0)

# Reconstructed second moment
v(t) ≈ r(t) ⊗ c(t) = r(t)·c(t)ᵀ

# Update (with additional features)
θ(t+1) = θ(t) - min(η, 1/√t)·m(t)/√v(t)
                 └─ adaptive LR ─┘
```

**Additional Innovations**:
1. **Adaptive Learning Rate**: No need to tune η!
   ```
   η_t = min(η_max, 1/√t)
   Automatically decays without scheduler
   ```

2. **Gradient Clipping**: Built-in stability
   ```
   RMS clipping: ||g||₂ ≤ threshold
   Prevents exploding gradients
   ```

3. **Relative Step Sizes**: Scale-invariant
   ```
   step_size ∝ ||θ|| (larger params get larger steps)
   ```

**Memory Comparison**:
```
Model: 7B parameters (typical for Llama, Mistral)
Precision: FP32

Optimizer    | Memory      | Total
-------------|-------------|-------
SGD          | 0           | 28 GB
Adam         | 56 GB       | 84 GB
Adafactor    | 0.4 GB      | 28.4 GB

Savings: 60× less than Adam!
```

**Production Case Study: Google's T5**
```
Task: Text-to-text transfer learning
Model: T5-11B (11 billion parameters)
Dataset: C4 (750 GB of text)

Optimizer: Adafactor
- No learning rate tuning (used default schedule)
- Batch size: 128 sequences
- Training: 1 million steps
- Hardware: 256 TPUv3 cores
- Training time: 1 month

Key results:
- Comparable quality to Adam (within 0.5% on GLUE)
- 3× less memory usage
- No hyperparameter tuning needed
- Successfully scaled to 11B params on limited hardware

Without Adafactor:
- Would need 3× more TPUs
- Or reduce model size to ~4B params
```

**Hyperparameter Guidelines**:
```python
# Adafactor rarely needs tuning, but if needed:
optimizer = Adafactor(
    model.parameters(),
    lr=None,              # Use adaptive LR (recommended)
    # lr=1e-3,            # Or specify manually
    eps=(1e-30, 1e-3),    # Numerical stability
    clip_threshold=1.0,   # Gradient clipping
    decay_rate=-0.8,      # LR decay schedule
    beta1=None,           # No first moment (saves memory)
    weight_decay=0.0,     # Usually not needed
    scale_parameter=True, # Relative step sizes
    relative_step=True,   # Adaptive LR
    warmup_init=True      # Gradual warmup
)
```

**Trade-offs**:
| Aspect | Advantage | Disadvantage |
|--------|-----------|--------------|
| Memory | 60× less than Adam | Approximation of v (slight accuracy loss) |
| Hyperparameters | No tuning needed | Less control over optimization |
| Convergence | Similar to Adam | Can be slower initially |
| Stability | Built-in clipping | Factorization can be unstable with extreme shapes |

**When to Use Adafactor**:
- ✅ Large models (>1B params) with memory constraints
- ✅ Training on limited hardware (consumer GPUs)
- ✅ When you can't tune learning rates
- ✅ Pretraining large LLMs (T5, UL2, Flan)
- ⚠️ Fine-tuning (AdamW often better, less memory critical)
- ❌ Small models (<100M params, memory not bottleneck)
- ❌ When you need exact Adam behavior

---

### 24.1.6 LAMB (Layer-wise Adaptive Moments for Batch Training)

**The Large Batch Training Problem**:
```
Standard recipe: Scale LR with batch size
- Batch 256 → LR 0.001
- Batch 2048 → LR 0.008 (linear scaling)

Problem: This breaks for very large batches!
- Batch 32K → LR 0.125 (explodes!)
- Different layers need different LR scales
```

**Real-world Motivation (Google BERT Pre-training)**:
```
Original BERT training:
- Batch size: 256
- Training time: 4 days on 16 TPU pods (64 chips)
- Cost: ~$1,000

Goal: Reduce to 76 minutes using larger batches
Target batch size: 32,768 (128× larger!)
Problem: Standard optimizers fail at this scale
```

**LAMB's Innovation: Layer-wise Adaptation**:
```
Instead of global learning rate η:

η_layer = η · ||W_layer|| / ||m_layer / √v_layer + ε||
          └─ global ─┘ └─ layer norm ─┘ └─ update norm ─┘

Key insight: Normalize update by layer's parameter norm
Effect: Each layer gets appropriate step size
```

**Mathematical Foundation**:
```
# Standard Adam update:
update = m̂ / (√v̂ + ε)
θ_new = θ - η · update

# LAMB update (per layer):
r₁ = ||θ_layer||₂           # Layer parameter norm
r₂ = ||update_layer||₂      # Update norm
η_layer = η · r₁ / r₂       # Adaptive LR
θ_new = θ - η_layer · update

Full algorithm:
1. Compute Adam-style m and v
2. Compute bias-corrected m̂ and v̂
3. Form update: u = m̂ / (√v̂ + ε)
4. Per layer: r₁ = ||θ||, r₂ = ||u||
5. Actual update: θ ← θ - η·(r₁/r₂)·u
```

**Why Layer-wise Adaptation Matters**:
```
Example: BERT model

Layer         | Param Norm | Grad Norm | Ratio
--------------|------------|-----------|-------
Embedding     | 15.2       | 0.003     | 5067
Layer 1       | 2.1        | 0.12      | 17.5
Layer 12      | 1.8        | 0.08      | 22.5
Output        | 0.4        | 0.15      | 2.7

Without LAMB:
- Same η for all layers → embedding barely moves
- Or η too high → output layer explodes

With LAMB:
- Each layer gets appropriate step size
- Balanced training across all layers
```

**Production Case Study: Google's BERT (LAMB paper)**
```
Task: Masked language modeling (BERT-Large)
Model: 340M parameters, 24 layers
Dataset: BooksCorpus + Wikipedia (3.3B words)

Standard training (baseline):
- Optimizer: Adam
- Batch size: 256
- Learning rate: 1e-4
- Warmup: 10k steps
- Training: 1M steps = 4 days on 16 TPU pods
- Perplexity: 3.99

LAMB training (optimized):
- Optimizer: LAMB
- Batch size: 32,768 (128× larger!)
- Learning rate: 0.00576 (computed by formula)
- Warmup: 5k steps
- Training: 8,599 steps = 76 minutes on 512 TPUs
- Perplexity: 3.99 (same quality!)

Key results:
- 76× faster wall-clock time
- Same final model quality
- Successfully scaled batch size 128×
- Enabled efficient use of massive parallelism

Cost analysis:
Standard: 4 days × 16 pods × $8/hr = $5,120
LAMB: 76 min × 512 TPUs × $8/hr = $520
Savings: 90% cost reduction!
```

**Batch Size Scaling Formula (LAMB Paper)**:
```
Learning rate scaling:
η = η_base × √(batch_size / batch_base)

BERT example:
batch_base = 256, η_base = 1e-4
batch_new = 32768

η_new = 1e-4 × √(32768 / 256)
      = 1e-4 × √128
      = 1e-4 × 11.31
      = 1.131e-3
      ≈ 0.00113 (used 0.00576 in practice after tuning)
```

**Hyperparameter Guidelines**:
```python
# LAMB for large batch training
optimizer = Lamb(
    model.parameters(),
    lr=0.00576,           # Scaled for batch size
    betas=(0.9, 0.999),   # Same as Adam
    eps=1e-6,             # Slightly larger for stability
    weight_decay=0.01,    # Important for large batches
)

# LR schedule (critical for LAMB)
scheduler = LinearWarmupCosineDecay(
    optimizer,
    warmup_steps=5000,    # ~10% of training
    total_steps=100000,
    min_lr=0.0
)
```

**LAMB vs Other Optimizers**:
| Optimizer | Max Batch Size | Speed (BERT) | Memory |
|-----------|----------------|--------------|--------|
| Adam      | 1K             | 4 days       | 2x     |
| AdamW     | 2K             | 2 days       | 2x     |
| LAMB      | 64K            | 76 min       | 2x     |
| LARS      | 32K            | 2 hours      | 1x     |

**When to Use LAMB**:
- ✅ Large batch training (>8K batch size)
- ✅ Data-parallel training across 100+ GPUs
- ✅ When training time is critical (production deadlines)
- ✅ Pretraining large models (BERT, RoBERTa, T5)
- ✅ When you have massive compute budget (512+ GPUs/TPUs)
- ⚠️ Fine-tuning (AdamW often better, smaller batches work)
- ❌ Small batches (<1K, LAMB overhead not worth it)
- ❌ Limited hardware (<8 GPUs, can't use large batches)

**Implementation Notes**:
```python
# LAMB not in PyTorch by default, use apex or timm
from apex.optimizers import FusedLAMB

optimizer = FusedLAMB(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0.01,
    bias_correction=True
)

# Critical: Use warmup + decay
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

---

### 24.1.7 RMSprop (Root Mean Square Propagation)

**Historical Context**:
- Developed by Geoffrey Hinton (Coursera lecture, 2012)
- Never formally published (!)
- Precursor to Adam
- Still used in specific domains (RL, RNNs)

**The Problem It Solves**:
```
Gradient magnitudes vary dramatically across parameters:

Parameter      | Typical Gradient | Problem
---------------|------------------|------------------
Embedding      | 1e-4             | Needs large LR
Hidden weights | 1e-1             | Needs medium LR  
Output weights | 1e+1             | Needs small LR

Fixed LR: Can't satisfy all parameters!
```

**Mathematical Foundation**:
```
v(t) = β · v(t-1) + (1-β) · (∇L)²
θ(t+1) = θ(t) - η / √(v(t) + ε) · ∇L

Components:
- v: Moving average of squared gradients
- β: Decay rate (typically 0.9 or 0.99)
- ε: Numerical stability (1e-8)
- η: Global learning rate (0.001 typical)
```

**Intuition**:
- Large historical gradients → large v → small effective LR
- Small historical gradients → small v → large effective LR
- Automatically adapts per-parameter learning rates

**Comparison with Momentum**:
```
Momentum: Accelerates in consistent directions
RMSprop: Adapts learning rate to gradient magnitude

Can combine both:
v(t) = β₂ · v(t-1) + (1-β₂) · (∇L)²
m(t) = β₁ · m(t-1) + (1-β₁) · ∇L
θ(t+1) = θ(t) - η · m(t) / √(v(t) + ε)

This is essentially Adam without bias correction!
```

**Production Case Study: DeepMind's DQN (Atari)**
```
Task: Reinforcement learning on Atari games
Model: CNN (3 conv layers + 2 FC)
Algorithm: Deep Q-Network (DQN)

Optimizer: RMSprop
- Learning rate: 0.00025
- β: 0.95 (faster decay than default)
- ε: 0.01 (larger for stability)
- Gradient clipping: [-1, 1]

Why RMSprop?
- RL gradients extremely noisy
- Reward scale varies per game (0-10K points)
- Needs per-parameter adaptation
- Momentum can be unstable in RL

Results:
- 29/49 Atari games: Superhuman performance
- Pong: 19.5 score (human: 9.3)
- Breakout: 401 score (human: 31)
- RMSprop crucial for stability
```

**Hyperparameter Guidelines**:
```python
# Standard RMSprop
optimizer = RMSprop(
    model.parameters(),
    lr=0.001,         # Higher than Adam (no momentum)
    alpha=0.99,       # Decay rate (PyTorch calls β "alpha")
    eps=1e-8,         # Numerical stability
    weight_decay=0,   # Usually not used with RMSprop
    momentum=0.0      # Can add momentum if desired
)

# RL-specific (DQN-style)
optimizer = RMSprop(
    model.parameters(),
    lr=0.00025,
    alpha=0.95,       # Faster decay for RL
    eps=0.01,         # Larger epsilon for stability
    centered=False    # False for standard, True for centered variant
)
```

**Centered RMSprop**:
```
Standard: v = E[g²]
Centered: v = E[g²] - E[g]²  (variance, not second moment)

Advantage: More stable updates
Disadvantage: Extra computation (need to track mean)
```

**When to Use RMSprop**:
- ✅ Reinforcement learning (DQN, A3C, PPO)
- ✅ Recurrent networks (LSTMs, GRUs)
- ✅ Non-stationary objectives (online learning)
- ✅ When Adam is unstable
- ⚠️ NLP/Transformers (Adam/AdamW preferred)
- ❌ Computer Vision (SGD+Momentum better)
- ❌ New projects (use Adam/AdamW instead)

**Why Adam Replaced RMSprop**:
1. Adam adds momentum (faster convergence)
2. Adam has bias correction (better early training)
3. Adam more stable across tasks
4. Adam better defaults (less tuning)

**RMSprop Still Used In**:
- Legacy RL codebases
- Specific RNN applications
- When Adam fails (rare cases)

---

### 24.1.8 Optimizer Selection Guide

**Decision Tree**:
```
Task Type?
├─ Computer Vision (ImageNet, COCO)
│  ├─ Have time to tune? → SGD + Momentum (best generalization)
│  └─ Fast prototyping? → AdamW (good enough, less tuning)
│
├─ NLP / Transformers (BERT, GPT, T5)
│  ├─ Fine-tuning? → AdamW (industry standard)
│  ├─ Pre-training <1B params? → AdamW
│  └─ Pre-training >10B params? → Adafactor (memory-efficient)
│
├─ Reinforcement Learning (RL)
│  ├─ Policy Gradients? → Adam
│  └─ Q-Learning (DQN)? → RMSprop
│
├─ Large Batch Training (>8K batch)
│  └─ → LAMB (enables massive parallelism)
│
└─ Memory Constrained (<16GB GPU)
   └─ → Adafactor (60× less than Adam)
```

**Quick Reference Table**:
| Use Case | Optimizer | Learning Rate | Batch Size | Notes |
|----------|-----------|---------------|------------|-------|
| ImageNet (ResNet) | SGD+Mom | 0.1 → 0.001 | 256 | Best Top-1 accuracy |
| BERT Fine-tuning | AdamW | 5e-5 | 32 | Industry standard |
| GPT-3 Pre-training | Adam | 6e-4 | 3.2M tokens | β₂=0.95 critical |
| T5-11B Training | Adafactor | Auto | 128 | Fits in memory |
| BERT 76-min | LAMB | 0.00576 | 32K | Extreme parallelism |
| DQN (Atari) | RMSprop | 2.5e-4 | 32 | RL stability |

**Hyperparameter Sensitivity Ranking** (high = needs careful tuning):
1. **SGD**: Very high (LR + momentum + schedule)
2. **RMSprop**: High (LR + decay rate)
3. **Adam**: Medium (LR sufficient, rest use defaults)
4. **AdamW**: Low (defaults work well)
5. **LAMB**: Medium (batch-dependent LR scaling)
6. **Adafactor**: Very low (works with no tuning!)

**Common Mistakes to Avoid**:
```python
# ❌ MISTAKE 1: Wrong weight decay implementation
optimizer = Adam(params, lr=1e-3)
loss = mse + 0.01 * l2_norm(params)  # Broken with Adam!

# ✅ CORRECT: Use AdamW
optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)

# ❌ MISTAKE 2: No learning rate schedule
optimizer = Adam(params, lr=1e-3)
train(100_epochs)  # Plateaus early

# ✅ CORRECT: Add warmup + decay
optimizer = AdamW(params, lr=1e-3)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps=500, total_steps=10000
)

# ❌ MISTAKE 3: Adam on CV without trying SGD
optimizer = Adam(params, lr=1e-3)
# Often 1-2% worse than SGD+Momentum!

# ✅ CORRECT: Try SGD first for CV
optimizer = SGD(params, lr=0.1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# ❌ MISTAKE 4: Using LAMB for small batches
optimizer = LAMB(params, lr=1e-3)  # Batch=128
# Overhead not worth it!

# ✅ CORRECT: LAMB only for large batches
if batch_size >= 8192:
    optimizer = LAMB(params, lr=scaled_lr)
else:
    optimizer = AdamW(params, lr=5e-5)

# ❌ MISTAKE 5: Forgetting bias correction
m = 0.9 * m + 0.1 * grad
theta -= lr * m  # Biased in early steps!

# ✅ CORRECT: Apply bias correction
m = 0.9 * m + 0.1 * grad
m_hat = m / (1 - 0.9**step)
theta -= lr * m_hat
```

---

### 24.1.9 Hyperparameter Tuning Strategies

**Learning Rate Tuning (Most Critical)**:

**Method 1: Learning Rate Range Test (fast.ai)**
```python
# Find LR by exponentially increasing from 1e-8 to 1
lrs, losses = [], []
lr = 1e-8

for batch in train_loader:
    optimizer.param_groups[0]['lr'] = lr
    loss = train_step(batch)
    
    lrs.append(lr)
    losses.append(loss)
    
    if loss > 10 * min(losses):  # Diverging
        break
    
    lr *= 1.1  # Exponential increase

# Plot and choose LR where loss decreases fastest
optimal_lr = lrs[losses.index(min(losses))] / 10
```

**Method 2: Grid Search (Systematic)**
```
Learning rates to try: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
Batch sizes to try: [16, 32, 64, 128, 256]
Weight decay: [0, 1e-5, 1e-4, 1e-3, 1e-2]

Total combinations: 6 × 5 × 5 = 150 runs
Use validation loss after 10% of training to prune bad configs
```

**Method 3: Bayesian Optimization (Efficient)**
```python
# Using Optuna
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    val_loss = train_and_evaluate(optimizer)
    
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"Best LR: {study.best_params['lr']}")
```

**Batch Size vs Learning Rate Scaling**:
```
Linear Scaling Rule (Goyal et al., Facebook):
If batch_size is multiplied by k, multiply LR by k

Example:
Batch 256, LR 0.1 → 93% accuracy
Batch 512, LR 0.2 → 93% accuracy ✓
Batch 8192, LR 3.2 → 92% accuracy (breaks down!)

Square Root Scaling (Hoffer et al.):
LR ∝ √batch_size (better for large batches)

Batch 256, LR 0.1
Batch 8192, LR 0.1 × √(8192/256) = 0.566 ✓
```

**Warmup Duration Heuristics**:
```
Task                 | Warmup Steps | Reasoning
---------------------|--------------|-------------------------
Fine-tuning (BERT)   | 500-1000     | Quick adaptation
Pre-training (GPT)   | 2000-10000   | Stable start critical
RL (DQN)             | 0            | No warmup needed
Vision (ImageNet)    | 5 epochs     | SGD+Momentum stable
```

**Production Checklist**:
```
□ Tried multiple learning rates (at least 5 values)
□ Tested with and without warmup
□ Compared 2-3 optimizers (e.g., SGD, Adam, AdamW)
□ Tuned weight decay (if using AdamW)
□ Chose appropriate scheduler (cosine, linear, step)
□ Validated on held-out set (not training set!)
□ Logged all hyperparameters (for reproducibility)
□ Measured wall-clock time (not just loss)
□ Tested generalization on test set
□ Documented final configuration
```

---

## INTERVIEW QUESTIONS (Hao Hoang Style)

### Question 1: Optimizer Memory Calculation
**Q**: You're training a 7B parameter model (Llama-2-7B) in FP32 precision. Calculate the GPU memory required for:
a) Model parameters
b) Adam optimizer states
c) Gradients
d) Total training memory (excluding activations)

Then: If you only have a 40GB A100 GPU, what optimizer and precision would you choose to fit the model?

**Expected Answer**:
```
a) Parameters: 7B × 4 bytes = 28 GB

b) Adam states:
   - First moment (m): 7B × 4 bytes = 28 GB
   - Second moment (v): 7B × 4 bytes = 28 GB
   - Total: 56 GB

c) Gradients: 7B × 4 bytes = 28 GB

d) Total: 28 + 56 + 28 = 112 GB (doesn't fit!)

Solution for 40GB GPU:
Option 1: Adafactor + FP32
- Parameters: 28 GB
- Adafactor states: ~0.5 GB (factorized)
- Gradients: 28 GB
- Total: 56.5 GB (still too much!)

Option 2: AdamW + FP16 mixed precision
- Parameters: 14 GB (FP16)
- Adam states: 28 GB (kept in FP32)
- Gradients: 14 GB (FP16)
- Master weights: 14 GB (FP32 copy)
- Total: 70 GB (still too much!)

Option 3: Adafactor + FP16
- Parameters: 14 GB
- Adafactor: ~0.3 GB
- Gradients: 14 GB
- Total: 28.3 GB ✓ Fits!

Best choice: Adafactor with mixed precision FP16
```

### Question 2: Adam vs SGD Trade-off
**Q**: You're training a ResNet-50 on ImageNet. Your teammate suggests using Adam because "it converges faster." You've heard SGD+Momentum generalizes better. Design an experiment to test both and explain:
a) What metrics to track
b) What would convince you to use Adam despite slower generalization
c) What hyperparameters need tuning for fair comparison

**Expected Answer**:
```
a) Metrics to track:
- Training loss & accuracy (convergence speed)
- Validation accuracy (generalization)
- Test accuracy (final performance)
- Wall-clock time per epoch
- Loss landscape sharpness (optional: Hessian eigenvalues)

b) Would use Adam if:
- Deadline is tight (production deadline in 1 week)
- Adam reaches 75% accuracy in 1 day, SGD needs 3 days
- Final accuracy diff <1% (75.2% vs 76.1% acceptable)
- Training cost matters more than 1% accuracy

c) Fair comparison hyperparameters:
SGD:
- LR: [0.01, 0.05, 0.1, 0.5] with momentum 0.9
- Schedule: MultiStep [30, 60, 90] or Cosine
- Warmup: 5 epochs
- Weight decay: 1e-4

Adam:
- LR: [1e-4, 5e-4, 1e-3, 5e-3]
- Schedule: Cosine decay or None
- Warmup: 5 epochs
- Weight decay: 1e-4 (or use AdamW)

Shared:
- Same batch size (256)
- Same augmentation
- Same training epochs (100)
- Same hardware (8× V100)

Expected result (based on literature):
SGD: 76.5% Top-1, 3 days
Adam: 75.8% Top-1, 1.5 days

Decision: Use SGD if accuracy matters, Adam if time critical
```

### Question 3: Weight Decay Deep Dive
**Q**: Explain why this code is problematic and how to fix it:
```python
model = GPT2()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

def compute_loss(batch):
    logits = model(batch)
    ce_loss = F.cross_entropy(logits, targets)
    l2_reg = sum(p.pow(2).sum() for p in model.parameters())
    return ce_loss + 0.01 * l2_reg

for batch in dataloader:
    loss = compute_loss(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Expected Answer**:
```
Problem: L2 regularization is incompatible with Adam!

Why it's broken:
1. L2 penalty adds to gradient: ∇L_total = ∇L_ce + λ·θ
2. Adam normalizes by √v (adaptive LR)
3. Parameters with large historical gradients:
   - Large v → small effective LR
   - L2 penalty gets scaled down → weak regularization
4. Parameters with small historical gradients:
   - Small v → large effective LR
   - L2 penalty gets scaled up → strong regularization
5. Result: Inconsistent regularization strength per parameter!

Fix 1: Use AdamW (decoupled weight decay)
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01  # This is NOT added to loss
)

def compute_loss(batch):
    logits = model(batch)
    return F.cross_entropy(logits, targets)  # No L2 term!

# AdamW applies: θ ← θ - lr*(grad + wd*θ) after gradient step
```

Fix 2: Use SGD (L2 reg works fine)
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.01  # Equivalent to L2 in SGD
)
```

Impact:
- Wrong way: Weight decay effect varies per parameter
- AdamW: Uniform weight decay, better generalization
- Empirical: +1-2% validation accuracy improvement
```

### Question 4: Large Batch Training
**Q**: Your company wants to reduce BERT pre-training time from 3 days to 3 hours using 256 GPUs instead of 8. Current setup:
- Batch size: 256 (8 GPUs × 32 per GPU)
- Learning rate: 1e-4
- Optimizer: Adam

What changes are needed to scale to 256 GPUs? Calculate new hyperparameters and explain potential issues.

**Expected Answer**:
```
Target setup:
- 256 GPUs × 32 per GPU = 8192 batch size (32× increase)
- Need to scale hyperparameters appropriately

Changes required:

1. Switch optimizer: Adam → LAMB
   Reason: LAMB designed for large batch training

2. Scale learning rate:
   Option A (Linear scaling): 1e-4 × 32 = 3.2e-3
   Option B (Sqrt scaling): 1e-4 × √32 = 5.66e-4
   
   Recommendation: Start with sqrt scaling (more stable)
   
3. Adjust warmup:
   - Old: 10,000 steps
   - New: 10,000 / 32 = 312 steps
   Reason: Same number of samples seen during warmup

4. Adjust total steps:
   - Old: 1,000,000 steps = 256M samples
   - New: 1,000,000 / 32 = 31,250 steps (same 256M samples)

5. Implementation:
```python
optimizer = LAMB(
    model.parameters(),
    lr=5.66e-4,           # Sqrt scaled
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0.01
)

scheduler = LinearWarmupCosineDecay(
    optimizer,
    warmup_steps=312,
    total_steps=31250,
    min_lr=0.0
)
```

Potential issues:

1. Gradient staleness
   - Large batch = less frequent updates
   - Model sees 32× more data before updating
   - Solution: LAMB's layer-wise adaptation helps

2. Reduced regularization
   - Large batch = less stochastic noise
   - Can overfit more easily
   - Solution: Increase weight decay to 0.02

3. Communication overhead
   - 256 GPUs need efficient all-reduce
   - Bandwidth bottleneck with many GPUs
   - Solution: Use hierarchical all-reduce, NCCL

4. Generalization gap
   - Very large batches may hurt generalization
   - Monitor validation perplexity carefully
   - May need slight adjustment to LR

Expected result:
- Training time: 3 hours ✓
- Final perplexity: Within 1% of baseline
- Cost: 256 GPU × 3 hrs vs 8 GPU × 72 hrs (same cost!)

Key insight: Time reduction, not cost reduction
```

### Question 5: Optimizer Debugging
**Q**: You're fine-tuning GPT-2 on a custom dataset. Training loss decreases but validation loss increases after epoch 2. You're using AdamW with lr=5e-5, weight_decay=0.0. Diagnose the problem and provide 3 solutions with trade-offs.

**Expected Answer**:
```
Diagnosis: Classic overfitting
- Training loss ↓ → model learning
- Validation loss ↑ → not generalizing
- Root cause: weight_decay=0.0 (no regularization!)

Solution 1: Add weight decay
```python
optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01  # Standard for fine-tuning
)
```
Trade-offs:
- ✅ Simplest fix, usually works
- ✅ No architectural changes
- ⚠️ May need to retune LR (try 1e-4, 2e-4)

Solution 2: Reduce learning rate + early stopping
```python
optimizer = AdamW(model.parameters(), lr=1e-5)
early_stopping = EarlyStopping(patience=3, min_delta=0.01)

for epoch in range(100):
    train_loss = train()
    val_loss = validate()
    if early_stopping(val_loss):
        break
```
Trade-offs:
- ✅ Prevents overfitting automatically
- ⚠️ Slower convergence (10× smaller LR)
- ⚠️ May underfit (stops too early)

Solution 3: Data augmentation + weight decay
```python
# Text augmentation
- Synonym replacement
- Back-translation
- Random deletion

optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01
)
```
Trade-offs:
- ✅ Best generalization
- ✅ Can train longer
- ❌ More complex (augmentation pipeline)
- ❌ Slower training (more data per epoch)

Recommendation order:
1. Try Solution 1 first (weight_decay=0.01)
2. If still overfits, add early stopping
3. If more accuracy needed, add augmentation

Expected improvement:
- Validation loss: Stops increasing, plateaus
- Val accuracy: +2-3% improvement
- Training time: Similar (unless augmentation)
```

### Question 6: Production Optimizer Choice
**Q**: You're deploying a recommendation model that trains daily on new user interactions (online learning). Requirements:
- Update model every hour with last hour's data (incremental updates)
- 100M user embeddings (memory constraint: 32GB GPU)
- Low latency training (<5 minutes per update)
- Stable convergence (can't have large performance swings)

Which optimizer and why? What hyperparameters?

**Expected Answer**:
```
Requirements analysis:
1. Online learning → need fast adaptation
2. Memory constrained → can't use Adam (2× memory)
3. Stability critical → need adaptive LR
4. Hourly updates → must converge quickly

Optimizer choice: Adafactor

Reasoning:
✅ Memory efficient: O(n) vs Adam's O(2n)
   - 100M embeddings × 256 dims = 25.6B params
   - Adafactor: ~100 GB
   - Adam: ~200 GB (doesn't fit!)

✅ Fast convergence: Adaptive LR per parameter
   - Don't need to retune LR each update
   - Auto-adjusts to data distribution

✅ Stability: Built-in gradient clipping
   - Prevents divergence on outlier batches
   - Critical for online learning

Configuration:
```python
optimizer = Adafactor(
    model.parameters(),
    lr=None,              # Use adaptive LR
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,   # Stability
    decay_rate=-0.8,
    beta1=None,           # No first moment (save memory)
    weight_decay=1e-4,    # Light regularization
    scale_parameter=True,
    relative_step=True,   # Adaptive LR critical
    warmup_init=False     # No warmup for online learning
)

# Update every hour
for hour in range(24):
    new_data = fetch_last_hour_interactions()
    
    # Train for 100 steps (~5 minutes)
    for _ in range(100):
        batch = sample_batch(new_data)
        loss = train_step(batch)
        optimizer.step()
    
    # Validate and deploy
    if val_metric_improved():
        deploy_model()
```

Alternative (if more memory available):
- AdamW with gradient checkpointing
- Allows ~1.5× larger model in same memory
- Faster convergence than Adafactor
- But still need 150GB (borderline)

Monitoring:
- Per-hour validation AUC (should be stable ±0.5%)
- Per-hour loss (watch for spikes)
- Memory usage (should stay <30GB)
- Training time (should be <5 min consistently)

Expected behavior:
- Hour 1: Loss 0.45, AUC 0.72
- Hour 2: Loss 0.44, AUC 0.73 (gradual improvement)
- No sudden drops (stable online learning)
```

---

*End of Part 1. Part 2 will cover Attention Mechanisms, Loss Functions, and Sampling Strategies.*
