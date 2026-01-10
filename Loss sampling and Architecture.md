# ML Fundamentals - Part 3: Loss Functions, Sampling Strategies & Model Architecture

**Complete Interview Preparation Guide**
*Sources: Research papers (Focal Loss, Contrastive Learning papers), Lilian Weng, Chip Huyen, Sebastian Raschka, Production systems at OpenAI/Google/Meta/Anthropic*

---

## 24.3 LOSS COMPUTATION DETAILS

### Overview: Why Loss Functions Matter

**Loss function = Training signal**:
- Wrong loss → model learns wrong thing
- Numerically unstable loss → training fails (NaN)
- Appropriate loss → fast convergence, good generalization

**Production Impact**:
- **Meta's LLama 2**: Switching from standard CE to custom loss improved RLHF by 8%
- **OpenAI GPT-4**: Multiple loss terms (language modeling + safety + coding)
- **Google BERT**: Masked LM loss + next sentence prediction (later dropped NSP)

---

### 24.3.1 Cross-Entropy: The Foundation

**Why Cross-Entropy for Classification?**
```
Information theory perspective:
- Entropy: Uncertainty in true distribution P
- Cross-entropy: Uncertainty when using Q to approximate P
- CE(P, Q) = H(P, Q) = -Σ p(x) log q(x)
- Minimizing CE ≡ Minimizing KL divergence

ML perspective:
- Equivalent to maximum likelihood estimation
- Encourages high probability for correct class
- Penalizes confident wrong predictions heavily
```

**Binary Cross-Entropy (BCE)**:
```
For binary classification (0 or 1):

BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

where:
- y ∈ {0, 1}: true label
- ŷ ∈ (0, 1): predicted probability

Example:
y=1 (positive class), ŷ=0.9 (confident correct)
BCE = -(1·log(0.9) + 0·log(0.1)) = -log(0.9) = 0.105 (low loss ✓)

y=1 (positive class), ŷ=0.1 (confident wrong)
BCE = -(1·log(0.1) + 0·log(0.9)) = -log(0.1) = 2.303 (high loss ✓)

y=1 (positive class), ŷ=0.5 (uncertain)
BCE = -(1·log(0.5) + 0·log(0.5)) = -log(0.5) = 0.693 (medium loss)
```

**Categorical Cross-Entropy**:
```
For multi-class classification (K classes):

CE = -Σ y_k · log(ŷ_k)

where:
- y: one-hot encoded true label [0,0,1,0,0] (k=3 is correct)
- ŷ: predicted probabilities [0.1, 0.2, 0.5, 0.15, 0.05]

Example (K=5 classes, true class k=3):
y = [0, 0, 1, 0, 0]
ŷ = [0.1, 0.2, 0.5, 0.15, 0.05]
CE = -(0·log(0.1) + 0·log(0.2) + 1·log(0.5) + 0·log(0.15) + 0·log(0.05))
   = -log(0.5) = 0.693

Only correct class contributes to loss!
```

**The Softmax Connection**:
```
In neural networks:

logits = model(x)  # [z₁, z₂, ..., z_K] (raw scores)
probs = softmax(logits)  # Convert to probabilities
loss = CE(y, probs)

Softmax: ŷ_k = exp(z_k) / Σ exp(z_j)

Combined: Called "Softmax Cross-Entropy"
```

---

### 24.3.2 Cross-Entropy Numerical Stability

**The Overflow/Underflow Problem**:
```
Naive implementation:

logits = [1000, 999, 998]  # Large values from deep network
probs = exp(logits) / sum(exp(logits))
      = [exp(1000), exp(999), exp(998)] / sum
      = [∞, ∞, ∞] / ∞  # OVERFLOW!

loss = -log(probs[0]) = -log(∞/∞) = NaN  # TRAINING BREAKS!
```

**Solution 1: LogSumExp Trick**:
```
Key insight: Use log-space arithmetic

Softmax mathematically equivalent:
ŷ_k = exp(z_k - max(z)) / Σ exp(z_j - max(z))

Example:
logits = [1000, 999, 998]
max_z = 1000
shifted = [0, -1, -2]
exp(shifted) = [1, 0.368, 0.135]
sum = 1.503
probs = [0.665, 0.245, 0.090] ✓ (numerically stable!)

Log-softmax (even better):
log_probs = shifted - log(sum)
          = [0, -1, -2] - log(1.503)
          = [-0.408, -1.408, -2.408]

Loss = -log_probs[correct_class]  # Direct, no overflow!
```

**Solution 2: Combined Softmax-CE** (Production Standard):
```
Instead of:
1. probs = softmax(logits)
2. loss = -log(probs[y])

Do:
loss = log(sum(exp(logits))) - logits[y]

This is numerically stable by design!
```

**PyTorch/TensorFlow Implementation**:
```python
# ❌ WRONG: Separate softmax + log
probs = F.softmax(logits, dim=-1)
loss = -torch.log(probs[range(batch), labels])
# Problem: Loses precision in softmax

# ✅ CORRECT: Fused operation
loss = F.cross_entropy(logits, labels)
# Internally uses log-softmax with numerical stability
```

**Production Case Study: GPT-3 Training**:
```
Problem: Vocabulary size = 50,257
At end of training, logits can be [-20, 30] range
Naive softmax: exp(30) = 1e13 (OK), exp(-20) = 2e-9 (OK)
BUT: Gradients involve exp(30) - exp(-20) ≈ 1e13 (loses precision)

Solution: PyTorch F.cross_entropy
- Uses log-softmax internally
- Stable even with extreme logit values
- No manual clipping needed

Result: GPT-3 trained for millions of steps without NaN losses
```

**Gradient Numerical Issues**:
```
Gradient of cross-entropy:
∂L/∂z_k = ŷ_k - y_k

Example:
y = [0, 0, 1, 0]
ŷ = [0.1, 0.2, 0.5, 0.2]
∂L/∂z = [0.1, 0.2, -0.5, 0.2]

Clean, bounded gradients (in [-1, 1])
No vanishing or exploding (unlike sigmoid/tanh derivatives)
```

**Underflow in Log**:
```
Problem: log(0) = -∞

Happens when:
- Model very confident: ŷ ≈ 0 or ŷ ≈ 1
- log(1e-45) = -103.6 (OK)
- log(1e-200) = -460.5 (OK)
- log(0) = -∞ (BAD!)

Solution: Epsilon clipping
ŷ_clipped = clip(ŷ, min=1e-7, max=1-1e-7)
loss = -log(ŷ_clipped)

PyTorch handles this automatically in F.cross_entropy
```

**When Things Go Wrong (Debugging)**:
```
Symptom: NaN loss appears suddenly

Checklist:
1. Check logit range:
   print(f"Logits: min={logits.min()}, max={logits.max()}")
   If |max| > 100, likely overflow issue
   
2. Check gradients:
   for name, param in model.named_parameters():
       if param.grad is not None:
           if torch.isnan(param.grad).any():
               print(f"NaN gradient in {name}")
   
3. Use grad clipping:
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   
4. Check data:
   if torch.isnan(batch).any():
       print("NaN in input data!")
       
5. Reduce learning rate:
   If NaN appears after N steps, halve LR and restart from checkpoint
```

---

### 24.3.3 Label Smoothing

**The Overconfidence Problem**:
```
Standard cross-entropy:
- Encourages ŷ → 1 for correct class
- Encourages ŷ → 0 for wrong classes
- Model becomes overconfident: [0.999, 0.0005, 0.0005]
- May harm calibration and generalization
```

**Label Smoothing Idea**:
```
Instead of hard labels: y = [0, 0, 1, 0, 0]
Use soft labels: y_smooth = [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]

where:
- ε: Smoothing parameter (typical 0.1)
- K: Number of classes

Example (K=5, ε=0.1):
Hard label: [0, 0, 1, 0, 0]
Soft label: [0.02, 0.02, 0.92, 0.02, 0.02]

Effect: Penalize overconfidence, encourage uncertainty
```

**Mathematical Formulation**:
```
y_smooth(k) = (1 - ε)·y(k) + ε/K

For correct class k:
y_smooth(k) = (1-ε)·1 + ε/K = 1 - ε + ε/K

For incorrect class j≠k:
y_smooth(j) = (1-ε)·0 + ε/K = ε/K

Loss:
CE_smooth = -Σ y_smooth(k) · log(ŷ_k)
          = -(1-ε+ε/K)·log(ŷ_correct) - Σ_{k≠correct} (ε/K)·log(ŷ_k)
```

**Gradient Impact**:
```
Standard CE gradient:
∂L/∂z_k = ŷ_k - y_k

With label smoothing:
∂L/∂z_k = ŷ_k - y_smooth(k)

Example (K=5, ε=0.1, correct class k=3):
Standard: ŷ = [0.05, 0.1, 0.7, 0.1, 0.05]
          ∂L/∂z = [0.05, 0.1, -0.3, 0.1, 0.05]

Smoothed: ŷ = [0.05, 0.1, 0.7, 0.1, 0.05]
          y_smooth = [0.02, 0.02, 0.92, 0.02, 0.02]
          ∂L/∂z = [0.03, 0.08, -0.22, 0.08, 0.03]

Effect: Smaller gradients → more cautious updates → less overconfidence
```

**Production Case Study: Google's Inception-v4**:
```
Task: ImageNet classification (1000 classes)
Model: Inception-v4 (48M parameters)

Baseline (no smoothing):
- Top-1 accuracy: 80.0%
- Top-5 accuracy: 95.2%
- Calibration error: 8.5% (poor calibration)
- Overconfident: Many predictions >0.99

With label smoothing (ε=0.1):
- Top-1 accuracy: 80.2% (+0.2%)
- Top-5 accuracy: 95.3% (+0.1%)
- Calibration error: 3.2% (much better!)
- Better uncertainty estimates

Key insight: Slight accuracy gain + much better calibration
Critical for production where probability matters (ranking, filtering)
```

**When to Use Label Smoothing**:
```
✅ Use when:
- Large number of classes (K > 100)
- Calibration matters (probabilities used in decision)
- Fine-tuning pre-trained models (prevents catastrophic forgetting)
- Noisy labels (smoothing acts as regularization)

❌ Don't use when:
- Binary classification (K=2, smoothing may hurt)
- Small datasets (<10K examples, may harm learning signal)
- Multi-label classification (labels not mutually exclusive)

Typical ε values:
- Image classification: 0.1
- Language modeling: 0.1-0.2
- Fine-tuning: 0.05 (gentler)
```

**Implementation**:
```python
# Manual implementation
def label_smoothing(labels, num_classes, epsilon=0.1):
    # labels: [batch_size] with integer class indices
    # Returns: [batch_size, num_classes] with smoothed labels
    
    batch_size = labels.size(0)
    smoothed = torch.full((batch_size, num_classes), epsilon / num_classes)
    smoothed.scatter_(1, labels.unsqueeze(1), 1.0 - epsilon + epsilon / num_classes)
    return smoothed

# PyTorch built-in (simpler)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = criterion(logits, labels)
```

---

### 24.3.4 Focal Loss (Handling Class Imbalance)

**The Class Imbalance Problem**:
```
Example: Object detection
- Background: 99% of pixels
- Objects: 1% of pixels

Standard CE:
- Loss dominated by easy negatives (background)
- Model learns to predict background everywhere
- Hard positives (small objects) get drowned out
```

**Focal Loss Innovation (Lin et al., RetinaNet)**:
```
Idea: Down-weight easy examples, up-weight hard examples

Focal Loss:
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

where:
- p_t: Predicted probability for true class
- α_t: Class weighting (handle class imbalance)
- γ: Focusing parameter (reduce loss for easy examples)

Comparison with CE:
CE = -log(p_t)
FL = -(1-p_t)^γ · log(p_t)

Key: (1-p_t)^γ term modulates loss
```

**How Focal Loss Works**:
```
Example (γ=2, α=1):

Easy example (p_t=0.9, model confident correct):
CE = -log(0.9) = 0.105
FL = -(1-0.9)² · log(0.9) = -0.01 · 0.105 = 0.00105
Reduction: 100× less loss! (easy examples contribute almost nothing)

Hard example (p_t=0.5, model uncertain):
CE = -log(0.5) = 0.693
FL = -(1-0.5)² · log(0.5) = -0.25 · 0.693 = 0.173
Reduction: 4× less loss (still significant)

Very hard example (p_t=0.1, model wrong):
CE = -log(0.1) = 2.303
FL = -(1-0.1)² · log(0.1) = -0.81 · 2.303 = 1.865
Reduction: 1.2× less (hard examples still dominate)
```

**The Focusing Parameter γ**:
```
γ controls how much to down-weight easy examples

γ=0: Focal Loss = Cross-Entropy (no change)
γ=1: Moderate focusing
γ=2: Strong focusing (recommended default)
γ=5: Very strong (may ignore too many examples)

Example: p_t=0.9 (easy example)
γ=0: (1-0.9)⁰ = 1.0 (no reduction)
γ=1: (1-0.9)¹ = 0.1 (10× reduction)
γ=2: (1-0.9)² = 0.01 (100× reduction)
γ=5: (1-0.9)⁵ = 0.00001 (100K× reduction!)
```

**Class Weighting α_t**:
```
α_t handles class imbalance (like weighted CE)

α_t = (# negative samples) / (# positive samples)

Example: 99% background, 1% object
α_background = 0.01 / 0.99 ≈ 0.01
α_object = 0.99 / 0.01 ≈ 99

Combined formula:
FL = -α_t · (1-p_t)^γ · log(p_t)

Effect: Hard objects get 99× weight + focusing
```

**Production Case Study: RetinaNet (Facebook AI)**:
```
Task: Object detection on COCO dataset
Challenge: 
- 100K objects
- ~10M background proposals
- Ratio: 1:100 (extreme imbalance)

Baseline (Faster R-CNN with CE):
- AP (Average Precision): 35.7
- Problem: Dominated by easy negatives
- Training unstable (gradient magnitude varies 100×)

RetinaNet with Focal Loss (γ=2, α=0.25):
- AP: 39.1 (+3.4 points!)
- Stable training (gradients balanced)
- 3× faster convergence

Why it works:
- Easy background: Loss ~0 (not wasting compute)
- Hard objects: Loss high (model focuses here)
- Training: 99% compute on 1% hard examples (efficient!)

Settings:
γ=2: Best trade-off (strong but not too strong)
α=0.25: Less weight on background (ratio 1:4, not 1:100)
```

**Gradient Analysis**:
```
Focal Loss gradient (w.r.t. logit z):

∂FL/∂z = α_t · [(1-p_t)^γ · (γ·p_t·log(p_t) + 1) · p_t - γ·(1-p_t)^(γ-1)·p_t·log(p_t)]

Compared to CE:
∂CE/∂z = p_t - y

Key differences:
1. Gradient magnitude depends on p_t (easy examples → small gradient)
2. Non-linear (CE gradient is linear in p_t)
3. Focuses gradient on hard examples automatically
```

**When to Use Focal Loss**:
```
✅ Use when:
- Severe class imbalance (ratio >10:1)
- Many easy examples dominate loss
- Detection tasks (object, segmentation)
- Hard negative mining needed

❌ Don't use when:
- Balanced classes (focal loss adds complexity for no gain)
- Small datasets (focusing may ignore too much)
- Already using hard negative mining (redundant)

Hyperparameter tuning:
γ: Start with 2, increase if still dominated by easy examples
α: Set to (# negative / # positive), then tune to balance gradient magnitude
```

**Implementation**:
```python
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    logits: [batch_size, num_classes]
    targets: [batch_size] (class indices)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # p_t
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# Multi-class with per-class alpha
def focal_loss_multiclass(logits, targets, gamma=2.0, alpha=None):
    if alpha is None:
        alpha = torch.ones(logits.size(1)) / logits.size(1)
    
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    
    # Get alpha for true class
    alpha_t = alpha[targets]
    
    loss = alpha_t * (1 - pt) ** gamma * ce
    return loss.mean()
```

---

### 24.3.5 Contrastive Losses (SimCLR, MoCo)

**Self-Supervised Learning via Contrastive Loss**:
```
Goal: Learn representations without labels
Method: Pull similar samples together, push different samples apart

Key idea:
- Positive pair: Augmented versions of same image (cat → cat_flipped)
- Negative pairs: Different images (cat vs dog)
- Learn embedding where similar items close, dissimilar far
```

**NT-Xent Loss (SimCLR - Normalized Temperature-scaled Cross-Entropy)**:
```
Given:
- Image i, create two augmentations: i_a, i_b (positive pair)
- Batch of N images → 2N augmented images
- For i_a: 1 positive (i_b), 2N-2 negatives (all others)

Similarity:
sim(z_i, z_j) = z_i^T · z_j / (||z_i|| · ||z_j||)  (cosine similarity)

Loss for positive pair (i, j):
L_i,j = -log(exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ))

where:
- τ: Temperature (typical 0.07-0.5)
- Denominator: Sum over all negatives (2N-2 samples)
```

**Temperature Parameter τ**:
```
Temperature controls sharpness of distribution

High temperature (τ=1.0):
sim/τ = [0.8, 0.6, 0.4, 0.2] / 1.0 = [0.8, 0.6, 0.4, 0.2]
softmax = [0.31, 0.23, 0.18, 0.14] (soft distribution)

Low temperature (τ=0.1):
sim/τ = [0.8, 0.6, 0.4, 0.2] / 0.1 = [8, 6, 4, 2]
softmax = [0.95, 0.04, 0.01, 0.00] (peaked distribution)

Effect:
- Low τ: Model must strongly separate positive from negatives (hard task)
- High τ: More forgiving (easier task)
- Typical: τ=0.07 (very low, strong discrimination)
```

**Production Case Study: SimCLR (Google)**:
```
Task: Self-supervised learning on ImageNet
Model: ResNet-50 (no labels used!)
Batch size: 4096 (critical for negatives!)
Augmentations: 
- Random crop + resize
- Color distortion
- Gaussian blur

Training:
- 1000 epochs
- SGD with momentum
- LARS optimizer (large batch)
- Temperature: τ=0.07

Results:
Labeled ImageNet (100% labels): 76.5% Top-1
SimCLR (0% labels): 69.3% Top-1 (linear eval)
SimCLR (1% labels): 73.9% Top-1 (fine-tuning)

Key insight: Can learn strong representations without labels!
Critical for domains with scarce labels (medical, scientific)

Why large batch matters:
- Batch 256: 510 negatives → 65.2% accuracy
- Batch 4096: 8190 negatives → 69.3% accuracy (+4.1%!)
- More negatives = harder task = better representations
```

**MoCo (Momentum Contrast - Facebook)**:
```
Problem with SimCLR: Needs huge batch size (expensive!)

MoCo solution: Maintain queue of negatives
- Queue size: 65536 (independent of batch size)
- Update queue with momentum encoder (slow-moving)
- Can use small batch sizes (256) but still 65K negatives

Momentum encoder:
θ_momentum = m · θ_momentum + (1-m) · θ_online
where m=0.999 (very slow update)

Why momentum?
- Fast encoder changes too quickly → queue becomes stale
- Slow encoder keeps queue consistent
- Balance: Queue updated slowly but not too slow
```

**Contrastive Loss vs Supervised**:
```
Supervised (cross-entropy):
- Requires labels: cat, dog, bird
- Learns: Decision boundaries between classes
- Data: Expensive (need labeled data)

Contrastive (NT-Xent):
- No labels needed
- Learns: General representations (semantic similarity)
- Data: Cheap (just augment images)
- Then: Fine-tune with small labeled dataset

When contrastive is better:
- Limited labels (<1% of data labeled)
- Transfer learning (pre-train on large unlabeled, fine-tune on small labeled)
- Domain shift (source domain unlabeled, target domain few labels)
```

**Implementation**:
```python
def nt_xent_loss(z_i, z_j, temperature=0.07):
    """
    z_i, z_j: [batch_size, embedding_dim] (positive pairs)
    """
    batch_size = z_i.size(0)
    
    # Concatenate positive pairs
    z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, emb_dim]
    
    # Compute similarity matrix
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    # sim_matrix: [2*batch_size, 2*batch_size]
    
    # Mask diagonal (self-similarity)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    
    # Positive pairs: (i, i+N) and (i+N, i)
    pos_sim = torch.cat([
        torch.diag(sim_matrix, batch_size),  # (i, i+N)
        torch.diag(sim_matrix, -batch_size)  # (i+N, i)
    ])
    
    # Compute loss
    sim_matrix = sim_matrix / temperature
    pos_sim = pos_sim / temperature
    
    loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
    return loss.mean()
```

---

### 24.3.6 Triplet Loss

**Metric Learning via Triplets**:
```
Goal: Learn embedding where distance reflects similarity

Triplet: (anchor, positive, negative)
- Anchor: Reference sample
- Positive: Similar to anchor (same class)
- Negative: Dissimilar to anchor (different class)

Constraint:
d(anchor, positive) + margin < d(anchor, negative)

Loss:
L = max(0, d(a,p) - d(a,n) + margin)
```

**Example (Face Recognition)**:
```
Anchor: Photo of Person A
Positive: Another photo of Person A (same person)
Negative: Photo of Person B (different person)

Embeddings (learned):
e_a = [0.1, 0.9, 0.3]
e_p = [0.15, 0.85, 0.35]  (close to anchor)
e_n = [0.8, 0.2, 0.6]  (far from anchor)

Distances:
d(a,p) = ||e_a - e_p|| = 0.07
d(a,n) = ||e_a - e_n|| = 1.05

Loss (margin=0.2):
L = max(0, 0.07 - 1.05 + 0.2) = max(0, -0.78) = 0 (satisfied! ✓)

If negative too close:
e_n_bad = [0.2, 0.8, 0.4]
d(a,n_bad) = 0.15
L = max(0, 0.07 - 0.15 + 0.2) = 0.12 (violated, has loss)
```

**Margin Hyperparameter**:
```
Margin: Minimum separation between positive and negative

Small margin (0.1):
- Easy to satisfy
- Weak separation
- May not generalize well

Large margin (1.0):
- Hard to satisfy
- Strong separation
- May be too strict (hard to optimize)

Typical: 0.2-0.5 depending on task
- Face recognition: 0.2 (tight)
- Image retrieval: 0.5 (looser)
```

**Triplet Mining (Critical)**:
```
Problem: Most triplets are easy (already satisfied)
- Random triplets: 99% have zero loss (wasted compute)
- Model learns nothing from easy triplets

Solution: Mine hard triplets

Hard negative: Negative closest to anchor
d(a, n_hard) = min{d(a, n) for all negatives}

Hard positive: Positive farthest from anchor
d(a, p_hard) = max{d(a, p) for all positives}

Semi-hard negative: Between hard and easy
d(a,p) < d(a, n_semi_hard) < d(a,p) + margin
```

**Production Case Study: FaceNet (Google)**:
```
Task: Face verification (same person or not?)
Model: Inception-based CNN → 128-dim embedding
Dataset: 200M face images, 8M identities

Training:
- Triplet loss with online mining
- Hard negative mining per mini-batch
- Margin: 0.2
- Embedding L2-normalized

Results:
LFW (Labeled Faces in the Wild): 99.63% accuracy
YouTube Faces: 95.12% accuracy

Key insight: Can verify faces with just embedding distance!
No classifier needed → generalizes to new identities

Inference:
1. Compute embedding for face A: e_A
2. Compute embedding for face B: e_B
3. Distance: d = ||e_A - e_B||
4. Decision: Same person if d < threshold (typically 0.6)
```

**Implementation**:
```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    anchor, positive, negative: [batch_size, embedding_dim]
    """
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

# With online hard negative mining
def triplet_loss_with_mining(embeddings, labels, margin=0.2):
    """
    embeddings: [batch_size, embedding_dim]
    labels: [batch_size] (class IDs)
    """
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2)
    
    # For each anchor, find hardest positive and negative
    losses = []
    for i in range(len(embeddings)):
        # Hardest positive (same class, maximum distance)
        pos_mask = (labels == labels[i]) & (torch.arange(len(labels)) != i)
        pos_distances = distances[i][pos_mask]
        if len(pos_distances) == 0:
            continue
        hardest_positive_dist = pos_distances.max()
        
        # Hardest negative (different class, minimum distance)
        neg_mask = labels != labels[i]
        neg_distances = distances[i][neg_mask]
        if len(neg_distances) == 0:
            continue
        hardest_negative_dist = neg_distances.min()
        
        # Compute loss
        loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
        losses.append(loss)
    
    return torch.stack(losses).mean()
```

---

### 24.3.7 Custom Loss Gradients

**When to Design Custom Loss**:
```
1. Multi-objective optimization
   - Example: Translation quality + fluency + brevity
   - Combine multiple losses with weights

2. Domain-specific constraints
   - Example: Physics-informed neural networks
   - Add physics equations as penalty terms

3. Special properties needed
   - Example: Robustness to outliers
   - Design loss that down-weights outliers
```

**Example: Multi-Task Learning Loss**:
```python
# Task 1: Classification (cross-entropy)
# Task 2: Regression (MSE)
# Task 3: Reconstruction (L1)

def multi_task_loss(outputs, targets, weights=[1.0, 1.0, 1.0]):
    cls_logits, reg_pred, recon_output = outputs
    cls_target, reg_target, recon_target = targets
    
    # Individual losses
    cls_loss = F.cross_entropy(cls_logits, cls_target)
    reg_loss = F.mse_loss(reg_pred, reg_target)
    recon_loss = F.l1_loss(recon_output, recon_target)
    
    # Weighted sum
    total_loss = (weights[0] * cls_loss + 
                  weights[1] * reg_loss + 
                  weights[2] * recon_loss)
    
    return total_loss, (cls_loss, reg_loss, recon_loss)

# Learnable weights (uncertainty weighting)
class MultiTaskLossLearnable(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))
    
    def forward(self, outputs, targets):
        cls_loss, reg_loss, recon_loss = compute_losses(outputs, targets)
        
        # Automatic weight learning
        loss = (cls_loss / (2 * self.log_vars[0].exp()) + self.log_vars[0] +
                reg_loss / (2 * self.log_vars[1].exp()) + self.log_vars[1] +
                recon_loss / (2 * self.log_vars[2].exp()) + self.log_vars[2])
        
        return loss
```

**Example: Huber Loss (Robust to Outliers)**:
```python
def huber_loss(pred, target, delta=1.0):
    """
    Combines MSE (small errors) and MAE (large errors)
    Less sensitive to outliers than MSE
    """
    error = pred - target
    abs_error = torch.abs(error)
    
    # Quadratic for small errors, linear for large
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)
    
    loss = torch.where(abs_error <= delta, quadratic, linear)
    return loss.mean()

# Why it's better than MSE:
# MSE: L = (pred - target)²
#      Error=10 → Loss=100 (heavily penalizes outliers)
# Huber: L = 0.5² for error<δ, δ(|error|-0.5δ) for error>δ  
#        Error=10 → Loss=9.5 (gentler on outliers)
```

---

## 24.4 SAMPLING STRATEGIES

### Overview: Text Generation as Search

**The Generation Problem**:
```
Given prompt: "The cat sat on the"
Vocabulary: 50,000 words
Next token probabilities: [0.001, 0.003, 0.45, 0.12, ...] (50K values)

Challenge: How to select next token?
- Deterministic: Always pick highest probability (boring, repetitive)
- Random: Sample uniformly (incoherent)
- Smart sampling: Balance quality and diversity
```

---

### 24.4.1 Greedy Decoding

**Algorithm: Always Pick argmax**:
```
At each step:
1. Model outputs probabilities: P(w₁), P(w₂), ..., P(w_V)
2. Select: w* = argmax P(w)
3. Append w* to sequence
4. Repeat until EOS or max length

Example:
Prompt: "The cat"
P(sat)=0.5, P(jumped)=0.3, P(ran)=0.2
Choose: "sat" (highest)
Continue: "The cat sat"
P(on)=0.6, P(down)=0.25, P(quietly)=0.15
Choose: "on"
Result: "The cat sat on..."
```

**Characteristics**:
```
Advantages:
✅ Fast (no beam search overhead)
✅ Deterministic (reproducible)
✅ Simple to implement

Disadvantages:
❌ Repetitive (gets stuck in loops)
❌ No exploration (misses better paths)
❌ Myopic (locally optimal ≠ globally optimal)
```

**The Repetition Problem**:
```
Example (real GPT-2 output with greedy):
Prompt: "I think that"
Output: "I think that I think that I think that I think that..."

Why?
- "I think that" is high probability continuation
- Once generated, context is "...I think that"
- Same pattern repeats → infinite loop!

Mitigation: Repetition penalty (see below)
```

**When to Use Greedy**:
```
✅ Factual tasks (translation, summarization)
✅ When determinism matters (test consistency)
✅ Short sequences (< 50 tokens)
✅ When best quality > diversity

❌ Creative generation (stories, poems)
❌ Long sequences (repetition becomes issue)
❌ When want variety in outputs
```

---

### 24.4.2 Beam Search

**Algorithm: Maintain Top-K Hypotheses**:
```
Instead of one sequence, keep beam_width candidates

Example (beam_width=3):
Step 1: "The"
  Candidates: ["The cat" (0.5), "The dog" (0.3), "The bird" (0.2)]

Step 2: For each candidate, extend with top words
  "The cat" → ["The cat sat" (0.25), "The cat jumped" (0.15), ...]
  "The dog" → ["The dog ran" (0.18), "The dog barked" (0.12), ...]
  "The bird" → ["The bird flew" (0.10), "The bird sang" (0.08), ...]

Keep top-3 overall:
1. "The cat sat" (score=0.25)
2. "The dog ran" (score=0.18)
3. "The cat jumped" (score=0.15)

Step 3: Continue until all beams generate EOS
```

**Scoring: Length Normalization**:
```
Problem: Longer sequences have lower scores
- Log probabilities are negative
- Longer = more terms summed = more negative

Score = Σ log P(w_t | w_{<t})
      = log P(w₁) + log P(w₂) + ... + log P(w_n)
      ∈ [-∞, 0]

Length penalty:
Score_normalized = (1/n^α) · Σ log P(w_t)

where α controls penalty:
- α=0: No normalization (favors short)
- α=1: Full normalization (favors long)
- α=0.7: Typical (slight favor to longer)

Example:
Seq 1: "The cat" (2 tokens, score=-1.5)
Seq 2: "The gray fluffy cat" (4 tokens, score=-3.2)

Without normalization:
Seq 1: -1.5 (better)
Seq 2: -3.2

With normalization (α=1):
Seq 1: -1.5/2 = -0.75
Seq 2: -3.2/4 = -0.8 (comparable)
```

**Beam Width Trade-off**:
```
| Width | Quality | Speed | Diversity |
|-------|---------|-------|-----------|
| 1 | Lowest (greedy) | Fastest | None |
| 3-5 | Good | Fast | Low |
| 10-20 | Best | Slow | Medium |
| 50+ | Diminishing returns | Very slow | High |

Typical: beam_width=4 or 5 (good balance)
```

**Production Case Study: Google Translate**:
```
Task: English → French translation
Model: Transformer (6 layers)

Greedy decoding:
- BLEU score: 37.2
- Speed: 50ms per sentence
- Quality: Good but sometimes awkward

Beam search (width=4, α=0.6):
- BLEU score: 41.0 (+3.8 points!)
- Speed: 120ms per sentence (2.4× slower)
- Quality: More fluent, natural

Beam search (width=10):
- BLEU score: 41.3 (+0.3 over width=4)
- Speed: 250ms per sentence (5× slower)
- Diminishing returns!

Conclusion: width=4 is sweet spot
```

**Implementation**:
```python
def beam_search(model, prompt, beam_width=5, max_length=50, alpha=0.7):
    # Initialize beam with prompt
    beams = [(prompt, 0.0)]  # (sequence, score)
    
    for _ in range(max_length):
        candidates = []
        
        for seq, score in beams:
            if seq[-1] == EOS_TOKEN:
                candidates.append((seq, score))
                continue
            
            # Get probabilities for next token
            logits = model(seq)
            probs = F.softmax(logits, dim=-1)
            
            # Expand with top-k tokens
            top_probs, top_indices = probs.topk(beam_width)
            
            for prob, idx in zip(top_probs, top_indices):
                new_seq = seq + [idx]
                new_score = score + torch.log(prob)
                candidates.append((new_seq, new_score))
        
        # Keep top beam_width candidates (with length penalty)
        def score_with_penalty(seq, score):
            return score / (len(seq) ** alpha)
        
        beams = sorted(candidates, 
                      key=lambda x: score_with_penalty(x[0], x[1]),
                      reverse=True)[:beam_width]
        
        # Stop if all beams finished
        if all(seq[-1] == EOS_TOKEN for seq, _ in beams):
            break
    
    return beams[0][0]  # Return best sequence
```

**When to Use Beam Search**:
```
✅ Translation (quality critical)
✅ Summarization (want best summary)
✅ Code generation (correctness matters)
✅ When quality > speed

❌ Creative writing (too deterministic)
❌ Real-time chat (too slow)
❌ When want diverse outputs
```

---

### 24.4.3 Top-k Sampling

**Algorithm: Sample from Top-k Highest Probabilities**:
```
At each step:
1. Model outputs: P(w₁), P(w₂), ..., P(w_V)
2. Select top-k highest probabilities
3. Renormalize to sum to 1
4. Sample from this distribution

Example (k=3):
Full distribution:
P(sat)=0.5, P(on)=0.3, P(jumped)=0.15, P(ran)=0.04, P(quietly)=0.01

Top-3:
P(sat)=0.5, P(on)=0.3, P(jumped)=0.15
Sum = 0.95

Renormalized:
P(sat)=0.526, P(on)=0.316, P(jumped)=0.158

Sample: Pick one according to these probabilities
Might get: "sat" (52.6% chance) or "on" (31.6%) or "jumped" (15.8%)
```

**Top-k Value Trade-off**:
```
Small k (k=3):
✅ High quality (only consider likely words)
✅ Coherent (stays on track)
❌ Less diverse (limited options)
❌ May miss creative alternatives

Large k (k=50):
✅ More diverse (more options)
✅ Creative (can pick unusual words)
❌ Lower quality (can pick unlikely words)
❌ Less coherent (may drift off-topic)

Typical: k=40-50 (balance)
```

**Production Case Study: GPT-2 (OpenAI)**:
```
Task: Story generation
Prompt: "Once upon a time"

Greedy decoding:
Output: "Once upon a time, there was a man who was a man who was a man who..."
Problem: Repetitive, boring

Top-k sampling (k=40):
Output 1: "Once upon a time, in a kingdom far away, there lived a brave knight..."
Output 2: "Once upon a time, there was a clever fox who loved to play tricks..."
Output 3: "Once upon a time, the world was covered in ice and snow..."

Quality: Diverse, creative, non-repetitive
Settings: k=40, temperature=0.9
```

**Adaptive k** (Dynamic):
```
Idea: k depends on distribution confidence

Confident distribution:
P(sat)=0.95, P(jumped)=0.03, P(ran)=0.02
Use small k=3 (distribution peaked, don't need many options)

Flat distribution:
P(sat)=0.25, P(jumped)=0.20, P(ran)=0.18, P(on)=0.15, ...
Use large k=10 (distribution uncertain, consider more options)

Metric: Entropy of distribution
H = -Σ P(w) log P(w)

Low entropy → small k
High entropy → large k
```

**Implementation**:
```python
def top_k_sampling(logits, k=40, temperature=1.0):
    """
    logits: [vocab_size] (raw model outputs)
    k: Number of top tokens to consider
    temperature: Controls randomness
    """
    # Apply temperature
    logits = logits / temperature
    
    # Get top-k
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Convert to probabilities
    probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample
    next_token_idx = torch.multinomial(probs, num_samples=1)
    next_token = top_k_indices[next_token_idx]
    
    return next_token
```

---

### 24.4.4 Top-p (Nucleus) Sampling

**Algorithm: Sample from Smallest Set with Cumulative Probability ≥ p**:
```
Instead of fixed k, use dynamic cutoff based on cumulative probability

Example (p=0.9):
Sorted probabilities:
P(sat)=0.5, P(on)=0.25, P(jumped)=0.15, P(ran)=0.05, P(quietly)=0.04, ...

Cumulative:
sat: 0.5
sat+on: 0.75
sat+on+jumped: 0.90 ← Cutoff here (≥ 0.9)

Nucleus: {sat, on, jumped}
Sample from these three

Contrast with top-k=3:
Same result in this example, but adapts to distribution!
```

**Adaptive Nucleus Size**:
```
Confident distribution (peaked):
P(sat)=0.92, P(jumped)=0.05, P(ran)=0.02, P(on)=0.01
Cumulative: sat (0.92) → Nucleus size = 1 (deterministic!)

Flat distribution (uncertain):
P(sat)=0.20, P(jumped)=0.18, P(ran)=0.15, P(on)=0.13, P(down)=0.10, ...
Cumulative needs 6 tokens to reach 0.90 → Nucleus size = 6

Advantage: Automatically adapts to model confidence!
```

**Top-p vs Top-k**:
```
Example 1 (confident):
Distribution: [0.8, 0.1, 0.05, 0.03, 0.01, 0.01]

Top-k (k=3): Consider [0.8, 0.1, 0.05] (fixed 3 tokens)
Top-p (p=0.9): Consider [0.8, 0.1] (only 2 needed)

Top-p is more selective when confident ✓

Example 2 (flat):
Distribution: [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, ...]

Top-k (k=3): Consider [0.15, 0.14, 0.13]
Top-p (p=0.9): Consider [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, ...]
              (needs 7 tokens)

Top-p is more explorative when uncertain ✓
```

**Production Case Study: GPT-3 (OpenAI)**:
```
Task: Various text generation (chat, completion, etc.)
Default settings: top-p=1.0 (consider all tokens, but with temperature)

Creative writing (stories):
- top-p=0.9
- temperature=0.9
- Result: Diverse, creative outputs

Factual tasks (QA, summarization):
- top-p=0.1
- temperature=0.2
- Result: Focused, accurate outputs

Code generation:
- top-p=0.95
- temperature=0.2
- Result: Correct but some variation in style

Key insight: top-p adapts to task naturally
No need to manually tune k for different distributions
```

**Combining Top-k and Top-p** (Production Standard):
```python
def top_k_top_p_sampling(logits, k=50, p=0.9, temperature=1.0):
    """
    Apply both filters: top-k AND top-p
    Use whichever is more restrictive
    """
    # Temperature scaling
    logits = logits / temperature
    
    # Sort descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # Top-k filter
    if k > 0:
        sorted_logits = sorted_logits[:k]
        sorted_indices = sorted_indices[:k]
    
    # Top-p filter
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    
    sorted_logits[sorted_indices_to_remove] = -float('Inf')
    
    # Sample
    probs = F.softmax(sorted_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, 1)
    next_token = sorted_indices[next_token_idx]
    
    return next_token
```

---

### 24.4.5 Temperature Scaling

**Temperature Parameter**: Controls randomness of sampling

**Mathematical Foundation**:
```
Standard softmax:
P(w_i) = exp(z_i) / Σ exp(z_j)

With temperature T:
P(w_i) = exp(z_i / T) / Σ exp(z_j / T)

Effect:
- T < 1: Sharpens distribution (more deterministic)
- T = 1: Standard softmax
- T > 1: Flattens distribution (more random)
```

**Temperature Effects (Example)**:
```
Logits: [2.0, 1.0, 0.5, 0.1]

T = 0.5 (low, sharp):
Scaled: [4.0, 2.0, 1.0, 0.2]
Probs: [0.78, 0.17, 0.04, 0.01]
→ Very confident, mostly picks first token

T = 1.0 (standard):
Scaled: [2.0, 1.0, 0.5, 0.1]
Probs: [0.53, 0.20, 0.12, 0.08]
→ Moderate confidence

T = 2.0 (high, flat):
Scaled: [1.0, 0.5, 0.25, 0.05]
Probs: [0.38, 0.23, 0.18, 0.14]
→ Low confidence, explores more

T = 10.0 (very high):
Scaled: [0.2, 0.1, 0.05, 0.01]
Probs: [0.27, 0.25, 0.24, 0.23]
→ Almost uniform (random)
```

**Choosing Temperature**:
```
Task               | Temperature | Reasoning
-------------------|-------------|------------------
Factual QA        | 0.2-0.5     | Want accuracy, not creativity
Translation       | 0.3-0.7     | Some flexibility, mostly accurate
Summarization     | 0.5-0.8     | Balance accuracy and style
Creative writing  | 0.7-1.0     | Want diversity and creativity
Brainstorming     | 1.0-1.5     | Maximum exploration
Random text       | 2.0+        | Mostly nonsense (not useful)
```

**Production Example: ChatGPT**:
```
Default: temperature=0.7
- Balances quality and creativity
- Not too repetitive (T not too low)
- Not too random (T not too high)

User can adjust (API):
- temperature=0: Deterministic (same answer every time)
- temperature=1: More creative
- temperature=2: Very random (rarely used)

Observation: Most users keep default 0.7
Shows sweet spot for general-purpose chat
```

**Implementation**:
```python
def sample_with_temperature(logits, temperature=1.0):
    if temperature == 0:
        # Deterministic (argmax)
        return logits.argmax()
    
    # Scale by temperature
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample
    return torch.multinomial(probs, 1)
```

---

### 24.4.6 Repetition Penalty

**The Repetition Problem**:
```
Greedy decoding often produces:
"I think that I think that I think that I think that..."

Or:
"The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."

Why?
- High-probability phrases get reinforced
- Once generated, become part of context
- Model predicts same pattern again
```

**Repetition Penalty Formula**:
```
Standard scoring:
score(w) = log P(w | context)

With repetition penalty:
score(w) = log P(w | context) / penalty^count(w)

where:
- count(w): How many times w appeared in generated text
- penalty: Typically 1.1-1.5

Effect:
- First occurrence: score(w) / 1.1^1 = score(w) / 1.1
- Second occurrence: score(w) / 1.1^2 = score(w) / 1.21
- Third occurrence: score(w) / 1.1^3 = score(w) / 1.33
```

**Example**:
```
Context: "The cat sat on"
Original probs:
P(the)=0.3, P(mat)=0.25, P(floor)=0.2, P(couch)=0.15, P(chair)=0.1

If "the" already appeared 2 times:
penalty = 1.2
score(the) = log(0.3) / 1.2^2 = -1.2 / 1.44 = -0.83
score(mat) = log(0.25) / 1 = -1.39 (not penalized yet)

After penalty, "mat" becomes more likely than "the"!
Prevents: "The cat sat on the the the..."
```

**Tuning Penalty**:
```
| Penalty | Effect |
|---------|--------|
| 1.0 | No penalty (baseline) |
| 1.1 | Gentle (still allows some repetition) |
| 1.2 | Moderate (good default) |
| 1.5 | Strong (very little repetition) |
| 2.0 | Extreme (may avoid common words too much) |

Sweet spot: 1.1-1.3 for most tasks
```

**Windowed Penalty** (Better):
```
Problem: Penalize ALL past occurrences?
- "The" is a common word, should be allowed to repeat
- But not in immediate context

Solution: Penalty window
- Only count occurrences in last N tokens (e.g., N=50)
- Allows "the" to appear again after 50 tokens
- Prevents immediate repetition

count(w) = # times w in last N tokens
```

**Production Example: GPT-3 (OpenAI)**:
```
Default settings:
- repetition_penalty: 1.0 (no penalty by default)
- frequency_penalty: 0.0 (alternative approach)
- presence_penalty: 0.0

Why no penalty by default?
- Let user choose (some tasks need repetition)
- Example: Poetry may intentionally repeat words
- Example: Code may repeat variable names (correct!)

When enabled (user request):
- Creative writing: penalty=1.2 (reduce repetition)
- Chat: penalty=1.1 (subtle reduction)
- Code: penalty=1.0 (don't interfere)
```

**Frequency vs Presence Penalty**:
```
Repetition penalty: score / penalty^count
- More repetitions → more penalty (nonlinear)

Frequency penalty: score - (frequency × penalty)
- Linear penalty based on frequency
- OpenAI API uses this

Presence penalty: score - (is_present × penalty)
- Binary: Penalize if word appeared at all
- Doesn't care about count

Example:
Word "the" appeared 5 times

Repetition (penalty=1.2):
score(the) / 1.2^5 = score(the) / 2.49

Frequency (penalty=0.5):
score(the) - (5 × 0.5) = score(the) - 2.5

Presence (penalty=0.5):
score(the) - 0.5 (same whether 1 or 5 occurrences)
```

---

### 24.4.7 Advanced: Min-p and Mirostat

**Min-p Sampling** (Alternative to Top-p):
```
Idea: Filter tokens below min_p × P(best_token)

Example (min_p=0.1):
Probs: [0.5, 0.25, 0.15, 0.08, 0.02]
P(best) = 0.5
Threshold = 0.1 × 0.5 = 0.05
Keep: [0.5, 0.25, 0.15, 0.08] (drop 0.02)

Advantage over top-p:
- Adapts to distribution shape
- Confident → keeps fewer (scaled down)
- Uncertain → keeps more (scaled up)
```

**Mirostat Sampling** (Perplexity Targeting):
```
Idea: Dynamically adjust temperature to target perplexity

Algorithm:
1. Set target perplexity τ (e.g., 5.0)
2. Compute current perplexity: H = -Σ P(w) log P(w)
3. If H > τ: Decrease temperature (more focused)
4. If H < τ: Increase temperature (more random)
5. Sample with adjusted temperature

Effect: Maintains consistent "surprise" level
- Too predictable → increase randomness
- Too random → increase focus

Production use: Rare (complex, not widely adopted)
```

---

## 24.5 MODEL ARCHITECTURE DETAILS

### 24.5.1 Parameter Counting

**Why Count Parameters**:
```
- Memory requirements: 1B params × 4 bytes (FP32) = 4GB
- Training cost: More params = slower, more expensive
- Inference speed: More params = slower generation
- Model selection: 7B vs 13B vs 70B?
```

**Transformer Parameter Breakdown**:
```
Given:
- d: Hidden dimension (e.g., 4096)
- V: Vocabulary size (e.g., 50,000)
- L: Number of layers (e.g., 32)
- h: Number of attention heads (e.g., 32)

1. Embeddings:
Token embeddings: V × d
Position embeddings: max_seq_len × d (if learned)
Total: ~V × d

2. Per-layer:
a) Self-attention:
   Q, K, V projections: 3 × (d × d) = 3d²
   Output projection: d × d = d²
   Total attention: 4d²

b) Feed-forward:
   First layer: d × d_ff (typically d_ff = 4d)
   Second layer: d_ff × d
   Total FFN: 2 × d × d_ff = 8d²

c) Layer norms:
   2 × (2d) = 4d (negligible compared to d²)

Per-layer total: 4d² + 8d² = 12d²

3. All layers:
L × 12d² + V × d

4. Output layer:
Final layer norm: 2d
Output projection: d × V
Total: d × V (shared with input embeddings usually)

TOTAL: L × 12d² + 2 × V × d
```

**Example: GPT-3 (175B)**:
```
Specs:
d = 12,288
L = 96
h = 96
V = 50,257
d_ff = 4 × d = 49,152

Calculation:
Embeddings: 50,257 × 12,288 = 617M
Per layer: 12 × (12,288)² = 1.81B
All layers: 96 × 1.81B = 174B
Output: 617M (shared with embeddings)

Total: 174B + 617M = 174.6B ≈ 175B ✓

Memory (FP32): 175B × 4 bytes = 700 GB
Memory (FP16): 175B × 2 bytes = 350 GB
```

**Example: Llama-2-7B**:
```
Specs:
d = 4,096
L = 32
h = 32
V = 32,000
d_ff = 11,008 (SwiGLU uses 2.7× d instead of 4×)

Calculation:
Embeddings: 32,000 × 4,096 = 131M
Per layer: 12 × (4,096)² = 201M (attention + FFN)
All layers: 32 × 201M = 6.43B
Output: 131M

Total: 6.43B + 131M = 6.56B ≈ 7B ✓
```

---

### 24.5.2 FLOPs Calculation (Forward & Backward)

**Why FLOPs Matter**:
- **Training time**: More FLOPs = longer training
- **Cost estimation**: FLOPs × $/FLOP = training cost
- **Hardware selection**: Match FLOPs to GPU capability

**Forward Pass FLOPs**:
```
For a matrix multiplication: C = A·B
A ∈ R^(m×k), B ∈ R^(k×n)
FLOPs = 2·m·k·n (2× for multiply-add)

Transformer forward:
Per layer:
1. Attention: Q·K^T + Softmax + Attention·V
   = 2·n·d·d + 2·n²·d + 2·n²·d = 4nd² + 4n²d
2. FFN: Two matmuls
   = 2·n·d·d_ff + 2·n·d_ff·d = 4nd·d_ff
Total per layer: 4nd² + 4n²d + 4nd·d_ff

All layers: L × (4nd² + 4n²d + 4nd·d_ff)

Total forward: ~6NL where N = total parameters
```

**Backward Pass FLOPs**:
```
Backward = 2× forward FLOPs
- Backprop through each operation costs same as forward
- Need to compute gradients w.r.t. inputs AND weights

Total training FLOPs per token:
Forward: 2N
Backward: 4N  
Total: 6N FLOPs per token

For full training:
FLOPs = 6 × N × D
where N = params, D = dataset size (tokens)
```

**Example: GPT-3 Training**:
```
N = 175B parameters
D = 300B tokens

FLOPs = 6 × 175B × 300B = 3.15 × 10^23 FLOPs

On V100 (125 TFLOPS):
Time = 3.15 × 10^23 / (125 × 10^12) = 2.52 million seconds
     = 29 days on 1 GPU
     = 6 hours on 10,000 GPUs ✓ (matches reality)
```

---

### 24.5.3 Memory Estimation

**Components**:
```
1. Model parameters: N × precision (4 bytes FP32, 2 bytes FP16)
2. Optimizer states: 2N for Adam (m and v)
3. Gradients: N
4. Activations: Depends on batch size and sequence length

Total training memory:
= Model (N) + Optimizer (2N) + Gradients (N) + Activations
= 4N + Activations

For inference:
= Model (N) + KV cache + Activations
```

**Activation Memory**:
```
Per layer, per sample:
= seq_len × batch_size × hidden_dim × precision

Example (seq=2048, batch=8, d=4096, FP16):
= 2048 × 8 × 4096 × 2 bytes = 128 MB per layer
× 32 layers = 4 GB for activations
```

**KV Cache** (Critical for Generation):
```
KV cache stores past keys and values for faster generation

Per layer:
2 × batch × seq_len × num_heads × head_dim × precision

Example (Llama-2-7B, batch=1, seq=4096, FP16):
= 2 × 1 × 4096 × 32 × 128 × 2
= 64 MB per layer
× 32 layers = 2 GB total KV cache

This grows linearly with sequence length!
At 128K context: 2GB × (128K/4K) = 64 GB just for KV cache!
```

---

## INTERVIEW QUESTIONS (Loss, Sampling, Architecture)

### Question 1: Numerical Stability Debugging
**Q**: Your cross-entropy loss suddenly becomes NaN after 10K training steps. Loss was stable before. Diagnose and fix:
```python
logits = model(batch)  # [32, 50000] (batch_size, vocab_size)
loss = F.cross_entropy(logits, targets)
```

**Expected Answer**:
```
Diagnosis steps:

1. Check logit statistics:
print(f"Logits: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")

If max > 100: Overflow likely
If any NaN: Problem earlier in model
If all same value: Dead neurons

2. Check gradients:
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")
        
3. Check learning rate:
current_lr = optimizer.param_groups[0]['lr']
If LR too high (>0.1), gradients explode

4. Check data:
if torch.isnan(batch).any() or torch.isinf(batch).any():
    print("Bad input data!")

Common causes and fixes:

A. Exploding logits (most common):
Fix 1: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

Fix 2: Lower learning rate
optimizer = AdamW(params, lr=current_lr / 10)

Fix 3: Layer normalization
Add LayerNorm before output layer

B. Numerical instability in softmax:
Fix: PyTorch F.cross_entropy already uses log-softmax trick (stable)
If implementing manually, use:
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs, targets)

C. Mixed precision issues:
If using FP16:
- Add loss scaling: scaler = GradScaler()
- Some operations need FP32 (layer norm, softmax)

Recommended fix (catches most cases):
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Add monitoring
if torch.isnan(loss):
    print("NaN loss detected!")
    print(f"Logits: min={logits.min()}, max={logits.max()}")
    # Save checkpoint, reduce LR, restart
```

### Question 2: Sampling Strategy Selection
**Q**: You're deploying a code completion model (like GitHub Copilot). Compare these strategies:
- Greedy
- Beam search (width=5)
- Top-p (p=0.95)

For each, predict:
a) Quality of generated code
b) Diversity of suggestions
c) Latency (speed)
d) Which would you choose and why?

**Expected Answer**:
```
Analysis:

1. Greedy:
Quality: High (picks most likely token each step)
- Usually syntactically correct
- Follows common patterns
Diversity: Zero (deterministic, same output every time)
Latency: Fastest (~50ms for 50 tokens)
- One forward pass per token
- No beam overhead
Issues:
- Repetitive (may get stuck in loops)
- No alternatives (user can't choose)

2. Beam Search (width=5):
Quality: Highest (considers multiple paths, picks best)
- 5 candidates maintain diversity in search
- Length normalization prevents short bias
Diversity: Medium (5 different completions)
Latency: Slowest (~250ms for 50 tokens, 5× slower)
- 5 forward passes per token
- Beam management overhead
Issues:
- Too slow for real-time (user types, waits 250ms)
- All 5 beams may be similar (low practical diversity)

3. Top-p (p=0.95, temp=0.2):
Quality: High (samples from high-probability tokens)
- Temperature 0.2 keeps it focused
- p=0.95 cuts off unlikely tokens
Diversity: High (every generation different)
- Can show 5 diverse suggestions quickly
Latency: Fast (~60ms for 50 tokens)
- One forward pass per token
- Slight sampling overhead
Issues:
- Occasional syntax errors (sampling randomness)
- May need multiple attempts to get best

RECOMMENDATION: Top-p (p=0.95, temperature=0.2-0.4)

Reasoning:
1. Speed critical: User expects <100ms latency
   - Greedy: 50ms ✓
   - Top-p: 60ms ✓
   - Beam: 250ms ✗ (too slow)

2. Diversity matters: Show multiple suggestions
   - Greedy: 1 suggestion ✗
   - Top-p: 5 diverse suggestions ✓
   - Beam: 5 similar suggestions ⚠️

3. Quality good enough: Code completion tolerates minor errors
   - User can edit or reject
   - Auto-complete, not auto-write
   - Top-p with low temperature (0.2-0.4) maintains quality

4. Real-world validation:
   - GitHub Copilot uses sampling (confirmed in papers)
   - Temperature ~0.2 for code (more deterministic than prose)
   - Multiple suggestions shown, user picks

Implementation:
def generate_code_completion(prompt, num_suggestions=5):
    completions = []
    for _ in range(num_suggestions):
        completion = model.generate(
            prompt,
            max_length=50,
            do_sample=True,
            top_p=0.95,
            temperature=0.2,
            repetition_penalty=1.1
        )
        completions.append(completion)
    
    # Rank by heuristics (syntax valid, length, etc.)
    return rank_completions(completions)

Edge case: Multi-line completions (50+ tokens)
- Increase temperature to 0.4 (more exploration)
- Add repetition penalty 1.2 (prevent loops)
- May sacrifice some quality for coherence
```

### Question 3: Parameter Scaling
**Q**: Your team debates: Train one 70B model or ensemble of seven 10B models? Same total parameters. Analyze trade-offs:
- Training cost
- Inference cost
- Model quality
- Deployment complexity

**Expected Answer**:
```
Comparison:

1. TRAINING COST:

One 70B model:
- Requires model parallelism (doesn't fit one GPU)
- Needs 8× A100 80GB (70B × 2 bytes (FP16) = 140GB > 80GB)
- Communication overhead between GPUs
- Training time: ~30 days on 8 GPUs
- Cost: 8 GPUs × 30 days × $3/hr × 24 = $17,280

Seven 10B models:
- Each fits on 1-2 GPUs (10B × 2 bytes = 20GB)
- Can train in parallel (no dependencies)
- 7× training jobs simultaneously
- Training time: ~10 days per model (smaller model faster)
- Cost: 7 models × 2 GPUs × 10 days × $3/hr × 24 = $10,080

Winner: Ensemble (40% cheaper, can parallelize)

2. INFERENCE COST:

One 70B model:
- Needs 4-8 GPUs for inference (model parallelism)
- Throughput: ~10 tokens/sec (limited by communication)
- Cost per 1M requests: 4 GPUs × time
- Memory: 140GB (KV cache, activations)

Seven 10B models:
- Each needs 1 GPU
- Total: 7 GPUs (one per model)
- Throughput: 7× ~30 tokens/sec = 210 tokens/sec
  BUT: Need to ensemble (run all 7), so actual: 30 tokens/sec
- Alternative: Route to one model (70 tokens/sec, but lower quality)
- Cost per 1M requests: 7 GPUs × time (running all)

Winner: 70B (fewer GPUs if model parallelism efficient)
       BUT if route to one 10B: Ensemble wins (cheaper, faster)

3. MODEL QUALITY:

One 70B model:
- Scaling laws: Larger model → better perplexity
- 70B typically 10-15% better than 10B on benchmarks
- Single coherent model (consistent style)
- Can learn more complex patterns

Seven 10B ensemble:
- Diversity: Each model learns different patterns
- Ensemble averaging reduces errors
- But: Individual models weaker
- Empirical: Ensemble of 7× 10B ≈ single 40B (not 70B)

Winner: 70B (better quality by ~20%)

4. DEPLOYMENT:

One 70B model:
- Simple: One model to serve
- Complex: Needs multi-GPU inference setup
- Updates: Update one model
- Versioning: One version to track

Seven 10B ensemble:
- Complex: Manage 7 models
- Simple: Each model on single GPU (easier scaling)
- Updates: Need to update all 7 (coordination needed)
- Versioning: 7 versions to track
- Partial updates: Can update one model without others

Winner: 70B (simpler conceptually, if infra supports multi-GPU)

FINAL RECOMMENDATION: One 70B model

Reasoning:
1. Quality matters most: 20% improvement significant
2. Inference cost comparable (both need multiple GPUs)
3. Simpler deployment (one model to manage)
4. Training cost difference (40%) one-time, quality gap ongoing

Exception - Choose ensemble if:
- Diversity critical (different model styles valued)
- Can route to single model (not true ensemble)
- Budget-constrained (training cost matters)
- Want gradual rollout (deploy models one-by-one)

Real-world evidence:
- OpenAI: Single large model (GPT-3 175B, GPT-4)
- Anthropic: Single large model (Claude)
- Google: Single large model (PaLM, Gemini)

Industry converged on "scale up, not out"
Mixture of Experts (MoE) is middle ground: Large sparse model
```

### Question 4: Memory Optimization
**Q**: You need to serve Llama-2-70B (70B params) but only have 4× A100 40GB GPUs (160GB total). Model needs 140GB just for weights (FP16). Calculate if feasible and what optimizations needed.

**Expected Answer**:
```
Memory breakdown:

1. Model weights:
70B params × 2 bytes (FP16) = 140 GB ✓ Fits (160 GB available)

2. KV cache (critical bottleneck):
Per layer: 2 × batch × seq_len × heads × head_dim × 2 bytes
= 2 × batch × 4096 × 8 × 128 × 2
= batch × 8 MB per layer
× 80 layers = batch × 640 MB

Batch=1: 640 MB ✓
Batch=4: 2.56 GB ✓
Batch=16: 10.2 GB ⚠️
Batch=32: 20.5 GB ✗ (too much!)

3. Activations:
~batch × seq_len × hidden_dim × 2 bytes
= batch × 4096 × 8192 × 2
= batch × 64 MB per layer (checkpointed, not all stored)

4. Total per batch=1:
Weights: 140 GB
KV cache: 640 MB
Activations: ~1 GB
Total: 142 GB < 160 GB ✓ Barely fits!

PROBLEM: Batch size = 1 (very inefficient!)
Need optimizations to increase batch size

OPTIMIZATION 1: Quantization (INT8)
Weights: 70B × 1 byte = 70 GB (2× reduction)
KV cache: 640 MB / 2 = 320 MB (2× reduction)
Total: 72 GB
Available for batching: 160 - 72 = 88 GB
Max batch: 88 GB / 1 GB per sample = 88 ✓

Trade-off: <1% quality loss (acceptable)

OPTIMIZATION 2: Flash Attention
KV cache unchanged, but reduces activation memory
Activation memory: 64 MB → 2 MB (30× reduction)
Gains modest (activations not bottleneck here)

OPTIMIZATION 3: Paged Attention (vLLM)
KV cache: 640 MB → 640 MB (same total)
BUT: Can share across requests (10-20× effective reduction)
Example: 10 requests sharing prompt prefix
Traditional: 10 × 640 MB = 6.4 GB
Paged: 640 MB + 10 × (unique parts) ≈ 2 GB
Enables batch 20+ with efficient memory use

OPTIMIZATION 4: Continuous Batching
Don't wait for all requests to finish
Stream in/out as they complete
Maximizes GPU utilization
Throughput: 3-5× improvement

FINAL CONFIGURATION:
- INT8 quantization (weights + KV cache)
- Flash Attention v2
- Paged Attention (vLLM)
- Continuous batching

Memory:
Weights: 70 GB
KV cache: ~2 GB (paged, 10 requests)
Activations: 200 MB
Total: 72 GB < 160 GB ✓

Batch size: Effective 10-20 requests
Throughput: ~30 tokens/sec (production-ready)
Quality: 99% of FP16 (INT8 minimal loss)

FEASIBILITY: YES, but requires all optimizations!

Without optimizations: Batch=1 (impractical)
With optimizations: Batch=10-20 (production-ready) ✓
```

---

*End of Part 3 - ML Fundamentals Complete*

**Total Coverage:**
- ✅ 24.1: Optimization Algorithms (SGD, Momentum, Adam, AdamW, Adafactor, LAMB, RMSprop)
- ✅ 24.2: Attention Variants (Softmax, Linear, Performer, Longformer, BigBird, Reformer, Flash)
- ✅ 24.3: Loss Functions (Cross-Entropy, Label Smoothing, Focal Loss, Contrastive, Triplet)
- ✅ 24.4: Sampling Strategies (Greedy, Beam, Top-k, Top-p, Temperature, Repetition Penalty)
- ✅ 24.5: Model Architecture (Parameter counting, FLOPs, Memory estimation)
- ✅ Production case studies from OpenAI, Google, Meta, Anthropic
- ✅ Hao Hoang-style interview questions with detailed answers
- ✅ Real-world trade-offs and decision frameworks
