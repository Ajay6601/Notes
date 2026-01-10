# ML Fundamentals - Part 2: Neural Networks, CNNs, RNNs, Loss Functions & Metrics

**Version 1.0 - Complete Interview-Ready Notes**

Sources: Ian Goodfellow's Deep Learning, Stanford CS231n/CS224n, Fast.ai, Andrej Karpathy, Sebastian Raschka, Chip Huyen, Production systems at Google/Meta/OpenAI, Hao Hoang interview materials.

---

## 14.2 NEURAL NETWORK BASICS

### Why Neural Networks?

**Historical Context**:
- 1958: Perceptron (Rosenblatt) - linear classifier
- 1986: Backpropagation (Rumelhart, Hinton, Williams) - made training possible
- 2006: Deep Learning renaissance (Hinton) - unsupervised pre-training
- 2012: AlexNet (Krizhevsky) - ImageNet breakthrough (CNNs dominate vision)
- 2017: Attention Is All You Need (Vaswani) - Transformers dominate NLP

**When to Use Neural Networks**:
```
Use NNs when:
✓ Complex non-linear patterns (image, audio, text)
✓ Large datasets (>100K samples for supervised)
✓ Have compute (GPUs)
✓ Accuracy > interpretability
✓ Can handle longer training times

Use Classical ML when:
✓ Tabular data with <100 features
✓ Small datasets (<10K samples)
✓ Need interpretability (healthcare, finance)
✓ Low latency critical (<1ms)
✓ Limited compute (CPU only)
```

---

### Feedforward Networks (Multi-Layer Perceptron)

**Architecture**:
```
Input Layer → Hidden Layer(s) → Output Layer

Example (3-layer network):
Input: x ∈ ℝⁿ
Hidden 1: h₁ = σ(W₁x + b₁) ∈ ℝᵐ¹
Hidden 2: h₂ = σ(W₂h₁ + b₂) ∈ ℝᵐ²
Output: ŷ = W₃h₂ + b₃ ∈ ℝᵏ

Where:
- σ = activation function (non-linearity)
- W = weight matrices
- b = bias vectors
```

**Why Multiple Layers?**:
```
Single layer (perceptron):
- Can only learn linear decision boundaries
- XOR problem: Not linearly separable

Two layers (1 hidden):
- Universal approximator (Cybenko, 1989)
- Can approximate any continuous function
- BUT: May need exponentially many neurons

Deep networks (many hidden):
- Learn hierarchical features
- Fewer parameters than shallow wide networks
- Example (Image): Edges → Shapes → Objects → Scenes
```

**Width vs Depth Trade-off**:
```
Wide networks (few layers, many neurons):
+ Easier to train (less vanishing gradient)
+ Faster training (more parallelizable)
- More parameters (memory intensive)
- Less feature reuse (redundant representations)

Deep networks (many layers, fewer neurons):
+ Hierarchical feature learning
+ Fewer parameters (more efficient)
+ Better generalization (often)
- Harder to train (vanishing/exploding gradients)
- Requires careful initialization

Production rule of thumb:
- Start with 2-3 hidden layers
- Increase depth if underfitting
- Increase width if have compute budget
```

---

### Backpropagation Algorithm

**Core Idea**: Compute gradients efficiently via chain rule.

**Forward Pass** (Compute predictions):
```
For layer l = 1 to L:
  z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾  (linear)
  a⁽ˡ⁾ = σ(z⁽ˡ⁾)             (activation)

Output: ŷ = a⁽ᴸ⁾
Loss: J = L(y, ŷ)
```

**Backward Pass** (Compute gradients):
```
For layer l = L down to 1:
  δ⁽ˡ⁾ = ∂J/∂z⁽ˡ⁾
       = (∂J/∂a⁽ˡ⁾) ⊙ σ'(z⁽ˡ⁾)     [chain rule]
       = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)  [recursive]
  
  ∂J/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
  ∂J/∂b⁽ˡ⁾ = δ⁽ˡ⁾

Where ⊙ = element-wise multiplication
```

**Computational Cost** (CRITICAL FOR INTERVIEWS):
```
Forward pass: 2N FLOPs
- N = number of parameters
- 2N because: multiply + add for each parameter

Backward pass: 4N FLOPs
- 2N for gradient computation (same as forward)
- 2N for gradient propagation (chain rule)

TOTAL: 6N FLOPs per sample per iteration

Example:
- Model: 7B parameters
- Batch size: 1M tokens
- FLOPs: 6 × 7B × 1M = 42 × 10¹⁵ = 42 PFLOPs

Interview trap: Many say "2N FLOPs" (only forward pass!)
Correct answer: "6N FLOPs" (forward + backward)
```

**Memory Requirements**:
```
Forward pass (inference):
- Store weights: N × bytes_per_param
- Store activations: batch_size × layer_sizes × bytes_per_activation
- Typically: Activations dominate for large batches

Backward pass (training):
- Store forward activations (needed for backward)
- Store gradients (same size as weights)
- Peak memory: ~2× forward pass memory

Gradient checkpointing:
- Recompute activations during backward (instead of storing)
- Trade-off: 33% more compute, 50% less memory
- Critical for training large models (GPT-3, Llama)
```

**Numerical Stability**:
```
Problems:
1. Vanishing gradients: σ' → 0 (deep networks)
2. Exploding gradients: σ' → ∞ (RNNs)
3. Numerical overflow: exp(1000) = inf

Solutions:
1. Careful activation choice (ReLU > sigmoid)
2. Gradient clipping (clip to [-1, 1])
3. Batch normalization (normalize activations)
4. Residual connections (skip connections)
5. Proper initialization (Xavier, He)
```

---

### Activation Functions

**Sigmoid** σ(x) = 1/(1 + e⁻ˣ):
```
Range: (0, 1)
Derivative: σ'(x) = σ(x)(1 - σ(x))

Pros:
+ Smooth, differentiable
+ Output interpretable as probability
+ Historically popular

Cons:
- Vanishing gradient (σ' ≈ 0 for |x| > 5)
- Not zero-centered (slows convergence)
- Expensive (exponential)

Use cases:
✓ Output layer (binary classification)
✗ Hidden layers (use ReLU instead)
```

**Tanh** tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ):
```
Range: (-1, 1)
Derivative: tanh'(x) = 1 - tanh²(x)

Pros:
+ Zero-centered (better than sigmoid)
+ Stronger gradient than sigmoid

Cons:
- Still vanishing gradient for |x| > 2
- Expensive (exponential)

Use cases:
✓ RNN hidden states (historically)
✓ When need zero-centered activations
✗ Deep networks (use ReLU)
```

**ReLU (Rectified Linear Unit)** ReLU(x) = max(0, x):
```
Range: [0, ∞)
Derivative: ReLU'(x) = 1 if x > 0, else 0

Pros:
+ No vanishing gradient (for x > 0)
+ Computationally cheap (just comparison)
+ Sparse activation (50% neurons = 0)
+ Enables deep networks

Cons:
- Dying ReLU problem (neurons stuck at 0)
- Not zero-centered
- Not differentiable at x=0 (doesn't matter in practice)

Use cases:
✓ Default for hidden layers (CNNs, MLPs)
✓ When training deep networks
✗ Output layer (use softmax/sigmoid)

Production note:
- Most CNNs use ReLU (AlexNet, VGG, ResNet)
- 90% of hidden layers in production use ReLU or variants
```

**Leaky ReLU** LeakyReLU(x) = max(0.01x, x):
```
Range: (-∞, ∞)
Derivative: 0.01 if x < 0, else 1

Pros:
+ Fixes dying ReLU (small gradient for x < 0)
+ All benefits of ReLU

Cons:
- Hyperparameter (leaky slope, typically 0.01)

Use cases:
✓ Alternative to ReLU when dying ReLU is problem
✓ GANs (common choice)
```

**ELU (Exponential Linear Unit)** ELU(x) = x if x > 0, else α(eˣ - 1):
```
Range: (-α, ∞)
Derivative: 1 if x > 0, else ELU(x) + α

Pros:
+ Zero-centered mean activation
+ Smooth everywhere (helps optimization)
+ Robust to noise

Cons:
- Expensive (exponential for x < 0)
- Hyperparameter α (typically 1.0)

Use cases:
✓ When need smooth activations
✓ Noise-robust networks
✗ When speed critical (use ReLU)
```

**GELU (Gaussian Error Linear Unit)** GELU(x) = x·Φ(x):
```
Φ(x) = CDF of standard normal distribution
Approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

Pros:
+ Smooth (differentiable everywhere)
+ State-of-the-art in Transformers (BERT, GPT)
+ Probabilistic interpretation

Cons:
- Expensive (tanh + polynomial)

Use cases:
✓ Transformer models (BERT, GPT, Llama)
✓ When accuracy > speed
✗ Inference-critical applications (use ReLU)

Production note:
- BERT, GPT-2, GPT-3 use GELU
- 10-20% slower than ReLU but 1-2% better accuracy
```

**Softmax** (Output layer):
```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)

Properties:
- Output sums to 1 (probability distribution)
- Differentiable
- Amplifies differences (max becomes dominant)

Use cases:
✓ Multi-class classification output layer
✗ Hidden layers (computationally expensive)

Numerical stability trick:
softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
(subtract max to prevent overflow)
```

**Swish (SiLU)** Swish(x) = x·σ(βx):
```
β = 1 typically (SiLU = Swish with β=1)

Pros:
+ Smooth, non-monotonic
+ Slightly better than ReLU (1-2%)
+ Self-gated (like LSTM gates)

Cons:
- More expensive than ReLU

Use cases:
✓ Mobile networks (MobileNet, EfficientNet)
✓ When 1-2% accuracy boost worth compute cost
✗ Latency-critical applications
```

**Production Selection Guide**:
```
CNNs: ReLU (default), Leaky ReLU (if dying ReLU problem)
RNNs: tanh (hidden), sigmoid (gates)
Transformers: GELU (BERT, GPT), ReLU (Llama, some variants)
Output layer: Softmax (multi-class), Sigmoid (binary), Linear (regression)
Speed-critical: ReLU (fastest)
Accuracy-critical: GELU or Swish
```

---

### Optimization Algorithms

**Gradient Descent Variants**:

**Batch Gradient Descent (BGD)**:
```
for epoch in epochs:
    gradient = compute_gradient(all_data)  # Full dataset
    theta = theta - lr * gradient

Pros:
+ Stable convergence (smooth gradient)
+ Guaranteed to find global minimum (convex) or good local minimum

Cons:
- Very slow (one update per epoch)
- Requires full dataset in memory
- Redundant computation (similar samples)
```

**Stochastic Gradient Descent (SGD)**:
```
for epoch in epochs:
    shuffle(data)
    for sample in data:
        gradient = compute_gradient(sample)  # Single sample
        theta = theta - lr * gradient

Pros:
+ Fast updates (N updates per epoch)
+ Works online (streaming data)
+ Escapes shallow local minima (noise helps)

Cons:
- Noisy convergence (high variance)
- Never truly converges (oscillates)
- Sensitive to learning rate
```

**Mini-Batch SGD** (Production Standard):
```
for epoch in epochs:
    shuffle(data)
    for batch in batches:
        gradient = compute_gradient(batch)  # Batch of B samples
        theta = theta - lr * gradient

Typical batch sizes:
- Computer Vision: 32, 64, 128, 256
- NLP: 16, 32, 64 (longer sequences)
- Recommendation: 512, 1024, 2048 (small embeddings)

Pros:
+ Balance between BGD and SGD
+ Efficient GPU utilization (parallel)
+ Smooth convergence (less noise than SGD)

Cons:
- Requires tuning batch size
```

**Batch Size Impact**:
```
Small batch (8-32):
+ Better generalization (noise acts as regularization)
+ Lower memory (fits larger models)
+ Faster convergence (more frequent updates)
- Noisy gradients
- Slower wall-clock time (less parallelism)

Large batch (256-1024):
+ Faster training (wall-clock time)
+ Stable gradients
+ Better GPU utilization
- Worse generalization ("sharp minima")
- Requires more memory
- May need LR scaling

Linear Scaling Rule (Goyal et al., 2017):
- If batch size × k, multiply LR × k
- Example: Batch 32 LR 0.1 → Batch 256 LR 0.8
- Works up to batch ~8K, then breaks down

Production:
- Start with batch=32 (good default)
- Increase if training too slow (and have memory)
- Monitor validation accuracy (large batch may hurt)
```

**Momentum Methods**:

**SGD with Momentum**:
```
v := βv - η∇J(θ)
θ := θ + v

β = 0.9 typically (momentum coefficient)

Intuition:
- Accumulates velocity in direction of persistent gradient
- Dampens oscillations
- Accelerates in consistent directions

Pros:
+ 2-10x faster than vanilla SGD
+ Less sensitive to learning rate

Cons:
- Additional hyperparameter β
```

**Nesterov Accelerated Gradient (NAG)**:
```
v := βv - η∇J(θ + βv)  # Look-ahead gradient
θ := θ + v

Intuition:
- Evaluates gradient at "look-ahead" position
- Corrects momentum if about to overshoot
- Slightly better than standard momentum

Pros:
+ Better convergence than momentum
+ Especially for convex functions

Cons:
- Slightly more complex
- Minimal improvement in deep learning (but widely used)

Production:
- PyTorch: SGD(momentum=0.9, nesterov=True)
- Stable default for CNNs
```

**Adaptive Learning Rate Methods**:

**RMSprop** (Hinton, 2012):
```
v := βv + (1-β)(∇J(θ))²  # Exponential moving average of squared gradient
θ := θ - η/√(v + ε) × ∇J(θ)

β = 0.9 typically
ε = 1e-8 (numerical stability)

Intuition:
- Adapts learning rate per parameter
- Large gradients → small effective LR
- Small gradients → large effective LR

Pros:
+ Works well with RNNs (prevents exploding gradients)
+ Less sensitive to learning rate choice

Cons:
- Can converge too fast (aggressive LR decay)
```

**Adam (Adaptive Moment Estimation)**:
```
m := β₁m + (1-β₁)∇J(θ)     [first moment: mean]
v := β₂v + (1-β₂)(∇J(θ))²   [second moment: variance]
m̂ := m/(1-β₁ᵗ)              [bias correction]
v̂ := v/(1-β₂ᵗ)              [bias correction]
θ := θ - η·m̂/(√v̂ + ε)

Default hyperparameters:
- β₁ = 0.9 (exponential decay rate for first moment)
- β₂ = 0.999 (exponential decay rate for second moment)
- η = 0.001 (learning rate)
- ε = 10⁻⁸ (numerical stability)

Pros:
+ Adaptive learning rates (per parameter)
+ Fast convergence
+ Works well out-of-the-box (good defaults)
+ Most popular optimizer (as of 2024)

Cons:
- More memory (stores m and v)
- Can overfit on small datasets (use AdamW)
```

**AdamW (Adam with Weight Decay)** - CRITICAL DIFFERENCE:
```
Adam (WRONG weight decay):
gradient = gradient + λ×θ  # Add to gradient
θ = θ - η×(m̂/(√v̂ + ε))    # Adaptive LR affects weight decay

AdamW (CORRECT weight decay):
θ = θ - η×(m̂/(√v̂ + ε) + λθ)  # Weight decay OUTSIDE adaptive LR

Why this matters:
- Adam: Weight decay gets scaled by adaptive LR (incorrect)
- AdamW: Weight decay applied uniformly (correct)
- Result: AdamW generalizes better (less overfitting)

Production standard:
- Use AdamW (not Adam) for training large models
- Llama, GPT-3, BERT all trained with AdamW
- Default in HuggingFace Transformers
```

**Learning Rate Schedules**:

**Step Decay**:
```python
# Drop LR by factor every N epochs
lr = initial_lr * (decay_factor ** (epoch // step_size))

Example:
- Initial LR: 0.1
- Decay factor: 0.1
- Step size: 30 epochs
- Schedule: 0.1 (epochs 0-29) → 0.01 (30-59) → 0.001 (60+)

When to use:
✓ Simple baseline
✓ Fixed training schedule (know epochs in advance)
```

**Exponential Decay**:
```python
lr = initial_lr * (decay_rate ** epoch)

Example:
- Initial LR: 0.1
- Decay rate: 0.95
- Epoch 0: 0.1
- Epoch 10: 0.1 × 0.95¹⁰ = 0.060
- Epoch 100: 0.1 × 0.95¹⁰⁰ = 0.006

Smoother than step decay
```

**Cosine Annealing**:
```python
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × epoch / T))

Where T = total epochs

Visualization:
LR starts at lr_max
Smoothly decays following cosine curve
Ends at lr_min

Benefits:
+ Smooth decay (no sudden drops)
+ Widely used in modern training (ResNet, Vision Transformers)

Variant: Cosine Annealing with Warm Restarts (SGDR)
- Restart LR periodically (sawtooth pattern)
- Helps escape local minima
```

**Learning Rate Warmup** (CRITICAL FOR TRANSFORMERS):
```python
# Linear warmup followed by decay
if epoch < warmup_epochs:
    lr = initial_lr * (epoch / warmup_epochs)  # Linear increase
else:
    lr = cosine_decay(epoch - warmup_epochs)   # Cosine decay

Why warmup needed:
- Large models (Transformers): Unstable at initialization
- Large batch sizes: Need gradual increase
- Adam with large LR: Exploding gradients at start

Typical warmup:
- Vision: 5-10 epochs (or 5K-10K iterations)
- NLP (Transformers): 10-40% of total steps
- BERT: 10K steps warmup (out of 1M steps)

Production:
- Always use warmup for Transformers
- May skip for CNNs (but doesn't hurt)
```

**Cyclical Learning Rates** (Leslie Smith, 2017):
```python
# Cycle between min and max LR
lr = lr_min + (lr_max - lr_min) × triangular_function(iteration)

Benefits:
+ Escapes local minima (LR increases periodically)
+ Faster convergence (fewer epochs)
+ Better generalization

1cycle policy (most popular):
1. Linear increase: lr_min → lr_max (30% of training)
2. Cosine decay: lr_max → lr_min (70% of training)

Fast.ai default:
- Widely used in fast.ai courses
- Often trains in 1/3 the epochs vs constant LR
```

**Mixed Precision Training**:

**FP32 vs FP16 vs BF16**:
```
FP32 (Full Precision):
- Range: ±10³⁸
- Precision: 7 decimal digits
- Memory: 4 bytes per parameter
- Speed: 1× (baseline)

FP16 (Half Precision):
- Range: ±65,504
- Precision: 3 decimal digits
- Memory: 2 bytes per parameter
- Speed: 2-3× faster (on Tensor Cores)
- Problem: Underflow (gradients → 0)

BF16 (Brain Float16):
- Range: ±10³⁸ (same as FP32)
- Precision: 2 decimal digits
- Memory: 2 bytes per parameter
- Speed: 2-3× faster
- Better than FP16 (no underflow issues)
```

**Automatic Mixed Precision (AMP)**:
```python
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        loss = model(batch)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

How it works:
1. Forward pass in FP16 (faster, less memory)
2. Scale loss by large factor (prevent underflow)
3. Backward pass in FP16 (compute gradients)
4. Unscale gradients (divide by scale factor)
5. Update weights in FP32 (master copy)

Benefits:
+ 2-3× speedup (on GPUs with Tensor Cores)
+ 50% memory reduction (enables larger batches/models)
+ <1% accuracy loss (often no loss)

Production standard:
- PyTorch: torch.cuda.amp
- TensorFlow: tf.keras.mixed_precision
- Almost always enabled in production (free speedup)
```

**Gradient Accumulation**:
```python
# Simulate large batch size with limited GPU memory
accumulation_steps = 4  # Simulate batch_size × 4

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset gradients

Example:
- GPU memory: 16GB
- Desired batch: 128
- Fits in memory: 32
- Accumulation steps: 128/32 = 4

Benefits:
+ Train with larger effective batch size
+ Same results as true large batch
+ No additional computation (just gradient accumulation)

Production use case:
- Training large models (GPT, Llama) on consumer GPUs
- DeepSpeed, Megatron use this internally
```

---

### Advanced CNN Layers

**Depthwise Separable Convolutions** (MobileNet):

**Standard Convolution**:
```
Input: H×W×C_in
Filter: K×K×C_in×C_out
Output: H×W×C_out

Computation: H×W × K×K×C_in×C_out
Parameters: K²×C_in×C_out

Example:
- Input: 56×56×128
- Filter: 3×3×128×256
- Computation: 56×56 × 3×3×128×256 = 924M FLOPs
- Parameters: 3×3×128×256 = 295K
```

**Depthwise Separable Convolution**:
```
Step 1: Depthwise Convolution
- Input: H×W×C_in
- Filter: K×K×1 (per channel)
- Output: H×W×C_in
- Computation: H×W × K×K×C_in

Step 2: Pointwise Convolution (1×1)
- Input: H×W×C_in
- Filter: 1×1×C_in×C_out
- Output: H×W×C_out
- Computation: H×W × C_in×C_out

Total computation: H×W × (K×K×C_in + C_in×C_out)

Reduction factor: (K²×C_in×C_out) / (K²×C_in + C_in×C_out)
                ≈ C_out / (K² + C_out/C_in)
                
For C_out = C_in and K=3: ~9× reduction

Example (same as above):
- Depthwise: 56×56 × 3×3×128 = 37M FLOPs
- Pointwise: 56×56 × 128×256 = 103M FLOPs
- Total: 140M FLOPs (vs 924M standard → 6.6× faster!)
- Parameters: 3×3×128 + 128×256 = 34K (vs 295K → 8.7× fewer)
```

**Why This Matters**:
```
MobileNetV1 (2017):
- Uses depthwise separable everywhere
- 4.2M parameters (vs 25M ResNet-50)
- 569M FLOPs (vs 3.86B ResNet-50)
- 70.6% ImageNet accuracy (vs 76% ResNet-50)
- Trade-off: 5.4% accuracy for 7× speedup

Production use cases:
✓ Mobile devices (limited compute)
✓ Edge deployment (IoT, embedded)
✓ Real-time applications (video processing)
✓ Cost-sensitive inference (cloud GPUs)
```

**Grouped Convolutions** (ResNeXt):

**Standard Convolution**:
```
Input: C_in channels
Output: C_out channels
One filter operates on ALL input channels
```

**Grouped Convolution**:
```
Input: Split C_in into G groups (C_in/G per group)
Each group processed independently
Output: G groups of C_out/G channels
Concatenate outputs

Parameters: (K²×C_in×C_out) / G

Example (ResNeXt):
- C_in = 256, C_out = 256
- Groups = 32
- Parameters: (3²×256×256) / 32 = 18K (vs 590K standard → 32× fewer)

Benefits:
+ Fewer parameters (reduction factor = G)
+ Faster computation (parallelizable)
+ Better accuracy (ResNeXt > ResNet with same FLOPs)

Why it works:
- Forces learning of diverse features (groups can't communicate)
- Reduces overfitting (fewer parameters)
- Resembles ensemble of G networks
```

**Dilated/Atrous Convolutions** (DeepLab):

**Standard Convolution**: Receptive field = K
```
3×3 filter: Receptive field = 3
```

**Dilated Convolution**: Receptive field = K + (K-1)×(r-1)
```
3×3 filter with dilation r=2:
- Filter: [x _ x _ x]
           [_ _ _ _ _]
           [x _ x _ x]  (gaps between elements)
- Receptive field: 3 + 2×1 = 5

3×3 filter with dilation r=3:
- Receptive field: 3 + 2×2 = 7

Benefits:
+ Larger receptive field (no additional parameters)
+ No downsampling needed (preserves resolution)
+ Captures multi-scale context

Use cases:
✓ Semantic segmentation (DeepLab, PSPNet)
✓ Audio generation (WaveNet)
✓ Time series (TCN - Temporal Convolutional Networks)
```

**Squeeze-and-Excitation (SE) Blocks** (SENet):

**Channel Attention Mechanism**:
```
Goal: Recalibrate channel-wise feature responses

Architecture:
1. Squeeze: Global Average Pooling (H×W×C → 1×1×C)
   - Aggregate spatial information per channel
   
2. Excitation: FC → ReLU → FC → Sigmoid
   - FC1: C → C/r (bottleneck, r=16 typical)
   - FC2: C/r → C
   - Sigmoid: Output attention weights (0,1)
   
3. Scale: Multiply input by attention weights
   - input × attention_weights (channel-wise)

Parameters: 2×C²/r (tiny addition, ~0.1% of total)

Example:
Input: 56×56×256
- Squeeze: 256 channels → 256 scalars (GAP)
- Excitation: 256 → 16 → 256 (two FC layers)
- Attention: 256 weights (one per channel)
- Scale: 56×56×256 × attention

Benefits:
+ 1-2% ImageNet accuracy improvement
+ <1% parameter increase
+ Minimal computation overhead
+ Plug-and-play (add to any architecture)

Production:
- SE-ResNet: ResNet + SE blocks (78.7% ImageNet, +2.7% vs ResNet)
- EfficientNet uses SE blocks by default
```

---

### Transfer Learning Strategies

**Feature Extraction vs Fine-Tuning**:

**Feature Extraction** (Frozen Backbone):
```python
# Load pretrained model
model = torchvision.models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(2048, num_classes)  # Only this trains

# Train only the new head
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

When to use:
✓ Small dataset (<10K images)
✓ Similar to ImageNet (natural images)
✓ Limited compute (fast training)

Expected:
- Training time: Minutes (only head trains)
- Accuracy: 80-90% (depends on similarity to ImageNet)
```

**Fine-Tuning** (Unfrozen Layers):
```python
# Load pretrained model
model = torchvision.models.resnet50(pretrained=True)

# Replace final layer
model.fc = nn.Linear(2048, num_classes)

# Fine-tune with small learning rate
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},  # Early layers: very small LR
    {'params': model.layer4.parameters(), 'lr': 1e-4}, # Late layers: small LR
    {'params': model.fc.parameters(), 'lr': 1e-3}      # New head: normal LR
])

When to use:
✓ Medium dataset (10K-100K images)
✓ Different from ImageNet (medical, satellite)
✓ Have compute budget

Discriminative learning rates:
- Early layers: Generic features (edges) → Small LR
- Late layers: Specific features → Medium LR
- New head: Random init → Large LR

Expected:
- Training time: Hours
- Accuracy: 85-95% (better than feature extraction)
```

**Progressive Unfreezing** (Fast.ai Strategy):
```python
# Stage 1: Train head only (5 epochs)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True
train(epochs=5, lr=1e-3)

# Stage 2: Unfreeze last block (5 epochs)
model.layer4.requires_grad = True
train(epochs=5, lr=1e-4)

# Stage 3: Unfreeze all (10 epochs)
for param in model.parameters():
    param.requires_grad = True
train(epochs=10, lr=1e-5)

Benefits:
+ Prevents catastrophic forgetting (gradual adaptation)
+ Faster convergence (head stabilizes first)
+ Better final accuracy

Production standard (Fast.ai):
- Used in fast.ai winners (Kaggle)
- 5-10% accuracy improvement vs full fine-tuning
```

---

### Debugging Neural Networks (Andrej Karpathy's Checklist)

**1. Overfit a Single Batch**:
```python
# Take one batch, try to get 100% accuracy
batch = next(iter(dataloader))
for i in range(1000):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Iteration {i}: Loss {loss.item()}")

Expected: Loss → 0 (model can memorize one batch)

If loss doesn't decrease:
- Learning rate too low (try 10x higher)
- Loss function wrong (check labels, predictions)
- Gradient flow broken (check with grad norms)
- Model too simple (add capacity)
```

**2. Check Loss at Initialization**:
```python
# Before training, check if loss makes sense
model = Model()
batch = next(iter(dataloader))
loss = model(batch)
print(f"Initial loss: {loss.item()}")

Expected:
- Cross-entropy (10 classes): -log(1/10) = 2.3
- Binary cross-entropy: -log(0.5) = 0.69

If loss is very different:
- Weights not initialized properly
- Loss function implemented incorrectly
- Labels wrong (check label distribution)
```

**3. Visualize Gradients**:
```python
# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item()}")

Warning signs:
- Grad norm = 0: Dead neurons (ReLU stuck at 0)
- Grad norm >> 1: Exploding gradients (clip or reduce LR)
- Grad norm = NaN: Numerical instability (reduce LR, check loss)

Tools:
- TensorBoard: Log gradient histograms
- PyTorch: torch.nn.utils.clip_grad_norm_()
```

**4. Monitor Training Curves**:
```python
# Plot train and val loss every epoch
train_losses = []
val_losses = []

for epoch in epochs:
    train_loss = train()
    val_loss = validate()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    plot(train_losses, val_losses)

Patterns:
- Both decreasing: Good (keep training)
- Train ↓, Val ↑: Overfitting (add regularization)
- Both high: Underfitting (increase capacity)
- Train flat: Learning rate too low or stuck
```

**5. Sanity Checks**:
```
Before training:
✓ Data augmentation off during validation
✓ Batch normalization in eval() mode during inference
✓ Dropout off during inference (model.eval())
✓ Loss function matches task (CE for classification, MSE for regression)
✓ Learning rate not too high (loss explodes) or too low (no progress)
✓ Batch size fits in memory (GPU doesn't OOM)

During training:
✓ Training loss decreasing
✓ Validation loss tracking training (within 2×)
✓ Gradient norms reasonable (0.01 to 10)
✓ Weights updating (check param.data before and after step)
✓ Accuracy increasing (not just loss decreasing)

Common bugs:
✗ Forgot model.train() / model.eval()
✗ Data not normalized (ImageNet mean/std)
✗ Labels wrong (0-indexed vs 1-indexed)
✗ Loss not averaged over batch (multiply by batch_size)
✗ Validation set in training augmentation
```

---

## Interview Questions (Additional Optimization & CNN)

### Question 21: Learning Rate Warmup
**Q**: "You're training a Transformer from scratch. Loss explodes in first 100 iterations then NaN. You're using Adam with LR=1e-4. What's wrong?"

**Expected Answer**:
- **Problem**: No learning rate warmup (Transformers unstable at initialization)

**Why warmup needed**:
```
At initialization:
- Weights random (large gradients)
- Adam's adaptive LR not calibrated yet
- First few updates can be huge (exploding gradients)

With warmup:
- LR starts at 0 (or very small)
- Gradually increases to target LR
- Gives Adam time to calibrate
- Prevents exploding gradients
```

**Solution**:
```python
# Linear warmup for 4000 steps
warmup_steps = 4000
def lr_schedule(step):
    if step < warmup_steps:
        return step / warmup_steps  # Linear increase 0 → 1
    else:
        return 1.0  # Then constant (or add decay)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

# Or use Hugging Face
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**Production settings**:
- BERT: 10K warmup steps (out of 1M)
- GPT: 2K warmup steps (out of 300K)
- Small models: 1% of total steps
- Large models: 5-10% of total steps

**Follow-up**: "After adding warmup, training is stable but very slow. Can you speed it up?"
- **Mixed precision** (AMP): 2-3× speedup
- **Gradient accumulation**: Larger effective batch size
- **Larger batch size**: If have GPU memory
- **Distributed training**: Multiple GPUs

---

### Question 22: Batch Size vs Learning Rate
**Q**: "You increase batch size from 32 to 256 (8×). Should you change learning rate?"

**Expected Answer**:
- **Yes, use linear scaling rule**

**Linear Scaling Rule** (Goyal et al., 2017):
```
If batch size × k, then learning rate × k

Example:
- Batch 32, LR 0.1 → Batch 256, LR 0.8

Why it works:
- Larger batch: Smoother gradient (less noise)
- Can take larger steps without overshooting
- Maintains same "progress" per epoch

Limitations:
- Works up to batch ~8K
- Beyond 8K: Sub-linear scaling (diminishing returns)
- May hurt generalization (large batches → sharp minima)
```

**Example** (ImageNet training):
```
Baseline:
- Batch size: 256
- Learning rate: 0.1
- Epochs: 90
- Accuracy: 76%

Scaled (8× batch):
- Batch size: 2048 (256 × 8)
- Learning rate: 0.8 (0.1 × 8)
- Epochs: 90
- Accuracy: 76% (same!)
- Wall-clock time: 3× faster (larger batches = fewer iterations)
```

**Caveats**:
```
Must also adjust:
1. Warmup: Longer warmup for large LR (5-10 epochs)
2. Weight decay: Scale by √k (not linearly)
3. Batch Normalization: May need larger momentum

If accuracy drops:
- Reduce LR scaling (use 0.5×k instead of 1×k)
- Increase warmup period
- Try LARS optimizer (for very large batches)
```

---

### Question 23: Mixed Precision Training
**Q**: "You enable mixed precision training (FP16). Training is 2× faster but validation accuracy drops from 85% to 60%. What happened?"

**Expected Answer**:
- **Gradient underflow** (FP16 range too small)

**Debugging**:
```python
# Check for gradient underflow
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm == 0 or grad_norm != grad_norm:  # Zero or NaN
            print(f"{name}: Gradient underflow")

Common causes:
1. Loss values too small (underflow in FP16)
2. Gradients too small (vanish in FP16)
3. Forget to use GradScaler (gradient scaling)
```

**Solution**:
```python
# Correct mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # CRITICAL: Use gradient scaler

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # Forward in FP16
        loss = model(batch)
    
    # MUST scale loss before backward
    scaler.scale(loss).backward()
    
    # MUST use scaler.step (not optimizer.step directly)
    scaler.step(optimizer)
    scaler.update()

Without GradScaler:
- Gradients computed in FP16
- Small gradients underflow to 0
- No learning (weights don't update)
```

**Alternative: Use BF16** (Brain Float 16):
```python
# BF16: Same range as FP32 (no underflow issues)
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss = model(batch)
loss.backward()
optimizer.step()

Benefits:
+ No GradScaler needed (no underflow)
+ 2× speedup (same as FP16)
+ Simpler code

Requirements:
- Ampere GPUs or newer (A100, RTX 3090+)
- PyTorch 1.10+
```

---

### Question 24: Depthwise Separable Convolutions
**Q**: "Your MobileNet model has 4.2M parameters but your ResNet-50 has 25M. Yet MobileNet is only 5% less accurate. How is this possible?"

**Expected Answer**:
- **Depthwise separable convolutions** (architectural innovation)

**Computation comparison**:
```
Standard 3×3 Conv:
- Input: 56×56×128
- Output: 56×56×256
- FLOPs: 56×56 × 3×3×128×256 = 924M
- Parameters: 3×3×128×256 = 295K

Depthwise Separable:
- Depthwise: 56×56 × 3×3×128 = 37M
- Pointwise: 56×56 × 128×256 = 103M
- Total: 140M FLOPs (6.6× fewer!)
- Parameters: 3²×128 + 128×256 = 34K (8.7× fewer!)

Trade-off:
- 8× fewer parameters
- 7× faster inference
- 5% accuracy loss (70.6% vs 76%)
```

**Why it works**:
```
Standard Conv: Mixes spatial and channel information together
Depthwise Separable: Separates spatial (depthwise) and channel (pointwise) mixing

Benefits:
+ Fewer parameters (less overfitting on small datasets)
+ Faster inference (critical for mobile)
+ Still captures important patterns

Limitations:
- Slightly less expressive (can't learn some interactions)
- Needs more layers to compensate
- 5% accuracy loss acceptable for 7× speedup
```

**Production use**:
```
When to use:
✓ Mobile/edge deployment (limited compute)
✓ Real-time applications (inference speed critical)
✓ Large-scale serving (reduce GPU costs)
✓ Small datasets (fewer parameters prevent overfitting)

When NOT to use:
✗ Accuracy critical (medical, autonomous vehicles)
✗ Have abundant compute (cloud GPUs)
✗ Research (trying to beat SOTA)
```

---

### Question 25: SE Blocks
**Q**: "Adding SE blocks to your ResNet improves accuracy by 2% but adds <1% parameters. How do SE blocks work and why are they so effective?"

**Expected Answer**:
- **Channel attention mechanism** (learns which channels are important)

**How SE blocks work**:
```
Goal: Recalibrate channel-wise feature responses

Example:
Input: 56×56×256 (H×W×C)

Step 1 - Squeeze (Global context):
- Global Average Pool: 256 channels → 256 scalars
- Each scalar = average activation of that channel
- Captures "what patterns are present?"

Step 2 - Excitation (Channel importance):
- FC1: 256 → 16 (bottleneck, reduce by 16×)
- ReLU
- FC2: 16 → 256
- Sigmoid → attention weights ∈ (0,1)
- Learns "which channels are important?"

Step 3 - Scale:
- Multiply input by attention weights (element-wise)
- Important channels amplified (weight ≈ 1)
- Unimportant channels suppressed (weight ≈ 0)

Parameters: 2 × C²/r (r=16)
- For C=256: 2 × 256²/16 = 8K parameters (tiny!)
```

**Why so effective**:
```
Intuition:
- Not all channels equally important
- Example: Edge detector irrelevant for smooth regions
- SE learns to emphasize relevant channels, suppress irrelevant

Analogy to attention:
- Self-attention in Transformers: Which tokens important?
- Channel attention in CNNs: Which channels important?

Empirical results:
- SE-ResNet-50: 78.7% ImageNet (+2.7% vs ResNet-50)
- SE-ResNeXt-50: 80.3% ImageNet (+2.2% vs ResNeXt-50)
- Cost: <1% parameter increase, <1% computation overhead
```

---

## 14.3.1 OBJECT DETECTION (ADVANCED)

### From Classification to Detection

**Task Hierarchy**:
```
Image Classification:
- Input: Image
- Output: Single label ("cat")
- Question: "What's in this image?"

Object Detection:
- Input: Image
- Output: Bounding boxes + labels
- Question: "What objects and where?"

Instance Segmentation:
- Input: Image
- Output: Pixel masks + labels
- Question: "What objects, where, and exact shape?"
```

---

### R-CNN Family (Region-based CNNs)

**R-CNN** (Girshick et al., 2014):

**Architecture**:
```
1. Region Proposal: Selective Search
   - Input: Image
   - Output: ~2000 region proposals (bounding boxes)
   - Algorithm: Hierarchical grouping of similar regions
   - Time: 1-2 seconds per image (CPU)

2. Feature Extraction: AlexNet/VGG
   - For each region: Warp to 224×224
   - Extract features: 4096-dim vector
   - Time: 2000 regions × 50ms = 100 seconds per image!

3. Classification: SVM
   - Train one SVM per class (20 classes → 20 SVMs)
   - Input: 4096-dim features
   - Output: Class scores

4. Bounding Box Regression:
   - Refine proposals (correct location)
   - Linear regression per class
   - Output: Adjusted box coordinates

Total time: ~47 seconds per image (too slow!)
```

**Training**:
```
Stage 1: Pre-train CNN on ImageNet (classification)
Stage 2: Fine-tune CNN on detection data
  - Positive: IoU > 0.5 with ground truth
  - Negative: IoU < 0.3
Stage 3: Train SVM (one per class)
Stage 4: Train bounding box regressors

Multi-stage training (complex, slow)
```

**Limitations**:
- Slow (47s per image)
- Expensive (2000× CNN forward passes)
- Multi-stage training
- Disk space (features for 2000 regions cached)

**Fast R-CNN** (Girshick, 2015):

**Key Innovation**: Single CNN pass (not 2000×)
```
1. Region Proposal: Selective Search (~2000 regions)

2. Single CNN Pass:
   - Input: Whole image (not cropped regions!)
   - Output: Feature map (e.g., 512×7×7)

3. RoI Pooling:
   - For each region proposal:
     - Project onto feature map
     - Pool to fixed size (7×7)
   - Output: 7×7×512 per region

4. Fully Connected Layers:
   - Two outputs:
     a. Classification (softmax over classes)
     b. Bounding box regression (4 values: x, y, w, h)

5. Multi-task Loss:
   L = L_cls + λ×L_box
   
   L_cls: Classification loss (cross-entropy)
   L_box: Smooth L1 loss for bounding box
```

**RoI Pooling** (Critical Innovation):
```
Problem: Region proposals have different sizes (100×100, 200×150, etc.)
Need: Fixed-size feature (for FC layers)

Solution: RoI Pooling
1. Divide region into fixed grid (e.g., 2×2)
2. Max pool within each grid cell
3. Output: 2×2 feature map (regardless of input size)

Example:
Region: 8×8 feature map → 2×2 RoI pool
Grid cell size: 4×4
Output: 2×2 (max pool each 4×4 cell)

Speed: 146× faster than R-CNN (0.32s vs 47s per image)
```

**Faster R-CNN** (Ren et al., 2016):

**Key Innovation**: Replace Selective Search with RPN (Region Proposal Network)
```
Architecture:
1. Backbone CNN: VGG or ResNet
   - Output: Feature map (512×H×W)

2. RPN (Region Proposal Network):
   - 3×3 conv on feature map
   - Two outputs per location:
     a. Objectness score (object vs background)
     b. Bounding box regression (4 values)
   - Anchors: 9 boxes per location (3 scales × 3 aspect ratios)
   - Output: ~300 proposals (after NMS)

3. RoI Pooling + Detection Head:
   - Same as Fast R-CNN
   - Classification + Box regression

4. End-to-End Training:
   - RPN and detection trained jointly
   - Alternating optimization or 4-step training
```

**Region Proposal Network (RPN)**:
```
At each spatial location (i,j) on feature map:
- Place 9 anchor boxes (predefined shapes):
  - Scales: 128², 256², 512² pixels
  - Aspect ratios: 1:1, 1:2, 2:1
  
For each anchor:
- Objectness: P(object) via 2-class softmax
- Box regression: (Δx, Δy, Δw, Δh) to refine anchor

Non-Maximum Suppression (NMS):
- Proposals overlap → Keep highest score, remove others
- IoU threshold: 0.7 typical
- Output: Top 300 proposals

Training:
- Positive anchor: IoU > 0.7 with any GT box
- Negative anchor: IoU < 0.3 with all GT boxes
- Ignore: 0.3 < IoU < 0.7
```

**Performance**:
```
Speed: 
- R-CNN: 47s per image
- Fast R-CNN: 2.3s per image (20× faster)
- Faster R-CNN: 0.2s per image (10× faster than Fast R-CNN, 200× faster than R-CNN!)

Accuracy (PASCAL VOC 2007):
- R-CNN: 66% mAP
- Fast R-CNN: 66.9% mAP
- Faster R-CNN: 69.9% mAP (+3% from RPN)

Real-time? No (0.2s = 5 FPS, not quite real-time 30 FPS)
```

---

### YOLO (You Only Look Once)

**Key Innovation**: Single-shot detection (no region proposals!)

**YOLOv1** (Redmon et al., 2016):

**Architecture**:
```
1. Divide image into S×S grid (S=7 typical)
   - Each grid cell predicts:
     - B bounding boxes (B=2, each with 5 values: x,y,w,h,confidence)
     - C class probabilities (C=20 for PASCAL VOC)

2. Backbone CNN: 24 conv layers + 2 FC layers
   - Input: 448×448×3
   - Output: 7×7×30 tensor
     - 7×7: Grid
     - 30: 2 boxes×5 + 20 classes = 10 + 20 = 30

3. Prediction (per grid cell):
   - Box 1: (x, y, w, h, confidence)
   - Box 2: (x, y, w, h, confidence)
   - Class probabilities: P(class|object) × 20

4. Final predictions:
   - Class-specific confidence: box_confidence × class_prob
   - NMS to remove duplicates
   - Output: Boxes with class labels
```

**Loss Function** (Complex!):
```
L = λ_coord Σ(x,y loss) + λ_coord Σ(w,h loss) + 
    Σ(confidence loss for objects) + 
    λ_noobj Σ(confidence loss for no objects) + 
    Σ(class loss)

Where:
- λ_coord = 5 (emphasize localization)
- λ_noobj = 0.5 (de-emphasize background)

Localization loss (x,y,w,h):
- (x,y): MSE of centers
- (w,h): MSE of square roots (to emphasize small boxes)

Confidence loss: MSE
Class loss: MSE (or cross-entropy in later versions)
```

**Pros**:
```
+ Very fast: 45 FPS (real-time!)
+ Single pass (end-to-end)
+ Sees whole image (better context than region-based)
+ Fewer background false positives
```

**Cons**:
```
- Lower accuracy than Faster R-CNN (63% vs 70% mAP)
- Struggles with small objects (grid cell limitation)
- Struggles with groups (max 2 objects per cell)
- Coarse localization (7×7 grid too coarse)
```

**YOLOv2/YOLO9000** (Redmon & Farhadi, 2017):

**Improvements**:
```
1. Batch Normalization (everywhere)
   - +2% mAP improvement
   
2. Higher Resolution: 448×448 (vs 224 in v1)
   - Better for small objects
   
3. Anchor Boxes (like Faster R-CNN):
   - 5 anchors per cell (learned from data via K-Means)
   - Removes grid cell limitation
   
4. Fine-grained features:
   - Pass-through layer (concatenate high-res features)
   - Better for small objects
   
5. Multi-scale Training:
   - Random input sizes (320, 352, ..., 608)
   - Model adapts to different resolutions

Performance:
- Speed: 40-90 FPS (depends on resolution)
- Accuracy: 76.8% mAP (vs 63% for v1, matches Faster R-CNN!)
```

**YOLOv3** (Redmon & Farhadi, 2018):

**Improvements**:
```
1. Multi-scale Predictions:
   - 3 different scales (like FPN)
   - Small objects: Fine-grained feature map
   - Large objects: Coarse feature map
   
2. Better Backbone: Darknet-53 (ResNet-like)
   - 53 conv layers
   - Residual connections
   
3. Binary Cross-Entropy Loss (vs MSE):
   - Better for multi-label (one object, multiple classes)
   
4. Logistic Regression (objectness):
   - Replaces softmax (handles overlapping objects)

Performance:
- Speed: 30 FPS (320×320) to 20 FPS (608×608)
- Accuracy: 57.9% mAP@IoU=0.5 (on COCO, harder than PASCAL)
- Trade-off: Slower than v2 but much more accurate
```

**YOLOv4, v5, v6, v7** (Continuous improvements):
```
YOLOv4 (Bochkovskiy et al., 2020):
+ CSPDarknet backbone (faster)
+ PANet (path aggregation)
+ Mosaic augmentation (mix 4 images)
+ 43.5% mAP@IoU=0.5 (COCO)

YOLOv5 (Ultralytics, 2020):
+ PyTorch implementation (v4 was Darknet)
+ Easier to use, better engineering
+ Auto-learning anchors
+ Multiple sizes (YOLOv5s/m/l/x)
+ Production-friendly (ONNX export, TensorRT)

YOLOv7 (2022):
+ 56.8% mAP (COCO)
+ 160 FPS (on V100 GPU)
+ State-of-the-art speed-accuracy trade-off

YOLOv8 (2023):
+ Anchor-free (simpler)
+ 53.9% mAP (COCO)
+ Improved training pipeline
```

**YOLO Production Use**:
```
When to use YOLO:
✓ Real-time detection (30+ FPS required)
✓ Surveillance, robotics, autonomous vehicles
✓ Edge deployment (YOLOv5s on Jetson Nano)
✓ Balance speed and accuracy

When NOT to use:
✗ Offline processing (use Faster R-CNN for best accuracy)
✗ Very small objects (use FPN-based detectors)
✗ High precision critical (use two-stage detectors)
```

---

### SSD (Single Shot Detector)

**Architecture** (Liu et al., 2016):
```
1. Base Network: VGG-16 (truncated at conv5_3)
   - Removes FC layers
   - Adds extra conv layers (conv6-11)

2. Multi-scale Feature Maps:
   - 6 different resolutions: 38×38, 19×19, 10×10, 5×5, 3×3, 1×1
   - Large feature maps: Small objects
   - Small feature maps: Large objects

3. Default Boxes (Anchors):
   - 4-6 boxes per feature map location
   - Different scales and aspect ratios
   - Total: ~8732 default boxes

4. Predictions (per default box):
   - Class scores: C+1 dimensions (C classes + background)
   - Box offsets: 4 dimensions (x,y,w,h adjustments)

5. Training:
   - Match GT boxes to default boxes (IoU > 0.5)
   - Hard negative mining (3:1 negative:positive ratio)
   - Multi-task loss: L = L_cls + α×L_loc (α=1 typical)
```

**Hard Negative Mining**:
```
Problem: 8732 boxes, only ~10 have objects (99.9% negative!)
- Training on all negatives → Model predicts "background" always

Solution: Hard negative mining
1. Compute loss for all negative boxes
2. Sort by loss (highest to lowest)
3. Keep top 3N negatives (where N = number of positives)
4. Ratio: 3:1 negatives:positives

Why hard negatives:
- High-loss negatives = difficult to classify (e.g., looks like object)
- Forces model to learn better discrimination
```

**Performance**:
```
Speed: 
- SSD300 (300×300 input): 59 FPS
- SSD512 (512×512 input): 22 FPS

Accuracy (PASCAL VOC 2007):
- SSD300: 77.2% mAP
- SSD512: 79.8% mAP
- Faster R-CNN: 73.2% mAP

SSD: Faster than Faster R-CNN AND more accurate!
```

**Comparison: YOLO vs SSD vs Faster R-CNN**:
```
| Model         | mAP   | FPS  | Strategy       |
|---------------|-------|------|----------------|
| Faster R-CNN  | 73.2% | 7    | Two-stage      |
| YOLOv3        | 57.9% | 30   | Single-shot    |
| SSD512        | 79.8% | 22   | Single-shot    |
| RetinaNet     | 59.1% | 14   | Single-shot+FL |

Production choice:
- Best accuracy: Faster R-CNN (if can afford 7 FPS)
- Best speed: YOLO (30+ FPS)
- Best balance: SSD (22 FPS, 79.8% mAP)
```

---

### RetinaNet (Focal Loss for Dense Object Detection)

**Motivation** (Lin et al., 2017):
```
Problem: Class imbalance in one-stage detectors
- 100K anchors per image
- ~10 contain objects (99.99% background!)
- Easy negatives dominate loss
- Model doesn't learn to detect objects well

Question: Why don't two-stage detectors have this problem?
Answer: RPN filters proposals (reduces negatives to ~1000, then 1:3 ratio)
```

**Focal Loss** (See Section 14.5 for math):
```
FL = -α(1-pₜ)^γ log(pₜ)

Easy negative (pₜ=0.99): (1-0.99)² = 0.0001 → Loss ≈ 0 (ignored)
Hard negative (pₜ=0.7): (1-0.7)² = 0.09 → Loss significant (learned from)

Result: Model focuses on hard examples (objects, difficult negatives)
```

**Architecture**:
```
1. Backbone: ResNet + FPN (Feature Pyramid Network)
   - Multi-scale features: P3, P4, P5, P6, P7
   - P3 (high-res): Small objects
   - P7 (low-res): Large objects

2. Classification Subnet:
   - 4 conv layers (3×3, 256 channels)
   - K×A outputs per location (K classes, A anchors)
   - Sigmoid activation (multi-label)

3. Box Regression Subnet:
   - 4 conv layers (3×3, 256 channels)
   - 4×A outputs per location (box coordinates)

4. Anchors:
   - 9 anchors per location (3 scales × 3 ratios)
   - Scales: 2⁰, 2^(1/3), 2^(2/3)
   - Ratios: 1:2, 1:1, 2:1
```

**Performance**:
```
COCO test-dev (2017):
- RetinaNet (ResNet-101): 39.1% mAP
- Faster R-CNN (ResNet-101): 36.2% mAP
- YOLOv3: 33.0% mAP
- SSD512: 31.2% mAP

RetinaNet: Best single-stage detector (as of 2017)

Speed: 14 FPS (slower than YOLO but faster than Faster R-CNN)
```

---

## 14.3.2 SEMANTIC SEGMENTATION

### FCN (Fully Convolutional Networks)

**Motivation** (Long et al., 2015):
```
Problem: Classification CNNs have FC layers
- FC layers require fixed input size
- Can't do dense prediction (pixel-wise)

Solution: Replace FC with 1×1 conv
- Accepts any input size
- Outputs spatial maps (not single vector)
```

**Architecture**:
```
Encoder (Downsampling):
1. VGG/ResNet backbone
2. Conv + pooling (5× downsampling)
3. Output: Coarse feature map (1/32 resolution)

Decoder (Upsampling):
1. Transposed Convolution (learnable upsampling)
   - 2× upsample per layer
   - 5 layers: 1/32 → 1/16 → 1/8 → 1/4 → 1/2 → 1/1
2. Skip connections (from encoder)
   - Combine coarse and fine features
   - Sharper boundaries
3. Final: 1×1 conv → K classes per pixel
```

**Transposed Convolution** (Deconvolution):
```
Standard Conv (downsampling):
Input: 4×4 → 3×3 filter, stride 2 → Output: 2×2

Transposed Conv (upsampling):
Input: 2×2 → 3×3 filter, stride 2 → Output: 4×4

How it works:
1. Insert zeros between input pixels (stride-1 gaps)
2. Pad input
3. Apply standard convolution
4. Result: Upsampled output

Learnable: Weights learned during training (better than bilinear)
Checkerboard artifacts: Can create patterns (use kernel_size divisible by stride)
```

**Skip Connections**:
```
Problem: Upsampling from 1/32 loses details
Solution: Add features from encoder layers

FCN-32s: Only final layer (coarse)
FCN-16s: Add skip from 1/16 (medium)
FCN-8s: Add skips from 1/8 and 1/16 (fine) ← Best

Example:
Decoder layer (1/16 res) + Encoder layer (1/16 res) = Combined features
→ Sharper boundaries, better small objects
```

**Performance**:
```
PASCAL VOC 2012:
- FCN-8s: 62.2% mIoU
- FCN-16s: 59.4% mIoU
- FCN-32s: 56.3% mIoU

Skip connections: +6% mIoU improvement!

Limitations:
- Still not very sharp boundaries
- Struggles with small objects
- Improved by: U-Net, DeepLab
```

---

### U-Net (Medical Image Segmentation)

**Motivation** (Ronneberger et al., 2015):
```
Medical imaging challenges:
- Limited data (100-1000 images typical)
- Need precise boundaries (tumor, organ)
- Class imbalance (1% tumor, 99% background)

U-Net solution: Strong skip connections + data augmentation
```

**Architecture (U-shape)**:
```
Encoder (Contracting Path):
[Conv-ReLU-Conv-ReLU-MaxPool] × 4
- 64 → 128 → 256 → 512 → 1024 channels
- Resolution: 1 → 1/2 → 1/4 → 1/8 → 1/16

Bottleneck:
- Lowest resolution (1/16)
- Highest channels (1024)

Decoder (Expansive Path):
[UpConv-Concat-Conv-ReLU-Conv-ReLU] × 4
- 1024 → 512 → 256 → 128 → 64 channels
- Resolution: 1/16 → 1/8 → 1/4 → 1/2 → 1

Skip Connections:
- Encoder features concatenated to decoder (not added)
- All spatial information preserved
- Critical for precise boundaries

Output:
- 1×1 conv → K classes per pixel
- Softmax (multi-class) or Sigmoid (binary)
```

**Key Differences from FCN**:
```
FCN: Add skip connections (element-wise)
U-Net: Concatenate skip connections (channel-wise)

U-Net: More symmetric (encoder = decoder size)
FCN: Asymmetric (large encoder, small decoder)

U-Net: Designed for small datasets (<1000 images)
FCN: Designed for large datasets (ImageNet-scale)
```

**Data Augmentation (Critical for U-Net)**:
```
Medical images: Limited data (100-1000 images)

Augmentation strategies:
1. Geometric: Rotation, flip, elastic deformation
2. Intensity: Brightness, contrast, gamma
3. Elastic deformation (specific to medical):
   - Random warping (mimics tissue deformation)
   - Critical for realistic augmentation

With augmentation: 35 training images → 35×100 = 3500 augmented images
Result: Comparable performance to models trained on 10K+ images
```

**Performance**:
```
Medical segmentation benchmarks:
- Cell segmentation (ISBI 2012): 92% IoU (vs 83% previous best)
- Microscopy images: 30 images → State-of-the-art!

Why U-Net succeeded:
+ Works with tiny datasets (30-100 images)
+ Precise boundaries (skip connections)
+ Fast inference (100ms per image)
+ Easy to train (stable, no complex tricks)

Production use:
- Medical imaging (tumors, organs, cells)
- Satellite imagery (buildings, roads)
- Document segmentation (text regions)
```

---

### DeepLab (Atrous/Dilated Convolutions)

**Motivation** (Chen et al., 2017):
```
Problem with pooling/striding:
- Reduces resolution (loses spatial info)
- Coarse predictions (need upsampling)

Solution: Dilated convolutions
- Large receptive field WITHOUT downsampling
- Preserve resolution throughout network
```

**Atrous Spatial Pyramid Pooling (ASPP)**:
```
Parallel dilated convolutions with different rates:

Input feature map → 
  → 1×1 conv (rate 1)
  → 3×3 conv (rate 6)
  → 3×3 conv (rate 12)
  → 3×3 conv (rate 18)
  → Global Average Pooling
  → Concatenate all → 1×1 conv → Output

Benefits:
+ Multi-scale context (different receptive fields)
+ No resolution loss
+ Captures both fine and coarse patterns

Example (3×3 dilated conv, rate=2):
Receptive field: 3 + (3-1)×(2-1) = 5
Parameters: Same as standard 3×3 (no increase!)
```

**DeepLabv3+ Architecture**:
```
Encoder:
- ResNet or Xception backbone
- ASPP module (multi-scale features)
- Output: 1/16 or 1/8 resolution

Decoder:
- Simple upsampling (4× bilinear interpolation)
- Concatenate with low-level encoder features
- 3×3 conv for refinement
- Final upsampling to full resolution

Output: Per-pixel class predictions
```

**Performance**:
```
PASCAL VOC 2012:
- DeepLabv3+: 89% mIoU (state-of-the-art in 2018)
- U-Net: 72% mIoU
- FCN-8s: 62% mIoU

COCO Stuff (164K images):
- DeepLabv3+: 45.7% mIoU

Why better:
+ Dilated convolutions (preserves resolution)
+ ASPP (multi-scale context)
+ Strong backbone (ResNet-101, Xception)
```

---

## 14.3.3 INSTANCE SEGMENTATION

### Mask R-CNN

**Motivation** (He et al., 2017):
```
Semantic segmentation: Classifies pixels (all "person" pixels = one class)
Instance segmentation: Separates individuals (person1, person2, person3)

Example:
Semantic: All people pixels = "person" class
Instance: Person at (x1,y1) = Instance 1, Person at (x2,y2) = Instance 2
```

**Architecture**:
```
Based on Faster R-CNN + Mask branch

1. Backbone: ResNet + FPN
   - Multi-scale features (P2-P5)

2. RPN: Region proposals (~1000 proposals)

3. RoI Align (not RoI Pooling):
   - RoI Pooling: Quantization (loses sub-pixel precision)
   - RoI Align: Bilinear interpolation (preserves alignment)
   - Critical for mask accuracy

4. Detection Head:
   - Classification: K+1 classes
   - Box regression: 4 values (x,y,w,h)

5. Mask Head (NEW):
   - FCN for each RoI
   - Output: 28×28 binary mask per class
   - Only compute mask for predicted class (efficient)

Multi-task loss:
L = L_cls + L_box + L_mask
```

**RoI Align** (Critical Innovation):
```
RoI Pooling problem:
- Input RoI: 7.3×7.3 → Quantize to 7×7
- Misalignment propagates (1-2 pixel error)
- Blurry masks (boundaries shift)

RoI Align solution:
- Keep floating-point coordinates
- Bilinear interpolation at sampling points
- No quantization (exact alignment)

Example:
RoI: (x=7.3, y=12.7, w=14.5, h=14.5)
RoI Pooling: Quantize to (7, 13, 14, 14) → Misaligned
RoI Align: Use (7.3, 12.7, 14.5, 14.5) → Exact alignment

Impact: +10-50% mask accuracy improvement!
```

**Performance**:
```
COCO 2017 Instance Segmentation:
- Mask R-CNN: 37.1% mAP (masks)
- Mask R-CNN: 39.8% mAP (boxes)

State-of-the-art for instance segmentation (2017-2020)

Speed: 5 FPS (200ms per image)
- Slower than YOLO but much more accurate
- Real-time variants exist (e.g., YOLACT, TensorMask)
```

**Production Use**:
```
Applications:
✓ Medical imaging (cell counting, organ segmentation)
✓ Autonomous vehicles (pedestrians, cars)
✓ Robotics (object manipulation, grasping)
✓ Photo editing (background removal, object selection)
✓ Retail (product counting, inventory)

Example (Medical):
- Cell segmentation: Distinguish overlapping cells
- Input: Microscopy image
- Output: Individual cell masks (even if overlapping)
- Critical: RoI Align for precise boundaries
```

---

## 14.3.4 GENERATIVE ADVERSARIAL NETWORKS (GANs)

### GAN Basics

**Motivation** (Goodfellow et al., 2014):
```
Problem: Generate realistic images from scratch
Traditional: Explicit density modeling (VAEs, PixelCNN)
GAN: Adversarial game (generator vs discriminator)
```

**Architecture**:
```
Generator G:
- Input: Random noise z ~ N(0,1) (e.g., 100-dim)
- Output: Fake image G(z) (e.g., 64×64×3)
- Goal: Fool discriminator

Discriminator D:
- Input: Real image or fake image
- Output: Probability [0,1] (real vs fake)
- Goal: Distinguish real from fake

Training (Minimax game):
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

Where:
- E[log D(x)]: Discriminator correctly identifies real images
- E[log(1-D(G(z)))]: Discriminator correctly identifies fake images
- Generator minimizes this (wants D to fail)
- Discriminator maximizes this (wants to succeed)
```

**Training Procedure**:
```
Alternate between:

Step 1: Train Discriminator (k steps, typically k=1)
1. Sample mini-batch of m real images {x⁽¹⁾, ..., x⁽ᵐ⁾}
2. Sample m noise samples {z⁽¹⁾, ..., z⁽ᵐ⁾}
3. Generate fake images: {G(z⁽¹⁾), ..., G(z⁽ᵐ⁾)}
4. Update D to maximize:
   (1/m) Σ[log D(x⁽ⁱ⁾) + log(1 - D(G(z⁽ⁱ⁾)))]

Step 2: Train Generator (1 step)
1. Sample m noise samples {z⁽¹⁾, ..., z⁽ᵐ⁾}
2. Update G to minimize:
   (1/m) Σ log(1 - D(G(z⁽ⁱ⁾)))
   
   Or equivalently, maximize:
   (1/m) Σ log D(G(z⁽ⁱ⁾))  ← Better gradients (non-saturating)
```

**Training Challenges**:

**Mode Collapse**:
```
Problem: Generator produces limited variety
- Learns to generate only few images that fool D
- Example: Generates same face repeatedly

Detection:
- Low diversity in generated samples
- Inception Score low (measures diversity)

Solutions:
1. Unrolled GAN: Look ahead k D steps when training G
2. Mini-batch discrimination: D sees multiple samples (encourages diversity)
3. Feature matching: G matches statistics of real data (not just fool D)
4. Label smoothing: Use 0.9 instead of 1.0 for real labels
```

**Unstable Training**:
```
Problems:
- D too strong: G gradient vanishes (can't learn)
- G too strong: D always fooled (can't provide signal)
- Oscillation: Neither converges

Solutions:
1. Balance training:
   - Train D more initially (k=5) until D strong
   - Then equal steps (k=1)
2. Learning rates:
   - D: 0.0001-0.0002
   - G: 0.0001 (same or slightly lower)
3. Architecture:
   - Use batch normalization (both G and D)
   - Use LeakyReLU (D), ReLU or tanh (G)
4. Label smoothing: Real=0.9, Fake=0.1 (smoothed)
```

**Evaluation Metrics**:
```
Inception Score (IS):
IS = exp(E[KL(p(y|x) || p(y))])

Where:
- p(y|x): Predicted class probabilities for generated image
- p(y): Marginal distribution over all generated images

High IS: Generated images are:
- Confident (low entropy p(y|x))
- Diverse (high entropy p(y))

Typical values:
- Random noise: IS ≈ 1
- MNIST GAN: IS ≈ 9
- ImageNet GAN: IS ≈ 50-100

Fréchet Inception Distance (FID):
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g))

Where:
- μ_r, Σ_r: Mean, covariance of real images (Inception features)
- μ_g, Σ_g: Mean, covariance of generated images

Lower FID: Generated images closer to real distribution

Typical values:
- FID < 10: Excellent (nearly indistinguishable)
- FID 10-50: Good
- FID > 50: Poor

Production standard: Report both IS and FID
```

**DCGAN** (Deep Convolutional GAN):

**Architecture Guidelines** (Radford et al., 2016):
```
Generator:
- Transposed conv layers (upsampling)
- Batch norm (all layers except output)
- ReLU activation (all layers except output)
- tanh output (normalize to [-1, 1])

Discriminator:
- Conv layers (downsampling)
- Batch norm (all layers except input)
- LeakyReLU activation (slope=0.2)
- Sigmoid output (probability)

No FC layers, no pooling (all convolutional)

Why these choices matter:
- Batch norm: Stabilizes training (critical for GANs)
- LeakyReLU (D): Allows gradients for negative activations
- No pooling: Lets network learn downsampling
```

**Production GANs**:
```
StyleGAN (NVIDIA, 2019):
- High-resolution (1024×1024)
- Style-based generator (control over generated features)
- FID: 4.4 on FFHQ (faces)
- Applications: Face generation, art, fashion

DALL-E (OpenAI, 2021):
- Text-to-image generation
- 12B parameters
- Applications: Creative design, content creation

Stable Diffusion (2022):
- Open-source text-to-image
- Latent diffusion (more efficient than GANs)
- Applications: Image generation, editing, inpainting
```

---

## 14.3.5 NEURAL STYLE TRANSFER

**Motivation** (Gatys et al., 2015):
```
Goal: Transfer artistic style from one image to another

Input:
- Content image: Photo of city
- Style image: Van Gogh's "Starry Night"
Output:
- Generated image: Photo in Van Gogh style
```

**Method**:
```
Use pre-trained VGG-19 (trained on ImageNet)

1. Content Representation:
   - Extract features from middle layer (conv4_2)
   - F^l[content]: Feature map at layer l
   
2. Style Representation (Gram Matrix):
   - Extract features from multiple layers (conv1_1, conv2_1, ..., conv5_1)
   - Gram matrix: G^l_ij = Σ F^l_ik × F^l_jk (correlation between features)
   - Captures texture, not spatial structure

3. Generated Image:
   - Start with noise or content image
   - Optimize image (not weights!) to:
     - Match content (F^l[generated] ≈ F^l[content])
     - Match style (G^l[generated] ≈ G^l[style])
```

**Loss Function**:
```
L_total = α×L_content + β×L_style

Content Loss:
L_content = (1/2) Σ (F^l_ij[generated] - F^l_ij[content])²

Style Loss:
L_style = Σ_l w_l × (1/(4N²M²)) Σ (G^l_ij[generated] - G^l_ij[style])²

Where:
- α, β: Weight hyperparameters (α/β controls style strength)
- w_l: Weight per layer (equal weighting typical)
- N: Number of feature maps
- M: Size of feature map

Optimization:
- Use L-BFGS or Adam
- 500-1000 iterations
- Time: 1-5 minutes on GPU
```

**Gram Matrix** (Style Representation):
```
Feature map: H×W×C (height × width × channels)
Gram matrix: C×C

G_ij = Σ_x Σ_y F_i(x,y) × F_j(x,y)

Intuition:
- Measures correlation between feature maps
- If filters i and j activate together → High G_ij
- Captures texture, ignores spatial layout

Example:
Two filters: "vertical edges" and "horizontal edges"
- Bricks (grid pattern): High correlation → High G_ij
- Tree (organic): Low correlation → Low G_ij

Style matching: Match these correlations (not exact features)
```

**Fast Style Transfer** (Johnson et al., 2016):
```
Problem: Original method is slow (1-5 min per image)
Solution: Train feedforward network to generate in one pass

Architecture:
Image Transformation Network:
- Input: Content image
- Encoder: 3 conv layers (downsampling)
- Residual blocks: 5-9 residual blocks
- Decoder: 3 transposed conv layers (upsampling)
- Output: Stylized image

Training:
- Train on 80K images (MS COCO)
- Loss: Same as original (content + style + variation)
- Train once per style (separate network for each style)

Speed: 1000× faster!
- Original: 1-5 minutes per image
- Fast: 20-50ms per image (real-time!)

Trade-off:
- Must train separate network per style
- Slightly lower quality (but acceptable)

Applications:
- Real-time video stylization
- Mobile apps (Prisma, Deep Art)
```

**Production Use**:
```
Instagram filters: Fast neural style transfer
Snapchat lenses: Real-time style transfer
Adobe apps: Style transfer tools

Example (Fashion):
- Content: Product photo
- Style: Artistic painting
- Output: Artistic product rendering
- Use: Marketing, creative design
```

---

## Interview Questions (Advanced CV)

### Question 31: YOLO vs Faster R-CNN
**Q**: "Your autonomous vehicle needs object detection. Latency requirement: <50ms (20 FPS). Accuracy requirement: >70% mAP. Should you use YOLO or Faster R-CNN?"

**Expected Answer**:
- **Choose YOLO** (meets both requirements)

**Analysis**:
```
Faster R-CNN:
- Accuracy: 73% mAP ✓
- Speed: 200ms (5 FPS) ✗ Too slow!

YOLOv5:
- Accuracy: 50-74% mAP (model size dependent) ✓
- Speed: 10-30ms (33-100 FPS) ✓ Meets requirement!

Recommendation: YOLOv5-medium
- Accuracy: 71% mAP (acceptable)
- Speed: 25ms (40 FPS) (comfortable margin)
- Model size: 80MB (fits on vehicle hardware)
```

**Optimizations**:
```
Further speedup:
1. TensorRT (NVIDIA optimization): 15ms (66 FPS)
2. Smaller model (YOLOv5s): 10ms but 68% mAP (-3%)
3. Lower resolution: 416×416 → 320×320 (1.5× faster)

Trade-off: Speed vs accuracy
- Autonomous vehicle: Accuracy critical (don't compromise below 70%)
- Surveillance: Speed critical (can accept 65% mAP)
```

**Follow-up**: "At night, small object detection drops to 40% mAP. How do you improve?"
```
Solutions:
1. Multi-scale training (mix day/night, various sizes)
2. Data augmentation (brightness, contrast adjustments)
3. Larger input resolution (640×640 vs 416×416)
4. Better backbone (CSPDarknet → EfficientNet)
5. Add infrared camera (thermal imaging)
6. FPN-style multi-scale predictions
```

---

### Question 32: Segmentation Loss Function
**Q**: "Your medical image segmentation: 99% background, 1% tumor. Cross-entropy loss gives poor results (model predicts all background). What loss do you use?"

**Expected Answer**:
- **Use Dice Loss** (handles extreme imbalance)

**Reasoning**:
```
Cross-Entropy problem:
- 99% correct by predicting all background!
- Loss: -(0.99×log(0.99) + 0.01×log(0.01)) = 0.056 (low, looks good)
- But: 0% tumor detected (useless for diagnosis)

Dice Loss:
- Dice = 2×|Prediction ∩ GT| / (|Prediction| + |GT|)
- If predict all background: Dice = 0 (high loss!)
- Forces model to predict tumor (even if 1%)

Combined Loss (Best):
L = 0.5×CE + 0.5×Dice
- CE: Pixel-wise accuracy
- Dice: Global overlap
- Result: Best of both worlds
```

**Expected Results**:
```
Cross-Entropy only:
- Accuracy: 99% (all background)
- Dice Score: 0% (no tumor detected)

Dice Loss only:
- Accuracy: 95% (some background errors)
- Dice Score: 85% (good tumor detection)

Combined (CE + Dice):
- Accuracy: 98% (few background errors)
- Dice Score: 90% (excellent tumor detection) ✓ Best!
```

**Follow-up**: "After using Dice loss, you get many false positives (healthy tissue marked as tumor). How do you reduce?"
```
Solutions:
1. Focal Loss (down-weight easy negatives)
2. Weighted Dice (higher weight on tumor class)
3. Post-processing: Remove small isolated predictions (<5 pixels)
4. Ensemble: Average 3-5 models (reduces false positives)
5. More training data (if possible)
```

---

### Question 33: Instance vs Semantic Segmentation
**Q**: "Your application needs to count individual people in crowded images. Should you use semantic segmentation or instance segmentation?"

**Expected Answer**:
- **Instance segmentation** (Mask R-CNN)

**Reasoning**:
```
Semantic Segmentation (e.g., DeepLab):
- Output: All "person" pixels = single class
- Can't distinguish individual people
- Counting: Impossible (all merged into one blob)

Instance Segmentation (e.g., Mask R-CNN):
- Output: Separate mask per person
- Person 1: Mask 1
- Person 2: Mask 2
- Counting: Count number of masks ✓

Example:
Image: 5 people standing close together
- Semantic: One big "person" blob (can't count)
- Instance: 5 separate masks (count = 5) ✓
```

**Implementation**:
```
Mask R-CNN:
1. Detect each person (bounding box)
2. Segment each person (pixel mask)
3. Count: Number of detected instances

Post-processing:
- NMS to remove duplicate detections
- Filter small masks (noise, partial people)
- Confidence threshold (only count if >0.7 confidence)

Expected accuracy:
- Crowd counting error: ±5% (on standard benchmarks)
```

**Follow-up**: "Mask R-CNN is too slow (200ms per image). You need <50ms. What do you do?"
```
Solutions:
1. YOLACT (real-time instance segmentation):
   - Speed: 30 FPS (33ms per image) ✓
   - Accuracy: 29% mAP (vs 37% Mask R-CNN)
   - Trade-off: Acceptable for counting

2. Lighter backbone:
   - ResNet-101 → ResNet-50 (2× faster)
   - 39% → 37% mAP (-2%)

3. Lower resolution:
   - 800×600 → 512×512 (1.5× faster)

4. TensorRT optimization:
   - NVIDIA TensorRT: 100ms → 40ms (2.5× speedup) ✓

5. Simpler approach (if counting only):
   - Detection only (no masks): Faster R-CNN
   - 70ms per image (faster than Mask R-CNN)
```

---

### Question 34: GAN Training Failure
**Q**: "Your GAN discriminator accuracy is 100% (always correctly identifies real vs fake) but generator produces noise. What's wrong?"

**Expected Answer**:
- **Discriminator too strong** (vanishing gradient for generator)

**Diagnosis**:
```
D accuracy 100%:
- D perfectly separates real and fake
- D(G(z)) ≈ 0 for all generated images
- Generator gradient: ∂log(1-D(G(z)))/∂G ≈ 0 (vanished!)
- G can't learn (no gradient signal)

Why this happened:
- D trained too many steps vs G
- D has more capacity than G
- D learning rate too high
```

**Solutions**:
```
1. Balance training ratio:
   - Current: Train D 5 steps, G 1 step
   - Fix: Train D 1 step, G 1 step (equal)
   - Or: Train D 1 step, G 2 steps (favor G)

2. Reduce D capacity:
   - Fewer layers in D
   - Smaller filters
   - Goal: D slightly weaker (gives G a chance)

3. Reduce D learning rate:
   - D: 0.0002 → 0.0001
   - G: Keep 0.0002
   - D learns slower (G catches up)

4. Noisy labels:
   - Real: Use 0.9 instead of 1.0
   - Fake: Use 0.1 instead of 0.0
   - "Label smoothing" - helps G get gradient

5. Feature matching:
   - G matches statistics of intermediate D features
   - Not just final output
   - More stable gradient

6. Switch loss (non-saturating):
   - Instead of: min_G log(1 - D(G(z)))  [saturates when D(G(z)) → 0]
   - Use: max_G log(D(G(z)))  [doesn't saturate]
```

**Monitoring GAN Health**:
```
Check every 100 iterations:
✓ D accuracy: Should be 60-80% (not 100%!)
✓ G loss and D loss: Should fluctuate (oscillate)
✓ Generated samples: Visual quality improving
✓ Mode collapse: Are samples diverse?

Warning signs:
✗ D accuracy = 100%: D too strong
✗ D accuracy = 50%: D too weak (random guessing)
✗ G loss = 0: Mode collapse
✗ Samples identical: Mode collapse
```

---

### Question 35: U-Net vs DeepLab
**Q**: "You're doing medical image segmentation. Dataset: 50 images. Should you use U-Net or DeepLabv3+?"

**Expected Answer**:
- **U-Net** (designed for small datasets)

**Reasoning**:
```
U-Net:
+ Designed for small data (30-100 images)
+ Strong skip connections (preserves details)
+ Data augmentation strategies (elastic deformation)
+ Fast to train (converges in 100-200 epochs)
+ Medical imaging standard

DeepLabv3+:
- Designed for large datasets (100K+ images)
- 50 images: Severe overfitting
- ResNet-101 backbone: 45M parameters (too many for 50 images)
- Needs ImageNet pre-training (might not help for medical)

Expected Results (50 medical images):
- U-Net: 85-90% Dice score ✓
- DeepLabv3+ (from scratch): 40-50% (overfits)
- DeepLabv3+ (ImageNet pre-trained): 70-75% (better but still worse than U-Net)
```

**Data Augmentation for Medical**:
```
U-Net paper strategy (30 images → state-of-the-art):
1. Elastic deformation (random warping):
   - Mimics tissue deformation
   - Critical for medical images
   
2. Rotation: ±10 degrees
3. Scaling: 0.9-1.1×
4. Flipping: Horizontal/vertical
5. Brightness/contrast: ±20%

Result: 30 images × 100 augmentations = 3000 effective images
```

**Follow-up**: "You have 100K medical images now. Should you switch to DeepLab?"
```
Maybe (try both):

Experiment:
1. U-Net: Train on 100K images
   - Expected: 92-95% Dice score
   
2. DeepLabv3+: Train on 100K images
   - Expected: 93-96% Dice score (+1-2% improvement)

Decision:
- If 1-2% improvement matters (cancer detection): Use DeepLab
- If not worth extra complexity: Stick with U-Net (simpler, well-understood)

Production:
- Most medical imaging still uses U-Net (proven, reliable)
- DeepLab gaining traction (when have large datasets)
```

---

## Key Takeaways (Computer Vision Advanced)

### Object Detection
- **Two-stage** (Faster R-CNN): Best accuracy, slower (7 FPS)
- **One-stage** (YOLO, SSD): Real-time, good accuracy
- **RetinaNet**: Focal loss solves class imbalance (best one-stage)
- **Production choice**: YOLOv5/v7 for speed, Faster R-CNN for accuracy

### Segmentation
- **Semantic** (FCN, DeepLab): Pixel classes (all "person" = one class)
- **Instance** (Mask R-CNN): Separate objects (person1, person2)
- **U-Net**: Small datasets (medical imaging standard)
- **DeepLab**: Large datasets (dilated conv + ASPP)

### GANs
- **Training**: Delicate balance (D vs G strength)
- **Metrics**: FID (lower better), IS (higher better)
- **Common failure**: Mode collapse, vanishing gradients
- **Production**: StyleGAN, Stable Diffusion (state-of-the-art)

### Style Transfer
- **Original**: Slow (1-5 min) but flexible (any style)
- **Fast**: Real-time (20ms) but one network per style
- **Production**: Fast style transfer (mobile apps, filters)

---

**Next: Document 3 will add Mathematical Theory and Advanced Learning topics...**

### Optimization Insights
- **Batch size**: Start with 32, scale with linear LR rule if increasing
- **Learning rate schedules**: Cosine annealing standard, always use warmup for Transformers
- **AdamW vs Adam**: Use AdamW (correct weight decay), not Adam
- **Mixed precision**: Always use (free 2× speedup), prefer BF16 over FP16

### CNN Innovations
- **Depthwise separable**: 7× speedup, 5% accuracy loss (MobileNet)
- **SE blocks**: 2% accuracy boost, <1% overhead (plug-and-play)
- **Grouped convolutions**: Fewer parameters, better accuracy (ResNeXt)
- **Dilated convolutions**: Larger receptive field, no downsampling (segmentation)

### Transfer Learning
- **Small data (<10K)**: Feature extraction (freeze backbone)
- **Medium data (10-100K)**: Fine-tuning (discriminative LR)
- **Different domain**: Progressive unfreezing (Fast.ai strategy)

### Debugging
- **Always**: Overfit single batch first (sanity check)
- **Monitor**: Train/val curves, gradient norms, weight updates
- **Common bugs**: Forgot model.eval(), wrong loss function, labels off-by-one

---

**Next Section**: Part 3 continues with RNNs, Loss Functions, and Evaluation Metrics...
```
θ := θ - η∇J(θ)

Where:
- η = learning rate
- ∇J(θ) = gradient on mini-batch

Pros:
+ Simple
+ Generalizes well (noise helps escape local minima)

Cons:
- Slow convergence (oscillates)
- Sensitive to learning rate
- Same learning rate for all parameters
```

**SGD with Momentum**:
```
v := βv - η∇J(θ)
θ := θ + v

β = 0.9 typically (momentum coefficient)

Intuition:
- Accumulates velocity in direction of persistent gradient
- Dampens oscillations
- Accelerates in consistent directions

Pros:
+ 2-10x faster than vanilla SGD
+ Less sensitive to learning rate

Cons:
- Hyperparameter β
```

**Adam (Adaptive Moment Estimation)**:
```
m := β₁m + (1-β₁)∇J(θ)     [first moment: mean]
v := β₂v + (1-β₂)(∇J(θ))²   [second moment: variance]
m̂ := m/(1-β₁ᵗ)              [bias correction]
v̂ := v/(1-β₂ᵗ)              [bias correction]
θ := θ - η·m̂/(√v̂ + ε)

Default hyperparameters:
- β₁ = 0.9 (exponential decay rate for first moment)
- β₂ = 0.999 (exponential decay rate for second moment)
- η = 0.001 (learning rate)
- ε = 10⁻⁸ (numerical stability)

Pros:
+ Adaptive learning rates (per parameter)
+ Fast convergence
+ Works well out-of-the-box (good defaults)
+ Most popular optimizer (as of 2024)

Cons:
- More memory (stores m and v)
- Can overfit on small datasets (use AdamW)

Use cases:
✓ Default optimizer for most tasks
✓ Transformers (BERT, GPT)
✓ When don't want to tune learning rate much
```

**AdamW (Adam with Weight Decay)**:
```
Same as Adam but:
θ := θ - η·(m̂/(√v̂ + ε) + λθ)

Where λ = weight decay coefficient (typically 0.01)

Why better than Adam:
- Original Adam: Weight decay inside adaptive learning rate (incorrect)
- AdamW: Weight decay outside (correct implementation)
- Better generalization (less overfitting)

Use cases:
✓ Training large language models (GPT-3, Llama)
✓ When have large models + overfitting concerns
✓ Default for Transformers (Hugging Face default)

Production note:
- Llama 2 trained with AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1)
- GPT-3 trained with AdamW
```

**Adafactor (Memory-Efficient Adam)**:
```
Reduces memory by:
- Not storing full second moment matrix
- Factorized approximation: v ≈ r·c (row × column)
- Memory: O(n + m) instead of O(n·m)

Use cases:
✓ Training very large models (T5, FLAN)
✓ When memory constrained
✗ When have enough memory (Adam/AdamW better)

Production note:
- T5 and Flan-T5 trained with Adafactor
- Saves 30-50% memory vs Adam
```

**Optimizer Selection Guide**:
```
Default: AdamW (works 90% of the time)
Computer Vision: SGD with momentum (often better generalization)
NLP/Transformers: AdamW (standard)
Memory constrained: Adafactor
Fine-tuning: Adam or AdamW with lower LR (1e-5 to 1e-4)
From scratch: AdamW with higher LR (1e-4 to 1e-3)

Learning rate:
- Too high: Loss diverges (NaN)
- Too low: Slow convergence
- Rule of thumb: Start with 1e-3, halve if unstable
```

---

### Regularization Techniques

**L1 Regularization (Lasso)**:
```
Loss = L(y, ŷ) + λΣ|θᵢ|

Effect:
- Sparse weights (many exactly zero)
- Feature selection

Use cases:
✓ When want interpretability (few important features)
✓ High-dimensional data with many irrelevant features
✗ Deep learning (L2 more common)
```

**L2 Regularization (Ridge / Weight Decay)**:
```
Loss = L(y, ŷ) + λΣθᵢ²

Effect:
- Small weights (closer to zero but not exactly zero)
- Prevents any single weight from dominating

Use cases:
✓ Default regularization for neural networks
✓ Prevents overfitting without losing features

Typical values:
- λ = 1e-4 to 1e-2 (tune via validation)
```

**Dropout (Srivastava et al., 2014)**:
```
Training:
- Randomly set neurons to 0 with probability p (typically 0.5)
- Scale remaining neurons by 1/(1-p)

Inference:
- Use all neurons (no dropout)
- Weights already scaled from training

Intuition:
- Forces network to not rely on any single neuron
- Like training an ensemble of 2ⁿ subnetworks
- Prevents co-adaptation

Pros:
+ Very effective (reduces overfitting by 20-50%)
+ Approximate ensemble without training multiple models

Cons:
- Increases training time (~2x)
- Hyperparameter p (typically 0.2-0.5)

Use cases:
✓ Fully connected layers (MLP, LSTM)
✓ When overfitting on training data
✗ Convolutional layers (less common, use sparingly)

Production note:
- AlexNet used dropout=0.5 (key to winning ImageNet)
- Transformers use dropout=0.1 (lighter than vision models)
- GPT-3 uses dropout=0.0 (no dropout, relies on scale)
```

**Batch Normalization (Ioffe & Szegedy, 2015)**:
```
For each mini-batch:
1. Normalize: x̂ = (x - μ_batch) / √(σ²_batch + ε)
2. Scale and shift: y = γx̂ + β (learnable parameters)

Where:
- μ_batch, σ²_batch = mean, variance of mini-batch
- γ, β = learned scale and shift
- ε = numerical stability (1e-5)

During inference:
- Use running mean/variance (computed during training)

Benefits:
+ Faster training (2-10x speedup)
+ Allows higher learning rates
+ Reduces sensitivity to initialization
+ Acts as regularization (slight noise from mini-batch statistics)
+ Enables very deep networks (100+ layers)

Cons:
- Adds parameters (2 per feature: γ, β)
- Behavior differs between train/inference (BatchNorm uses different statistics)
- Small batch sizes: Unstable statistics (batch size < 8)

Use cases:
✓ CNNs (almost always)
✓ Deep networks (>10 layers)
✗ RNNs (use Layer Normalization instead)
✗ Small batches (batch size < 8)

Production note:
- ResNet, Inception, most CNNs use BatchNorm
- Place after linear layer, before activation
```

**Layer Normalization (Ba et al., 2016)**:
```
Normalize across features (not batch):
x̂ = (x - μ_layer) / √(σ²_layer + ε)

Where:
- μ_layer, σ²_layer = mean, variance across features for single sample

Benefits:
+ Works with batch size = 1 (no batch dependency)
+ Same behavior train/inference
+ Works well with sequences (RNNs, Transformers)

Use cases:
✓ Transformers (BERT, GPT, Llama)
✓ RNNs, LSTMs
✓ When batch size is small or varies
✗ CNNs (BatchNorm usually better)

Production note:
- All Transformer models use LayerNorm
- GPT, BERT, Llama all use LayerNorm
- Pre-LN (before attention) > Post-LN (after attention) for stability
```

**Data Augmentation**:
```
Image:
- Flip, rotate, crop, zoom
- Color jitter, brightness, contrast
- Cutout, mixup, cutmix

Text:
- Synonym replacement
- Back-translation
- Random insertion/deletion

Benefits:
+ Artificially increase dataset size
+ Reduces overfitting
+ Improves generalization

Use cases:
✓ Computer vision (almost always)
✓ NLP (less common, more careful)
✓ When have limited data
```

**Early Stopping**:
```
Monitor validation loss:
1. Train until validation loss stops improving
2. Save model at best validation loss
3. Stop if no improvement for N epochs (patience)

Typical patience: 5-10 epochs

Pros:
+ Prevents overfitting
+ Saves compute (don't train too long)

Cons:
- Need validation set
- Might stop too early (validation loss can plateau then improve)

Production standard:
- Always use early stopping
- Patience = 5-10 for small models, 3-5 for large models
```

---

### Initialization Strategies

**Why Initialization Matters**:
```
Bad initialization:
- Vanishing gradients: All activations → 0
- Exploding gradients: All activations → ∞
- Dead neurons: Stuck at 0, never activate
- Slow convergence: Loss barely decreases

Good initialization:
- Activations have reasonable scale (not too small/large)
- Gradients flow properly
- Fast convergence
```

**Zero Initialization** (WRONG):
```
W = 0, b = 0

Problem:
- All neurons compute same function (symmetry)
- All gradients identical
- Network never learns different features

Never use zero initialization for weights!
Can use zero for biases.
```

**Random Initialization** (Naive):
```
W ~ N(0, 0.01), b = 0

Problem:
- 0.01 too small for deep networks (vanishing activations)
- 1.0 too large (exploding activations)

Need: Scale variance based on layer size
```

**Xavier/Glorot Initialization (2010)**:
```
W ~ N(0, √(2/(n_in + n_out)))
or
W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

Where:
- n_in = number of input units
- n_out = number of output units

Intuition:
- Keep variance of activations constant across layers
- Keep variance of gradients constant during backprop

Use cases:
✓ Networks with tanh or sigmoid activations
✗ ReLU networks (use He initialization)

Derivation:
- Assume linear activations: y = Wx
- Var(y) = n_in · Var(W) (if inputs independent)
- Want: Var(y) = Var(x) → Var(W) = 1/n_in
- Symmetric for forward/backward → Var(W) = 2/(n_in + n_out)
```

**He Initialization (2015)** (for ReLU):
```
W ~ N(0, √(2/n_in))

Intuition:
- ReLU zeros out half the neurons
- Need 2× variance to compensate
- Factor of 2 accounts for ReLU killing half the gradient

Use cases:
✓ Networks with ReLU, Leaky ReLU, ELU
✓ Default for most modern networks

Production standard:
- PyTorch default for Conv2d, Linear
- Use He initialization unless using tanh (then Xavier)
```

**LeCun Initialization**:
```
W ~ N(0, √(1/n_in))

Use cases:
✓ SELU activation (self-normalizing networks)
✗ ReLU (use He)
```

**Initialization Summary**:
```
| Activation | Initialization |
|------------|----------------|
| tanh       | Xavier         |
| sigmoid    | Xavier         |
| ReLU       | He             |
| Leaky ReLU | He             |
| ELU        | He             |
| SELU       | LeCun          |
| Linear     | Xavier         |

Default: He initialization (works for most cases)
```

---

### Overfitting Prevention Checklist

**Detection**:
```
Signs of overfitting:
- Training loss ↓, validation loss ↑ (divergence)
- Train accuracy >> val accuracy (>10% gap)
- Model performs perfectly on train, poorly on test
- High variance in predictions (sensitive to input changes)
```

**Solutions (in order of effectiveness)**:
```
1. Get more data (if possible):
   - Best solution (if feasible)
   - 10x data → ~3-5% accuracy improvement

2. Data augmentation:
   - Artificially increase data
   - Image: flip, rotate, crop
   - Text: back-translation, paraphrasing

3. Regularization:
   - L2 (weight decay): λ = 1e-4 to 1e-2
   - Dropout: p = 0.2 to 0.5
   - Early stopping: patience = 5-10

4. Simplify model:
   - Reduce layers (5 → 3)
   - Reduce neurons per layer (512 → 256)
   - Trade-off: May underfit

5. Ensemble:
   - Train multiple models
   - Average predictions
   - Reduces variance

6. Cross-validation:
   - K-fold (k=5 or 10)
   - Detect overfitting to specific split

Production approach:
1. Start with regularization (L2 + dropout)
2. Add early stopping
3. If still overfitting, simplify model
4. If still overfitting, get more data
```

---

## 14.3 CNN ARCHITECTURES

### Why CNNs for Images?

**Problems with Fully Connected Networks**:
```
Image: 224×224×3 = 150,528 pixels
First layer: 150,528 × 1000 neurons = 150M parameters!

Problems:
- Too many parameters (overfitting, memory)
- No spatial structure (treats pixels independently)
- No translation invariance (object at different positions = different features)
```

**CNN Solution**:
```
Local connectivity: Neuron connected to small region (3×3, 5×5)
Weight sharing: Same filter applied across entire image
Translation invariance: Detect edges anywhere in image

Result:
- Fewer parameters (1000x reduction)
- Captures spatial structure
- Learns hierarchical features (edges → shapes → objects)
```

---

### Convolutional Layers

**Operation**:
```
Input: H×W×C (height × width × channels)
Filter: F×F×C (typically F=3 or 5)
Output: (H-F+1)×(W-F+1)×K (K = number of filters)

Computation (for one output pixel):
output[i,j,k] = Σ Σ Σ input[i+m, j+n, c] × filter[m, n, c, k] + bias[k]
                m n c

Where: m,n ∈ {0,...,F-1}, c ∈ {0,...,C-1}
```

**Parameters**:
```
Number of parameters:
- Filters: F × F × C × K
- Biases: K
- Total: (F²×C + 1) × K

Example (Conv layer):
- Input: 224×224×3
- Filter: 3×3, 64 filters
- Parameters: (3²×3 + 1) × 64 = 1,792

vs Fully connected:
- Parameters: 224×224×3 × 64 = 9,633,792
- Reduction: 5,000x fewer parameters!
```

**Stride**:
```
Stride s: Move filter by s pixels

Output size: ⌊(H - F)/s⌋ + 1

Example:
- Input: 7×7, Filter: 3×3
- Stride 1: Output 5×5 (spatial resolution preserved)
- Stride 2: Output 3×3 (downsampling by 2)

Use cases:
- Stride 1: Standard convolution (preserve resolution)
- Stride 2: Downsampling (instead of pooling)
```

**Padding**:
```
Padding p: Add p pixels around border (typically zeros)

Output size: ⌊(H + 2p - F)/s⌋ + 1

"SAME" padding: Output size = Input size (when s=1)
- p = (F-1)/2
- Example: F=3 → p=1, F=5 → p=2

"VALID" padding: No padding (p=0)
- Output shrinks by F-1 pixels

Use cases:
- SAME: Preserve spatial resolution (common in ResNet)
- VALID: Allow natural shrinking (common in early CNNs)
```

**1×1 Convolutions**:
```
Filter: 1×1×C×K

Effect:
- Channel mixing (linear combination of channels)
- Dimensionality reduction/expansion
- Add non-linearity (if followed by activation)

Use cases:
✓ Bottleneck layers (reduce channels before expensive 3×3 conv)
✓ Network-in-Network (Lin et al., 2013)
✓ Inception modules (GoogLeNet)
✓ ResNet bottleneck blocks

Example (ResNet bottleneck):
64 channels → 1×1 conv → 16 channels → 3×3 conv → 16 channels → 1×1 conv → 64 channels
Reduces 3×3 conv cost by 4x
```

---

### Pooling Layers

**Max Pooling**:
```
Operation: Take maximum value in pooling window

Common: 2×2 window, stride 2
- Downsamples by 2x (224×224 → 112×112)
- No parameters

Benefits:
+ Translation invariance (small shifts don't matter)
+ Reduces computation (fewer pixels)
+ Captures strongest activation

Cons:
- Lossy (discards information)
- No learning (fixed operation)
```

**Average Pooling**:
```
Operation: Take average value in pooling window

Use cases:
✓ Global Average Pooling (GAP): Entire feature map → 1 value
✓ Final layer before classification (instead of fully connected)
✗ Downsampling (max pooling better)

Production note:
- ResNet uses GAP before classification (instead of FC)
- Reduces parameters: 2048 channels × 1000 classes = 2M (GAP) vs 2048×7×7 × 1000 = 100M (FC)
```

**Stride vs Pooling**:
```
Modern trend: Stride instead of pooling
- Strided convolution: Learnable downsampling
- Pooling: Fixed downsampling

VGG (2014): Conv + MaxPool
ResNet (2015): Conv + MaxPool (early layers), then strided conv
EfficientNet (2019): Mostly strided conv (no pooling)
```

---

### ResNet (Residual Networks)

**Motivation**:
```
Problem: Deep networks (>20 layers) perform worse than shallow
- Not overfitting (training error higher)
- Optimization problem (gradients vanish)

Hypothesis: Deep networks hard to optimize to identity mapping
- If deeper network could learn identity (F(x) = x), at least match shallow
- But hard for deep networks to learn exact identity
```

**Residual Connection** (He et al., 2015):
```
Instead of learning H(x), learn residual F(x):
H(x) = F(x) + x

Architecture:
x → [Conv-BN-ReLU-Conv-BN] → Add x → ReLU → output
     └──────────────┬────────────┘
                 F(x)
    
    output = ReLU(F(x) + x)

Intuition:
- Easy to learn F(x) = 0 (just set weights to zero)
- Then H(x) = x (identity mapping)
- Network can go deeper without hurting performance
```

**ResNet Architecture**:
```
ResNet-50:
- 50 layers (including shortcut connections)
- Structure: Conv → MaxPool → [Residual Block] × 16 → GAP → FC
- Residual blocks: Bottleneck design (1×1 → 3×3 → 1×1)
- Parameters: 25M

ResNet-152:
- 152 layers
- Same structure, more blocks
- Parameters: 60M

Key innovation: Can train 100+ layers (before: max ~20 layers)
```

**Why Residual Connections Work**:
```
Gradient flow:
∂Loss/∂x = ∂Loss/∂output × (∂F(x)/∂x + 1)
                                          ↑
                          Always have gradient from identity!

Even if ∂F(x)/∂x → 0 (vanishing), still have ∂Loss/∂x from identity path.

Result:
- Gradients flow directly to early layers
- Enables training very deep networks (1000+ layers possible)
```

**Production Impact**:
```
ImageNet (2015):
- Before ResNet: VGG-16 (16 layers, 7.3% error)
- ResNet-152: 152 layers, 3.57% error (human-level: ~5%)

Object detection:
- ResNet backbone: Standard (Faster R-CNN, Mask R-CNN)

Influence:
- Residual connections everywhere: Transformers (attention + residual), U-Net (skip connections)
```

---

### VGG (Visual Geometry Group)

**Architecture** (Simonyan & Zisserman, 2014):
```
Philosophy: Very deep networks with small filters (3×3)

VGG-16:
- 16 layers (13 conv, 3 FC)
- Only 3×3 filters (stride 1, padding 1)
- MaxPool (2×2, stride 2) after conv blocks
- Double channels after each pool (64 → 128 → 256 → 512 → 512)

Structure:
[Conv3-64] × 2 → MaxPool → [Conv3-128] × 2 → MaxPool → 
[Conv3-256] × 3 → MaxPool → [Conv3-512] × 3 → MaxPool → 
[Conv3-512] × 3 → MaxPool → FC-4096 → FC-4096 → FC-1000

Parameters: 138M (mostly in FC layers)
```

**Why 3×3 Filters?**:
```
Two 3×3 conv (receptive field 5×5):
- Parameters: 2 × (3²×C²) = 18C²
- With non-linearity between

One 5×5 conv (receptive field 5×5):
- Parameters: 5²×C² = 25C²
- No non-linearity

Stack of 3×3 > Single large filter:
+ Fewer parameters (18C² < 25C²)
+ More non-linearities (deeper → more expressiveness)
- More layers (slower)
```

**Limitations**:
```
Problems:
- Too many parameters (138M, mostly FC)
- Slow training
- Prone to overfitting (needs heavy regularization)

Solutions (later models):
- Replace FC with GAP (ResNet, Inception)
- Reduce parameters (EfficientNet)
```

---

### Inception / GoogLeNet

**Motivation** (Szegedy et al., 2014):
```
Problem: What filter size to use? (1×1, 3×3, 5×5?)
Solution: Use all of them! (Inception module)
```

**Inception Module**:
```
Input → 1×1 conv → Output 1
     → 1×1 conv → 3×3 conv → Output 2
     → 1×1 conv → 5×5 conv → Output 3
     → MaxPool → 1×1 conv → Output 4
     
Concatenate [Output 1, 2, 3, 4] → Final Output

Key: 1×1 conv before expensive 3×3/5×5 (bottleneck)
- Reduces channels: 256 → 64 → 3×3 → 256
- Reduces computation by 4x
```

**GoogLeNet (Inception v1)**:
```
- 22 layers
- 9 Inception modules
- Parameters: 7M (20x fewer than VGG!)
- ImageNet: 6.7% error (2014 winner)

Key innovations:
+ Inception module (multi-scale features)
+ 1×1 bottleneck (reduce computation)
+ Global Average Pooling (no FC layers)
+ Auxiliary classifiers (help gradient flow)
```

**Inception v2, v3, v4**:
```
Improvements:
- Factorized convolutions (5×5 → two 3×3, or 3×3 → 3×1 and 1×3)
- Batch Normalization
- Residual connections (Inception-ResNet)
```

---

### EfficientNet

**Motivation** (Tan & Le, 2019):
```
Problem: How to scale CNNs efficiently?
- Scale depth? (more layers)
- Scale width? (more channels)
- Scale resolution? (larger images)

Traditional: Scale one dimension (arbitrary)
EfficientNet: Compound scaling (scale all three systematically)
```

**Compound Scaling**:
```
Depth: d = α^φ
Width: w = β^φ
Resolution: r = γ^φ

Subject to: α × β² × γ² ≈ 2
           α ≥ 1, β ≥ 1, γ ≥ 1

Where:
- α, β, γ = scaling coefficients (grid search: α=1.2, β=1.1, γ=1.15)
- φ = compound coefficient (user-specified)

Example (φ = 1):
- Depth: 1.2x layers
- Width: 1.1x channels
- Resolution: 1.15x image size
- FLOPs: 2x (balanced scaling)
```

**Architecture**:
```
EfficientNet-B0 (Baseline):
- Mobile Inverted Bottleneck (MBConv blocks)
- Squeeze-and-Excitation (SE) blocks
- Parameters: 5.3M
- ImageNet: 77.3% top-1

EfficientNet-B7 (Scaled):
- Same architecture, scaled by φ = 2.2
- Parameters: 66M
- ImageNet: 84.3% top-1 (state-of-the-art in 2019)

Key: Better accuracy-efficiency trade-off than ResNet, Inception
```

---

### Vision Transformers (ViT)

**Motivation** (Dosovitskiy et al., 2020):
```
Question: Can Transformers (from NLP) work for vision?
Answer: Yes, with enough data (no inductive bias like CNNs)
```

**Architecture**:
```
1. Split image into patches (16×16 pixels)
   - Image 224×224 → 14×14 = 196 patches

2. Flatten patches (16×16×3 = 768 dimensions)

3. Linear projection (patch embedding)
   - 768 → D dimensions (typically D=768)

4. Add position embeddings (learned)
   - No built-in spatial structure (unlike CNNs)

5. Transformer encoder
   - Multi-head self-attention
   - Layer normalization
   - MLP (feedforward)

6. Classification head
   - Use [CLS] token (like BERT)

Parameters (ViT-Base):
- Patches: 16×16
- Layers: 12
- Hidden size: 768
- Heads: 12
- Parameters: 86M
```

**ViT vs CNN**:
```
CNNs:
+ Inductive bias (locality, translation invariance)
+ Data efficient (works with 100K images)
+ Faster inference (optimized hardware)
- Fixed receptive field (limited by kernel size)

ViTs:
+ Global receptive field (self-attention sees all patches)
+ Scalable (larger model → better performance)
- Needs massive data (100M+ images) or good pre-training
- Slower inference (attention is O(n²))

Production:
- CNNs: When data limited (<1M images), need speed
- ViTs: When have massive data (100M+), need accuracy
- Hybrid: ViT with CNN stem (best of both)
```

---

### Image Preprocessing and Augmentation

**Preprocessing** (Training):
```
Standard pipeline (ImageNet):
1. Resize to 256×256 (shorter side)
2. Random crop to 224×224
3. Random horizontal flip (p=0.5)
4. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Mean/std from ImageNet

Code:
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

**Preprocessing** (Inference):
```
1. Resize to 256×256
2. Center crop to 224×224 (not random!)
3. Normalize (same as training)

No augmentation during inference (want deterministic predictions)
```

**Advanced Augmentation**:
```
AutoAugment (Google, 2018):
- Search for optimal augmentation policy
- Example: Rotate(30°) + Posterize(4) + ...

RandAugment (Google, 2019):
- Simplified AutoAugment
- Two hyperparameters: N (num operations), M (magnitude)
- Typical: N=2, M=9

Mixup (Zhang et al., 2017):
- Mix two images: x = λx₁ + (1-λ)x₂
- Mix labels: y = λy₁ + (1-λ)y₂
- λ ~ Beta(α, α), α=0.2 typical

Cutmix (Yun et al., 2019):
- Cut patch from image 1, paste into image 2
- Mix labels proportionally to patch area

Production impact:
- AutoAugment: +1% ImageNet accuracy
- RandAugment: Similar to AutoAugment, simpler
- Mixup/Cutmix: +1-2% accuracy, standard in modern training
```

---

## Interview Questions (Hao Hoang Style)

### Question 1: Backprop Computation (CRITICAL TRAP)
**Q**: "Your model has 7 billion parameters. Estimate the FLOPs for one forward and one backward pass with batch size 1."

**Expected Answer**:
- **Forward pass**: 2N FLOPs = 2 × 7B = 14 GFLOPs
  - 2N because: multiply + add for each parameter
- **Backward pass**: 4N FLOPs = 4 × 7B = 28 GFLOPs
  - 2N for gradient computation (same as forward)
  - 2N for gradient propagation (chain rule)
- **TOTAL**: 6N = 42 GFLOPs per sample

**Common traps**:
- Saying "2N FLOPs total" (only forward!)
- Forgetting gradient propagation (chain rule adds 2N)
- Not accounting for batch size (multiply by batch size)

**Follow-up**: "How does batch size affect this?"
- FLOPs scale linearly with batch size
- Batch size 1024: 42 × 1024 = 43 TFLOPs
- Wall-clock time: Sub-linear speedup (due to parallelization, overhead)

---

### Question 2: Activation Function Selection
**Q**: "You're training a 100-layer CNN. It's not converging (loss stuck at 0.7). You're using sigmoid activation. What's wrong and how do you fix it?"

**Expected Answer**:
- **Problem**: Vanishing gradient (sigmoid saturates, σ' ≈ 0 for |x| > 2)
  - After 100 layers: gradient × (0.1)^100 ≈ 0
  - Network can't learn (gradients too small)
  
- **Solutions**:
  1. **Use ReLU** (best fix):
     - No saturation for x > 0
     - Gradient either 0 or 1 (no vanishing)
  2. **Residual connections** (if must use sigmoid):
     - Gradients bypass sigmoid (flow through identity)
  3. **Batch Normalization**:
     - Keeps activations in reasonable range (prevents saturation)
  4. **Better initialization** (He):
     - Prevents initial saturation

**Red flag**: Suggesting learning rate changes without addressing activation

**Follow-up**: "After switching to ReLU, loss is NaN. What happened?"
- **Exploding gradients** (ReLU has no upper bound)
- **Solutions**:
  - Gradient clipping (clip to [-1, 1])
  - Lower learning rate (0.1 → 0.01)
  - Batch Normalization (stabilizes activations)

---

### Question 3: Overfitting in CNNs
**Q**: "Your CNN gets 95% train accuracy, 70% validation accuracy. Dataset size: 10K images. What do you do?"

**Expected Answer**:
- **Diagnosis**: Overfitting (25% gap)
  - 10K images small for CNNs (typically need 100K+)
  
- **Solutions (in order)**:
  1. **Data augmentation** (biggest impact):
     - Flip, rotate, crop, color jitter
     - Artificially 5-10x data
     - Expected: Gap reduces to 10-15%
  2. **Regularization**:
     - Dropout (p=0.5) in FC layers
     - L2 weight decay (λ=1e-4)
  3. **Simplify model**:
     - Reduce layers (ResNet-50 → ResNet-18)
     - Reduce channels (512 → 256)
  4. **Transfer learning** (if applicable):
     - Pre-train on ImageNet (1M images)
     - Fine-tune on 10K images
     - Expected: 85-90% val accuracy

**Red flag**: Jumping to "reduce learning rate" without addressing overfitting

**Follow-up**: "After augmentation, val accuracy is 82% but inference is slow (100ms per image). Need <10ms. What do you do?"
- Model compression:
  - Pruning (remove 50% of weights, <5% accuracy loss)
  - Quantization (FP32 → INT8, 4x speedup)
  - Knowledge distillation (train smaller student model)
  - MobileNet/EfficientNet (smaller architectures)

---

### Question 4: Receptive Field
**Q**: "You have three 3×3 conv layers (stride 1, no padding). What's the receptive field?"

**Expected Answer**:
- **Layer 1**: Receptive field = 3×3
- **Layer 2**: Each neuron sees 3×3 from L1, each L1 neuron sees 3×3 → 5×5 total
- **Layer 3**: 7×7

**Formula**: RF = 1 + Σ(kernel_size - 1) × Π(strides before)
- Three 3×3: RF = 1 + (3-1) + (3-1) + (3-1) = 7

**Insight**: Stack of small filters > Single large filter
- Three 3×3 (RF=7): Parameters = 3 × (3²×C²) = 27C²
- One 7×7 (RF=7): Parameters = 7²×C² = 49C²
- Savings: 45% fewer parameters + 2 more non-linearities

**Follow-up**: "How does stride affect receptive field?"
- Stride s: Multiplies subsequent layers' RF by s
- Example: 3×3 (stride 1), 3×3 (stride 2), 3×3 (stride 1)
  - RF = 1 + 2 + 2×2 + 2×1 = 9 (not 7!)

---

### Question 5: Batch Normalization Trap
**Q**: "Your model with BatchNorm trains perfectly (99% train accuracy) but gets 60% test accuracy. Batch size during training: 64. Batch size during inference: 1. What's wrong?"

**Expected Answer**:
- **Problem**: BatchNorm uses batch statistics (mean, std)
  - Training: Computed from batch of 64 (stable statistics)
  - Inference batch size 1: "Batch" mean/std from single sample (unstable!)
  
- **Correct inference**: Use running mean/std (accumulated during training)
  - PyTorch: model.eval() switches to running stats
  - TensorFlow: training=False

- **Debugging**:
  - Check: model.eval() called before inference?
  - Check: Running stats saved in checkpoint?
  - Check: Running stats updated during training? (requires model.train())

**Red flag**: Not knowing BatchNorm behaves differently in train vs inference

**Follow-up**: "After fixing, accuracy is 85% (better) but inference is slow. Can you remove BatchNorm?"
- **No** (accuracy will drop significantly)
- **Alternative**: Fold BatchNorm into convolution (inference optimization)
  - Combine Conv + BN into single Conv layer
  - Mathematically equivalent but faster
  - Speedup: 10-20% inference time

---

### Question 6: ResNet vs VGG
**Q**: "You need to choose between ResNet-50 (25M params) and VGG-16 (138M params) for a production system. Latency budget: 10ms per image. Which do you choose and why?"

**Expected Answer**:
- **Choice**: ResNet-50

**Reasoning**:
1. **Accuracy**: ResNet-50 better (76% vs 71% ImageNet top-1)
2. **Speed**: ResNet-50 faster despite more layers
   - Fewer parameters (25M vs 138M)
   - No giant FC layers (GAP instead)
   - Inference: ~7ms (ResNet-50) vs ~15ms (VGG-16) on GPU
3. **Memory**: 100MB (ResNet-50) vs 528MB (VGG-16)
4. **Training**: ResNet trains faster (convergence in fewer epochs)

**Only choose VGG if**:
- Need simpler architecture (easier to understand/debug)
- Using very old hardware (VGG more compatible)

**Production note**:
- 95% of production CNNs use ResNet or EfficientNet (not VGG)
- VGG mostly for educational purposes now

**Follow-up**: "Still too slow. Need <5ms. What do you do?"
- **Switch to smaller model**:
  - ResNet-18 (11M params): ~3ms, 70% accuracy (-6%)
  - MobileNetV2 (3.5M params): ~2ms, 72% accuracy (-4%)
  - EfficientNet-B0 (5.3M params): ~4ms, 77% accuracy (+1%!)
- **Or optimize inference**:
  - TensorRT (NVIDIA): 2-3x speedup
  - Quantization (INT8): 4x speedup
  - Pruning: 30-50% speedup

---

### Question 7: Transfer Learning
**Q**: "You have 5K images for a binary classification task (dogs vs cats). Training from scratch gives 60% accuracy. What's your strategy?"

**Expected Answer**:
- **Problem**: Too little data for CNN from scratch (need 100K+)

**Solution: Transfer Learning**:
1. **Load pre-trained model** (ImageNet):
   - ResNet-50 pre-trained (76% ImageNet accuracy)
   
2. **Replace final layer**:
   - Original: 1000 classes (ImageNet)
   - New: 2 classes (dogs vs cats)
   
3. **Freeze early layers** (feature extractor):
   - Layers 1-40: Frozen (generic features: edges, shapes)
   - Layers 41-50 + new head: Trainable
   
4. **Train with small learning rate**:
   - Frozen layers: 0 (not updated)
   - Trainable layers: 1e-4 (10x smaller than from-scratch)
   
5. **Expected accuracy**: 95%+ (with 5K images!)

**Fine-tuning schedule**:
```
Epochs 1-5: Only train new head (freeze all conv layers)
- LR = 1e-3
- Expected: 85% accuracy

Epochs 6-15: Unfreeze last residual block + head
- LR = 1e-4
- Expected: 92% accuracy

Epochs 16-30: Unfreeze all layers (fine-tune)
- LR = 1e-5 (very small!)
- Expected: 95%+ accuracy
```

**Red flag**: Training from scratch with <10K images

**Follow-up**: "What if your task is medical imaging (X-rays)?"
- **Still use ImageNet pre-training** (surprisingly effective!)
- Generic features (edges, textures) transfer well
- Studies show: ImageNet → X-rays gives 5-10% boost
- Alternative: Pre-train on similar medical dataset (if available)

---

### Question 8: Model Size Estimation
**Q**: "Estimate memory footprint of ResNet-50 for inference (batch size 1)."

**Expected Answer**:
- **Parameters**: 25M
  - FP32: 25M × 4 bytes = 100 MB
  - FP16: 25M × 2 bytes = 50 MB
  
- **Activations** (intermediate outputs):
  - Input: 224×224×3 = 150K pixels
  - After conv1: 112×112×64 = 800K
  - After residual blocks: 7×7×2048 = 100K
  - Total activations: ~10M values
  - FP32: 10M × 4 = 40 MB
  
- **Total inference memory**:
  - FP32: 100 MB (params) + 40 MB (activations) = **140 MB**
  - FP16: 50 MB + 20 MB = **70 MB**

**Training memory** (additional):
- Gradients: Same size as parameters (100 MB)
- Optimizer states (Adam): 2× parameters (200 MB)
- **Total training**: 140 MB + 100 MB + 200 MB = **440 MB** (FP32)

**Follow-up**: "Need to reduce memory by 50%. What do you do?"
1. **FP16 inference**: 70 MB (50% reduction) ✓
2. **Quantization (INT8)**: 35 MB (75% reduction)
3. **Pruning**: Remove 30% of weights → 70 MB
4. **Smaller model**: ResNet-18 (11M params) → 44 MB

---

### Question 9: CNN Failure Modes
**Q**: "Your CNN works well on validation set (90% accuracy) but fails catastrophically on specific images (0% accuracy on images with watermarks). Why and how do you fix it?"

**Expected Answer**:
- **Problem**: Spurious correlation (shortcut learning)
  - Training data: All dog images have watermark, cat images don't
  - Model learned: Watermark → Dog (not actual dog features!)
  - Test data: Dog without watermark → Model fails
  
- **Detection**:
  - Saliency maps (Grad-CAM): Shows model focuses on watermark
  - Counterfactual examples: Add/remove watermark → Prediction flips
  
- **Solutions**:
  1. **Data cleaning**: Remove watermarks from training data
  2. **Data augmentation**: Add synthetic watermarks randomly to both classes
  3. **Adversarial training**: Penalize reliance on watermarks
  4. **Regularization**: Encourage model to use diverse features

**Similar failure modes**:
- Background bias: Model learns grass (background) instead of cow
- Texture bias: Model learns texture instead of shape
- Color bias: Model learns color instead of object

**Production debugging**:
- Always visualize attention (Grad-CAM, SHAP)
- Check: Is model using expected features?
- Test: Adversarial examples (modify irrelevant features)

---

### Question 10: Architecture Selection
**Q**: "You're building a mobile app for real-time object detection. Device: iPhone 12 (6-core CPU, 4GB RAM). Latency requirement: <50ms per frame. Which architecture do you choose?"

**Expected Answer**:
- **Requirements analysis**:
  - Mobile device (CPU, limited RAM)
  - Real-time (<50ms → 20 FPS)
  - Object detection (not just classification)
  
- **Architecture choice**: **MobileNetV2 or EfficientNet-B0**

**Comparison**:
```
| Model           | Params | Latency | Accuracy | Memory |
|-----------------|--------|---------|----------|--------|
| ResNet-50       | 25M    | 150ms   | 76%      | 100MB  | ✗ Too slow
| MobileNetV2     | 3.5M   | 40ms    | 72%      | 14MB   | ✓ Good
| EfficientNet-B0 | 5.3M   | 45ms    | 77%      | 20MB   | ✓ Best
| MobileNetV3     | 5.4M   | 35ms    | 75%      | 20MB   | ✓ Good
```

- **Recommendation**: EfficientNet-B0
  - Best accuracy (77%)
  - Meets latency (<50ms)
  - Fits in memory (20MB < 4GB)

**Optimizations**:
1. **Quantization** (FP32 → INT8): 4x faster, 75MB → 5MB
2. **Core ML** (Apple's framework): Additional 1.5-2x speedup
3. **Pruning**: Remove 30% weights → 25% speedup

**Production note**:
- Test on actual device (simulator inaccurate)
- Profile: Use Xcode Instruments
- Batch size 1 (process frames individually)

---

## Key Takeaways for Interviews

### Neural Network Fundamentals
- **Backprop computation**: 6N FLOPs (not 2N!)
- **Activation functions**: ReLU default, GELU for Transformers
- **Regularization**: Dropout (FC), BatchNorm (CNN), L2 weight decay
- **Initialization**: He for ReLU, Xavier for tanh

### CNNs
- **Modern architecture**: ResNet or EfficientNet (not VGG)
- **Transfer learning**: Always use pre-trained on ImageNet (even for medical)
- **Inference optimization**: Quantization (4x), TensorRT (2-3x), pruning (30%)
- **Receptive field**: Stack of small filters > Single large filter

### Production Lessons
- **Google**: EfficientNet (compound scaling)
- **Meta**: ResNet everywhere (vision backbone)
- **Apple**: MobileNet/EfficientNet (on-device)
- **NVIDIA**: TensorRT for deployment

### Interview Red Flags to Avoid
1. Saying "2N FLOPs" for backprop (correct: 6N)
2. Using VGG in production (use ResNet/EfficientNet)
3. Training CNNs from scratch with <10K images (use transfer learning)
4. Not knowing BatchNorm behaves differently in train vs inference
5. Choosing model without considering inference latency

---

**Next Document**: Part 3 will cover RNN/LSTMs (14.4), Loss Functions (14.5), Evaluation Metrics (14.6), and Overfitting/Regularization (14.7).
