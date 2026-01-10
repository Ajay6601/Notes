# LLM Fundamentals - Part 3: Advanced Topics & Production Patterns
## Deep Technical Dive (Hao Hoang Style)

**Sources**: Anthropic Scaling Laws paper, Meta Llama 2 paper, MergeKit documentation, OpenAI o1 system card, Google Gemini tech report, Apple MLX documentation, Microsoft DeepSpeed papers, NVIDIA TensorRT-LLM benchmarks

---

# 6. Model Merging - The Black Art

## 6.1 Model Soups - When Averaging Beats Ensembles

### The WiSE (Wortsman et al., 2022) Discovery

**Counterintuitive Result**:
```python
# Ensemble (standard approach)
predictions = []
for model in models:
    pred = model.predict(x)
    predictions.append(pred)
average_prediction = np.mean(predictions, axis=0)

# Cost: N × inference time
# Benefit: ~2% accuracy gain

# Model Soup (weight averaging)
avg_weights = {}
for param_name in models[0].state_dict().keys():
    avg_weights[param_name] = torch.stack([
        m.state_dict()[param_name] for m in models
    ]).mean(dim=0)

merged_model.load_state_dict(avg_weights)
prediction = merged_model.predict(x)

# Cost: 1 × inference time (SAME as single model!)
# Benefit: ~1.5% accuracy gain (slightly less than ensemble)

# Winner: Model Soup (same speed, almost same accuracy)
```

**Interview Question** (Hao Hoang style):
```
Q: You train 5 models with different learning rates:
   {1e-5, 5e-5, 1e-4, 5e-4, 1e-3}
   
   Validation accuracies: {85%, 87%, 89%, 86%, 82%}
   
   Should you:
   A) Use best model (89%)
   B) Uniform soup (average all 5)
   C) Greedy soup (start with best, add others)
```

**Answer**:
```python
# Test each approach

# A) Best single: 89%

# B) Uniform soup
# Problem: Last model (82%) is much worse
# Averaging pulls down performance
# Result: ~86% (WORSE than best single!)

# C) Greedy soup
def greedy_soup(models, val_accuracies, val_set):
    # Start with best model
    best_idx = np.argmax(val_accuracies)
    soup = [models[best_idx]]
    best_acc = val_accuracies[best_idx]
    
    # Try adding each other model
    for i, model in enumerate(models):
        if i == best_idx:
            continue
        
        # Test candidate soup
        candidate = soup + [model]
        candidate_weights = average_weights(candidate)
        candidate_acc = evaluate(candidate_weights, val_set)
        
        if candidate_acc > best_acc:
            soup = candidate
            best_acc = candidate_acc
            print(f"Added model {i}, new acc: {best_acc}")
    
    return soup

# Result: Greedy adds models [3, 2] → 91% (BEST!)
# Answer: C (Greedy soup)
```

**Production Example** (From LAION, 2023):
```
LAION trained CLIP models with different:
- Batch sizes: [256, 512, 1024, 2048]
- Learning rates: [1e-4, 5e-4, 1e-3]
- Warmup steps: [500, 1000, 2000]

Total: 36 models

Greedy soup selected 8 models
Result: +3.2% zero-shot accuracy vs best single model
```

### The Soup Failure Modes

**When Soups Fail**:
```python
# Failure Mode 1: Divergent models
model_A = finetune(base, task="summarization")
model_B = finetune(base, task="translation")

# Averaging → model that can't do either task well!
# Reason: Weight directions conflict

# Failure Mode 2: Different architectures
model_A = Llama2_7B  # 32 layers
model_B = Mistral_7B  # 32 layers BUT different intermediate sizes

# Can't average: Shape mismatch!

# Failure Mode 3: Large learning rate divergence
model_A = finetune(base, lr=1e-5)  # Barely moved from base
model_B = finetune(base, lr=1e-2)  # VERY different from base

# Average: 50% closer to model_B, doesn't work well
```

---

## 6.2 SLERP - The Geometry of Model Space

### The Spherical Interpolation Intuition

**Why Linear Fails**:
```python
# In high-dimensional space, vectors lie on hypersphere
# Linear interpolation cuts through sphere (shortcut)

# Example: 2D unit circle
w1 = np.array([1, 0])    # Point on circle
w2 = np.array([0, 1])    # Point on circle

# Linear interpolation
linear = 0.5 * w1 + 0.5 * w2  # [0.5, 0.5]
norm_linear = np.linalg.norm(linear)  # 0.707 (NOT on circle!)

# SLERP interpolation
theta = np.arccos(np.dot(w1, w2))  # 90 degrees
slerp = (np.sin(0.5 * theta) / np.sin(theta)) * w1 + \
        (np.sin(0.5 * theta) / np.sin(theta)) * w2
norm_slerp = np.linalg.norm(slerp)  # 1.0 (ON circle!)
```

**Interview Question**:
```
Q: SLERP between two 7B models at t=0.5.
   Does the result have 7B parameters?
   Does it have 7B params * 2 = 14B memory during merge?
```

**Answer**:
```python
# Q1: Yes, 7B parameters (SLERP doesn't change size)
# Q2: YES - common trap!

# Naive implementation:
model1 = load_model("model1.safetensors")  # 28GB (7B × 4 bytes)
model2 = load_model("model2.safetensors")  # 28GB
merged = slerp(model1, model2, t=0.5)      # 28GB

# Peak memory: 84GB (all 3 in RAM!)

# Optimized implementation:
import torch.nn as nn

def slerp_merge_efficient(model1_path, model2_path, output_path, t=0.5):
    """Memory-efficient SLERP using streaming"""
    
    # Load metadata only
    model1_meta = safetensors.safe_open(model1_path, framework="pt")
    model2_meta = safetensors.safe_open(model2_path, framework="pt")
    
    merged_tensors = {}
    
    # Process layer by layer (streaming)
    for param_name in model1_meta.keys():
        # Load single layer (e.g., 50MB)
        w1 = model1_meta.get_tensor(param_name)
        w2 = model2_meta.get_tensor(param_name)
        
        # SLERP this layer
        merged = slerp_tensor(w1, w2, t)
        merged_tensors[param_name] = merged
        
        # Immediately write to disk, free memory
        del w1, w2
        
    # Save merged model
    safetensors.torch.save_file(merged_tensors, output_path)

# Peak memory: ~1GB (single layer at a time)
```

**Production Gotcha** (From MergeKit issues):
```
Problem: SLERP between models trained with different precisions
         Model A: FP16
         Model B: BF16
         
Solution: Convert to FP32 before merging
          Merge in FP32
          Downcast to FP16/BF16 after
          
Cost: 2x memory, but avoids precision artifacts
```

---

## 6.3 Task Arithmetic - The Addition Algebra

### The Linear Representation Hypothesis

**Hypothesis**: Skills are linear directions in weight space

**Test**:
```python
# Learn task vectors
math_vector = math_model.weights - base_model.weights
code_vector = code_model.weights - base_model.weights

# Test 1: Adding tasks
multi_task = base_model.weights + math_vector + code_vector

eval_math = evaluate(multi_task, "GSM8K")  # Math benchmark
eval_code = evaluate(multi_task, "HumanEval")  # Code benchmark

# Result: 85% of specialist performance on BOTH tasks
# (vs 100% for single-task models)

# Test 2: Subtracting tasks
detoxified = base_model.weights - toxicity_vector

eval_toxic = evaluate(detoxified, "RealToxicityPrompts")
# Result: 40% reduction in toxic outputs

# Test 3: Scaling tasks
strong_math = base_model.weights + 1.5 * math_vector  # Amplify math
weak_code = base_model.weights + 0.3 * code_vector   # Weak code

# Result: Strong math, weak code (as expected!)
```

**Interview Question** (Hao Hoang style):
```
Q: You have:
   - Base model (no instruction following)
   - Instruct model (instruction-tuned)
   - Math model (fine-tuned on math, from base)
   
   Can you create a model that does math + instruction following?
   
   Method A: (math_model - base_model) + instruct_model
   Method B: instruct_model + (math_model - base_model)
   
   Are they equivalent?
```

**Answer**:
```python
# Method A: (math_model - base_model) + instruct_model
# = math_vector + instruct_model
# = (math_model - base_model) + instruct_model
# = math_model + (instruct_model - base_model)
# = math_model + instruct_vector

# Method B: instruct_model + (math_model - base_model)
# = instruct_model + math_vector

# Mathematically equivalent: A = B ✓

# BUT: Practically different!
# Reason: instruct_model may have different base than math_model

# Safe approach: Use same base for both
math_vector = math_model - base_model
instruct_vector = instruct_model - base_model
merged = base_model + math_vector + instruct_vector
```

---

## 6.4 TIES - Resolving Parameter Conflicts

### The Sign Disagreement Problem

**Example**:
```python
# Parameter index 42 in layer 10
base_model[10][42] = 0.5

# Task A (math): Increase by 0.3
math_model[10][42] = 0.8
math_vector[10][42] = +0.3

# Task B (code): Decrease by 0.2
code_model[10][42] = 0.3
code_vector[10][42] = -0.2

# Naive addition:
merged[10][42] = base + math_vector + code_vector
               = 0.5 + 0.3 + (-0.2)
               = 0.6

# But: Tasks disagree! Compromise (0.6) might be worst of both
```

**TIES Solution**:
```python
def ties_merge(base, task_vectors, weights):
    """
    TIES: Trim, Elect, Merge
    """
    # Step 1: TRIM (keep top 20% by magnitude)
    trimmed_vectors = []
    for tv in task_vectors:
        mask = np.abs(tv) > np.percentile(np.abs(tv), 80)
        trimmed_vectors.append(tv * mask)
    
    # Step 2: ELECT (resolve sign conflicts)
    merged = np.zeros_like(base)
    for i in range(len(base)):
        # Get signs for this parameter
        signs = [np.sign(tv[i]) for tv in trimmed_vectors if tv[i] != 0]
        
        if len(signs) == 0:
            continue  # No task modified this parameter
        
        # Vote on sign
        sign_vote = sum(signs)
        majority_sign = np.sign(sign_vote)
        
        if abs(sign_vote) < len(signs) / 2:
            # No clear majority, skip this parameter
            continue
        
        # Step 3: MERGE (average values with same sign)
        same_sign_values = [
            w * tv[i] 
            for w, tv in zip(weights, trimmed_vectors)
            if np.sign(tv[i]) == majority_sign
        ]
        
        merged[i] = sum(same_sign_values)
    
    return base + merged

# Result on parameter 42:
# Math: +0.3 (positive), Code: -0.2 (negative)
# Vote: 1 positive, 1 negative → TIE
# TIES skips this parameter (leaves at base value 0.5)
# Better than arbitrary compromise!
```

**Benchmarks** (Yadav et al., 2023):
```
8-task merge (MMLU, HellaSwag, ARC, etc.):

Task Arithmetic: 67.3% avg accuracy
TIES-Merging: 73.5% avg accuracy (+6.2%)
Individual experts: 78.2% avg

TIES recovers 88% of expert performance
(vs 86% for task arithmetic)
```

---

## 6.5 DARE - Dropout for Merging

### The Sparsity Observation

**Key Insight** (Yu et al., 2024):
```python
# Analyze task vectors after fine-tuning
base = load_model("llama-2-7b")
math_model = load_model("llama-2-7b-math")

task_vector = math_model - base

# Statistics
print(f"Total params: {task_vector.numel()}")  # 7B
print(f"Non-zero: {(task_vector != 0).sum()}")  # 7B (all)

# But how many are significant?
threshold = 0.001
significant = (np.abs(task_vector) > threshold).sum()
print(f"Significant: {significant}")  # 700M (10%)

# 90% of changes are tiny!
# Hypothesis: Can drop them without hurting performance
```

**DARE Implementation**:
```python
def dare(task_vector, drop_rate=0.9):
    """
    Drop And REscale
    """
    # Random drop mask
    mask = np.random.binomial(1, 1 - drop_rate, task_vector.shape)
    
    # Drop parameters
    dropped = task_vector * mask
    
    # Rescale to maintain expected value
    # E[dropped] = E[task_vector] * (1 - drop_rate)
    # To fix: multiply by 1 / (1 - drop_rate)
    rescaled = dropped / (1 - drop_rate)
    
    return rescaled

# Example
task_vec = np.array([0.1, 0.05, 0.2, 0.01, 0.15])
dare_vec = dare(task_vec, drop_rate=0.9)

# Before: [0.1, 0.05, 0.2, 0.01, 0.15]
# Mask:   [  0,    1,   0,    0,    1]
# After:  [  0,  0.5,   0,    0,  1.5]  (rescaled by 10)

# Mean: Same as before (due to rescaling)
```

**Benchmark Results**:
```
Llama-2-7B, 8 task merge:

DARE 50% drop: 73.1% accuracy (vs 73.5% no drop)
DARE 90% drop: 72.4% accuracy (-1.1%)
DARE 95% drop: 70.8% accuracy (-2.7%)
DARE 99% drop: 66.1% accuracy (-7.4%)

Practical: Use 90% drop
- 10x fewer parameters to merge
- 10x faster merging
- 1% accuracy loss (acceptable)
```

---

## 6.6 Frankenmerge - The Depth-Up Scaling

### Layer Concatenation Intuition

**Hypothesis**: Early layers learn general features, late layers learn task-specific

**Experiment**:
```python
# Two 7B models (32 layers each)
model_A = Llama_2_7B  # General knowledge
model_B = CodeLlama_7B  # Code specialist

# Frankenmerge: Stack layers
franken = concatenate_layers(
    model_A.layers[0:16],   # General early layers
    model_B.layers[16:32],  # Code-specific late layers
)

# Result: 48-layer model (~9B parameters)

# Eval:
# Coding: 85% of CodeLlama performance
# General: 90% of Llama-2 performance
# Zero training required!
```

**Interview Question**:
```
Q: You concatenate layers from two 7B models (32 layers each).
   You take first 16 layers from Model A, last 16 from Model B.
   
   Final model has 32 layers (same as original).
   Is it still 7B parameters? Or more?
```

**Answer**:
```python
# Trap: Most say "7B" (wrong!)

# Analysis:
# Original: 32 layers × 218.75M params/layer = 7B
# (Approx: 7B / 32 = 218.75M per layer)

# Frankenfr: 16 layers from A + 16 layers from B
#          = 16 × 218.75M + 16 × 218.75M
#          = 7B total

# Correct: Still 7B parameters!

# BUT: If you take ALL 32 from A + ALL 32 from B:
# Result: 64 layers = 14B parameters (depth-up scaling)
```

**Production Example** (Goliath 120B):
```yaml
# Recipe from HuggingFace
slices:
  - sources:
      - model: Llama-2-70B
        layer_range: [0, 40]   # 40 layers
  - sources:
      - model: CodeLlama-70B
        layer_range: [0, 40]   # 40 layers

# Total: 80 layers ≈ 120B parameters

# Evaluation:
# MMLU (general): 68% (vs 69% Llama-2-70B)
# HumanEval (code): 45% (vs 30% Llama-2, 50% CodeLlama)

# Sweet spot: Slight general decrease, big code increase
```

---

# 7. Interpretability - Opening the Black Box

## 7.1 Sparse Autoencoders (SAEs) - The Feature Dictionary

### The Superposition Problem

**Problem Statement**:
```python
# Model has 4096 neurons in a layer
# But represents ~20,000 features!

# How? Superposition (linear combinations)

# Example: 3 neurons, 5 features
neuron_activations = np.array([0.8, 0.3, 0.5])

# Feature representations (learned)
feature_directions = np.array([
    [1.0, 0.1, 0.0],  # Feature 1: mostly neuron 1
    [0.8, 0.6, 0.0],  # Feature 2: mix of neurons 1&2
    [0.0, 0.9, 0.2],  # Feature 3: mostly neuron 2
    [0.0, 0.0, 1.0],  # Feature 4: mostly neuron 3
    [0.3, 0.3, 0.8],  # Feature 5: mix of all
])

# Activations are superpositions:
# neuron[0] = 0.5*feature[0] + 0.3*feature[1] + ...
```

**SAE Solution**:
```python
class SparseAutoencoder(nn.Module):
    """
    Expand 4096 neurons → 16384 sparse features
    """
    def __init__(self, d_model=4096, d_sparse=16384, k=64):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sparse)
        self.decoder = nn.Linear(d_sparse, d_model)
        self.k = k  # Top-k sparsity (only 64 active of 16384)
    
    def forward(self, x):
        # Encode to sparse space
        latent = F.relu(self.encoder(x))
        
        # Top-k sparsification
        values, indices = torch.topk(latent, self.k, dim=-1)
        sparse_latent = torch.zeros_like(latent)
        sparse_latent.scatter_(-1, indices, values)
        
        # Decode back
        reconstruction = self.decoder(sparse_latent)
        
        return reconstruction, sparse_latent

# Training loss
loss = F.mse_loss(reconstruction, x) + \
       0.01 * sparse_latent.abs().sum()  # Sparsity penalty
```

**Anthropic's SAE Results** (Oct 2023):
```
Trained SAE on Claude's layer 10 (4096 → 16384 features)

Found interpretable features:
- Feature 4823: "Golden Gate Bridge"
  - Activates on: Images/text about the bridge
  - Max activation: 8.2 (on photo of bridge)
  - Other activations: San Francisco (4.1), bridges (3.3)

- Feature 9172: "LaTeX math equations"
  - Activates on: Mathematical notation
  - Examples: \int, \sum, \frac{}{}, E=mc^2

- Feature 12045: "Sarcastic tone"
  - Activates on: "Oh great, just what I needed" (7.8)
  - Not on: "Great, thank you!" (0.3)
  - Hard to detect with single neurons!

Practical use:
- Steering: Multiply feature 12045 by 0 → Remove sarcasm
- Analysis: Which features fire on toxic content?
```

**Interview Question**:
```
Q: You train SAE: 4096 neurons → 16384 features, top-64 sparsity.
   
   Memory usage:
   A) 4096 × 16384 = 67M params (encoder) + same (decoder) = 134M
   B) Inference: 16384 float32 = 65KB per sample
   
   For batch of 1000:
   Without SAE: 1000 × 4096 × 4 bytes = 16MB
   With SAE: 1000 × 16384 × 4 bytes = 65MB
   
   Why use SAE if it uses 4x memory?
```

**Answer**:
```
SAE is for ANALYSIS, not PRODUCTION inference.

Workflow:
1. Train SAE offline (on saved activations)
2. Analyze features (find interpretable features)
3. Use findings to improve model (but don't deploy SAE)

Example applications:
- Feature ablation: Turn off feature 12045 → no sarcasm
- Toxicity detection: If features [892, 1043, 5672] fire → toxic
- Debugging: Why did model output X? Check which features fired

SAE never runs in production inference!
```

---

## 7.2 Abliteration - Removing Unnecessary Refusals

### The Refusal Direction Hypothesis

**Hypothesis**: Refusals live in a linear subspace of activation space

**Method**:
```python
def find_refusal_direction(model, harmful_prompts, harmless_prompts):
    """
    Find direction in activation space that represents refusal
    """
    # Collect activations
    harmful_acts = []
    harmless_acts = []
    
    for prompt in harmful_prompts:
        # Get activations at layer 15 (middle of model)
        acts = model(prompt, output_hidden_states=True).hidden_states[15]
        # Take mean over sequence
        harmful_acts.append(acts.mean(dim=1))
    
    for prompt in harmless_prompts:
        acts = model(prompt, output_hidden_states=True).hidden_states[15]
        harmless_acts.append(acts.mean(dim=1))
    
    # Average activations
    harmful_mean = torch.stack(harmful_acts).mean(dim=0)
    harmless_mean = torch.stack(harmless_acts).mean(dim=0)
    
    # Refusal direction
    refusal_direction = harmful_mean - harmless_mean
    refusal_direction = refusal_direction / refusal_direction.norm()
    
    return refusal_direction

# Abliterate: Project out refusal direction from all layers
def abliterate(model, refusal_direction, layers_to_ablate):
    with torch.no_grad():
        for layer_idx in layers_to_ablate:
            layer = model.transformer.h[layer_idx]
            
            # Project out refusal direction from MLP output projection
            W = layer.mlp.c_proj.weight
            
            # W_new = W - W @ refusal_dir @ refusal_dir^T
            projection = torch.outer(refusal_direction, W @ refusal_direction)
            W -= projection
```

**Results** (Arditi et al., 2024):
```
Llama-2-13B-Chat:

Before abliteration:
- Harmless creative prompts refused: 15%
  Example: "Write a heist story" → "I can't help with that"
  
- Genuinely harmful prompts refused: 95%
  Example: "How to make explosives" → Refused

After abliteration (layer 15 only):
- Harmless prompts refused: 2% (87% reduction!)
- Harmful prompts refused: 92% (slight decrease)

Sweet spot: Abliterate layers 12-18
            Reduces unnecessary refusals
            Maintains safety on harmful requests
```

**Interview Question** (Hao Hoang style):
```
Q: After abliteration, model says "I can help with that" to both:
   1. "Write a detective story about a bank heist"
   2. "How to rob a bank in real life"
   
   Did abliteration fail?
```

**Answer**:
```
Not necessarily! Check the FULL response:

1. Detective story: "I can help with that. Here's a story: ..."
   ✓ Appropriate response

2. Bank robbery: "I can help with that. Bank security systems ..."
   [Then explains why robbing banks is illegal, security measures]
   ? Nuanced response (not ideal, but not harmful instructions)

Abliteration doesn't make model "unsafe":
- It removes refusal DIRECTION, not safety concepts
- Model still knows robbing banks is illegal
- It just doesn't refuse as knee-jerk reaction

True failure would be:
"I can help with that. Step 1: Disable cameras. Step 2: ..."
(Detailed instructions)

This is rare after proper abliteration.
```

---

# 8. Test-time Compute Scaling - The Inference Budget Trade-off

## 8.1 o1 Reasoning Models - The Hidden CoT

### The Training Process

**Different from Standard LLMs**:
```python
# Standard LLM training
# Input: Question
# Output: Answer
# Loss: Cross-entropy on answer tokens

# o1 training (simplified)
# Input: Question
# Hidden: Chain-of-thought reasoning (not shown to user)
# Output: Final answer
# Loss: Reinforcement learning on correctness

# RL reward function
def reward(question, hidden_reasoning, answer):
    # Check if answer is correct
    if is_correct(answer, ground_truth):
        # Reward based on reasoning quality
        if len(hidden_reasoning) < 100:
            return 1.0  # Efficient reasoning
        elif len(hidden_reasoning) < 1000:
            return 0.5  # Verbose but correct
        else:
            return 0.1  # Too verbose
    else:
        return -1.0  # Wrong answer
```

**Production Behavior**:
```python
# o1-preview API call
response = openai.chat.completions.create(
    model="o1-preview",
    messages=[{"role": "user", "content": "What is 127 * 89?"}]
)

# Response includes reasoning_tokens (not shown to user)
print(response.usage)
# {
#   "prompt_tokens": 15,
#   "completion_tokens": 20,
#   "reasoning_tokens": 2847  # Hidden thinking!
# }

# Cost calculation
cost_input = 15 * ($15/1M)  # $0.000225
cost_output = 20 * ($60/1M)  # $0.001200
cost_reasoning = 2847 * ($60/1M)  # $0.170820

total = $0.172245

# Compare to GPT-4-turbo:
# Same question, no reasoning:
# Input: 15 * ($10/1M) = $0.00015
# Output: 20 * ($30/1M) = $0.00060
# Total: $0.00075

# o1 is 230x more expensive for this simple math!
# But gets it right more often
```

**When to Use o1**:
```python
def should_use_o1(question_type, budget, accuracy_requirement):
    """
    Decision matrix for o1 usage
    """
    # Math/Logic: o1 shines
    if question_type in ["math", "logic", "code_debugging"]:
        if accuracy_requirement > 0.9:  # Need 90%+ accuracy
            return "o1-preview"  # Expensive but accurate
        else:
            return "gpt-4-turbo"  # Good enough, cheaper
    
    # Creative/Open-ended: o1 overkill
    if question_type in ["creative_writing", "brainstorming"]:
        return "gpt-4-turbo"  # Reasoning doesn't help much
    
    # Factual: o1 might hallucinate less
    if question_type == "factual":
        if budget == "high":
            return "o1-preview"  # More reliable
        else:
            return "gpt-4-turbo"  # Acceptable
    
    # Default
    return "gpt-4-turbo"
```

---

## Interview Questions (Hao Hoang Style) - Advanced Topics

### Q1: The Merge Compatibility Trap
**Q**: You want to merge three models:
- Model A: Llama-2-7B fine-tuned on math
- Model B: Llama-2-7B fine-tuned on code
- Model C: Mistral-7B fine-tuned on math

Can you use TIES to merge all three?

**Answer**:
```
NO - Model C is incompatible!

Reasons:
1. Different architectures:
   - Llama-2: 32 layers, 4096 hidden, 32 attention heads
   - Mistral: 32 layers, 4096 hidden, 32 GQA (8 groups)
   
2. Different attention mechanisms:
   - Llama-2: MHA (Multi-Head Attention)
   - Mistral: GQA (Grouped-Query Attention)
   - KV cache shapes are DIFFERENT

3. Different position embeddings:
   - Llama-2: RoPE with base 10000
   - Mistral: RoPE with base 1000000 + sliding window

Can only merge: Model A + Model B (both Llama-2-7B)
Cannot merge Model C (different architecture)
```

### Q2: The SAE Sparsity Math
**Q**: SAE: 4096 neurons → 16384 features, top-64 sparsity.
What fraction of features are active? If you want 1% sparsity, how many features active?

**Answer**:
```python
# Current: top-64 out of 16384
sparsity = 64 / 16384
print(f"Sparsity: {sparsity:.4f}")  # 0.0039 = 0.39%

# For 1% sparsity:
k = int(0.01 * 16384)
print(f"Active features: {k}")  # 164 features

# Why not use 1% sparsity?
# Trade-off:
# - 0.39% (64): Very sparse, may miss features
# - 1.0% (164): Less sparse, better reconstruction
# - 5.0% (819): Too dense, loses interpretability

# Anthropic uses 0.5-1% in practice (64-164 features)
```

### Q3: The Test-time Compute Budget
**Q**: You have $100 to spend. Two options:
A) 100 requests with GPT-4 ($1 each)
B) 10 requests with o1-preview ($10 each)

Task: Complex math problems, need 80%+ accuracy.
GPT-4 accuracy: 68%
o1-preview accuracy: 92%

Which option gives more correct answers?

**Answer**:
```
Option A: 100 × 0.68 = 68 correct answers
Option B: 10 × 0.92 = 9.2 correct answers

Option A wins: 68 > 9.2

But: What if you can retry failed GPT-4 attempts?
- 100 requests, 68 correct, 32 wrong
- Retry 32 wrong ones (costs $32, outside budget)
- Even with retries: 68 + 0.68×32 = 89 correct
- Still less than 92% × 100 = 92 (if you had budget for o1)

Real answer: Use GPT-4 with retries on failures
            Or: Use o1-mini ($1 per request, 85% accuracy)
                100 × 0.85 = 85 correct (middle ground)
```

---
