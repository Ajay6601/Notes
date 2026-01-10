# ML Fundamentals - Part 1: Classical ML & Neural Network Basics

**Version 1.0 - Complete Interview-Ready Notes**

Sources: Chip Huyen's ML Interviews Book, Andriy Burkov's ML Engineering, Ian Goodfellow's Deep Learning, Fast.ai, Sebastian Raschka, Sunny Savita, Hao Hoang's materials, production case studies from Google, Meta, Netflix, Uber.

---

## 14.1 CLASSICAL ML

### Why Classical ML Still Matters in 2024+

**Real-world context**: Despite the LLM hype, 80% of production ML is still classical ML (Chip Huyen, 2023). Why?
- **Interpretability**: Credit scoring, healthcare (regulatory requirements)
- **Cost**: $0.001 per prediction vs $0.10 for LLM inference
- **Latency**: 1-10ms vs 100-1000ms for deep learning
- **Data efficiency**: Works with 100-1000 samples; deep learning needs 10K-1M+
- **Resource efficiency**: Runs on CPU; no GPU needed

**When to use Classical ML vs Deep Learning**:
```
Classical ML → Tabular data, <10K samples, need interpretability, low latency
Deep Learning → Images, text, audio, >100K samples, accuracy > interpretability
```

**Case Study - Uber's ETA Prediction (2015-2019)**:
- Initially tried LSTMs (failed - overfitting, slow)
- Switched to Gradient Boosting (XGBoost)
- Result: 15% better accuracy, 10x faster inference
- Learning: Tabular data (location, traffic, weather) → GB beats DL

---

### Linear Regression

**Core Concept**: Find best-fit line minimizing squared errors.

**Mathematical Foundation**:
```
Hypothesis: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
Cost Function: J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
Gradient: ∇J(θ) = (1/m) Xᵀ(Xθ - y)
Update: θ := θ - α∇J(θ)
```

**Closed-form Solution (Normal Equation)**:
```
θ = (XᵀX)⁻¹Xᵀy

When to use:
- Features < 10,000 (matrix inversion is O(n³))
- No regularization needed
- Small dataset (fits in memory)

When NOT to use:
- Features > 10,000 (use gradient descent)
- XᵀX is singular (use regularization)
- Online learning needed (use SGD)
```

**Assumptions (Critical for Interviews)**:
1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features aren't highly correlated

**Checking Assumptions**:
```python
# 1. Linearity: Scatter plot
# 2. Independence: Durbin-Watson test (should be ~2)
# 3. Homoscedasticity: Residual plot (should show no pattern)
# 4. Normality: Q-Q plot, Shapiro-Wilk test
# 5. Multicollinearity: VIF (Variance Inflation Factor) < 10
```

**Regularization Techniques**:

**Ridge Regression (L2)**:
```
Cost: J(θ) = MSE + λΣθⱼ²
Effect: Shrinks coefficients toward zero (never exactly zero)
Use when: Many features, multicollinearity
λ selection: Cross-validation (typical range: 0.01 to 100)
```

**Lasso Regression (L1)**:
```
Cost: J(θ) = MSE + λΣ|θⱼ|
Effect: Shrinks coefficients to exactly zero (feature selection)
Use when: Want sparse model, feature selection
Limitation: If features correlated, picks one arbitrarily
```

**Elastic Net (L1 + L2)**:
```
Cost: J(θ) = MSE + λ₁Σ|θⱼ| + λ₂Σθⱼ²
Effect: Combines benefits of Ridge and Lasso
Use when: Many correlated features + want feature selection
```

**Production Case Study - Netflix Prize (2006-2009)**:
- Challenge: Predict movie ratings
- Linear regression baseline: RMSE = 1.05
- Regularized linear models (Ridge): RMSE = 0.95
- Key learning: Feature engineering > complex models
- Features used: User avg rating, movie avg rating, time of day, day of week
- **Interview trap**: Don't jump to complex models without trying linear first!

**Common Interview Traps**:
1. **Trap**: "Linear regression always needs feature scaling"
   - **Truth**: Only for gradient descent. Normal equation doesn't need it.
   
2. **Trap**: "More features always improve accuracy"
   - **Truth**: Overfitting! Use regularization or feature selection.
   
3. **Trap**: "R² close to 1 means good model"
   - **Truth**: Could be overfitting. Check test R², not just train.

---

### Logistic Regression

**Core Concept**: Classification via probability estimation (NOT regression despite name).

**Mathematical Foundation**:
```
Hypothesis: h(x) = σ(θᵀx) where σ(z) = 1/(1 + e⁻ᶻ)
Decision boundary: θᵀx = 0
Cost Function: J(θ) = -(1/m) Σ[y log(h(x)) + (1-y)log(1-h(x))]
Gradient: ∇J(θ) = (1/m) Xᵀ(h(x) - y)  [same form as linear regression!]
```

**Why Log Loss (Not MSE)?**:
```
MSE with sigmoid → Non-convex (multiple local minima)
Log Loss with sigmoid → Convex (single global minimum)
Log Loss penalizes confident wrong predictions heavily
```

**Probability Interpretation**:
```
h(x) = 0.7 means:
- 70% probability of class 1
- 30% probability of class 0
- If threshold = 0.5, predict class 1
```

**Threshold Selection (Critical)**:
```
Default: 0.5 (assumes balanced classes)
Imbalanced data: Adjust based on cost

Example - Credit Card Fraud:
- False Positive (block legit transaction): $0 loss
- False Negative (miss fraud): $100 loss
- Optimal threshold: Lower than 0.5 (catch more fraud)

Threshold tuning:
1. Plot precision-recall curve
2. Plot ROC curve
3. Select threshold maximizing F1 or minimizing cost
```

**Multiclass Classification**:

**One-vs-Rest (OvR)**:
```
K classes → Train K binary classifiers
Classifier 1: Class 1 vs all others
Classifier 2: Class 2 vs all others
...
Prediction: argmax(P(class k))
```

**One-vs-One (OvO)**:
```
K classes → Train K(K-1)/2 classifiers
Example: 5 classes → 10 classifiers
More expensive but more accurate
```

**Softmax Regression (Multinomial)**:
```
h(x) = softmax(Wx + b)
softmax(z)ⱼ = e^(zⱼ) / Σe^(zₖ)
Properties: Σh(x) = 1, h(x) ∈ [0,1]
More efficient than OvR for many classes
```

**Production Case Study - Gmail Spam Detection (2004-2015)**:
- Initially: Naive Bayes
- 2010+: Logistic Regression with feature engineering
- Features: Word frequencies, sender reputation, link count, HTML ratio
- Regularization: L1 (Lasso) for feature selection
- Result: 99.9% accuracy, <0.1% false positive rate
- Key: Feature engineering (bag of words → TF-IDF → word embeddings)

**Case Study - LinkedIn Connection Recommendations**:
- Problem: Predict probability user accepts connection request
- Model: Logistic Regression (2010-2015), then upgraded to GBDT
- Features: Mutual connections, profile similarity, messaging history
- Threshold tuning: Balance precision (don't spam) vs recall (show opportunities)
- Challenge: Class imbalance (99% reject, 1% accept)
- Solution: Down-sample negatives, calibrate probabilities

**Calibration (Often Missed in Interviews)**:
```
Uncalibrated: Model outputs 0.7, but only 50% of such predictions are correct
Calibrated: Model outputs 0.7, and 70% of such predictions are correct

Calibration methods:
1. Platt Scaling: Fit logistic regression on outputs
2. Isotonic Regression: Non-parametric, monotonic
3. Beta Calibration: For imbalanced data

Why it matters:
- Healthcare: P(disease) must be accurate for treatment decisions
- Finance: P(default) used for risk calculations
- Ranking: Combine multiple models (need comparable probabilities)
```

**Interview Traps**:
1. **Trap**: "Logistic regression is for regression"
   - **Truth**: It's for classification (outputs probabilities)

2. **Trap**: "Always use 0.5 threshold"
   - **Truth**: Threshold depends on cost and class balance

3. **Trap**: "Can't handle multiclass"
   - **Truth**: OvR, OvO, or Softmax all work

4. **Trap**: "Linear decision boundary limits it"
   - **Truth**: Add polynomial features or use kernel trick

---

### Decision Trees

**Core Concept**: Recursive binary splits to partition feature space.

**Algorithm (CART - Classification and Regression Trees)**:
```
1. Select best feature and split point (max information gain)
2. Split data into left and right child nodes
3. Recursively repeat on child nodes
4. Stop when: max_depth reached, min_samples_leaf, no improvement
```

**Split Criteria**:

**For Classification**:
```
Gini Impurity: G = 1 - Σpᵢ²
- Range: [0, 0.5] (binary), [0, 1-1/K] (K classes)
- 0 = pure node, 0.5 = maximum impurity (binary)
- Fast to compute

Entropy: H = -Σpᵢ log₂(pᵢ)
- Range: [0, log₂(K)]
- 0 = pure node, log₂(K) = maximum impurity
- Slightly slower but more precise

Information Gain: IG = H(parent) - Σ(n_child/n_parent)·H(child)

Example:
Parent: [5+, 5-] → Gini = 1 - (0.5² + 0.5²) = 0.5
Split 1: [4+, 1-] and [1+, 4-] → Avg Gini = 0.32
Split 2: [5+, 0-] and [0+, 5-] → Avg Gini = 0.0 ← Best split!
```

**For Regression**:
```
MSE: (1/n) Σ(yᵢ - ȳ)²
MAE: (1/n) Σ|yᵢ - ȳ|

Variance Reduction: Var(parent) - Σ(n_child/n_parent)·Var(child)
```

**Hyperparameters (Critical for Interviews)**:
```
max_depth: Maximum tree depth
- Too high → Overfitting
- Too low → Underfitting
- Typical: 3-10 for interpretation, 10-20 for accuracy

min_samples_split: Minimum samples required to split
- Default: 2
- Increase for noisy data (10-50)

min_samples_leaf: Minimum samples in leaf node
- Default: 1
- Increase to prevent overfitting (5-20)

max_features: Number of features to consider for split
- Classification: sqrt(n_features)
- Regression: n_features/3
- Rationale: Decorrelates trees in Random Forest

max_leaf_nodes: Limit number of leaf nodes
- Alternative to max_depth
- Grows best-first (more balanced tree)
```

**Advantages**:
1. **Interpretable**: Can visualize decision path
2. **No feature scaling needed**: Works with raw features
3. **Handles mixed data**: Numeric and categorical
4. **Non-linear**: Captures complex patterns
5. **Feature importance**: Built-in ranking

**Disadvantages**:
1. **Overfitting**: Memorizes training data if not pruned
2. **Instability**: Small data change → completely different tree
3. **Greedy**: Locally optimal splits, not globally optimal
4. **Biased toward features with many levels**
5. **Poor extrapolation**: Can't predict outside training range

**Pruning**:
```
Pre-pruning (Early Stopping):
- Set max_depth, min_samples_leaf during training
- Faster but may stop too early

Post-pruning (Cost-Complexity Pruning):
- Grow full tree
- Remove nodes that don't improve validation score
- Controlled by α parameter
- More accurate but slower
```

**Production Case Study - Zillow Zestimate (Home Price Estimation)**:
- Initial model: Decision trees (2006)
- Problem: High variance (small data changes → big price changes)
- Solution: Ensemble methods (Random Forest, then Gradient Boosting)
- Features: Square footage, beds, baths, location, school district, crime rate
- Challenge: Trees can't extrapolate (predict prices for luxury homes)
- Fix: Cap predictions at 99th percentile, use linear regression for outliers

**Interview Traps**:
1. **Trap**: "Decision trees don't overfit"
   - **Truth**: They overfit badly without pruning!

2. **Trap**: "More depth always better"
   - **Truth**: Depth > 15 often overfits (memorizes noise)

3. **Trap**: "Can handle missing values natively"
   - **Truth**: Depends on implementation. Scikit-learn doesn't (as of 2024). XGBoost does.

4. **Trap**: "Feature importance from single tree is reliable"
   - **Truth**: Unstable. Use ensemble (Random Forest) for stable importance.

---

### K-Nearest Neighbors (KNN)

**Core Concept**: Classify based on majority vote of K nearest neighbors.

**Algorithm**:
```
Training: Store all data (lazy learning)
Prediction:
1. Compute distance to all training points
2. Find K nearest neighbors
3. Classification: Majority vote
   Regression: Average of K neighbors
```

**Distance Metrics**:
```
Euclidean: d = √(Σ(xᵢ - yᵢ)²)
- Most common
- Assumes all features equally important

Manhattan: d = Σ|xᵢ - yᵢ|
- Better when features on different scales
- More robust to outliers

Minkowski: d = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
- p=1: Manhattan
- p=2: Euclidean
- p=∞: Chebyshev

Cosine: similarity = (x·y)/(||x||×||y||)
- For high-dimensional sparse data (text)
- Ignores magnitude, focuses on direction
```

**Choosing K**:
```
K = 1: Overfitting (noise sensitive)
K = N: Underfitting (predicts majority class always)

Rule of thumb: K = √N
- For N=100: K ≈ 10
- For N=10,000: K ≈ 100

Cross-validation:
- Try K ∈ {1, 3, 5, 7, 11, 15, 21, 31, 51}
- Odd K prevents ties (for binary classification)
- Choose K minimizing validation error
```

**Optimization**:
```
Naive: O(N×d) per query (N samples, d dimensions)

KD-Tree: O(log N) per query
- Binary tree partitioning space
- Works well for d < 20
- Breaks down in high dimensions

Ball Tree: O(log N) per query
- Better than KD-Tree for d > 20
- Uses ball (hypersphere) instead of hyperplane

Approximate Nearest Neighbors (ANN):
- HNSW, Annoy, FAISS
- O(log N) with 95%+ accuracy
- Essential for large-scale (N > 1M)
```

**Advantages**:
```
+ Simple, no training phase
+ Non-parametric (no assumptions)
+ Naturally multi-class
+ Updates easily (just add data)
```

**Disadvantages**:
```
- Slow prediction (O(N) naive, O(log N) with tree)
- Memory intensive (stores all training data)
- Curse of dimensionality (d > 50)
- Sensitive to scale (must normalize)
- Sensitive to irrelevant features
```

**Production Note**:
```
Rarely used in production ML:
- Too slow for real-time (>10ms latency)
- Memory intensive (stores all data)
- Better alternatives exist (logistic regression, trees)

Use cases:
✓ Recommendation systems (collaborative filtering)
✓ Anomaly detection (if point far from all neighbors)
✓ Baseline model (simple, interpretable)
```

---

### Naive Bayes

**Core Concept**: Apply Bayes' theorem with "naive" independence assumption.

**Bayes' Theorem**:
```
P(y|X) = P(X|y)P(y) / P(X)

Where:
- P(y|X): Posterior probability (class given features)
- P(X|y): Likelihood (features given class)
- P(y): Prior probability (class frequency)
- P(X): Evidence (normalizing constant)

For classification:
ŷ = argmax P(y|X) = argmax P(X|y)P(y)

"Naive" assumption:
P(X|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
(features independent given class)
```

**Variants**:
```
Gaussian Naive Bayes:
- Continuous features
- P(xᵢ|y) ~ N(μy, σ²y)
- Use: Real-valued features (heights, weights, prices)

Multinomial Naive Bayes:
- Count features (word frequencies)
- P(xᵢ|y) ~ Multinomial(θy)
- Use: Text classification (bag of words, TF-IDF)

Bernoulli Naive Bayes:
- Binary features (word present/absent)
- P(xᵢ|y) ~ Bernoulli(py)
- Use: Document classification (word occurrence)
```

**Laplace Smoothing**:
```
Problem: P(xᵢ|y) = 0 if feature not seen in training
Solution: Add small constant α (typically 1)

P(xᵢ=k|y) = (count(xᵢ=k, y) + α) / (count(y) + α×K)

Where K = number of possible values for xᵢ
```

**Advantages**:
```
+ Fast training and prediction
+ Works well with small data
+ Handles high dimensions well
+ Naturally multi-class
+ Probabilistic predictions
```

**Disadvantages**:
```
- Independence assumption rarely holds
- Can't model feature interactions
- Poor probability estimates (overconfident)
- Sensitive to irrelevant features
```

**Production Case Study - Email Spam Classification**:
```
Problem: Classify emails as spam/ham
Features: 10K words (TF-IDF)
Model: Multinomial Naive Bayes

Training:
- 50K emails (5% spam)
- Training time: <1 second
- Model size: <1 MB

Results:
- Accuracy: 97%
- Precision: 92% (of flagged spam, 92% are spam)
- Recall: 85% (of actual spam, 85% caught)
- Inference: <1ms per email

Why Naive Bayes:
- High-dimensional sparse data (TF-IDF)
- Fast training and inference
- Good enough accuracy for email filtering
- Low resource requirements
```

---

## Interview Questions (Additional KNN, SVM, Clustering)

### Question 11: KNN vs Logistic Regression
**Q**: "You have 100K training samples and need <5ms inference latency. Should you use KNN or Logistic Regression?"

**Expected Answer**:
- **Choose Logistic Regression**

**Reasoning**:
1. **Latency**:
   - KNN: O(N) = 100K distance computations per prediction
     - With KD-Tree: O(log N) ≈ 17 comparisons
     - Still 5-10ms typical
   - Logistic Regression: O(d) where d = features
     - Typically <1ms for d<1000
   
2. **Memory**:
   - KNN: Stores all 100K samples (100-500 MB)
   - Logistic Regression: Stores weights only (1-10 MB)

3. **Scalability**:
   - KNN: Slows down as data grows
   - Logistic Regression: Constant inference time

**Only use KNN if**:
- Need non-linear boundaries (but use kernel SVM or tree-based instead)
- Data is small (<10K samples)
- Accuracy far more important than speed

---

### Question 12: SVM Kernel Selection
**Q**: "Your SVM with linear kernel gets 75% accuracy. RBF kernel gets 95% but takes 100x longer to train. Which do you use?"

**Expected Answer**:
- **It depends on data size and production requirements**

**Analysis**:
1. **Data size**:
   - Small (<10K samples): Use RBF (training time acceptable)
   - Large (>100K samples): Consider linear or switch to gradient boosting

2. **Accuracy requirements**:
   - Critical application (medical): Use RBF (95% accuracy worth it)
   - Non-critical (recommendation): Linear might suffice (75%)

3. **Training frequency**:
   - Train once: RBF (one-time cost)
   - Frequent retraining: Linear (daily updates feasible)

4. **Inference latency**:
   - Both similar inference time
   - RBF slightly slower (kernel computation)

**Best approach**:
1. Start with RBF on subset (10K samples)
2. If training too slow, try:
   - Linear SVM on full data
   - Gradient Boosting (often 90%+ accuracy, faster than RBF SVM)
   - Neural network (if have GPU)

**Production choice**: Gradient Boosting (XGBoost/LightGBM)
- Accuracy: Often matches or beats RBF SVM (90-95%)
- Training: Faster than RBF SVM on large data
- Inference: Similar or faster
- Handles non-linear without kernel tuning

---

### Question 13: K-Means Failure Mode
**Q**: "Your K-Means clustering produces poor results. Elbow plot shows no clear elbow. What's wrong and what do you try?"

**Expected Answer**:
- **Problem**: Data likely not well-suited for K-Means

**Diagnostics**:
1. **Visualize data** (PCA to 2D):
   - Check if clusters are spherical (K-Means assumption)
   - Check for non-convex shapes (K-Means fails)
   
2. **Check silhouette scores**:
   - Low scores (<0.3): Overlapping clusters
   - Negative scores: Wrong clustering

3. **Common causes**:
   - **Different densities**: DBSCAN instead
   - **Non-spherical clusters**: DBSCAN or Spectral Clustering
   - **Too many features**: Apply PCA first (reduce to 10-50 dims)
   - **Outliers**: Preprocess (remove outliers) or use K-Medoids

**Alternative algorithms**:
```
DBSCAN:
✓ Arbitrary cluster shapes
✓ Handles outliers
✗ Requires tuning ε and MinPts

Hierarchical Clustering:
✓ No need to specify K
✓ Dendrogram visualization
✗ O(n³) complexity (slow)

Gaussian Mixture Models (GMM):
✓ Soft clustering (probabilities)
✓ Handles elliptical clusters
✗ More complex (EM algorithm)

Spectral Clustering:
✓ Non-convex clusters
✓ Uses graph structure
✗ Slow (eigenvalue decomposition)
```

**Production approach**:
1. Try K-Means first (fastest)
2. If poor results, diagnose issue (visualize)
3. Switch algorithm based on data characteristics
4. Validate with domain knowledge (do clusters make sense?)

---

### Question 14: PCA Trap
**Q**: "You apply PCA to reduce 1000 features to 50. Accuracy drops from 90% to 60%. What went wrong?"

**Expected Answer**:
- **Most likely**: **Forgot to fit PCA on training set only**

**Common mistakes**:
```python
# WRONG: Fit PCA on all data (train + test)
pca = PCA(n_components=50)
X_all_reduced = pca.fit_transform(X_all)  # Data leakage!

# CORRECT: Fit PCA on training only
pca = PCA(n_components=50)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)  # Use training PCA
```

**Other potential issues**:
1. **Standardization**:
   - Forgot to standardize before PCA
   - Features with large variance dominate
   
2. **Information loss**:
   - 50 components explain <80% variance
   - Important features in low-variance components
   
3. **Non-linear relationships**:
   - PCA is linear (can't capture non-linear patterns)
   - Try Kernel PCA or keep original features

**Debugging steps**:
1. Check variance explained: `pca.explained_variance_ratio_.sum()`
   - If <90%, try more components
2. Check if standardized: `X.mean()` should be ~0
3. Verify train-only fitting (no data leakage)
4. Try without PCA (maybe 1000 features aren't too many)

**Production lesson**:
- PCA useful when n_features >> n_samples (curse of dimensionality)
- Not always needed (tree-based models handle high dims well)
- Always validate on held-out test set

---

### Random Forests

**Core Concept**: Ensemble of decision trees trained on bootstrapped data with random feature sampling.

**Algorithm (Breiman, 2001)**:
```
1. For b = 1 to B (number of trees):
   a. Draw bootstrap sample from training data (sample with replacement)
   b. Train decision tree on bootstrap sample
   c. At each split, randomly sample m features (typically sqrt(n_features))
   d. Choose best split from m features (not all features)
2. Prediction:
   - Classification: Majority vote
   - Regression: Average prediction
```

**Key Components**:

**Bootstrap Aggregating (Bagging)**:
```
Original data: N samples
Bootstrap sample: N samples drawn WITH replacement
→ ~63% unique samples, ~37% repeated (1 - 1/e ≈ 0.632)

Out-of-Bag (OOB) samples:
- Samples not in bootstrap (~37%)
- Used for validation (no need for separate validation set)
- OOB error ≈ validation error
```

**Random Feature Sampling**:
```
At each split:
- Classification: m = sqrt(n_features)
- Regression: m = n_features/3

Why it helps:
- Decorrelates trees (reduces variance)
- Prevents strong features from dominating
- Example: If one feature is very strong, all trees would use it
  → Trees become correlated → less benefit from ensembling
```

**Hyperparameters**:
```
n_estimators: Number of trees
- More trees → Better (but diminishing returns after ~100-500)
- No overfitting risk (unlike single tree depth)
- Typical: 100-500 trees

max_depth: Tree depth
- Default: None (grow until pure)
- Practical: 10-30 (balance accuracy vs speed)

min_samples_split: Minimum samples to split
- Default: 2
- Increase for noisy data: 10-50

min_samples_leaf: Minimum samples in leaf
- Default: 1
- Increase to smooth: 5-20

max_features: Features to sample per split
- "sqrt": sqrt(n_features) (classification)
- "log2": log2(n_features)
- float: fraction of features
- Smaller → More decorrelation, less overfitting

bootstrap: Whether to use bootstrap
- True: Bagging
- False: Pasting (use all data for each tree)

oob_score: Whether to compute OOB error
- True: Free validation estimate
- False: Faster training
```

**Feature Importance**:
```
Method 1: Mean Decrease in Impurity (MDI)
- Sum of impurity decreases for all splits using this feature
- Normalized across all features
- Fast but biased toward high-cardinality features

Method 2: Mean Decrease in Accuracy (MDA) / Permutation Importance
- Shuffle feature values, measure accuracy drop
- More reliable but slower
- Not biased by cardinality

Production tip:
- Use MDA for final importance ranking
- Use MDI for quick feature screening
```

**Advantages**:
1. **Robust**: Reduces overfitting vs single tree
2. **Handles high-dimensional data**: Works with many features
3. **Parallelizable**: Each tree trains independently
4. **OOB validation**: No need for separate validation set
5. **Handles missing data**: (XGBoost/LightGBM, not sklearn)
6. **Feature importance**: Identify key predictors

**Disadvantages**:
1. **Black box**: Less interpretable than single tree
2. **Memory intensive**: Stores B full trees
3. **Slow inference**: Must query B trees (vs 1 tree)
4. **Poor extrapolation**: Predictions bounded by training data
5. **Not optimal for very high-dimensional sparse data** (text → use linear models)

**Production Case Study - Airbnb Price Prediction**:
- Problem: Predict optimal listing price
- Model: Random Forest (500 trees, depth=20)
- Features: Location, property type, amenities, reviews, availability
- Why RF over single tree: Prices vary wildly (stability needed)
- Why RF over gradient boosting: Gradient boosting overfits to outliers
- Result: 85% of predictions within 15% of actual price
- Inference: 50ms (acceptable for pricing recommendation)

**Case Study - Kaggle Titanic (Classic Benchmark)**:
- Baseline (Decision Tree): 79% accuracy
- Random Forest (100 trees): 83% accuracy
- Feature importance: Sex > Age > Fare > Pclass
- Key learning: RF improves over single tree by ~4-5% typically

**Interview Traps**:
1. **Trap**: "More trees always better"
   - **Truth**: Diminishing returns after ~100-500 trees. Check OOB error plateau.

2. **Trap**: "Random Forest can't overfit"
   - **Truth**: It can (if trees too deep). Control via max_depth, min_samples_leaf.

3. **Trap**: "RF is always better than single tree"
   - **Truth**: For interpretability, use single tree (with pruning). For accuracy, use RF or GB.

4. **Trap**: "RF works well with high-cardinality categorical features"
   - **Truth**: Biased toward such features. One-hot encode or use target encoding.

---

### Gradient Boosting (XGBoost, LightGBM, CatBoost)

**Core Concept**: Sequentially build trees, each correcting errors of previous ensemble.

**Mathematical Foundation**:
```
Objective: Minimize L(y, F(x))
Algorithm:
1. Initialize: F₀(x) = argmin Σ L(yᵢ, γ)
2. For m = 1 to M:
   a. Compute negative gradient (pseudo-residuals):
      rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]|F=Fₘ₋₁
   b. Fit tree hₘ(x) to pseudo-residuals {rᵢₘ}
   c. Update: Fₘ(x) = Fₘ₋₁(x) + ν·hₘ(x)
      where ν = learning rate (shrinkage)
3. Output: F(x) = Σνhₘ(x)

Intuition:
- Tree 1: Learns main pattern
- Tree 2: Learns residuals of Tree 1
- Tree 3: Learns residuals of Tree 1+2
- ...
```

**Key Differences from Random Forest**:
```
Random Forest:
- Trees independent (parallel training)
- Full-depth trees
- Reduces variance (averaging)
- Bootstrap + random features

Gradient Boosting:
- Trees sequential (must train in order)
- Shallow trees (depth=3-8 typical)
- Reduces bias (iterative improvement)
- Learns from residuals
```

**XGBoost (Extreme Gradient Boosting)**:

**Key Innovations**:
```
1. Regularization:
   Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)
   Ω(f) = γT + (λ/2)Σwⱼ²
   where T = number of leaves, wⱼ = leaf weights

2. Sparsity-aware split finding:
   - Handles missing values natively
   - Learns optimal direction for missing values

3. Weighted quantile sketch:
   - Approximate split finding for large data
   - Reduces computation from O(#features × #samples) to O(#features × #buckets)

4. Cache-aware access:
   - Optimized data layout for CPU cache
   - Block structure for out-of-core computation

5. Column sampling:
   - Like Random Forest (decorrelates trees)
   - Per-tree, per-level, or per-split sampling
```

**Hyperparameters (Critical)**:
```
n_estimators (num_boost_round):
- Number of trees
- More → better (but diminishing returns + slower)
- Typical: 100-1000
- Use early stopping to find optimal

learning_rate (eta):
- Shrinkage factor (0, 1]
- Smaller → better generalization (but needs more trees)
- Typical: 0.01-0.3
- Rule: eta × n_estimators ≈ constant

max_depth:
- Tree depth
- Typical: 3-10 (shallow trees for boosting!)
- Deeper → more complex interactions (but overfits)

min_child_weight:
- Minimum sum of instance weight in child
- Higher → more conservative (less overfitting)
- Typical: 1-10

subsample:
- Fraction of samples for each tree
- Typical: 0.5-1.0
- <1.0 → stochastic boosting (like bagging)

colsample_bytree:
- Fraction of features per tree
- Typical: 0.3-1.0
- Decorrelates trees

gamma (min_split_loss):
- Minimum loss reduction for split
- Higher → more conservative
- Typical: 0-5

reg_alpha (L1):
- L1 regularization on weights
- Typical: 0-1

reg_lambda (L2):
- L2 regularization on weights
- Typical: 0-1
```

**LightGBM (Microsoft)**:

**Key Innovations**:
```
1. Gradient-based One-Side Sampling (GOSS):
   - Keep instances with large gradients (more informative)
   - Randomly sample instances with small gradients
   - Reduces data size while maintaining accuracy

2. Exclusive Feature Bundling (EFB):
   - Bundle mutually exclusive features (sparse features)
   - Example: One-hot encoded features
   - Reduces feature dimension

3. Histogram-based algorithm:
   - Bins continuous features into discrete buckets
   - Faster split finding: O(#buckets) vs O(#samples)
   - Typical: 255 buckets

4. Leaf-wise growth (vs level-wise):
   Level-wise (XGBoost): [1] → [1,1] → [1,1,1,1]
   Leaf-wise (LightGBM): [1] → [1,1] → [1,1,1] → [1,1,1,1]
   - More accurate (splits best leaf)
   - Risk: Overfitting if not controlled (use max_depth)
```

**When to use LightGBM**:
- Large datasets (>10K rows, >100 features)
- Need for speed (3-15x faster than XGBoost)
- Categorical features (native handling)
- Memory constraints (histogram bins save memory)

**CatBoost (Yandex)**:

**Key Innovations**:
```
1. Ordered Boosting:
   - Eliminates prediction shift (target leakage in gradients)
   - Uses different data orders for different trees
   - More robust but slower training

2. Native categorical handling:
   - Target encoding (ordered target statistics)
   - Combinations of categorical features
   - No need for one-hot or label encoding

3. Symmetric trees:
   - All leaves at same depth use same split condition
   - Faster inference (table lookup vs tree traversal)
```

**When to use CatBoost**:
- High-cardinality categoricals (user IDs, product IDs)
- Need for robustness (less overfitting)
- Small datasets (<10K rows)
- Baseline model (less hyperparameter tuning needed)

**Production Case Study - Uber Demand Forecasting**:
- Problem: Predict ride demand per region per 15-min window
- Model evolution:
  - 2014: Linear regression → MAPE 25%
  - 2015: Random Forest → MAPE 20%
  - 2016: XGBoost → MAPE 15%
  - 2018: LightGBM → MAPE 13% (3x faster)
- Features: Historical demand, events, weather, time, holidays
- Hyperparameters (LightGBM):
  - num_leaves: 31
  - learning_rate: 0.05
  - n_estimators: 500
  - max_depth: 8
- Inference: <5ms per prediction (required for real-time routing)

**Case Study - Kaggle Competitions**:
- **XGBoost**: Dominated 2014-2017 (17/29 winning solutions)
- **LightGBM**: Gaining traction 2018+ (speed advantage)
- **CatBoost**: Winning on small/medium datasets with categoricals
- Ensemble strategy: Blend all three (XGB + LGBM + CatBoost)

**Interview Traps**:
1. **Trap**: "Gradient boosting doesn't overfit"
   - **Truth**: It overfits easily! Use learning_rate, early_stopping, max_depth.

2. **Trap**: "XGBoost is always best"
   - **Truth**: LightGBM faster on large data. CatBoost better on small data + categoricals.

3. **Trap**: "More trees always better"
   - **Truth**: Overfits after optimal point. Use early_stopping_rounds.

4. **Trap**: "Hyperparameters don't matter much"
   - **Truth**: learning_rate × n_estimators is critical. max_depth controls complexity.

5. **Trap**: "Can handle missing values natively"
   - **Truth**: XGBoost and LightGBM yes. CatBoost learns during training.

---

### SVM (Support Vector Machines)

**Core Concept**: Find hyperplane that maximally separates classes.

**Mathematical Foundation**:
```
Primal problem:
min (1/2)||w||² + C Σξᵢ
subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

Where:
- w: Weight vector (defines hyperplane)
- b: Bias term
- ξᵢ: Slack variables (allow some misclassification)
- C: Regularization parameter (trade-off between margin and error)

Dual problem (used in practice):
max Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
subject to: 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0

Where:
- αᵢ: Lagrange multipliers (dual variables)
- K(xᵢ,xⱼ): Kernel function
```

**Kernel Trick**:
```
Linear kernel: K(x,z) = x·z
- Fast, interpretable
- Use when: Data linearly separable

Polynomial kernel: K(x,z) = (γx·z + r)^d
- Degree d (typically 2-4)
- Use when: Polynomial decision boundary

RBF (Gaussian) kernel: K(x,z) = exp(-γ||x-z||²)
- Most popular (handles non-linear well)
- γ: Controls influence of single training example
- Use when: Complex, non-linear boundary

Sigmoid kernel: K(x,z) = tanh(γx·z + r)
- Neural network-like
- Less common in practice
```

**Hyperparameters**:
```
C (Regularization):
- C → ∞: Hard margin (no misclassification allowed)
- C small: Soft margin (allow misclassification)
- Typical: 0.1 to 10 (tune via cross-validation)
- Trade-off: Margin width vs training error

γ (RBF kernel):
- γ large: Small influence radius (overfitting risk)
- γ small: Large influence radius (underfitting risk)
- Typical: 0.001 to 10
- Rule of thumb: 1/(n_features × variance)
```

**Advantages**:
```
+ Effective in high dimensions (n_features > n_samples)
+ Memory efficient (only stores support vectors)
+ Versatile (different kernels for different problems)
+ Robust to outliers (depends on C)
```

**Disadvantages**:
```
- Slow on large datasets (O(n²) to O(n³) training time)
- Requires feature scaling (sensitive to scale)
- Difficult to interpret (especially with non-linear kernels)
- Hyperparameter tuning crucial (C, γ)
- No probability estimates (by default)
```

**When to Use**:
```
✓ Medium-sized datasets (100 - 10K samples)
✓ High-dimensional data (text classification, genomics)
✓ Clear margin of separation exists
✓ Need robust classifier

✗ Large datasets (>10K samples, use linear SVM or logistic regression)
✗ Many noise/overlapping classes (use Random Forest)
✗ Need probability estimates (use logistic regression or calibrate SVM)
```

**Production Case Study - Text Classification (spam detection)**:
```
Problem: Email spam classification
Dataset: 5K emails, 10K features (TF-IDF)
Model: SVM with RBF kernel

Hyperparameters:
- C = 1.0
- γ = 0.01 (tuned via grid search)
- kernel = 'rbf'

Results:
- Accuracy: 98%
- Support vectors: 450 (9% of data)
- Training time: 2 seconds
- Inference: <1ms per email

Why SVM worked:
- High-dimensional sparse data (TF-IDF vectors)
- Clear separation (spam vs legitimate)
- Small dataset (5K samples)
```

---

### K-Means Clustering

**Core Concept**: Partition data into K clusters by minimizing within-cluster variance.

**Algorithm**:
```
1. Initialize: Randomly select K centroids
2. Repeat until convergence:
   a. Assignment: Assign each point to nearest centroid
   b. Update: Recompute centroids as mean of assigned points

Objective: min Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
Where:
- μᵢ: Centroid of cluster i
- Cᵢ: Set of points in cluster i
```

**Initialization Methods**:
```
Random:
- Select K random points as centroids
- Problem: Sensitive to initialization (different results each run)

K-Means++:
- Select first centroid randomly
- For remaining centroids:
  - Choose point with probability ∝ D(x)² (distance to nearest centroid)
- Result: Better initial centroids, faster convergence
- Standard in scikit-learn (default)

Multiple runs:
- Run K-Means N times with different initializations
- Choose run with lowest inertia (within-cluster sum of squares)
- Typical: n_init=10 in scikit-learn
```

**Choosing K (Number of Clusters)**:
```
Elbow method:
- Plot inertia vs K
- Look for "elbow" (diminishing returns)
- Problem: Elbow not always clear

Silhouette score:
- s(i) = (b(i) - a(i)) / max(a(i), b(i))
- a(i): Mean distance to points in same cluster
- b(i): Mean distance to points in nearest cluster
- Range: [-1, 1] (higher is better)
- Choose K maximizing average silhouette

Gap statistic:
- Compare inertia to random data
- Choose K where gap is maximized

Domain knowledge:
- Often best: Use business context (e.g., customer segments)
```

**Advantages**:
```
+ Simple, fast (O(n×K×d×iterations))
+ Scales to large datasets
+ Guaranteed to converge (to local minimum)
+ Easy to interpret
```

**Disadvantages**:
```
- Assumes spherical clusters (equal variance)
- Sensitive to initialization (use K-Means++)
- Sensitive to outliers (use K-Medoids instead)
- Must specify K in advance
- Only linear cluster boundaries
```

**Variants**:
```
K-Medoids (PAM):
- Uses actual data points as centroids (not means)
- More robust to outliers
- Slower (O(n²))

Mini-Batch K-Means:
- Uses random batches (faster on large data)
- Slightly lower quality but 10-100x faster
- Good for >100K samples

Fuzzy C-Means:
- Soft assignment (each point has probability for each cluster)
- Better for overlapping clusters
```

**Production Case Study - Customer Segmentation (E-commerce)**:
```
Problem: Segment 100K customers into groups
Features: Purchase frequency, avg order value, recency, category preferences
K = 5 (business decision: 5 marketing campaigns)

Preprocessing:
- Standardize features (mean=0, std=1)
- PCA: 20 features → 10 dimensions (retain 95% variance)

Model: K-Means++ (n_init=20, max_iter=300)

Results:
- Cluster 1: High-value frequent buyers (5%)
- Cluster 2: Occasional big spenders (15%)
- Cluster 3: Frequent small buyers (30%)
- Cluster 4: Inactive churned (20%)
- Cluster 5: One-time buyers (30%)

Business impact:
- Targeted campaigns per segment
- 20% increase in marketing ROI
```

---

### DBSCAN (Density-Based Clustering)

**Core Concept**: Group points that are closely packed, mark outliers as noise.

**Algorithm**:
```
Parameters:
- ε (eps): Neighborhood radius
- MinPts: Minimum points to form dense region

Point types:
- Core point: Has ≥ MinPts points within ε
- Border point: Within ε of core point, but not core itself
- Noise point: Not core, not within ε of any core

Algorithm:
1. For each unvisited point p:
   a. Mark p as visited
   b. Find all points within ε (neighborhood)
   c. If |neighborhood| < MinPts: Mark p as noise
   d. Else: Start new cluster
      - Add all points in neighborhood to cluster
      - Expand cluster recursively from each core point
```

**Advantages over K-Means**:
```
+ No need to specify K (number of clusters)
+ Can find arbitrarily shaped clusters (not just spherical)
+ Robust to outliers (marks them as noise)
+ Can find clusters of different densities
```

**Disadvantages**:
```
- Sensitive to ε and MinPts (requires tuning)
- Struggles with varying densities
- High-dimensional data (curse of dimensionality)
- O(n²) complexity (slow for large datasets)
  - Use Ball Tree or KD Tree for O(n log n)
```

**Choosing Parameters**:
```
ε (eps):
- Plot k-distance graph (distance to kth nearest neighbor)
- Look for "knee" in sorted distances
- Typical: ε = distance at knee

MinPts:
- Rule of thumb: MinPts = 2 × dimensions
- Minimum: MinPts = 3 (practical lower bound)
- Higher MinPts: Fewer, denser clusters

Grid search:
- Try multiple (ε, MinPts) combinations
- Evaluate with silhouette score or domain knowledge
```

**Production Case Study - Anomaly Detection (Network Intrusion)**:
```
Problem: Detect network intrusions (anomalous traffic patterns)
Data: 1M network connections, 41 features
Model: DBSCAN (ε=0.5, MinPts=5)

Preprocessing:
- Normalize features to [0,1]
- PCA: 41 → 15 dimensions

Results:
- Normal traffic: 3 dense clusters (98% of data)
- Anomalies: Noise points (2% of data)
- True positive rate: 94% (caught 94% of attacks)
- False positive rate: 3% (acceptable)

Why DBSCAN:
- Don't know number of attack types (K unknown)
- Attacks are outliers (low density)
- Normal traffic forms dense clusters
```

---

### PCA (Principal Component Analysis)

**Core Concept**: Dimensionality reduction via orthogonal linear transformation.

**Mathematical Foundation**:
```
Goal: Find directions of maximum variance

1. Center data: X̃ = X - mean(X)
2. Compute covariance matrix: C = (1/n)X̃ᵀX̃
3. Eigenvalue decomposition: C = VΛVᵀ
   - V: Eigenvectors (principal components)
   - Λ: Eigenvalues (variance explained)
4. Project: Z = X̃V_k (keep top k eigenvectors)

Variance explained by component i:
- λᵢ / Σλⱼ

Cumulative variance (first k components):
- Σⁱ₌₁ᵏ λᵢ / Σⱼ λⱼ
```

**Choosing Number of Components**:
```
Cumulative variance:
- Keep k components explaining ≥ 95% variance
- Plot cumulative variance vs k (scree plot)

Kaiser criterion:
- Keep components with λᵢ > 1 (for standardized data)
- Rule of thumb only

Elbow method:
- Plot eigenvalues vs component number
- Look for "elbow"

Cross-validation:
- Train model on k components
- Choose k minimizing validation error
```

**Advantages**:
```
+ Reduces dimensionality (speeds up training)
+ Removes correlated features
+ Visualization (project to 2D or 3D)
+ Noise reduction (small eigenvalues ≈ noise)
+ No hyperparameters (except k)
```

**Disadvantages**:
```
- Linear transformation only (can't capture non-linear)
- Loses interpretability (PCs are combinations of features)
- Sensitive to scale (must standardize first)
- Assumes variance = importance (not always true)
```

**Variants**:
```
Kernel PCA:
- Non-linear dimensionality reduction (RBF kernel)
- Can capture non-linear patterns
- Slower, less interpretable

Incremental PCA:
- For large datasets (doesn't fit in memory)
- Process data in batches
- Slightly less accurate but scalable

Sparse PCA:
- L1 penalty on loadings (sparse eigenvectors)
- More interpretable (fewer features per component)
```

**Production Case Study - Image Compression**:
```
Problem: Compress face images (64×64 = 4096 pixels)
Data: 10K face images

PCA:
- Components: 150 (explain 95% variance)
- Compression ratio: 4096 → 150 (27x reduction)

Results:
- Original: 4096 features
- PCA: 150 features
- Training time: 10x faster (SVM classifier)
- Accuracy: 95% (vs 96% with all features, acceptable loss)
- Storage: 27x smaller

Reconstruction quality:
- 50 components: Recognizable but blurry
- 150 components: Nearly identical to original
- 500 components: Indistinguishable from original
```

---

### Comparison: When to Use What

```
Decision Tree:
✓ Need interpretability (visualize tree, explain predictions)
✓ Small dataset (<1K rows)
✓ Quick baseline
✗ Accuracy not critical
✗ Unstable (high variance)

Random Forest:
✓ Need robustness (stable predictions)
✓ Moderate dataset (1K-100K rows)
✓ Have parallel compute (GPUs)
✓ Feature importance needed
✗ Not the highest accuracy
✗ Slow inference (B trees)

Gradient Boosting (XGBoost):
✓ Need highest accuracy
✓ Tabular data
✓ Medium dataset (10K-1M rows)
✓ Can tune hyperparameters
✗ Slower than RF/LGBM
✗ Sensitive to hyperparameters

LightGBM:
✓ Large dataset (>100K rows, >100 features)
✓ Need speed (training + inference)
✓ Have categorical features
✓ Memory constraints
✗ Small data (overfits)
✗ Fewer CPU cores (<4)

CatBoost:
✓ High-cardinality categoricals (user IDs, product IDs)
✓ Small-medium dataset (100-100K rows)
✓ Need robustness (less tuning)
✓ Baseline model (default params work well)
✗ Slower training than LGBM
✗ Large dataset (use LGBM)
```

**Production Decision Framework**:
```
1. Start with XGBoost (good default)
2. If too slow → LightGBM
3. If many categoricals → CatBoost
4. If need interpretability → Random Forest or single Decision Tree
5. If need stability over accuracy → Random Forest
6. Ensemble all three for Kaggle competitions
```

---

## Interview Questions (Hao Hoang Style)

### Question 1: Feature Importance Trap
**Q**: "Your Random Forest shows feature X has highest importance. Your XGBoost shows feature Y. Which is correct?"

**Expected Answer**:
- Neither is definitively "correct" - they measure different things
- RF importance: Mean decrease in impurity (biased toward high-cardinality)
- XGBoost importance: Weight (how often used), gain (average gain), or cover
- **Better approach**: Permutation importance (model-agnostic)
- **Production practice**: Use SHAP values for consistent, interpretable importance
- Red flag: Trusting single-model importance without validation

**Follow-up**: "How would you validate feature importance?"
- Check consistency across multiple models
- Permutation importance on validation set
- SHAP values
- Remove feature and check accuracy drop
- Check correlation with target

---

### Question 2: Overfitting Detection
**Q**: "Your XGBoost gets 99% accuracy on training, 85% on validation. Is this overfitting? What do you do?"

**Expected Answer**:
- Yes, 14% gap indicates overfitting
- **Diagnosis**:
  - Check learning curves (train vs validation)
  - Check if more data helps (add 20% data, see if gap closes)
- **Solutions** (in order of preference):
  1. **Early stopping**: Stop when validation score plateaus (most effective)
  2. **Reduce model complexity**: Lower max_depth (8→5), increase min_child_weight
  3. **Regularization**: Increase reg_alpha, reg_lambda
  4. **Add more data**: If train curve hasn't plateaued
  5. **Feature selection**: Remove noisy features (check importance)
  6. **Lower learning rate**: More stable but needs more trees
- **Red flag**: Jumping to "add regularization" without checking learning curves

**Follow-up**: "After fixing overfitting, validation accuracy is 88% but test is 80%. What happened?"
- **Validation set overfitting**: Tuned hyperparameters on validation, now validation ≠ test
- **Solution**: Use K-fold cross-validation for hyperparameter tuning
- **Or**: Keep test set truly held-out (no tuning decisions based on test)

---

### Question 3: Gradient Boosting Computation
**Q**: "You're training XGBoost with 100 trees, depth 5, on 1M samples. Estimate training time if one tree takes 10 seconds."

**Expected Answer**:
- **Naive**: 100 trees × 10 sec = 1000 sec = 16.7 minutes
- **Actual**: Slower due to:
  - Each tree trains on residuals (requires prediction from previous ensemble)
  - Overhead: Split finding, histogram building
  - Better estimate: 100 × 12 sec = 20 minutes
- **Speedups**:
  - Use `tree_method='hist'` (histogram-based): 2-3x faster
  - Subsample data (subsample=0.8): 1.25x faster
  - Use fewer features (colsample_bytree=0.8): 1.25x faster
  - Switch to LightGBM: 3-15x faster
- **Red flag**: Not considering sequential nature of boosting

---

### Question 4: Bias-Variance Trade-off
**Q**: "Your model has high bias. Would you use Random Forest or Gradient Boosting? Why?"

**Expected Answer**:
- **High bias** = underfitting (training accuracy low)
- **Solution**: Need more complex model
- **Gradient Boosting** (correct choice):
  - Reduces bias iteratively
  - Each tree corrects previous errors
  - Can overfit (high variance) but controlled via learning rate, early stopping
- **Random Forest** (wrong choice):
  - Reduces variance (averaging)
  - Doesn't reduce bias much (trees see same patterns)
  - Use when already have low bias but high variance
- **Interview tip**: Be clear on bias-variance framework

**Follow-up**: "What if model has high variance?"
- Use Random Forest (variance reduction via averaging)
- Or use Gradient Boosting with regularization (learning_rate, max_depth, early_stopping)
- Add more training data (variance decreases as N increases)

---

### Question 5: Categorical Encoding
**Q**: "You have a categorical feature 'City' with 1000 unique values. One-hot encoding gives 1000 features. Your XGBoost crashes. What do you do?"

**Expected Answer**:
- **Problem**: High-cardinality categorical → sparse data → memory explosion
- **Solutions**:
  1. **Target encoding**: Mean target per category (watch for leakage!)
  2. **Frequency encoding**: Replace with count of occurrences
  3. **CatBoost**: Native categorical handling (ordered target statistics)
  4. **Embedding**: Learn low-dimensional representation (neural net)
  5. **Grouping**: Merge rare categories (e.g., cities with <100 samples → "Other")
  6. **Leave-one-out encoding**: Target mean excluding current row
  7. **Hash encoding**: Hash to fixed number of buckets (collisions OK)
- **Red flag**: Blindly one-hot encoding without checking cardinality
- **Production**: CatBoost or target encoding most common

**Follow-up**: "How do you prevent target leakage in target encoding?"
- Use cross-validation: Encode each fold using other folds' statistics
- Add smoothing: Weighted average of category mean and global mean
- Use CatBoost (handles this automatically with ordered target statistics)

---

### Question 6: Missing Values
**Q**: "Your dataset has 30% missing values in a key feature. XGBoost handles missing values. Do you still need to impute?"

**Expected Answer**:
- **It depends on why data is missing**:
  - **MCAR** (Missing Completely at Random): XGBoost learns optimal direction automatically → No imputation needed
  - **MAR** (Missing at Random): Depends on other features → XGBoost can handle
  - **MNAR** (Missing Not at Random): Missingness is informative → Create "is_missing" indicator + let XGBoost learn
- **Experimentation**:
  - Try no imputation (let XGBoost handle)
  - Try mean/median imputation
  - Try advanced imputation (KNN, MICE)
  - Compare validation scores
- **Production tip**: XGBoost's native handling often works best
- **Red flag**: Imputing without checking if it helps

**Follow-up**: "What if you're using Logistic Regression instead?"
- Must impute (LR doesn't handle missing values)
- Options: Mean/median, KNN, model-based (MICE), or add "is_missing" flag

---

### Question 7: Class Imbalance
**Q**: "Your fraud detection dataset has 99.9% legitimate, 0.1% fraud. Your Random Forest gets 99.9% accuracy. Is this good?"

**Expected Answer**:
- **No! Accuracy is misleading with imbalance**
- **Problem**: Model predicts "legitimate" for everything → 99.9% accuracy but 0% fraud caught
- **Correct metrics**:
  - **Precision**: Of predicted fraud, how many are actual fraud? (avoid false alarms)
  - **Recall**: Of actual fraud, how many did we catch? (don't miss fraud)
  - **F1-score**: Harmonic mean of precision and recall
  - **PR-AUC**: Area under precision-recall curve (better than ROC-AUC for imbalanced)
- **Solutions**:
  1. **Resampling**:
     - Undersample majority (lose data but faster)
     - Oversample minority (SMOTE, ADASYN)
     - Hybrid (undersample + oversample)
  2. **Class weights**:
     - `class_weight='balanced'` (sklearn)
     - `scale_pos_weight` (XGBoost) = #negatives / #positives
  3. **Threshold tuning**: Lower decision threshold (catch more fraud, accept more false alarms)
  4. **Ensemble**: Train multiple models on different resampled datasets
  5. **Focal loss**: Focuses on hard examples (deep learning)
- **Production**: Class weights + threshold tuning most common

**Follow-up**: "How do you choose the decision threshold?"
- Plot precision-recall curve
- Select threshold based on business cost:
  - False positive cost (block legit transaction): $0
  - False negative cost (miss fraud): $100
  - Optimal threshold: Minimize expected cost
- Or: Maximize F1-score if costs unknown

---

### Question 8: Inference Speed
**Q**: "Your XGBoost model has 500 trees, depth 10. Inference takes 100ms. You need <10ms. What do you do?"

**Expected Answer**:
- **Diagnosis**:
  - 500 trees × 2^10 leaves/tree = 512K leaf lookups per prediction
  - Each lookup: ~0.0002ms → Total ≈ 100ms (matches)
- **Solutions** (in order of impact):
  1. **Reduce trees**: 500 → 100 (5x speedup, slight accuracy loss)
     - Use early stopping to find optimal
  2. **Reduce depth**: 10 → 6 (4x faster, 64 leaves vs 1024)
  3. **Model compression**: Convert to ONNX (1.5-2x speedup)
  4. **Quantization**: FP32 → FP16 or INT8 (2x speedup)
  5. **Approximate**: Use first N trees only (e.g., 50 trees for quick estimate)
  6. **Switch model**: Use LightGBM (symmetric trees → faster lookup)
  7. **Distillation**: Train smaller model (depth 5, 50 trees) to mimic large model
  8. **Hardware**: GPU inference (10-100x speedup for large batches)
- **Production**: Usually reduce depth + early stopping gets to <10ms

**Follow-up**: "After optimization, accuracy drops 2%. Is this acceptable?"
- **Depends on application**:
  - Real-time bidding: Speed critical, 2% loss OK
  - Medical diagnosis: Accuracy critical, 100ms acceptable
- **Framework**: Always measure business impact, not just accuracy

---

### Question 9: Hyperparameter Tuning
**Q**: "You need to tune 5 hyperparameters for XGBoost. You have 1 hour of compute budget. What's your strategy?"

**Expected Answer**:
- **Not Grid Search**: 5 params × 5 values each = 3125 combinations (too many)
- **Strategy**:
  1. **Manual tuning first** (15 min):
     - Start with defaults
     - Tune most important params manually:
       - n_estimators: 100 → 500 (use early stopping)
       - learning_rate: 0.3 → 0.1
       - max_depth: 6 → 5
  2. **Random Search** (30 min):
     - Sample 50-100 random combinations
     - More efficient than grid (Bergstra & Bengio, 2012)
     - Covers broader space
  3. **Bayesian Optimization** (15 min):
     - Use Optuna or Hyperopt
     - Intelligent sampling (explores promising regions)
     - 20-30 trials typically enough
- **Key**: Most gains from n_estimators, learning_rate, max_depth
- **Red flag**: Grid searching all combinations (wastes compute)

**Follow-up**: "Which hyperparameters matter most?"
- **XGBoost**: learning_rate, max_depth, n_estimators (80% of gains)
- **LightGBM**: num_leaves, learning_rate (leaf-wise growth sensitive)
- **Random Forest**: n_estimators, max_depth (but RF less sensitive overall)

---

### Question 10: Production Debugging
**Q**: "Your XGBoost model performs well on validation (85% accuracy) but poorly in production (70%). What do you check?"

**Expected Answer**:
- **Training-serving skew** (most common):
  1. **Feature computation difference**:
     - Validation: Features from batch pipeline (correct)
     - Production: Features from real-time API (bug?)
     - **Check**: Log production features, compare to validation
  2. **Data distribution shift**:
     - Validation: January data
     - Production: July data (seasonality)
     - **Check**: Plot feature distributions (train vs prod)
  3. **Missing values handled differently**:
     - Validation: Imputed with train mean
     - Production: Imputed with zero (bug)
     - **Check**: Missing value counts
  4. **Categorical encoding mismatch**:
     - Validation: Target encoding with full dataset
     - Production: Target encoding with partial data (different means)
     - **Check**: Log encoded values
  5. **Feature leakage in training** (caught by prod):
     - Training: Accidentally used future information
     - Production: Future info not available
     - **Check**: Recreate validation with production logic
- **Debugging process**:
  1. Log production inputs and predictions
  2. Sample 100 prod inputs, run through validation pipeline
  3. Compare predictions (should match)
  4. If different, inspect feature-by-feature
- **Prevention**: Shadow mode (run prod pipeline on validation data before deploying)

**Follow-up**: "Production accuracy recovered to 83% but still below validation 85%. Acceptable?"
- **Probably yes** - validation set may have been "easier" or model slightly overfit
- **Monitor**: If drops further, investigate
- **Baseline**: Track validation → test → production gap over time

---

## Key Takeaways for Interviews

### Classical ML Still Dominates Production
- 80% of ML in production is classical ML (Chip Huyen, 2023)
- Why: Interpretability, cost, latency, data efficiency
- Know when to use each algorithm (decision framework)

### Gradient Boosting vs Random Forest
- **GB**: Higher accuracy, more sensitive, slower
- **RF**: More robust, less tuning, parallelizable
- **Production**: Start with XGBoost, switch to LightGBM if too slow

### Interview Red Flags to Avoid
1. Jumping to complex models without trying linear first
2. Using accuracy for imbalanced data
3. Not checking for overfitting (train-val gap)
4. Grid searching 5+ hyperparameters
5. Not considering training-serving skew

### Production Lessons from Big Tech
- **Uber**: XGBoost → LightGBM for speed (3x faster)
- **Netflix**: Feature engineering > complex models
- **LinkedIn**: Threshold tuning based on business cost
- **Airbnb**: Random Forest for stability over accuracy

---

**Next Document**: Part 2 will cover Neural Networks (14.2), CNNs (14.3), RNNs (14.4), Loss Functions (14.5), and Evaluation Metrics (14.6).
