# Section 12: MLOps and Infrastructure - Complete Notes

**Version 1.0 - Comprehensive Production Guide**
**Sources**: Chip Huyen (Designing ML Systems), MLOps books, Google SRE, company engineering blogs, production case studies

---

## 12.1 Training Infrastructure

### Overview
Training infrastructure is the backbone of ML development. At scale, companies train hundreds to thousands of models simultaneously, requiring robust orchestration, resource management, and experiment tracking systems.

**Key Challenges**:
- Multi-user resource contention (10+ teams competing for GPUs)
- Long-running jobs (days to weeks for large models)
- Experiment reproducibility across different hardware
- Cost management ($100K+ monthly GPU bills)
- Fair resource allocation

---

### Cluster Management

#### **Kubernetes for ML Workloads**

**Why Kubernetes?**
- Container orchestration at scale
- Declarative configuration (YAML manifests)
- Auto-scaling based on workload
- Multi-tenancy support (namespace isolation)
- Cloud-agnostic (AWS EKS, GCP GKE, Azure AKS)

**Key Components**:
1. **Pods**: Smallest deployable unit (1+ containers)
2. **Nodes**: Worker machines (GPU nodes, CPU nodes)
3. **Services**: Load balancing and service discovery
4. **Persistent Volumes**: Storage for datasets/checkpoints
5. **ConfigMaps/Secrets**: Configuration and credentials

**ML-Specific Kubernetes Tools**:
- **Kubeflow**: End-to-end ML platform on K8s
  - Pipelines (workflow orchestration)
  - Katib (hyperparameter tuning)
  - Training operators (TFJob, PyTorchJob)
- **KubeFlow Serving**: Model deployment
- **Volcano**: Batch scheduling for ML jobs
- **NVIDIA GPU Operator**: GPU resource management

**Production Example (Spotify)**:
- 2000+ K8s nodes for ML workloads
- Dynamic GPU allocation (1-64 GPUs per job)
- Auto-scaling during peak training times
- Cost: 40% reduction via spot instances + right-sizing

**Kubernetes Challenges**:
- Complexity (steep learning curve)
- Overhead (pod scheduling latency 5-30 seconds)
- GPU utilization gaps (pods don't always fill GPUs 100%)
- Networking complexity (multi-node training)

---

#### **Slurm (Simple Linux Utility for Resource Management)**

**Why Slurm?**
- HPC-focused (designed for supercomputers)
- Lower overhead than Kubernetes
- Better for tightly-coupled distributed jobs (MPI, NCCL)
- Simpler than K8s for pure training workloads
- Used by academic institutions and research labs

**Architecture**:
- **slurmctld**: Controller daemon (master)
- **slurmd**: Compute node daemon (workers)
- **srun**: Launch parallel jobs
- **sbatch**: Submit batch jobs
- **squeue**: View job queue

**Key Features**:
1. **Fair-share scheduling**: Allocate resources based on usage history
2. **Priority queues**: High-priority jobs preempt low-priority
3. **Resource limits**: CPU/GPU/memory quotas per user/group
4. **Job arrays**: Submit thousands of similar jobs
5. **Gang scheduling**: Reserve all resources before starting (no partial starts)

**Slurm Job Submission Example**:
```bash
#!/bin/bash
#SBATCH --job-name=llm_training
#SBATCH --nodes=4              # 4 nodes
#SBATCH --gres=gpu:8           # 8 GPUs per node (32 total)
#SBATCH --time=48:00:00        # Max 48 hours
#SBATCH --partition=gpu        # GPU partition
#SBATCH --qos=high             # High priority

srun python train_llm.py --distributed
```

**Production Example (OpenAI - reported)**:
- 10,000+ GPU cluster managed by Slurm
- Multi-day training jobs (GPT-3: weeks)
- Checkpoint/restart for fault tolerance
- Priority-based scheduling (research vs production)

**Slurm vs Kubernetes**:
| Feature | Slurm | Kubernetes |
|---------|-------|------------|
| **Use Case** | Training (HPC) | Training + Serving |
| **Overhead** | Low | Medium |
| **Complexity** | Medium | High |
| **Multi-tenancy** | User-based | Namespace-based |
| **GPU Support** | Native | Via plugins |
| **Dynamic Scaling** | Limited | Excellent |
| **Best For** | Research labs | Production ML platforms |

**Decision Criteria**:
- **Slurm**: Pure training workloads, HPC environment, lower complexity
- **Kubernetes**: Full ML lifecycle, cloud-native, microservices architecture

---

### Resource Allocation and Scheduling

#### **Resource Types**
1. **Compute**: CPUs, GPUs (A100, H100, V100)
2. **Memory**: RAM (256GB-2TB per node), GPU memory (40GB-80GB)
3. **Storage**: Local SSD (fast I/O), NFS (shared datasets), object storage (S3)
4. **Network**: InfiniBand (400Gbps), Ethernet (100Gbps)

#### **Scheduling Strategies**

**1. First-Come-First-Served (FCFS)**
- Simple, fair
- Problem: Long jobs block short jobs (convoy effect)
- Used in small teams (<10 users)

**2. Fair-Share Scheduling**
- Track historical usage per user/team
- Users with lower past usage get higher priority
- Formula: `priority = base_priority / (1 + usage_factor)`
- **Example**: User A used 100 GPU-hours, User B used 10 → User B gets priority

**3. Priority Queues**
- Multiple queues: high, medium, low
- High-priority jobs can preempt low-priority
- **Example** (Netflix):
  - Production models: High priority
  - Experimentation: Medium priority
  - Hyperparameter sweeps: Low priority (preemptible)

**4. Backfilling**
- Run small jobs in gaps left by large jobs
- Improves GPU utilization (80% → 95%)
- Example: 64-GPU job waiting → run 8-GPU jobs in the meantime

**5. Gang Scheduling**
- All resources must be available before starting
- Prevents deadlocks in distributed training
- Used for multi-node jobs (8+ nodes)

#### **Resource Quotas**

**Per-User Quotas**:
```
User: alice
- Max GPUs: 16 (at once)
- Max CPU: 128 cores
- Max jobs: 50 (queued)
- Storage: 10TB
```

**Per-Team Quotas**:
```
Team: ML Research
- Max GPUs: 128 (shared among team)
- Max memory: 2TB
- Priority: Medium
- Budget: $50K/month
```

**Dynamic Quotas**:
- Increase quotas for teams near deadlines
- Burst capacity: Allow temporary over-quota (150% for 24 hours)
- Idle resource reclamation: Unused quotas redistributed

---

### Job Scheduling (Priorities, Queues)

#### **Job Lifecycle**
1. **Submitted**: Job enters queue
2. **Pending**: Waiting for resources
3. **Running**: Actively training
4. **Completed**: Finished successfully
5. **Failed**: Error occurred
6. **Preempted**: Interrupted by higher-priority job

#### **Priority Calculation**

**Multi-factor Priority**:
```
priority = base_priority + 
           urgency_factor + 
           fair_share_adjustment - 
           usage_penalty + 
           deadline_boost
```

**Example**:
- Base priority: 100 (team level)
- Urgency: +50 (critical production model)
- Fair-share: +30 (team under-utilized GPUs this week)
- Usage penalty: -20 (this user ran many jobs recently)
- Deadline boost: +100 (job due in 2 days)
- **Final priority: 260**

#### **Queue Management**

**Multiple Queues**:
1. **Interactive queue**: Short jobs (<1 hour), high priority
   - Use case: Debugging, prototyping
   - Limit: 4 GPUs max
2. **Batch queue**: Long jobs (1-48 hours), medium priority
   - Use case: Full training runs
   - Limit: 64 GPUs max
3. **Preemptible queue**: Any duration, low priority
   - Use case: Hyperparameter sweeps, low-cost training
   - Limit: Unlimited (but can be killed anytime)
   - Cost: 50-80% cheaper (spot instances)

**Production Example (Uber)**:
- 3 queues: critical (production models), standard (research), spot (experiments)
- Critical queue: 5-minute SLA for job start
- Standard queue: 30-minute SLA
- Spot queue: Best-effort (no SLA)

#### **Preemption Policies**

**Checkpoint-Based Preemption**:
1. Job checkpoints every 10 minutes
2. High-priority job arrives
3. Low-priority job receives SIGTERM (graceful shutdown)
4. Job saves checkpoint and exits
5. Job re-queued with saved state

**Live Migration** (advanced):
- Move running job to different node
- Used in Google's Borg system
- Requires specialized infrastructure

---

### Experiment Tracking (MLflow, W&B, Neptune)

**Why Experiment Tracking?**
- Track 100s-1000s of experiments
- Reproduce results (code version, data version, hyperparameters)
- Compare models (which config performed best?)
- Collaboration (share results with team)
- Debugging (what went wrong in run #374?)

#### **MLflow**

**Components**:
1. **Tracking**: Log parameters, metrics, artifacts
2. **Projects**: Reproducible runs (conda env, Docker)
3. **Models**: Package models for deployment
4. **Registry**: Version and stage models (staging, production)

**Key Features**:
- Open-source (Apache 2.0)
- Language-agnostic (Python, R, Java, REST API)
- Pluggable backend (local, S3, Azure, GCS)
- UI for visualization

**Logging Example**:
```python
import mlflow

mlflow.start_run()
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)
mlflow.log_metric("train_loss", 0.5, step=100)
mlflow.log_metric("val_accuracy", 0.92, step=100)
mlflow.log_artifact("model.pth")  # Save model file
mlflow.end_run()
```

**Production Use** (Databricks):
- 10M+ experiments tracked
- Integration with Spark for distributed training
- Auto-logging for popular frameworks (PyTorch, TensorFlow)

**Pros**:
- Simple, lightweight
- Self-hosted option (no vendor lock-in)
- Good for small-to-medium teams

**Cons**:
- Limited collaboration features (no comments, sharing)
- UI less polished than commercial tools
- No built-in hyperparameter optimization

---

#### **Weights & Biases (W&B)**

**Key Features**:
1. **Real-time metric tracking**: Live dashboard updates
2. **System metrics**: GPU utilization, memory, CPU (automatic)
3. **Media logging**: Images, audio, video, tables
4. **Hyperparameter sweeps**: Integrated Bayesian optimization
5. **Collaboration**: Comments, reports, sharing
6. **Artifact tracking**: Dataset versioning, model versioning

**Advanced Features**:
- **Sweeps**: Automated hyperparameter search
- **Reports**: Publish findings (like Jupyter notebook)
- **Alerts**: Slack/email when metrics exceed thresholds
- **Model registry**: Track model lineage

**Example Integration**:
```python
import wandb

wandb.init(project="llm-training", config={"lr": 0.001})
wandb.config.update({"batch_size": 32})

for epoch in range(10):
    loss = train_epoch()
    wandb.log({"loss": loss, "epoch": epoch})
    
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(...)})
wandb.finish()
```

**Production Example (OpenAI - public mentions)**:
- Track GPT model training (loss curves, gradient norms)
- System metrics (GPU utilization across 1000s of GPUs)
- Alert on anomalies (loss spike, NaN gradients)

**Pros**:
- Best-in-class UI/UX
- Excellent collaboration features
- Strong community and tutorials
- Free tier for individuals

**Cons**:
- Cloud-only (no self-hosted option for free tier)
- Costs scale with usage
- Some features locked behind paid tiers

---

#### **Neptune.ai**

**Key Features**:
1. **Metadata store**: Centralized tracking for all ML metadata
2. **Model registry**: Built-in versioning and staging
3. **Notebooks integration**: Track Jupyter experiments
4. **Team collaboration**: Project-level access control
5. **Custom dashboards**: Flexible metric visualization

**Differentiation**:
- More structured than W&B (better for compliance/audit)
- Better for large teams (100+ ML engineers)
- Stronger model registry features
- Enterprise-focused (SSO, RBAC, audit logs)

**Example**:
```python
import neptune

run = neptune.init_run(project="team/llm")
run["parameters"] = {"lr": 0.001, "batch_size": 32}
run["train/loss"].append(0.5)
run["sys/gpu_usage"].append(85.5)
run.stop()
```

**Pros**:
- Enterprise-grade (compliance, security)
- Excellent for regulated industries (finance, healthcare)
- Strong model governance features

**Cons**:
- Steeper learning curve
- More expensive than W&B
- Smaller community

---

#### **Tool Comparison**

| Feature | MLflow | W&B | Neptune |
|---------|--------|-----|---------|
| **Cost** | Free (self-hosted) | Free tier + paid | Paid (free tier limited) |
| **Deployment** | Self-hosted or cloud | Cloud only | Cloud or self-hosted |
| **UI Quality** | Basic | Excellent | Very good |
| **Collaboration** | Limited | Excellent | Excellent |
| **Compliance** | DIY | Limited | Strong |
| **Hyperparameter Tuning** | No (use Optuna) | Built-in (Sweeps) | Limited |
| **Best For** | Small teams, cost-conscious | Research, startups | Enterprises, regulated |

**Decision Tree**:
- **Budget-constrained or need self-hosting**: MLflow
- **Fast prototyping, research-focused**: W&B
- **Enterprise with compliance needs**: Neptune

**Multi-Tool Strategy** (Common in large orgs):
- W&B for research/experimentation (fast iteration)
- MLflow for production models (self-hosted registry)
- Neptune for compliance-heavy projects (audit trails)

---

### Hyperparameter Tuning (Grid, Random, Bayesian, Optuna)

**Why Hyperparameter Tuning?**
- Small hyperparameter changes = large accuracy gains (1-5%)
- Manual tuning is time-consuming and non-systematic
- Automated tuning finds better configurations faster

**Search Space Example**:
```
learning_rate: [1e-5, 1e-4, 1e-3]
batch_size: [16, 32, 64, 128]
num_layers: [6, 12, 24]
hidden_size: [512, 768, 1024]
dropout: [0.1, 0.2, 0.3]
```
**Total combinations**: 3 × 4 × 3 × 3 × 3 = **324 trials**

#### **Grid Search**

**Approach**: Try every combination in the search space

**Pros**:
- Exhaustive (guaranteed to find best in grid)
- Reproducible
- Simple to understand

**Cons**:
- Exponential cost (doubles with each parameter)
- Wastes compute on bad regions
- Not practical for large search spaces (324 trials too many)

**When to use**:
- Small search spaces (<50 trials)
- Critical hyperparameters (must find optimal)
- When you have unlimited compute

**Example**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [1e-4, 1e-3],
    'batch_size': [32, 64]
}
# Runs 2 × 2 = 4 trials
```

---

#### **Random Search**

**Approach**: Sample random combinations from search space

**Key Insight** (Bergstra & Bengio, 2012):
- Most hyperparameters don't matter much
- Random search explores important dimensions better
- **10-20x more efficient than grid search**

**Example**:
```
Trial 1: lr=3e-4, batch=48, layers=9
Trial 2: lr=7e-5, batch=96, layers=15
Trial 3: lr=2e-3, batch=20, layers=18
```

**Pros**:
- Much faster than grid search
- Discovers unexpected good combinations
- Easy to parallelize (independent trials)

**Cons**:
- No learning between trials (wastes compute on bad regions)
- May miss optimal if unlucky
- Need more trials for high confidence

**When to use**:
- First exploration of search space
- Large search spaces (100+ dimensions)
- Time-constrained tuning (1 day budget)

**Production Example** (Uber):
- Random search for initial hyperparameters
- 50 trials in 12 hours (parallel on 50 GPUs)
- Found 95% of grid search quality in 5% of time

---

#### **Bayesian Optimization**

**Approach**: Build probabilistic model of objective function, use it to guide search

**Key Idea**:
1. Run a few random trials (5-10)
2. Fit Gaussian Process (GP) to results
3. GP predicts: mean (expected performance) + uncertainty
4. Next trial: Maximize **acquisition function** (balance exploitation vs exploration)
5. Repeat steps 2-4

**Acquisition Functions**:
- **EI (Expected Improvement)**: How much better than current best?
- **UCB (Upper Confidence Bound)**: Optimistic estimate (mean + k × std)
- **PI (Probability of Improvement)**: Probability of beating current best

**Pros**:
- Sample-efficient (finds good config in 10-50 trials)
- Learns from past trials (smarter than random)
- Handles expensive evaluations (each trial = hours)

**Cons**:
- Slower per iteration (GP fitting overhead)
- Struggles with high dimensions (>20 parameters)
- Requires smooth objective function

**When to use**:
- Expensive evaluations (each trial = 1+ hours)
- Moderate search space (<20 dimensions)
- Need near-optimal (not just good enough)

**Libraries**:
- **Optuna**: Best general-purpose library
- **Hyperopt**: Early library (Tree-structured Parzen Estimators)
- **Ax** (Meta): Advanced, production-grade
- **Ray Tune**: Distributed tuning

---

#### **Optuna (Recommended)**

**Why Optuna?**
- Define-by-run API (Pythonic, flexible)
- Pruning: Stop bad trials early (save compute)
- Multi-objective optimization (accuracy + latency)
- Distributed tuning (parallel trials)
- Visualization (parameter importance, contour plots)

**Example**:
```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    
    model = train_model(lr, batch_size)
    accuracy = evaluate(model)
    
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best params:", study.best_params)
print("Best accuracy:", study.best_value)
```

**Pruning (Early Stopping)**:
```python
def objective(trial):
    for epoch in range(100):
        accuracy = train_epoch()
        
        # Report intermediate value
        trial.report(accuracy, epoch)
        
        # Prune if trial is unpromising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy
```

**Multi-Objective**:
```python
def objective(trial):
    # Optimize accuracy AND latency
    accuracy = ...
    latency = ...
    return accuracy, latency

study = optuna.create_study(directions=["maximize", "minimize"])
```

**Production Example** (Netflix):
- Optuna for hyperparameter tuning on recommendation models
- 200 trials in parallel (distributed across GPUs)
- Pruning saves 40% compute (bad trials stopped early)
- Found 2% accuracy improvement over manual tuning

---

#### **Hyperparameter Tuning Best Practices**

**1. Start with Random Search**
- Quick exploration (50-100 trials)
- Identify promising regions
- Cheap and parallelizable

**2. Refine with Bayesian Optimization**
- Focus on promising regions
- 20-50 trials
- Use pruning to save compute

**3. Use Warm Starting**
- Start from known good config (literature, previous experiments)
- Fine-tune around that point

**4. Track Everything**
- Log ALL hyperparameters (even ones you didn't tune)
- Log system metrics (GPU utilization, memory)
- Save checkpoints for best trials

**5. Consider Multi-Fidelity**
- Train on subset of data first (10% → 100%)
- Short training first (10 epochs → 100 epochs)
- Use cheap proxy metrics (validation loss at epoch 5)

**6. Don't Overfit to Validation Set**
- Use separate test set for final evaluation
- Early stopping on validation, report on test
- If tuning for >100 trials, refresh validation set

---

### AutoML Frameworks

**What is AutoML?**
Automated Machine Learning: automate model selection, feature engineering, hyperparameter tuning, and architecture search.

#### **AutoML Components**

**1. Neural Architecture Search (NAS)**
- Automatically design neural network architectures
- Search space: layers, connections, operations
- Methods: Reinforcement learning, evolutionary algorithms, gradient-based
- **Problem**: Extremely expensive (1000s of GPU-hours)

**2. Hyperparameter Optimization (HPO)**
- Same as above (Bayesian optimization, etc.)

**3. Feature Engineering**
- Automated feature selection
- Feature transformation (log, polynomial, interactions)
- Embedding learning

**4. Model Selection**
- Try multiple algorithms (XGBoost, LightGBM, Neural nets)
- Ensemble best models

---

#### **Popular AutoML Tools**

**1. AutoGluon (AWS)**
- Automated tabular, text, image, multimodal ML
- **Key Feature**: Stacking/ensembling (combines multiple models)
- 3 lines of code:
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='price')
predictor.fit(train_data)
predictions = predictor.predict(test_data)
```

**Pros**:
- State-of-the-art accuracy (wins Kaggle competitions)
- Fast (smart defaults, early stopping)
- Handles missing data, categorical features automatically

**Cons**:
- Black box (hard to interpret)
- Large models (ensemble = slow inference)
- Tabular-focused (less good for LLMs)

---

**2. H2O AutoML**
- Open-source, distributed
- Supports classification, regression, time-series
- **Key Feature**: Leaderboard (compares all models)

**Example**:
```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()
aml = H2OAutoML(max_runtime_secs=3600)  # 1 hour budget
aml.train(y='target', training_frame=train)

leaderboard = aml.leaderboard  # See all models tried
best_model = aml.leader  # Best model
```

**Pros**:
- Distributed (scales to large datasets)
- Good for production (Java/Python model export)
- Explainability (SHAP, variable importance)

**Cons**:
- Setup complexity (cluster management)
- Older codebase

---

**3. FLAML (Microsoft)**
- Fast, Lightweight AutoML
- **Key Feature**: Cost-frugal tuning (minimize cost, not just maximize accuracy)
- Economical hyperparameter optimization

**Example**:
```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="classification", time_budget=600)
```

**Pros**:
- Very fast (10x faster than competitors)
- Cost-aware (good for limited budgets)
- Integrates with scikit-learn

**Cons**:
- Less mature than others
- Smaller community

---

**4. AutoKeras**
- AutoML for Keras/TensorFlow
- NAS for deep learning
- **Key Feature**: Architecture search for images, text

**Example**:
```python
import autokeras as ak

clf = ak.ImageClassifier(max_trials=10)
clf.fit(x_train, y_train, epochs=10)
```

**Pros**:
- Easy for deep learning
- Good for computer vision

**Cons**:
- Slow (NAS is expensive)
- Limited to Keras/TensorFlow

---

**5. Ray Tune + Optuna**
- Distributed hyperparameter tuning
- Supports any ML framework (PyTorch, TensorFlow, XGBoost)
- **Key Feature**: Scalability (1000s of trials in parallel)

**Example**:
```python
from ray import tune

def train_model(config):
    model = create_model(config["lr"], config["batch_size"])
    accuracy = train_and_eval(model)
    return {"accuracy": accuracy}

analysis = tune.run(
    train_model,
    config={
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64])
    },
    num_samples=100
)
```

**Pros**:
- Most scalable (cloud-native)
- Framework-agnostic
- Active development

**Cons**:
- Requires Ray cluster setup
- Steep learning curve

---

#### **When to Use AutoML**

**Use AutoML when**:
- Quick baseline needed (hackathon, prototype)
- Non-expert users (product managers, analysts)
- Standard problems (tabular data, classification)
- Limited time (hours, not days)

**Don't Use AutoML when**:
- Custom architectures needed (LLMs, specialized models)
- Interpretability critical (black-box is unacceptable)
- Extreme performance needed (hand-tuned beats AutoML by 1-2%)
- Domain-specific knowledge available (AutoML can't incorporate this)

**Hybrid Approach** (Common):
1. Use AutoML for quick baseline (1 day)
2. Understand what AutoML found (which models, hyperparameters)
3. Hand-tune top model for extra 1-2% (1 week)

---

## 12.2 CI/CD for ML

### Overview
CI/CD for ML is different from software CI/CD. Traditional CI/CD tests code; ML CI/CD tests code + data + models. Changes in any of these can break production.

**Key Differences**:
| Traditional CI/CD | ML CI/CD |
|-------------------|----------|
| Test code | Test code + data + models |
| Deterministic | Non-deterministic (randomness) |
| Fast tests (<5 min) | Slow tests (retraining = hours) |
| Single environment | Multiple environments (training, serving) |
| Code versioning | Code + data + model versioning |

---

### Model Testing Strategies

#### **1. Unit Tests for ML Code**

**What to Test**:
- Data preprocessing functions
- Feature engineering logic
- Model inference logic
- Custom loss functions
- Training loop components

**Example (PyTest)**:
```python
def test_preprocess_text():
    input_text = "Hello World!"
    expected = "hello world"
    assert preprocess(input_text) == expected

def test_model_output_shape():
    model = create_model()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, 1000)  # 1000 classes

def test_loss_is_positive():
    predictions = torch.tensor([0.8, 0.2])
    targets = torch.tensor([1, 0])
    loss = compute_loss(predictions, targets)
    assert loss >= 0
```

**Coverage Goal**: 70-80% of ML code

---

#### **2. Integration Tests**

**What to Test**:
- End-to-end training pipeline
- Model loading and inference
- Data pipeline (fetch data → preprocess → train)
- Model saving and checkpointing

**Example**:
```python
def test_training_pipeline():
    # Load small dataset
    train_data = load_dummy_data(num_samples=100)
    
    # Train for 1 epoch
    model = train_model(train_data, epochs=1)
    
    # Check model was trained
    assert model is not None
    assert os.path.exists("checkpoints/model.pth")

def test_inference_pipeline():
    model = load_model("models/production_v1.pth")
    sample_input = create_sample_input()
    
    prediction = model.predict(sample_input)
    
    assert prediction is not None
    assert 0 <= prediction <= 1  # Valid probability
```

**Run Frequency**: On every pull request

---

#### **3. Model Quality Tests**

**What to Test**:
- Model accuracy on holdout test set
- Model doesn't degrade (regression test)
- Model performance on critical subgroups

**Example**:
```python
def test_model_accuracy():
    model = load_latest_model()
    test_data = load_test_set()
    
    accuracy = evaluate(model, test_data)
    
    assert accuracy >= 0.85  # Minimum acceptable accuracy

def test_no_regression():
    new_model = load_latest_model()
    old_model = load_model("models/production_v1.pth")
    test_data = load_test_set()
    
    new_accuracy = evaluate(new_model, test_data)
    old_accuracy = evaluate(old_model, test_data)
    
    # New model must be at least as good
    assert new_accuracy >= old_accuracy - 0.01  # Allow 1% tolerance

def test_fairness():
    model = load_latest_model()
    
    # Test on demographic subgroups
    accuracy_group_a = evaluate(model, test_data_group_a)
    accuracy_group_b = evaluate(model, test_data_group_b)
    
    # Difference should be small
    assert abs(accuracy_group_a - accuracy_group_b) < 0.05
```

**Run Frequency**: Before deployment (nightly or on-demand)

---

### Data Validation Tests

**Why Data Validation?**
- Data is the #1 cause of ML failures in production
- Schema changes break models (new column, missing feature)
- Data drift causes model degradation
- Outliers cause inference errors

#### **Schema Validation**

**What to Check**:
- Column names and types
- Required columns present
- Value ranges (min/max)
- Categorical values (known categories only)

**Example (Great Expectations)**:
```python
import great_expectations as ge

df = ge.read_csv("data/input.csv")

# Schema checks
df.expect_table_column_count_to_equal(10)
df.expect_column_to_exist("age")
df.expect_column_values_to_be_of_type("age", "int")

# Value range checks
df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
df.expect_column_values_to_be_in_set("country", ["US", "UK", "CA"])

# Null checks
df.expect_column_values_to_not_be_null("user_id")

# Save expectations
df.save_expectation_suite("data_validation_suite.json")
```

**Run Frequency**: Every data ingestion (real-time or batch)

---

#### **Statistical Validation**

**What to Check**:
- Distribution shifts (training vs serving data)
- Outlier detection
- Missing value rates

**Example**:
```python
def test_data_distribution():
    train_data = load_training_data()
    new_data = load_new_batch()
    
    # KS test for distribution shift
    from scipy.stats import ks_2samp
    stat, pvalue = ks_2samp(train_data['age'], new_data['age'])
    
    assert pvalue > 0.05  # No significant shift

def test_missing_rate():
    df = load_new_batch()
    
    for col in df.columns:
        missing_rate = df[col].isna().mean()
        assert missing_rate < 0.1  # <10% missing allowed
```

**Production Example (Airbnb)**:
- Monitor 100+ features for distribution shifts
- Alert if KS test p-value < 0.01 (significant shift)
- Automatically retrain model if shift detected

---

### Automated Retraining Pipelines

**Why Automated Retraining?**
- Models degrade over time (concept drift, data drift)
- Manual retraining is slow and error-prone
- Need to adapt to new data continuously

#### **Retraining Triggers**

**1. Time-based**:
- Daily: Recommendation models (user behavior changes fast)
- Weekly: Fraud detection (new fraud patterns)
- Monthly: Churn prediction (slower changes)

**2. Performance-based**:
- Accuracy drops below threshold (85% → 82%)
- Precision/recall degradation
- Increased error rate in production

**3. Data-based**:
- New training data available (1M+ new samples)
- Distribution shift detected (KS test)
- Concept drift detected (model's predictions don't match reality)

**4. Manual**:
- New features added
- Hyperparameter changes
- Architecture updates

---

#### **Retraining Pipeline Architecture**

**Components**:
1. **Trigger**: Detect when to retrain
2. **Data Preparation**: Fetch and preprocess new data
3. **Training**: Train model on new data
4. **Evaluation**: Validate on holdout set
5. **Deployment**: If better, deploy to production
6. **Monitoring**: Track new model's performance

**Example (Airflow DAG)**:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('model_retraining', schedule_interval='@daily')

fetch_data = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_latest_data,
    dag=dag
)

validate_data = PythonOperator(
    task_id='validate_data',
    python_callable=run_data_validation,
    dag=dag
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_new_model,
    dag=dag
)

evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_on_test_set,
    dag=dag
)

deploy_model = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_if_better,
    dag=dag
)

# Pipeline
fetch_data >> validate_data >> train_model >> evaluate_model >> deploy_model
```

---

#### **Production Examples**

**Uber (Michelangelo platform)**:
- Automated retraining for 1000+ models
- Daily retraining for demand prediction
- Weekly retraining for fraud detection
- Model automatically deployed if accuracy improves by >1%

**Netflix**:
- A/B test new models before full deployment
- Champion-challenger setup (old model vs new model)
- Deploy new model only if statistically significant improvement

---

### Model Deployment Pipelines

#### **Deployment Strategies**

**1. Blue-Green Deployment**:
- Two environments: Blue (current), Green (new)
- Deploy new model to Green
- Test Green (smoke tests, canary traffic)
- Switch traffic: Blue → Green (instant cutover)
- Keep Blue as rollback option (24-48 hours)

**Pros**: Instant rollback, zero downtime
**Cons**: 2x infrastructure cost (both environments running)

---

**2. Canary Deployment**:
- Deploy new model to small % of traffic (1-5%)
- Monitor metrics (latency, accuracy, errors)
- Gradually increase traffic (5% → 25% → 50% → 100%)
- Rollback if metrics degrade

**Example (Kubernetes)**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model
  ports:
    - port: 80

---
# Old model (95% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v1
spec:
  replicas: 19  # 95% of 20 pods
  selector:
    matchLabels:
      app: model
      version: v1

---
# New model (5% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v2
spec:
  replicas: 1  # 5% of 20 pods
  selector:
    matchLabels:
      app: model
      version: v2
```

**Pros**: Safe (gradual rollout), easy rollback
**Cons**: Slower deployment (days), monitoring overhead

---

**3. Shadow Deployment**:
- New model serves traffic in parallel (shadow mode)
- Old model's predictions returned to user
- New model's predictions logged for comparison
- No user impact (new model doesn't affect users)

**When to use**: Test new model without risk

**Example**:
```python
def predict(input):
    # Production model
    pred_v1 = model_v1.predict(input)
    
    # Shadow model (async)
    asyncio.create_task(log_shadow_prediction(model_v2, input))
    
    return pred_v1  # Only v1 affects user

async def log_shadow_prediction(model, input):
    pred_v2 = model.predict(input)
    log_to_database(input, pred_v2)
```

**Pros**: Zero user impact, real-world testing
**Cons**: 2x compute cost, comparison lag

---

**4. A/B Testing**:
- Split users into groups (A: old model, B: new model)
- Measure business metrics (CTR, conversion, revenue)
- Deploy new model if statistically significant improvement

**Example (Experimentation Platform)**:
```python
def predict(user_id, input):
    # Assign user to group (consistent hashing)
    group = hash(user_id) % 2
    
    if group == 0:
        return model_v1.predict(input)  # Control
    else:
        return model_v2.predict(input)  # Treatment
```

**Pros**: Measures business impact, statistically rigorous
**Cons**: Slow (need 1000s of samples), complex analysis

---

### Smoke Testing

**What is Smoke Testing?**
Quick sanity checks after deployment to ensure model works.

**What to Test**:
1. Model loads successfully
2. Inference completes without errors
3. Predictions are in valid range
4. Latency is acceptable
5. No memory leaks

**Example**:
```python
def smoke_test():
    # Test 1: Model loads
    model = load_model("models/production_v2.pth")
    assert model is not None
    
    # Test 2: Inference works
    sample_input = create_sample_input()
    prediction = model.predict(sample_input)
    assert prediction is not None
    
    # Test 3: Valid output
    assert 0 <= prediction <= 1
    
    # Test 4: Latency check
    start = time.time()
    _ = model.predict(sample_input)
    latency = time.time() - start
    assert latency < 0.1  # <100ms
    
    # Test 5: Batch inference
    batch_inputs = [create_sample_input() for _ in range(32)]
    predictions = model.predict_batch(batch_inputs)
    assert len(predictions) == 32
    
    print("✓ All smoke tests passed")
```

**Run Frequency**: Immediately after deployment (automated)

**Production Example (Stripe)**:
- Smoke tests run on every deployment
- 20+ checks (model load, inference, latency, error rate)
- Deployment blocked if any test fails
- Tests complete in <2 minutes

---

### Regression Testing

**What is Regression Testing?**
Ensure new model doesn't perform worse than old model on critical test cases.

**What to Test**:
1. Accuracy on benchmark dataset (no degradation)
2. Performance on edge cases
3. Fairness across subgroups
4. Latency and throughput
5. Memory usage

**Example**:
```python
def regression_test():
    old_model = load_model("models/production_v1.pth")
    new_model = load_model("models/production_v2.pth")
    test_set = load_benchmark_test_set()
    
    # Test 1: Overall accuracy
    old_acc = evaluate(old_model, test_set)
    new_acc = evaluate(new_model, test_set)
    assert new_acc >= old_acc - 0.01  # Allow 1% regression
    
    # Test 2: Critical edge cases
    edge_cases = load_edge_cases()
    old_edge_acc = evaluate(old_model, edge_cases)
    new_edge_acc = evaluate(new_model, edge_cases)
    assert new_edge_acc >= old_edge_acc
    
    # Test 3: Latency regression
    old_latency = measure_latency(old_model)
    new_latency = measure_latency(new_model)
    assert new_latency <= old_latency * 1.2  # Allow 20% slower
    
    # Test 4: Memory regression
    old_memory = measure_memory(old_model)
    new_memory = measure_memory(new_model)
    assert new_memory <= old_memory * 1.5  # Allow 50% more memory
    
    print("✓ All regression tests passed")
```

**Production Example (Google)**:
- 1000+ regression tests for each model
- Tests run on every model update (weekly)
- Automatic rollback if >5% accuracy regression
- Human review required for latency regression

---

## 12.3 Model Governance

### Overview
Model governance ensures ML systems are trustworthy, compliant, fair, and auditable. Critical for regulated industries (finance, healthcare, insurance) and large enterprises.

**Why Model Governance?**
- **Regulatory compliance**: GDPR, CCPA, FCRA, ECOA
- **Risk management**: Model failures can cost millions
- **Fairness**: Avoid discrimination (race, gender, age)
- **Auditability**: Explain model decisions to regulators
- **Reproducibility**: Recreate model from scratch

---

### Model Documentation (Model Cards)

**What are Model Cards?**
Standardized documentation for ML models (introduced by Google, 2019).

**Sections**:
1. **Model Details**: Architecture, version, training date
2. **Intended Use**: What problem does it solve?
3. **Factors**: Demographic groups, environments
4. **Metrics**: Accuracy, precision, recall (overall and per-group)
5. **Training Data**: Source, size, biases
6. **Evaluation Data**: How was model tested?
7. **Ethical Considerations**: Potential harms, biases
8. **Caveats and Recommendations**: When NOT to use model

**Example (Simplified)**:
```
Model Card: Credit Risk Classifier v3.2

Model Details:
- Architecture: XGBoost (500 trees, max_depth=6)
- Training Date: 2024-01-15
- Developers: ML Team @ FinTech Inc.

Intended Use:
- Predict credit default risk (binary classification)
- Use case: Loan approval decisions
- NOT for: Employment screening, insurance pricing

Metrics:
- Overall Accuracy: 87%
- Precision: 82% (minimize false positives)
- Recall: 78%
- AUC: 0.91

Fairness Metrics (by demographic):
- Male accuracy: 88%
- Female accuracy: 86% (within 2% tolerance)
- Race: No significant difference (tested 5 groups)

Training Data:
- Source: Internal loan database (2019-2023)
- Size: 1M loan applications
- Known biases: Historical lending bias toward high-income zip codes

Ethical Considerations:
- Model may perpetuate historical biases
- Human review required for edge cases (borderline scores)
- Appeals process in place

Caveats:
- Do NOT use for loans >$500K (not trained on this range)
- Do NOT use for non-US applicants (different credit systems)
- Retrain quarterly to adapt to economic changes
```

**Tools**:
- **Model Card Toolkit** (Google): Generate model cards from metadata
- **HuggingFace Model Cards**: Built into Hub
- **W&B Reports**: Generate documentation from experiments

**Production Example (OpenAI)**:
- Model cards for GPT-3, GPT-4 (public)
- Details: Capabilities, limitations, safety mitigations
- Updated with each major release

---

### Datasheets for Datasets

**What are Datasheets?**
Documentation for datasets (analogous to model cards).

**Sections**:
1. **Motivation**: Why was dataset created?
2. **Composition**: What's in the dataset? (samples, features, labels)
3. **Collection Process**: How was data collected?
4. **Preprocessing**: What cleaning/transformations were applied?
5. **Uses**: Recommended uses (and discouraged uses)
6. **Distribution**: How is dataset shared?
7. **Maintenance**: Who maintains it? Update frequency?

**Example (Simplified)**:
```
Datasheet: Customer Support Tickets Dataset v2.0

Motivation:
- Enable ML models for ticket routing and response generation
- Improve support team efficiency

Composition:
- 500K support tickets (2020-2023)
- Fields: ticket_text, category (10 classes), resolution_time, customer_id
- Languages: English (95%), Spanish (5%)
- No PII (emails/names redacted)

Collection Process:
- Exported from Zendesk API
- Random sampling (no selection bias)
- Manual review for quality (10% sample)

Preprocessing:
- Text anonymization (regex for emails, phone numbers)
- HTML tags removed
- Non-English tickets excluded (except Spanish)
- Duplicates removed (fuzzy matching)

Known Biases:
- Technical tickets over-represented (40% of dataset)
- Billing tickets under-represented (5%)
- Weekend tickets excluded (support closed weekends)

Recommended Uses:
- Ticket classification (intended use)
- Response generation (fine-tuning LLMs)

Discouraged Uses:
- Sentiment analysis (tickets are biased negative)
- Customer segmentation (missing demographic data)

Maintenance:
- Updated quarterly (incremental)
- Maintained by Data Engineering team
```

**Tools**:
- **Datasheets for Datasets Toolkit** (Google)
- **Data Version Control (DVC)**: Track dataset versions
- **Great Expectations**: Validate dataset schema

---

### Bias Detection and Mitigation

**Types of Bias**:
1. **Historical bias**: Training data reflects past discrimination
2. **Representation bias**: Some groups under-represented
3. **Measurement bias**: Labels are biased (subjective judgments)
4. **Aggregation bias**: Model treats diverse groups as one
5. **Evaluation bias**: Test set not representative

---

#### **Bias Detection**

**1. Statistical Parity** (Demographic Parity):
```
P(Y_pred = 1 | Group A) = P(Y_pred = 1 | Group B)
```
- Model predicts positive outcome at same rate for all groups

**Example**:
```python
def demographic_parity(predictions, groups):
    rate_group_a = predictions[groups == 'A'].mean()
    rate_group_b = predictions[groups == 'B'].mean()
    
    disparity = abs(rate_group_a - rate_group_b)
    return disparity

# Acceptable if disparity < 0.05 (5% difference)
```

**Limitation**: Equal outcome rates may not be fair if base rates differ

---

**2. Equalized Odds**:
```
P(Y_pred = 1 | Y_true = 1, Group A) = P(Y_pred = 1 | Y_true = 1, Group B)  # Equal TPR
P(Y_pred = 1 | Y_true = 0, Group A) = P(Y_pred = 1 | Y_true = 0, Group B)  # Equal FPR
```
- Model has same error rates across groups

**Example**:
```python
def equalized_odds(predictions, labels, groups):
    # True Positive Rate (TPR)
    tpr_a = ((predictions == 1) & (labels == 1) & (groups == 'A')).sum() / ((labels == 1) & (groups == 'A')).sum()
    tpr_b = ((predictions == 1) & (labels == 1) & (groups == 'B')).sum() / ((labels == 1) & (groups == 'B')).sum()
    
    # False Positive Rate (FPR)
    fpr_a = ((predictions == 1) & (labels == 0) & (groups == 'A')).sum() / ((labels == 0) & (groups == 'A')).sum()
    fpr_b = ((predictions == 1) & (labels == 0) & (groups == 'B')).sum() / ((labels == 0) & (groups == 'B')).sum()
    
    tpr_disparity = abs(tpr_a - tpr_b)
    fpr_disparity = abs(fpr_a - fpr_b)
    
    return tpr_disparity, fpr_disparity
```

---

**3. Predictive Parity**:
```
P(Y_true = 1 | Y_pred = 1, Group A) = P(Y_true = 1 | Y_pred = 1, Group B)
```
- Precision is equal across groups

**When to use each**:
- **Demographic parity**: When equal opportunity is important (hiring, promotions)
- **Equalized odds**: When error rates matter (recidivism, medical diagnosis)
- **Predictive parity**: When precision matters (fraud detection, spam filtering)

**Impossibility Theorem** (Chouldechova, 2017):
- Cannot satisfy all three simultaneously (if base rates differ)
- Must choose which fairness metric to prioritize

---

#### **Bias Mitigation**

**Pre-processing** (Fix training data):
1. **Resampling**: Over-sample minority group, under-sample majority
2. **Reweighting**: Give more weight to minority samples
3. **Synthetic data**: Generate synthetic samples for minority group (SMOTE)

**In-processing** (Modify training):
1. **Adversarial debiasing**: Train model to be unable to predict sensitive attribute
2. **Fairness constraints**: Add fairness loss term (e.g., demographic parity loss)
3. **Fair representation learning**: Learn features that are fair

**Post-processing** (Adjust predictions):
1. **Threshold optimization**: Different decision thresholds per group
2. **Calibration**: Adjust probabilities to equalize metrics

**Example (Threshold Optimization)**:
```python
def optimize_thresholds(model, X, y, groups):
    # Find threshold for each group
    thresholds = {}
    
    for group in ['A', 'B']:
        X_group = X[groups == group]
        y_group = y[groups == group]
        
        probs = model.predict_proba(X_group)[:, 1]
        
        # Find threshold that maximizes F1 score
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= threshold).astype(int)
            f1 = f1_score(y_group, preds)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        thresholds[group] = best_threshold
    
    return thresholds

# Apply different thresholds
def predict_fair(model, X, groups, thresholds):
    probs = model.predict_proba(X)[:, 1]
    predictions = np.zeros(len(X))
    
    for group in ['A', 'B']:
        mask = (groups == group)
        predictions[mask] = (probs[mask] >= thresholds[group]).astype(int)
    
    return predictions
```

---

### Fairness Metrics (Demographic Parity, Equalized Odds)

**Summary Table**:
| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Demographic Parity** | P(Ŷ=1\|A) = P(Ŷ=1\|B) | Equal opportunity scenarios (hiring) |
| **Equalized Odds** | TPR and FPR equal across groups | When error rates critical (criminal justice) |
| **Predictive Parity** | PPV equal across groups | When precision matters (fraud detection) |
| **Calibration** | E[Y\|Ŷ=p,A] = p for all groups | When probabilities need to be accurate |

**Tools**:
- **Fairlearn** (Microsoft): Bias detection + mitigation
- **AI Fairness 360** (IBM): 70+ fairness metrics
- **Aequitas** (UChicago): Fairness audit toolkit
- **What-If Tool** (Google): Visual fairness analysis

**Production Example (FICO)**:
- Credit scoring models audited for fairness
- Test for disparate impact (80% rule): 
  - If approval rate for group A < 0.8 × approval rate for group B → Bias
- Regular audits (quarterly)
- Third-party validation

---

### Compliance (GDPR, CCPA)

**Key Regulations**:

**1. GDPR (General Data Protection Regulation)** - EU:
- **Right to Explanation**: Users can ask "why was I denied?"
- **Right to be Forgotten**: Delete user data on request
- **Data Minimization**: Collect only necessary data
- **Purpose Limitation**: Use data only for stated purpose
- **Penalties**: Up to 4% of global revenue (€20M max)

**ML Implications**:
- Models must be explainable (SHAP, LIME)
- Cannot use sensitive attributes (race, religion) directly
- Must handle data deletion (retrain model?)
- Audit trails for model decisions

---

**2. CCPA (California Consumer Privacy Act)** - California, US:
- **Right to Know**: What data is collected
- **Right to Delete**: Delete personal data
- **Right to Opt-Out**: Opt out of data sale
- **Penalties**: $2,500 per violation ($7,500 if intentional)

**ML Implications**:
- Similar to GDPR (transparency, deletion)
- Opt-out affects personalization models

---

**3. FCRA (Fair Credit Reporting Act)** - US:
- Credit decisions must be explainable
- **Adverse Action Notices**: Explain why credit denied
- Required: Top 4 factors that led to denial

**Example (Credit Denial)**:
```
Your application was denied based on:
1. High credit utilization (75% vs 30% average)
2. Short credit history (2 years vs 10 years average)
3. Recent missed payment (1 in last 6 months)
4. High debt-to-income ratio (45% vs 35% average)
```

**ML Implications**:
- Must extract feature importance (SHAP values)
- Top 4 features → adverse action reasons
- Cannot be black box

---

**4. ECOA (Equal Credit Opportunity Act)** - US:
- Prohibits discrimination in credit (race, gender, age, etc.)
- Penalties for disparate impact

**ML Implications**:
- Cannot use protected attributes (even indirectly)
- Must audit for disparate impact
- Example: Zip code correlates with race → careful usage

---

#### **Compliance Best Practices**

**1. Explainability**:
- Use interpretable models (linear, tree-based) when possible
- Add explainability layer (SHAP) for complex models
- Generate explanations for every prediction (production)

**2. Data Governance**:
- Track data lineage (where did each feature come from?)
- Document data retention policies
- Implement data deletion workflows

**3. Audit Trails**:
- Log every model prediction (input, output, model version)
- Log data access (who accessed what, when)
- Retention: 7 years (typical compliance requirement)

**4. Privacy by Design**:
- Minimize data collection (only what's needed)
- Anonymize data where possible
- Use differential privacy for sensitive data

**5. Third-Party Audits**:
- External audit of models (fairness, accuracy)
- Annual compliance reviews
- Document everything

---

### Audit Trails

**What to Log**:
1. **Model Training**:
   - Training data version
   - Hyperparameters
   - Training duration
   - Final metrics (accuracy, loss)
   - Model artifacts (weights, config)
   - Git commit hash (code version)

2. **Model Deployment**:
   - Deployment timestamp
   - Model version deployed
   - Deployment method (canary, blue-green)
   - Who deployed (user ID)
   - Rollback plan

3. **Model Inference**:
   - Input features (user ID, features)
   - Model prediction (class, probability)
   - Model version used
   - Timestamp
   - Latency (ms)

4. **Data Access**:
   - Who accessed data
   - When
   - What data (tables, rows)
   - Purpose (training, analysis, debugging)

**Example (Logging)**:
```python
import logging
import json

logger = logging.getLogger("ml_audit")

def predict_and_log(model, user_id, features):
    start_time = time.time()
    
    prediction = model.predict(features)
    
    latency = time.time() - start_time
    
    # Log to audit trail
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "model_version": model.version,
        "features": features.tolist(),
        "prediction": float(prediction),
        "latency_ms": latency * 1000
    }
    
    logger.info(json.dumps(log_entry))
    
    return prediction
```

**Storage**:
- **Short-term** (30 days): Hot storage (PostgreSQL, MongoDB)
- **Long-term** (7 years): Cold storage (S3 Glacier, Azure Archive)

**Retention Policies**:
- PII: Encrypt and delete after retention period
- Aggregate metrics: Keep indefinitely (no PII)
- Full logs: 7 years (compliance)

**Production Example (Financial Services)**:
- Every credit decision logged (10M+ per day)
- Stored in immutable log (WORM - Write Once Read Many)
- Audit trail reviewed monthly by compliance team
- External auditors access logs annually

---

### Ethical AI Practices

**Key Principles**:

**1. Transparency**:
- Document model limitations (what it can't do)
- Explain how model makes decisions
- Share model cards publicly (when appropriate)

**2. Accountability**:
- Assign ownership (who's responsible for model?)
- Define escalation process (model fails → who fixes?)
- Regular audits (internal and external)

**3. Fairness**:
- Test for bias (across demographics)
- Mitigate bias (pre/in/post-processing)
- Monitor fairness in production (ongoing)

**4. Privacy**:
- Minimize data collection
- Anonymize data
- Differential privacy (add noise for privacy)
- Secure storage (encryption at rest and in transit)

**5. Safety**:
- Red-teaming (adversarial testing)
- Fail-safes (graceful degradation)
- Human oversight (for high-stakes decisions)

**6. Robustness**:
- Handle edge cases
- Adversarial robustness (defend against attacks)
- Monitor for drift (data, concept)

---

### Model Risk Management

**Risk Categories**:

**1. Model Risk**:
- Model is inaccurate (low precision/recall)
- Model degrades over time (drift)
- Model fails on edge cases

**Mitigation**:
- Regular retraining
- Drift monitoring
- Ensemble models (reduce variance)

---

**2. Data Risk**:
- Training data is biased
- Data pipeline breaks (missing features)
- Data leakage (test data in training)

**Mitigation**:
- Data validation (Great Expectations)
- Schema checks
- Holdout test set (never touched during training)

---

**3. Implementation Risk**:
- Bug in code (wrong loss function)
- Training-serving skew (features computed differently)
- Incorrect model loading

**Mitigation**:
- Code reviews
- Unit tests (100% coverage for critical code)
- Integration tests

---

**4. Operational Risk**:
- Model service crashes
- Latency spikes (slow inference)
- Resource exhaustion (OOM)

**Mitigation**:
- Load testing
- Auto-scaling
- Circuit breakers

---

**5. Compliance Risk**:
- Model violates GDPR (no explainability)
- Model discriminates (disparate impact)
- Audit trail incomplete

**Mitigation**:
- Model cards
- Fairness audits
- Comprehensive logging

---

**Risk Assessment Matrix**:
| Risk Level | Likelihood | Impact | Action |
|------------|------------|--------|--------|
| **Critical** | High | High | Immediate mitigation, continuous monitoring |
| **High** | High | Medium | Mitigation plan, monthly review |
| **Medium** | Medium | Medium | Document risk, quarterly review |
| **Low** | Low | Low | Accept risk |

**Example (Loan Approval Model)**:
- **Critical Risk**: Model discriminates by race → Fairness audit, bias mitigation
- **High Risk**: Model degrades (concept drift) → Monthly retraining, drift monitoring
- **Medium Risk**: Latency spike (99th percentile >1s) → Load testing, auto-scaling
- **Low Risk**: Rare edge case (missing feature) → Graceful fallback

---

## 12.4 Cost Optimization

### Overview
ML workloads are expensive: Training (GPU-days), inference (millions of predictions/day), storage (TBs of data). Cost optimization can reduce spend by 50-80% without sacrificing quality.

**Typical Cost Breakdown**:
- Training: 40% (GPUs, data storage, experiment tracking)
- Inference: 50% (model serving, auto-scaling)
- Data storage: 5% (S3, databases)
- Other: 5% (networking, monitoring)

---

### Spot Instances (AWS, GCP, Azure)

**What are Spot Instances?**
- Spare cloud capacity sold at 50-90% discount
- Can be interrupted (preempted) with 2-minute notice
- Best for: Fault-tolerant workloads (training, batch inference)

**Cloud Provider Names**:
- **AWS**: Spot Instances
- **GCP**: Preemptible VMs
- **Azure**: Spot VMs

**Pricing Example (AWS p3.2xlarge - V100 GPU)**:
- On-Demand: $3.06/hour
- Spot: $0.92/hour (70% discount)
- Savings: $2.14/hour × 24 hours × 30 days = **$1,540/month per GPU**

**When to Use Spot Instances**:
1. **Training**: Checkpointing every 10-30 minutes
   - If interrupted, resume from last checkpoint
   - Minimal work lost (<30 minutes)
2. **Hyperparameter Tuning**: Many short jobs
   - Each trial is independent
   - Interruption just delays completion
3. **Batch Inference**: Non-time-sensitive
   - Process data in batches
   - Interruption → reprocess batch

**When NOT to Use**:
1. **Real-time Inference**: Cannot tolerate interruptions
2. **Long-running Jobs** (>24 hours without checkpoints)
3. **Critical Production Workloads**: Need reliability

---

#### **Spot Instance Best Practices**

**1. Checkpoint Frequently**:
```python
def train_with_checkpointing(model, data_loader, checkpoint_dir):
    for epoch in range(100):
        for batch_idx, batch in enumerate(data_loader):
            loss = train_step(model, batch)
            
            # Checkpoint every 100 batches (~10 minutes)
            if batch_idx % 100 == 0:
                save_checkpoint(model, epoch, batch_idx, checkpoint_dir)
        
        # Checkpoint end of epoch
        save_checkpoint(model, epoch + 1, 0, checkpoint_dir)

def resume_from_checkpoint(checkpoint_dir):
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    model, epoch, batch_idx = load_checkpoint(latest_checkpoint)
    return model, epoch, batch_idx
```

**2. Use Spot Fleet (Multiple Instance Types)**:
- Don't rely on single instance type (might be unavailable)
- Request multiple types: `[p3.2xlarge, p3.8xlarge, g4dn.xlarge]`
- Fallback strategy: If spot unavailable → on-demand

**3. Handle Interruptions Gracefully**:
```python
import signal
import sys

def handle_interruption(signum, frame):
    print("Spot instance interruption detected. Saving checkpoint...")
    save_checkpoint(model, epoch, batch_idx, "checkpoints/")
    sys.exit(0)

# AWS sends SIGTERM 2 minutes before termination
signal.signal(signal.SIGTERM, handle_interruption)
```

**4. Monitor Spot Pricing**:
- Track historical prices (AWS Spot Pricing History)
- Avoid peak hours (cheaper at night)
- Use Spot Advisor (AWS tool) for best instance types

**Production Example (Netflix)**:
- 80% of training on spot instances
- Automated checkpoint/resume system
- Estimated savings: $5M/year
- Interruption rate: 5% (95% complete without interruption)

---

### Reserved Capacity

**What is Reserved Capacity?**
- Commit to using instances for 1-3 years
- Get 30-70% discount vs on-demand
- Best for: Predictable, long-running workloads

**Types**:

**1. Standard Reserved Instances**:
- Fixed instance type, region
- Largest discount (up to 72%)
- Cannot change instance type

**2. Convertible Reserved Instances**:
- Can change instance type (within family)
- Moderate discount (up to 54%)
- More flexibility

**3. Savings Plans** (newer, recommended):
- Commit to $/hour spend (not instance type)
- Flexibility: Any instance type, region, OS
- Discount: Up to 72%

**Example (AWS Savings Plan)**:
```
Commit: $100/hour for 1 year
On-Demand Cost: $200/hour
Savings Plan Cost: $100/hour
Savings: $100/hour × 24 × 365 = $876,000/year
```

**When to Use Reserved Capacity**:
1. **Production Inference**: Always-on model serving
2. **Baseline Training**: Continuous training pipelines
3. **Data Processing**: Regular ETL jobs

**Strategy (Hybrid)**:
- Reserved: Cover baseline usage (70%)
- On-Demand: Handle spikes (20%)
- Spot: Cost-optimized training (10%)

**Production Example (Airbnb)**:
- Reserved instances for production inference (24/7)
- Spot instances for experimentation
- On-demand for peak traffic
- Total savings: 60% vs pure on-demand

---

### Right-Sizing Instances

**What is Right-Sizing?**
Matching instance type to workload requirements (CPU, memory, GPU).

**Common Mistakes**:
1. **Over-provisioning**: Using p3.8xlarge (4 GPUs) when 1 GPU is enough
2. **Under-utilizing memory**: 256GB instance with 50GB used
3. **Wrong GPU type**: Using expensive A100 for inference (overkill)

**Right-Sizing Process**:

**Step 1: Monitor Resource Usage**:
```python
# Track GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1

# Track memory usage
free -h

# Track CPU usage
top
```

**Step 2: Analyze Utilization**:
- **GPU Utilization < 50%**: Downsize GPU or batch size too small
- **Memory Usage < 60%**: Downsize instance type
- **CPU Utilization < 30%**: Downsize CPU cores

**Step 3: Experiment with Smaller Instances**:
```
Before: p3.8xlarge (4x V100, $12.24/hour)
Test: p3.2xlarge (1x V100, $3.06/hour)
Result: Same training time (GPU utilization 90%)
Savings: $9.18/hour × 24 × 30 = $6,610/month
```

**GPU Selection Guide**:
| Use Case | GPU Type | Cost/Hour (AWS) |
|----------|----------|-----------------|
| **Training (large models)** | A100 (40GB) | $32.77 |
| **Training (medium models)** | V100 (16GB) | $3.06 |
| **Training (small models)** | T4 (16GB) | $0.526 |
| **Inference (latency-critical)** | T4 | $0.526 |
| **Inference (batch)** | CPU (c5.xlarge) | $0.17 |

**Production Example (Uber)**:
- Analyzed 1000+ training jobs
- Found 40% were over-provisioned (using 4 GPUs, only needed 1)
- Right-sizing saved 30% on training costs ($2M/year)

---

### Model Compression Techniques

(Covered in detail in Section 4.1 Quantization and 4.2 Model Compression)

**Quick Summary**:
1. **Quantization**: INT8 (4x smaller, 3x faster)
2. **Pruning**: Remove 50% of weights (minimal accuracy loss)
3. **Knowledge Distillation**: Train small model to mimic large model
4. **Low-Rank Factorization**: Decompose weight matrices

**Cost Impact**:
- Model size: 7B params → 1.8GB (FP16) → 0.9GB (INT8) → 0.5GB (INT4)
- Inference cost: 10 requests/sec on A100 → 40 requests/sec (INT8)
- **Savings**: 4x throughput = 75% cost reduction

---

### Training Optimization

**1. Mixed Precision Training** (FP16/BF16):
- **Memory**: 2x less (32-bit → 16-bit)
- **Speed**: 2-3x faster (Tensor Cores)
- **Cost**: 50-70% reduction

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in data_loader:
    with autocast():  # FP16 forward pass
        loss = model(batch)
    
    scaler.scale(loss).backward()  # FP32 gradients
    scaler.step(optimizer)
    scaler.update()
```

**Savings Example**:
- Training time: 10 days → 4 days (FP16)
- Cost: $3.06/hour × 240 hours = $734 → $294 (60% savings)

---

**2. Gradient Accumulation**:
- Simulate large batch size on small GPU
- Accumulate gradients over N batches before update

```python
accumulation_steps = 4

for i, batch in enumerate(data_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Cost Impact**:
- Use 1x A100 instead of 4x A100 (4x cheaper)
- Training time: Slightly longer (10-20%)
- **Net savings**: 3-3.5x

---

**3. Gradient Checkpointing**:
- Trade compute for memory (recompute activations)
- 50% memory reduction, 20% slower

```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, x):
    return checkpoint(model, x)
```

**When to use**: Memory-constrained (cannot fit larger batch size)

---

**4. Efficient Data Loading**:
- Parallelize data loading (multiple workers)
- Pin memory (faster CPU → GPU transfer)
- Prefetch batches (hide I/O latency)

```python
data_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # 4 parallel workers
    pin_memory=True,    # Faster transfer
    prefetch_factor=2   # Prefetch 2 batches
)
```

**Savings**: 20-30% faster training (GPU stays busy)

---

### Inference Optimization

(Covered in Section 4: Inference Optimization)

**Quick Summary**:
1. **Quantization**: INT8 inference (4x faster)
2. **Batching**: Process multiple requests together
3. **Model Serving**: vLLM, TensorRT (optimized engines)
4. **Caching**: Cache frequent predictions

**Cost Example**:
- Baseline: 1000 req/sec on 10x A100 = $327/hour
- Optimized (INT8 + batching): 1000 req/sec on 3x A100 = $98/hour
- **Savings**: $229/hour × 24 × 30 = $164,880/month

---

### Budget Monitoring and Alerts

**1. Set Budget Limits**:
```yaml
Monthly Budget: $50,000
Breakdown:
  - Training: $20,000 (40%)
  - Inference: $25,000 (50%)
  - Storage: $2,500 (5%)
  - Other: $2,500 (5%)
```

**2. Cost Allocation Tags**:
```yaml
Tags:
  - Team: ml-research
  - Project: recommendation-model
  - Environment: production
  - Cost-Center: engineering
```

**3. Alerts**:
```
Alert 1: 80% of budget spent → Warning
Alert 2: 95% of budget spent → Critical (auto-stop non-critical jobs)
Alert 3: Unusual spike (>2x daily average) → Investigate
```

**4. Cost Dashboards**:
- Daily spend by team/project
- Cost trends (week-over-week, month-over-month)
- Top 10 most expensive resources
- Savings opportunities (idle resources, over-provisioned)

**Tools**:
- **AWS Cost Explorer**: Visualize AWS spending
- **GCP Cost Management**: GCP spending analysis
- **Kubecost**: Kubernetes cost monitoring
- **CloudHealth**: Multi-cloud cost optimization
- **Infracost**: Infrastructure cost estimation (IaC)

**Production Example (Pinterest)**:
- Cost dashboard updated daily
- Alerts at 75%, 90%, 100% of budget
- Auto-stop non-critical training jobs at 95%
- Monthly cost reviews with each team
- Result: 20% cost reduction in 6 months

---

### Cost Allocation by Team/Project

**Why Cost Allocation?**
- Accountability (teams own their costs)
- Budget planning (historical trends)
- Cost optimization (identify waste)

**Tagging Strategy**:
```yaml
Resource Tags:
  - team: ml-research
  - project: llm-fine-tuning
  - environment: dev | staging | prod
  - owner: alice@company.com
  - cost-center: 12345
```

**Chargeback Model**:
```
Team A:
  - Training: $15,000
  - Inference: $10,000
  - Storage: $2,000
  - Total: $27,000

Team B:
  - Training: $5,000
  - Inference: $15,000
  - Storage: $1,000
  - Total: $21,000
```

**Cost Optimization Incentives**:
- Budget per team (exceed = justify to management)
- Cost savings shared (50% to team budget, 50% to company)
- Quarterly cost reviews (teams present optimizations)

**Production Example (Lyft)**:
- Chargeback model for all ML teams
- Monthly cost reports (team breakdown)
- Gamification: Teams compete for lowest cost per model
- Result: 35% cost reduction year-over-year

---

## 12.5 Observability

### Overview
Observability is understanding system internal state from external outputs. For ML systems: logs, metrics, traces help debug failures, optimize performance, and ensure reliability.

**Three Pillars of Observability**:
1. **Logs**: Discrete events (errors, warnings, info)
2. **Metrics**: Numerical measurements over time (latency, throughput)
3. **Traces**: Request flow through distributed systems

---

### Logging Strategies (Structured Logging)

**Unstructured Logging** (BAD):
```python
print("Model prediction failed for user 12345 at 2024-01-15 10:30:00")
```
- Hard to parse
- No context (what error?)
- Cannot aggregate/analyze

**Structured Logging** (GOOD):
```python
import logging
import json

logger = logging.getLogger("ml_service")

logger.info(json.dumps({
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "error",
    "event": "prediction_failed",
    "user_id": "12345",
    "model_version": "v2.3",
    "error": "ValueError: Invalid input shape",
    "latency_ms": 45,
    "trace_id": "abc123"
}))
```
- Machine-readable (JSON)
- Rich context (user, model, error)
- Easy to query/aggregate

---

#### **What to Log**

**1. Request/Response**:
```python
logger.info({
    "event": "prediction_request",
    "user_id": user_id,
    "model_version": model.version,
    "input_shape": input.shape,
    "timestamp": datetime.now().isoformat()
})

prediction = model.predict(input)

logger.info({
    "event": "prediction_response",
    "user_id": user_id,
    "prediction": prediction,
    "confidence": confidence,
    "latency_ms": latency,
    "timestamp": datetime.now().isoformat()
})
```

**2. Errors**:
```python
try:
    prediction = model.predict(input)
except Exception as e:
    logger.error({
        "event": "prediction_error",
        "user_id": user_id,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "stack_trace": traceback.format_exc()
    })
    raise
```

**3. Model Metrics**:
```python
logger.info({
    "event": "model_metrics",
    "model_version": "v2.3",
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.87,
    "f1": 0.88,
    "timestamp": datetime.now().isoformat()
})
```

**4. Data Quality Issues**:
```python
if has_missing_values(input):
    logger.warning({
        "event": "data_quality_issue",
        "issue_type": "missing_values",
        "columns": missing_columns,
        "user_id": user_id
    })
```

---

#### **Log Levels**

**Standard Levels** (from least to most severe):
1. **DEBUG**: Detailed diagnostic info (development only)
2. **INFO**: General informational messages (normal operation)
3. **WARNING**: Unexpected but handled (degraded performance)
4. **ERROR**: Error occurred (request failed)
5. **CRITICAL**: System-level failure (service down)

**Example Configuration**:
```python
logging.basicConfig(
    level=logging.INFO,  # Production: INFO or WARNING
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Production Settings**:
- Development: DEBUG (verbose)
- Staging: INFO
- Production: WARNING (reduce log volume)

**Log Volume Management**:
- 1000 requests/sec × 5 log lines/request = 5000 logs/sec
- 5000 logs/sec × 86400 sec/day = 432M logs/day
- Storage: 432M × 1KB/log = 432GB/day → $10-20/day (S3)

**Sampling** (reduce volume):
```python
import random

if random.random() < 0.1:  # Sample 10%
    logger.info({...})
```

---

### Distributed Tracing

**Why Distributed Tracing?**
ML systems are distributed (API gateway → model service → feature store → database). Tracing shows request flow across services.

**Example Request Flow**:
```
User → API Gateway → Auth Service → Model Service → Feature Store → Database
  |         |              |              |               |            |
 50ms      10ms           5ms           100ms           50ms        30ms
```
**Total latency**: 245ms  
**Bottleneck**: Model Service (100ms)

---

#### **OpenTelemetry (Standard)**

**Concepts**:
1. **Span**: Single operation (e.g., "predict" function)
2. **Trace**: Collection of spans (end-to-end request)
3. **Context**: Metadata (user_id, trace_id)

**Example (Python)**:
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter

# Setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument code
def predict(user_id, features):
    with tracer.start_as_current_span("model_prediction") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("model_version", "v2.3")
        
        with tracer.start_as_current_span("feature_engineering"):
            processed_features = preprocess(features)
        
        with tracer.start_as_current_span("model_inference"):
            prediction = model.predict(processed_features)
        
        span.set_attribute("prediction", prediction)
        return prediction
```

**Visualization** (Jaeger UI):
```
Trace ID: abc123
Total Duration: 245ms

├─ model_prediction (245ms)
   ├─ feature_engineering (50ms)
   │  ├─ fetch_user_data (30ms)
   │  └─ normalize_features (20ms)
   └─ model_inference (100ms)
      ├─ load_model (10ms)
      └─ forward_pass (90ms)
```

---

#### **Tracing Tools**

**1. Jaeger** (Open-source):
- Distributed tracing
- Service dependency graph
- Latency analysis

**2. Zipkin** (Open-source):
- Similar to Jaeger
- Lightweight

**3. AWS X-Ray**:
- Native AWS integration
- Automatic instrumentation (Lambda, ECS)

**4. Google Cloud Trace**:
- GCP-native
- Integrates with Cloud Logging

**5. Datadog APM**:
- Commercial (expensive)
- Best-in-class UI
- Auto-instrumentation

**Production Example (Shopify)**:
- Jaeger for distributed tracing
- 10M+ traces per day
- Identifies bottlenecks (database queries, slow API calls)
- Reduced P99 latency from 2s to 500ms

---

### Metrics Collection (Prometheus, CloudWatch)

**What Metrics to Track?**

**1. Request Metrics**:
- **Request rate** (req/sec)
- **Error rate** (errors/sec, % of requests)
- **Latency** (P50, P95, P99, max)
- **Throughput** (requests handled/sec)

**2. Model Metrics**:
- **Prediction distribution** (class distribution)
- **Confidence scores** (average, P50, P95)
- **Model version** (which version served request)

**3. System Metrics**:
- **CPU utilization** (%)
- **Memory usage** (GB, %)
- **GPU utilization** (%)
- **GPU memory** (GB, %)
- **Disk I/O** (MB/s)
- **Network I/O** (MB/s)

**4. Business Metrics**:
- **Prediction correctness** (accuracy, when labels available)
- **User engagement** (CTR, conversion)
- **Revenue impact** ($ per prediction)

---

#### **Prometheus (Industry Standard)**

**Architecture**:
1. **Metrics Exporter**: Expose metrics endpoint (e.g., `/metrics`)
2. **Prometheus Server**: Scrape metrics (pull model)
3. **Time-Series Database**: Store metrics
4. **Grafana**: Visualize metrics (dashboards)

**Example (Expose Metrics)**:
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
prediction_count = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization')

# Instrument code
@prediction_latency.time()
def predict(input):
    prediction = model.predict(input)
    prediction_count.inc()
    return prediction

# Update metrics periodically
def update_metrics():
    accuracy = evaluate_model()
    model_accuracy.set(accuracy)
    
    gpu_util = get_gpu_utilization()
    gpu_utilization.set(gpu_util)

# Start metrics server
start_http_server(8000)  # Expose at :8000/metrics
```

**Prometheus Queries (PromQL)**:
```promql
# Request rate (per second)
rate(predictions_total[5m])

# P95 latency
histogram_quantile(0.95, prediction_latency_seconds_bucket)

# Error rate
rate(prediction_errors_total[5m]) / rate(predictions_total[5m])

# GPU utilization average (last 10 minutes)
avg_over_time(gpu_utilization_percent[10m])
```

---

#### **CloudWatch (AWS)**

**Native AWS Integration**:
- Auto-collects metrics from EC2, Lambda, SageMaker
- Custom metrics via PutMetricData API

**Example (Custom Metrics)**:
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='MLService',
    MetricData=[
        {
            'MetricName': 'PredictionLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds',
            'Timestamp': datetime.now()
        },
        {
            'MetricName': 'ModelAccuracy',
            'Value': accuracy,
            'Unit': 'Percent'
        }
    ]
)
```

**CloudWatch Alarms**:
```yaml
Alarm: HighLatency
Metric: PredictionLatency
Threshold: P95 > 500ms
Period: 5 minutes
Action: Send SNS notification → PagerDuty
```

---

#### **Metrics Best Practices**

**1. Cardinality**:
- Avoid high-cardinality labels (user_id, request_id)
- OK: `model_version`, `endpoint`, `status_code`
- BAD: `user_id` (millions of unique values → Prometheus OOM)

**2. Aggregation**:
- Use histograms for latency (not averages)
- Percentiles (P50, P95, P99) more meaningful than mean

**3. Retention**:
- High-resolution (1s): 1 day
- Medium-resolution (1min): 7 days
- Low-resolution (1hour): 1 year

**4. Alerting**:
- Alert on symptoms (high latency) not causes (CPU high)
- Actionable alerts only (avoid noise)

---

### Alerting (PagerDuty, OpsGenie)

**Alerting Principles**:
1. **Actionable**: Alert if human intervention needed
2. **Timely**: Alert immediately (not 1 hour later)
3. **Contextualized**: Include debugging info (which model? which user?)
4. **Escalated**: On-call engineer → manager → VP (if unresolved)

**Alert Severity**:
- **P1 (Critical)**: Production down, data loss
  - Response time: 15 minutes
  - Escalation: Immediate page
- **P2 (High)**: Degraded performance, errors
  - Response time: 1 hour
  - Escalation: Slack + email
- **P3 (Medium)**: Minor issues, non-critical
  - Response time: Next business day
  - Escalation: Ticket
- **P4 (Low)**: Informational
  - Response time: Best effort

---

#### **Alert Examples**

**1. High Error Rate**:
```yaml
Alert: HighErrorRate
Condition: error_rate > 5% for 5 minutes
Severity: P2
Action: Page on-call engineer
Runbook: https://wiki.company.com/ml-service-errors
```

**2. Model Accuracy Drop**:
```yaml
Alert: ModelAccuracyDrop
Condition: model_accuracy < 0.85
Severity: P2
Action: Slack notification to ML team
Runbook: Check for data drift, retrain model
```

**3. High Latency**:
```yaml
Alert: HighLatency
Condition: P95_latency > 500ms for 10 minutes
Severity: P2
Action: Page on-call engineer
Runbook: Check model server, scale up instances
```

**4. Resource Exhaustion**:
```yaml
Alert: HighMemoryUsage
Condition: memory_usage > 90% for 5 minutes
Severity: P1
Action: Auto-scale + page on-call
Runbook: Scale horizontally, investigate memory leak
```

---

#### **Alert Fatigue Prevention**

**Problems**:
- Too many alerts → engineers ignore them
- False positives → desensitization
- Alert storms (100s of alerts at once)

**Solutions**:

**1. Alert Aggregation**:
- Group related alerts (all instances down → 1 alert)
- Deduplicate (same alert every 5 minutes → 1 alert)

**2. Alert Suppression**:
- Maintenance windows (no alerts during deployment)
- Known issues (alert on fix, not repeated failures)

**3. Smart Thresholds**:
- Dynamic thresholds (adapt to traffic patterns)
- Example: Latency alert at 500ms during day, 200ms at night

**4. Runbooks**:
- Every alert links to runbook (how to fix)
- Example: "High latency → Check model server logs → Restart if OOM"

**Production Example (Stripe)**:
- Average 2 alerts/day (down from 50/day before optimization)
- 95% of alerts actionable (required human intervention)
- Alert response time: <5 minutes (P1), <30 minutes (P2)

---

### Debugging Production Issues

**Common ML Production Issues**:

**1. Model Serving Failures**:
- **Symptom**: 500 errors, timeouts
- **Causes**: OOM, model file corrupted, dependency version mismatch
- **Debug**: Check logs, test locally, verify model file

**2. Prediction Quality Degradation**:
- **Symptom**: Accuracy drops, user complaints
- **Causes**: Data drift, model staleness, feature pipeline broken
- **Debug**: Compare input distributions, check feature values, evaluate on test set

**3. Latency Spikes**:
- **Symptom**: P99 latency increases (100ms → 2s)
- **Causes**: Cold start, GC pauses, network congestion, database slow query
- **Debug**: Distributed tracing, profile code, check system metrics

**4. Memory Leaks**:
- **Symptom**: Memory usage grows over time, eventual OOM
- **Causes**: Not releasing GPU memory, caching too much, circular references
- **Debug**: Memory profiler, track object counts, check GPU memory

---

#### **Debugging Workflow**

**Step 1: Reproduce Issue**:
- Collect example request (input that causes error)
- Test locally (can you reproduce?)
- If not reproducible: Check production logs for context

**Step 2: Narrow Down**:
- **Binary search**: Which component failed? (API → model → feature store)
- **Differential analysis**: What changed? (new deployment, data change)
- **Logs**: Error stack trace, warnings before failure

**Step 3: Hypothesize**:
- Form hypothesis (e.g., "Model OOM due to large batch size")
- Test hypothesis (check memory usage logs)
- Iterate (if wrong hypothesis, form new one)

**Step 4: Fix**:
- Implement fix (reduce batch size, add error handling)
- Test fix (unit tests, integration tests)
- Deploy fix (canary → full rollout)

**Step 5: Prevent Recurrence**:
- Add test case for this failure
- Add monitoring/alerting
- Document in runbook

---

### Root Cause Analysis

**What is Root Cause Analysis (RCA)?**
Systematic process to identify underlying cause of failure (not just symptoms).

**5 Whys Technique**:
```
Problem: Model server crashed

Why? → OOM error
Why? → Memory usage spiked to 32GB
Why? → Batch size increased to 1024
Why? → Engineer changed config to improve throughput
Why? → No memory profiling before deployment

Root Cause: Lack of pre-deployment testing for memory usage
Fix: Add memory profiling to CI/CD pipeline
```

---

#### **Incident Report Template**

```markdown
# Incident Report: Model Server Outage

**Date**: 2024-01-15
**Duration**: 10:30 AM - 11:15 AM (45 minutes)
**Severity**: P1 (Production down)

## Summary
Model serving API returned 500 errors for 45 minutes, affecting 10k users.

## Timeline
- 10:30 AM: Deployment of v2.4 started
- 10:32 AM: Error rate spiked to 100%
- 10:35 AM: Alerts fired (high error rate, latency)
- 10:40 AM: On-call engineer paged
- 10:45 AM: Root cause identified (OOM)
- 11:00 AM: Rollback initiated
- 11:15 AM: Service restored

## Root Cause
Batch size increased from 32 to 1024 in v2.4, causing OOM (memory usage: 8GB → 32GB).

## Impact
- 10k users affected (errors on prediction requests)
- 5k requests failed
- Revenue impact: ~$500 (estimated)

## Resolution
- Immediate: Rolled back to v2.3
- Long-term: Added memory profiling to CI/CD

## Action Items
- [x] Add memory profiling step to deployment pipeline (Owner: Alice, Due: 2024-01-20)
- [x] Update runbook for OOM errors (Owner: Bob, Due: 2024-01-18)
- [ ] Load test before production deployment (Owner: Charlie, Due: 2024-01-25)

## Lessons Learned
- Memory profiling should be mandatory before deployment
- Load testing with production traffic volume
- Canary deployment could have caught this (only affected 5% initially)
```

---

### Log Aggregation Tools

**Why Log Aggregation?**
- Centralized logging (100+ servers → 1 dashboard)
- Fast search (find errors in billions of logs)
- Correlation (link logs across services)
- Retention (store logs for compliance, debugging)

---

#### **ELK Stack (Elasticsearch, Logstash, Kibana)**

**Architecture**:
1. **Logstash**: Collect logs from servers, parse, transform
2. **Elasticsearch**: Store and index logs (search engine)
3. **Kibana**: Visualize logs (dashboards, queries)

**Example Flow**:
```
ML Service → Logstash → Elasticsearch → Kibana
   (JSON logs)    (parse)     (index)     (visualize)
```

**Kibana Query (Find Errors)**:
```
level: "error" AND service: "model-server" AND timestamp: [now-1h TO now]
```

**Pros**:
- Open-source (free)
- Powerful search (full-text, aggregations)
- Highly customizable

**Cons**:
- Complex setup (3 components)
- Expensive to scale (Elasticsearch resource-intensive)
- Maintenance overhead

---

#### **Splunk**

**Commercial log aggregation platform**.

**Pros**:
- Enterprise-grade (reliability, support)
- Advanced analytics (ML-powered anomaly detection)
- Compliance features (audit trails, retention)

**Cons**:
- Very expensive ($150-500/GB ingested)
- Vendor lock-in

**When to use**: Large enterprises with compliance needs.

---

#### **Datadog Logs**

**Cloud-native log aggregation**.

**Pros**:
- Easy setup (agent-based)
- Integrates with Datadog metrics/tracing
- Good UI/UX

**Cons**:
- Expensive (volume-based pricing)
- Limited customization

---

#### **CloudWatch Logs (AWS)**

**Native AWS logging**.

**Pros**:
- Zero setup (auto-collects from AWS services)
- Cheap ($0.50/GB ingested)
- Integrates with AWS ecosystem

**Cons**:
- Basic UI (no advanced analytics)
- AWS-only (not multi-cloud)

---

#### **Tool Comparison**

| Feature | ELK | Splunk | Datadog | CloudWatch |
|---------|-----|--------|---------|------------|
| **Cost** | Free (self-hosted) | $$ | $$ | $ |
| **Setup** | Complex | Easy | Easy | Zero |
| **Search** | Excellent | Excellent | Good | Basic |
| **Analytics** | Good | Excellent | Good | Basic |
| **Scalability** | DIY | Excellent | Excellent | Excellent |
| **Best For** | Tech-savvy teams | Enterprises | Startups/scale-ups | AWS-only |

**Recommendation**:
- **Small teams**: CloudWatch (AWS) or Datadog (multi-cloud)
- **Medium teams**: ELK (self-hosted) or Datadog
- **Large enterprises**: Splunk or Datadog

---

## Interview Questions

### Section 12.1: Training Infrastructure

**Q1: You're managing a cluster with 100 GPUs and 10 teams. Team A needs 64 GPUs for a critical training job, but all GPUs are in use. How do you handle this?**

**Expected Answer**:
1. **Priority-based preemption**: If Team A has higher priority (e.g., production model training), preempt lower-priority jobs
2. **Checkpointing**: Preempted jobs save checkpoint and re-queue
3. **Spot instances**: Spin up additional spot instances for Team A (if budget allows)
4. **Queue management**: Estimate wait time, communicate to Team A
5. **Resource quotas**: Review quotas (is Team A over their fair share?)
6. **Long-term**: Capacity planning (need more GPUs? reserved instances?)

**Follow-up: How do you prevent this situation in the future?**
- Fair-share scheduling (track historical usage)
- Reserved capacity for critical jobs
- Burst capacity (spot instances)
- Better communication (advance notice for large jobs)

---

**Q2: Your training job uses 4x A100 GPUs but each GPU is only 40% utilized. What could be the issue and how would you debug?**

**Expected Answer**:

**Possible Causes**:
1. **Small batch size**: Not fully utilizing GPU
2. **Data loading bottleneck**: GPU waiting for data (CPU-bound)
3. **Synchronization overhead**: Multi-GPU communication inefficient
4. **Model architecture**: Not enough compute (small model)

**Debugging Steps**:
1. **Check GPU utilization**: `nvidia-smi` (continuous monitoring)
2. **Profile data loading**: Add timers (data loading time vs training time)
3. **Increase batch size**: Experiment with 2x, 4x batch size
4. **Optimize data loading**: More workers (`num_workers=8`), prefetching
5. **Profile communication**: Check AllReduce time (distributed training)
6. **Consider mixed precision**: Tensor Cores need sufficient compute

**Solution**:
- If data-bound: Optimize data pipeline
- If compute-bound: Increase batch size or use larger model
- If communication-bound: Optimize distributed training (gradient accumulation, reduce communication frequency)

---

**Q3: Explain the difference between Kubernetes and Slurm for ML workloads. When would you use each?**

**Expected Answer**:

| Aspect | Kubernetes | Slurm |
|--------|------------|-------|
| **Use Case** | Full ML lifecycle (train + serve) | Pure training/HPC |
| **Complexity** | High | Medium |
| **Overhead** | Medium (pods, scheduling) | Low (direct job submission) |
| **GPU Support** | Via plugins (NVIDIA GPU operator) | Native |
| **Flexibility** | High (microservices, auto-scaling) | Limited (batch jobs) |
| **Best For** | Cloud-native, production systems | Research, HPC environments |

**When to use Kubernetes**:
- Full ML platform (training + inference + monitoring)
- Cloud deployment (AWS, GCP, Azure)
- Microservices architecture
- Need auto-scaling and dynamic resource allocation

**When to use Slurm**:
- Research labs (pure training workloads)
- On-premise HPC clusters
- Long-running multi-node jobs (tightly coupled)
- Lower complexity requirements

---

### Section 12.2: CI/CD for ML

**Q4: You deployed a new model to production and accuracy dropped from 92% to 85%. Walk through your debugging process.**

**Expected Answer**:

**Step 1: Rollback (if critical)**:
- Immediately rollback to previous model (minimize user impact)
- Verify old model accuracy restored

**Step 2: Data Quality**:
- Check input data distribution (training vs production)
- Schema changes (missing columns, type changes)
- Feature drift (statistical tests: KS test, PSI)
- Sample production inputs (manually inspect)

**Step 3: Model Issues**:
- Test new model on holdout test set (does it match offline eval?)
- Training-serving skew (features computed differently?)
- Model file corruption (checksum verification)

**Step 4: Code Issues**:
- Review deployment changes (new code, dependencies)
- Test inference pipeline (unit tests, integration tests)
- Check preprocessing logic (normalization, encoding)

**Step 5: Environment**:
- Library versions (PyTorch, NumPy, etc.)
- Hardware differences (CPU vs GPU inference)
- Batch size effects (trained on 32, serving on 1?)

**Step 6: Root Cause**:
- Once identified, document in incident report
- Add tests to prevent recurrence
- Update monitoring/alerting

**Follow-up: How do you prevent this in the future?**
- Shadow deployment (test new model without affecting users)
- Gradual rollout (canary: 1% → 5% → 25% → 100%)
- Automated regression tests (accuracy check before deployment)
- Data validation (schema, distribution checks)

---

**Q5: Design a CI/CD pipeline for ML models. What stages would you include?**

**Expected Answer**:

**Pipeline Stages**:

1. **Code Commit** (GitHub, GitLab):
   - Trigger: Push to main branch or PR

2. **Unit Tests** (5 min):
   - Test data preprocessing
   - Test model inference
   - Test loss functions
   - Coverage: 70%+

3. **Data Validation** (10 min):
   - Schema checks (Great Expectations)
   - Distribution checks (KS test, PSI)
   - Data quality metrics

4. **Model Training** (2-4 hours):
   - Train on training set
   - Hyperparameter: Use config file (YAML)
   - Log to MLflow/W&B

5. **Model Evaluation** (10 min):
   - Evaluate on validation set
   - Metrics: Accuracy, precision, recall, F1, AUC
   - Compare to baseline (must be ≥ baseline)

6. **Integration Tests** (15 min):
   - End-to-end inference test
   - Latency test (< 100ms)
   - Load test (100 req/sec)
   - Memory test (no leaks)

7. **Model Packaging** (5 min):
   - Export model (ONNX, TorchScript)
   - Create Docker image
   - Push to registry

8. **Canary Deployment** (1-24 hours):
   - Deploy to 5% of traffic
   - Monitor metrics (error rate, latency, accuracy)
   - Auto-rollback if metrics degrade

9. **Full Deployment** (1 hour):
   - Deploy to 100% of traffic
   - Blue-green switch
   - Monitor for 24 hours

10. **Post-Deployment**:
    - Update model registry (mark as production)
    - Generate model card
    - Notify stakeholders

**Tools**:
- CI/CD: GitHub Actions, GitLab CI, Jenkins
- Orchestration: Airflow, Kubeflow Pipelines
- Deployment: Kubernetes, AWS SageMaker
- Monitoring: Prometheus, Datadog

---

### Section 12.3: Model Governance

**Q6: Explain how you would implement GDPR's "Right to Explanation" for a credit scoring model.**

**Expected Answer**:

**GDPR Requirement**:
- User has right to know why credit was denied
- Explanation must be clear and actionable

**Implementation**:

**1. Use Interpretable Features**:
- Avoid opaque features (user_id embeddings)
- Use transparent features (income, credit history, debt-to-income ratio)

**2. Model Explainability**:
- **SHAP (SHapley Additive exPlanations)**:
  - Compute feature importance per prediction
  - Example: `credit_utilization: +0.3, payment_history: -0.2, ...`
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Approximate complex model locally with simple model
  - Generate explanations per request

**3. Adverse Action Notice**:
```
Your credit application was denied based on:
1. High credit utilization (75% vs 30% average) → Impact: +30%
2. Short credit history (2 years vs 10 years average) → Impact: +20%
3. Recent missed payment (1 in last 6 months) → Impact: +15%
4. High debt-to-income ratio (45% vs 35% average) → Impact: +10%
```

**4. Actionable Recommendations**:
```
To improve your chances:
- Reduce credit utilization below 30%
- Make on-time payments for next 6 months
- Reduce debt-to-income ratio below 36%
```

**5. Human Review**:
- User can appeal decision
- Human reviews model explanation + additional context
- Human can override model

**6. Audit Trail**:
- Log every prediction (features, model version, explanation)
- Retention: 7 years (compliance requirement)
- Immutable logs (WORM storage)

---

**Q7: How do you detect and mitigate bias in a hiring recommendation system?**

**Expected Answer**:

**Step 1: Bias Detection**

**1. Statistical Parity**:
```python
# Check if hiring rate is equal across demographics
hire_rate_male = predictions[gender == 'M'].mean()
hire_rate_female = predictions[gender == 'F'].mean()

disparity = abs(hire_rate_male - hire_rate_female)
# If disparity > 0.05 (5%), bias exists
```

**2. Equalized Odds**:
```python
# Check if error rates are equal across demographics
tpr_male = true_positive_rate(predictions, labels, gender == 'M')
tpr_female = true_positive_rate(predictions, labels, gender == 'F')

fpr_male = false_positive_rate(predictions, labels, gender == 'M')
fpr_female = false_positive_rate(predictions, labels, gender == 'F')

# Both should be similar (< 5% difference)
```

**3. Audit Across Intersections**:
- Test for race, gender, age, disability
- Test for intersections (e.g., Black women vs white men)

---

**Step 2: Bias Mitigation**

**Pre-processing** (Fix training data):
1. **Balance dataset**: Equal samples per demographic group
2. **Remove biased labels**: Audit historical hiring decisions (may reflect past discrimination)
3. **Sensitive attribute removal**: Don't use gender, race directly

**In-processing** (Modify training):
1. **Adversarial debiasing**:
   - Train model to be unable to predict gender from features
   - Two objectives: Predict hire + Unable to predict gender
2. **Fairness constraints**:
   - Add demographic parity loss term
   - Penalize model if hire rates differ across groups

**Post-processing** (Adjust predictions):
1. **Threshold optimization**:
   - Different thresholds per group (to equalize hire rates)
   - Example: threshold_male = 0.6, threshold_female = 0.5
2. **Calibration**:
   - Adjust probabilities to achieve fairness metric

**Step 3: Ongoing Monitoring**:
- Monitor hire rates by demographic (monthly)
- Alert if disparity exceeds threshold (5%)
- Annual third-party audit

**Step 4: Human-in-the-Loop**:
- Model provides recommendations, not final decisions
- Human reviews borderline cases
- Human can override model (with justification)

---

### Section 12.4: Cost Optimization

**Q8: Your monthly ML training bill is $100K. How would you optimize costs by 50%?**

**Expected Answer**:

**Analysis Phase** (Week 1):
1. **Cost breakdown**: Training (60%), Inference (30%), Storage (10%)
2. **Resource utilization**: Check GPU utilization across all jobs
3. **Identify waste**: Idle resources, over-provisioned instances

**Optimization Strategies**:

**1. Spot Instances** (30-40% savings):
- Move 70-80% of training to spot instances
- Savings: $100K × 0.7 × 0.7 = $49K saved
- Requirement: Checkpointing every 10-30 minutes

**2. Right-Sizing** (10-15% savings):
- Analyze GPU utilization (if <50%, downsize)
- Example: 4x V100 → 1x V100 (save 75%)
- Savings: $100K × 0.15 = $15K

**3. Mixed Precision Training** (10-20% time savings):
- FP16 training (2x faster → 50% cost reduction)
- Savings: $100K × 0.6 × 0.5 × 0.2 = $6K

**4. Training Optimization**:
- Early stopping (stop unpromising runs)
- Hyperparameter tuning with pruning (Optuna)
- Savings: 20% fewer GPU-hours = $12K

**5. Reserved Instances** (30-50% discount for baseline):
- Reserve 30% of usage (always-on jobs)
- Savings: $100K × 0.3 × 0.4 = $12K

**Total Savings**:
- Spot: $49K
- Right-sizing: $15K
- Mixed precision: $6K
- Training optimization: $12K
- Reserved: $12K
- **Total: $94K (94% saved → exceeds 50% goal)**

**Implementation Timeline**:
- Week 2: Enable spot instances (high impact, low effort)
- Week 3: Right-size instances (audit all jobs)
- Week 4: Mixed precision (code changes, testing)
- Month 2: Reserved instances (commit for 1 year)
- Ongoing: Continuous monitoring and optimization

---

**Q9: Spot instances vs Reserved instances vs On-demand. When would you use each?**

**Expected Answer**:

| Use Case | Instance Type | Why |
|----------|---------------|-----|
| **Production Inference** (24/7) | Reserved | Predictable, 30-70% cheaper, always available |
| **Training** (fault-tolerant) | Spot | 70-90% cheaper, checkpointing handles interruptions |
| **Experimentation** (ad-hoc) | On-Demand | Flexibility, no commitment, pay-as-you-go |
| **Baseline Training** (continuous) | Reserved | Predictable usage, significant savings |
| **Peak Traffic** (burst capacity) | On-Demand | Short duration, auto-scaling |

**Hybrid Strategy** (Recommended):
- **Reserved** (30-40%): Cover baseline usage (always-on inference, continuous training)
- **Spot** (40-50%): Cost-optimized training (interruption-tolerant)
- **On-Demand** (10-20%): Handle spikes, critical workloads, fallback for spot

**Example (Netflix)**:
- Reserved: Production inference (recommendation API)
- Spot: Experimentation, hyperparameter tuning
- On-Demand: Traffic spikes (new show release), fallback for spot

**Decision Criteria**:
- **Criticality**: Production → Reserved, Experimentation → Spot
- **Predictability**: Predictable → Reserved, Variable → Spot/On-Demand
- **Duration**: Long-term (>1 year) → Reserved, Short-term → On-Demand
- **Fault-tolerance**: Tolerant → Spot, Intolerant → Reserved/On-Demand

---

### Section 12.5: Observability

**Q10: Your model's P99 latency spiked from 100ms to 2s. How do you debug this?**

**Expected Answer**:

**Step 1: Confirm and Scope**:
- Check metrics dashboard (Grafana, CloudWatch)
- Confirm spike (not just noisy data point)
- Scope: All users or specific segment? (region, model version)

**Step 2: Distributed Tracing**:
- Pull traces for slow requests
- Identify bottleneck (which component is slow?)
```
Example Trace:
Total: 2000ms
├─ API Gateway: 10ms
├─ Auth: 5ms
├─ Model Service: 1900ms ← Bottleneck
   ├─ Feature Fetch: 50ms
   └─ Model Inference: 1850ms ← Root cause
```

**Step 3: Model Inference Analysis**:

**Possible Causes**:
1. **Cold start**: Model not loaded in memory
2. **Large batch**: Batch size too large (OOM, slow)
3. **GC pause**: Garbage collection (Python/JVM)
4. **Slow GPU transfer**: CPU → GPU transfer slow
5. **Network congestion**: Multi-node inference

**Debugging**:
- Check model loading time (first request vs subsequent)
- Check batch size (recent config change?)
- Check memory usage (approaching limit → swapping, GC)
- Check GPU utilization (low util → transfer bottleneck)
- Check system metrics (CPU, memory, network)

**Step 4: Logs**:
- Search for errors, warnings around spike time
- Check for OOM errors, timeouts
- Check for deployment events (new model deployed?)

**Step 5: Correlation**:
- Did traffic spike? (more requests → queuing)
- Did model change? (new version deployed?)
- Did data change? (larger inputs → slower inference)

**Step 6: Root Cause (Example)**:
- New model version deployed with larger architecture
- Inference time: 50ms → 1850ms (37x slower)
- Solution: Rollback to previous version OR optimize new model (quantization, smaller batch size)

**Prevention**:
- Load testing before deployment
- Canary deployment (catch issue on 5% traffic)
- Latency alerts (P99 > 500ms)

---

**Q11: Design a monitoring and alerting strategy for an ML model serving 1M requests/day.**

**Expected Answer**:

**Metrics to Track**:

**1. Request Metrics**:
- Request rate (req/sec)
- Error rate (% of requests)
- Latency (P50, P95, P99)
- Throughput (successful requests/sec)

**2. Model Metrics**:
- Prediction distribution (class imbalance)
- Confidence scores (average, distribution)
- Model version (which version served each request)

**3. System Metrics**:
- CPU utilization (%)
- Memory usage (GB)
- GPU utilization (%)
- GPU memory (GB)

**4. Business Metrics**:
- Prediction correctness (when ground truth available)
- User engagement (CTR, conversion)
- Revenue impact

---

**Alerting Strategy**:

**1. Latency Alerts**:
```yaml
Alert: HighLatency
Condition: P99 latency > 500ms for 10 minutes
Severity: P2
Action: Slack notification + PagerDuty
Runbook: Check traces, scale up, investigate bottleneck
```

**2. Error Rate Alerts**:
```yaml
Alert: HighErrorRate
Condition: Error rate > 1% for 5 minutes
Severity: P1
Action: Page on-call immediately
Runbook: Check logs, rollback if recent deployment, investigate
```

**3. Model Accuracy Alerts**:
```yaml
Alert: ModelAccuracyDrop
Condition: Accuracy < 85% (when labels available)
Severity: P2
Action: Slack notification to ML team
Runbook: Check for data drift, retrain model, investigate input distribution
```

**4. Resource Alerts**:
```yaml
Alert: HighMemoryUsage
Condition: Memory > 90% for 5 minutes
Severity: P1
Action: Auto-scale + page on-call
Runbook: Scale horizontally, investigate memory leak
```

**5. Data Drift Alerts**:
```yaml
Alert: DataDrift
Condition: Input distribution shift (KS test p-value < 0.01)
Severity: P3
Action: Ticket to ML team
Runbook: Investigate cause, consider retraining
```

---

**Dashboards**:

**1. Real-time Dashboard** (Grafana):
- Request rate, error rate, latency (last 1 hour)
- System metrics (CPU, memory, GPU)
- Model predictions (class distribution)

**2. Model Performance Dashboard**:
- Accuracy over time (when labels available)
- Prediction distribution
- Confidence score distribution
- Model version breakdown

**3. Cost Dashboard**:
- Inference cost per 1K requests
- Daily/monthly spend
- Cost by model version

---

**Logging**:
- Structured logs (JSON)
- Log every request (input, output, latency, model version)
- Retention: 30 days (hot), 1 year (cold)
- Sampling: 100% for errors, 10% for success (reduce volume)

---

**Tracing**:
- Distributed tracing (Jaeger, Zipkin)
- Trace every request (100% sampling for first week, then 1%)
- Identify bottlenecks (slow components)

---

**Monitoring Tools**:
- Metrics: Prometheus + Grafana
- Logs: ELK Stack or Datadog
- Tracing: Jaeger or Datadog APM
- Alerting: PagerDuty + Slack

---

---

## RAPID-FIRE INTERVIEW PREP: KEY FACTS TO MEMORIZE

### Quick Numbers (Memorize These!)

**GPU Costs (AWS On-Demand, 2024)**:
- V100 (16GB): **$3.06/hour** (p3.2xlarge)
- A100 (40GB): **$32.77/hour** (p4d.24xlarge)
- H100 (80GB): **~$50/hour** (p5.48xlarge)
- Spot discount: **70-90%** off
- Reserved discount: **30-70%** off

**Spot Instance Interruption Rates**:
- p3 instances: **5-10%** interruption rate
- g4 instances: **3-5%** interruption rate
- Checkpoint every **10-30 minutes** to minimize loss

**Training Speed-ups**:
- Mixed precision (FP16): **2-3x faster**
- Gradient accumulation: Same speed, **4x less memory**
- Gradient checkpointing: **50% less memory, 20% slower**
- Multi-GPU (4 GPUs): **3.5x speedup** (not 4x due to communication)

**Experiment Tracking Market Share**:
- Weights & Biases: **~40%** (startups/research)
- MLflow: **~30%** (enterprises, self-hosted)
- Neptune: **~15%** (regulated industries)
- Others: **~15%**

**Hyperparameter Tuning**:
- Random search: **10-20x** more efficient than grid search
- Bayesian optimization: Finds near-optimal in **20-50 trials**
- Grid search on 5 params (3 values each): **243 trials** (3^5)

**Kubernetes Overhead**:
- Pod startup time: **5-30 seconds**
- Node provisioning: **2-5 minutes**
- Good for: Jobs **>10 minutes** duration

**CI/CD Pipeline Times**:
- Unit tests: **5 minutes**
- Integration tests: **15 minutes**
- Model training: **2-4 hours** (small model)
- Canary deployment: **1-24 hours** (monitor before full rollout)

**Compliance Fines**:
- GDPR: Up to **4% of global revenue** or €20M
- CCPA: **$2,500 per violation** ($7,500 intentional)
- Average data breach cost: **$4.45M** (IBM 2023)

**Monitoring Best Practices**:
- Log retention: **30 days hot, 1 year cold**
- Metrics resolution: **1s for 1 day, 1min for 7 days, 1hr for 1 year**
- Alert response: **<5 min** (P1), **<30 min** (P2)
- Log sampling: **100% errors, 10% success**

**Observability Costs**:
- CloudWatch Logs: **$0.50/GB** ingested
- Datadog Logs: **$1.70/GB** ingested
- Splunk: **$150-500/GB** ingested
- ELK (self-hosted): **$0.10-0.30/GB** (compute + storage)

---

## MISSING PRODUCTION PATTERNS (ADDED)

### 12.1A Advanced Training Infrastructure Patterns

#### **GPU Pooling and Dynamic Allocation**

**THE REAL PROBLEM (From Reddit r/MachineLearning & HN discussions)**:

You join a company with 100 GPUs. Sounds great, right? Wrong.

**Day 1 Reality**:
- Team A: Reserved 40 GPUs (using 15, rest idle)
- Team B: Reserved 30 GPUs (using 8, rest idle)  
- Team C: Reserved 20 GPUs (using 20, fully utilized)
- Team D: Reserved 10 GPUs (using 2, rest idle)
- **You**: Need 8 GPUs. All "reserved." Wait 3 days.

**Actual GPU Utilization**: 45/100 = 45% (rest sitting idle!)

**Why This Happens** (Real stories from Blind, TeamBlind, Reddit):

1. **"Just in case" reservations**: 
   - PM: "We MIGHT need 40 GPUs next week for the big experiment"
   - Reality: Experiment takes 3 days to prep, GPUs sit idle
   
2. **Long-running jobs with low utilization**:
   - Engineer starts 7-day training job
   - Job crashes after 2 hours (out of memory)
   - GPUs locked for 7 days, actually unused after 2 hours
   - Engineer on vacation, doesn't notice
   
3. **No accountability**:
   - "It's reserved, I can do whatever I want"
   - No one tracks actual usage vs reservation
   
4. **No preemption**:
   - Low-priority hyperparameter sweep holds 16 GPUs
   - High-priority production model training blocked
   - Can't preempt (no system in place)

**Real Cost** (Uber's 2019 blog post numbers):
- 1000 GPUs at $3/hour = $3,000/hour = $72,000/day
- 45% utilization = wasting $32,400/day = $1M/month wasted
- CFO not happy

---

**WHY TRADITIONAL SOLUTIONS FAILED**:

**Attempt 1: "Be reasonable, people!"**
- Result: Nobody changed behavior
- Why: No incentive, no enforcement

**Attempt 2: Quota system (each team gets fixed GPUs)**
- Problem: Team A needs 50 GPUs this week, 5 next week
- Team B needs 5 GPUs this week, 50 next week
- Fixed quotas don't adapt → still wasted

**Attempt 3: First-come-first-served queue**
- Problem: Long jobs (7 days) block short jobs (1 hour)
- "Convoy effect" - short jobs wait forever
- Engineers game the system (submit placeholder jobs to hold spot)

---

**HOW UBER ACTUALLY SOLVED THIS** (Engineering blog + Peloton paper):

**Dynamic GPU Pool with Intelligent Scheduling**:

**Component 1: No Permanent Reservations**
```yaml
Old way:
  Team A: 40 GPUs (permanent)
  Team B: 30 GPUs (permanent)
  
New way:
  Shared pool: 100 GPUs
  Request GPUs when needed
  Return GPUs when done
```

**Component 2: Real-time Utilization Monitoring**
```python
# Monitor every job every 30 seconds
while job_running:
    gpu_util = nvidia_smi_query()  # Returns 0-100%
    
    if gpu_util < 20% for 15 minutes:
        # Send warning to engineer
        send_slack_message(
            f"⚠️ Your job {job_id} has <20% GPU utilization "
            f"for 15 minutes. Will be killed in 5 minutes if not fixed."
        )
        
        wait(5 minutes)
        
        if still_low_utilization:
            # Save checkpoint and kill
            save_checkpoint(job_id)
            kill_job(job_id)
            release_gpus(job_id)
            
            log_to_dashboard(
                f"Job {job_id} killed due to low utilization. "
                f"Engineer notified. Checkpoint saved."
            )
```

**Why this works**:
- Data loading too slow? Engineer sees warning, fixes data pipeline
- Debugging with pdb? Warning reminds engineer to kill job manually
- Job actually needs low utilization? Engineer can justify (gets exemption)

**Component 3: Priority-Based Preemption**
```python
# Job submission with priority
submit_job(
    gpus_needed=16,
    priority="high",  # production model training
    preemptible=False  # cannot be killed
)

submit_job(
    gpus_needed=8,
    priority="low",  # hyperparameter sweep
    preemptible=True  # can be killed if needed
)

# Scheduler logic
def schedule_job(new_job):
    available_gpus = count_free_gpus()
    
    if available_gpus >= new_job.gpus_needed:
        allocate_gpus(new_job)
    else:
        # Not enough GPUs. Can we preempt?
        if new_job.priority == "high":
            # Find lowest priority preemptible jobs
            victims = find_preemptible_jobs(
                priority="low",
                gpus_needed=new_job.gpus_needed
            )
            
            if victims:
                for job in victims:
                    checkpoint_and_kill(job)  # Save progress
                    requeue_job(job)  # Will restart later
                
                allocate_gpus(new_job)
            else:
                queue_job(new_job)  # Wait for GPUs
        else:
            queue_job(new_job)  # Low priority, must wait
```

**Real Example** (from Uber's metrics):
```
11:00 AM: Team A running hyperparameter sweep (32 GPUs, priority=low)
11:15 AM: Team B submits production model training (32 GPUs, priority=high)
11:15 AM: Scheduler preempts Team A's jobs
  - Team A jobs: Save checkpoint, killed
  - Team B jobs: Start immediately
11:20 AM: Team B training starts (no wait!)
6:00 PM: Team B training completes, releases GPUs
6:01 PM: Team A jobs auto-resume from checkpoint

Result: 
  - Team B: Zero wait time (critical for production)
  - Team A: Lost 5 hours of progress, BUT low-priority (acceptable)
  - Company: High-priority work never blocked
```

**Component 4: Fair-Share Scheduling**
```python
# Track historical usage per team
team_usage = {
    "Team A": 10000,  # GPU-hours used this month
    "Team B": 5000,
    "Team C": 2000,
}

# Adjust priority based on usage
def calculate_priority(team, base_priority):
    avg_usage = sum(team_usage.values()) / len(team_usage)
    team_usage_ratio = team_usage[team] / avg_usage
    
    # Teams that used less get priority boost
    adjusted_priority = base_priority / team_usage_ratio
    
    return adjusted_priority

# Example:
# Team A used 2x average -> priority penalty
# Team C used 0.4x average -> priority boost
```

**Why this matters**:
- Prevents one team from hogging all GPUs all month
- "Rich get richer" problem solved
- Every team gets fair access over time

---

**RESULTS FROM UBER** (Published metrics):

**Before Dynamic Pooling**:
- GPU Utilization: 45% (1000 GPUs, only 450 in use)
- Average queue time: 8 hours (engineers waiting for GPUs)
- Cost: $3M/month ($72k/day × 30 × 1.4x overhead from idle)
- Engineer satisfaction: 3.2/5 (frustrated by waiting)

**After Dynamic Pooling**:
- GPU Utilization: 85% (1000 GPUs, 850 in use)
- Average queue time: 15 minutes (high priority), 2 hours (low priority)
- Cost savings: $1.2M/month (40% reduction)
- Engineer satisfaction: 4.5/5 (less waiting, more productive)

**Key Metrics** (from their dashboard):
- Preemptions per day: ~50 (out of 500 jobs = 10%)
- Preempted jobs successfully resumed: 98% (checkpoint system works!)
- Time to preempt and restart: <5 minutes (fast checkpoint saving)
- Jobs killed for low utilization: ~20/day (engineers now monitor usage!)

---

**REAL IMPLEMENTATION DETAILS** (From their open-source Peloton):

**Component 5: Gang Scheduling** (All-or-nothing GPU allocation)
```python
# Problem: Distributed training needs 8 GPUs
# If only 6 available, job can't start
# Partial allocation = wasted resources

# Gang scheduling: Reserve all 8 or none
def gang_schedule(job):
    required_gpus = job.num_gpus
    
    # Atomic reservation: lock all GPUs at once
    with gpu_pool_lock:
        available_gpus = get_available_gpus()
        
        if len(available_gpus) >= required_gpus:
            allocate_gpus(job, available_gpus[:required_gpus])
            return True
        else:
            # Don't allocate partial GPUs
            return False  # Job stays in queue
```

**Why this matters**:
- Distributed training with 7/8 GPUs = doesn't work (synchronization barrier)
- Allocating 7 GPUs and waiting for 8th = wasted 7 GPUs
- Gang scheduling prevents this waste

**Component 6: Backfilling** (Use gaps intelligently)
```python
# Main job queue
queue = [
    Job(id=1, gpus=64, duration=8_hours, priority=high),  # Waiting
    Job(id=2, gpus=8, duration=1_hour, priority=low),    # Waiting
    Job(id=3, gpus=4, duration=30_min, priority=low),    # Waiting
]

# Current state: 60 GPUs in use, 40 GPUs free
# Job 1 needs 64 GPUs, must wait for 4 more to free up
# Expected wait: 2 hours (when current job finishes)

# Backfilling: Run small jobs in the gap!
def backfill():
    free_gpus = 40
    time_until_job1_starts = 2_hours
    
    # Can we run Job 2 (8 GPUs, 1 hour) without delaying Job 1?
    if Job2.gpus <= free_gpus and Job2.duration < time_until_job1_starts:
        schedule(Job2)  # Yes! Run Job 2 now
        free_gpus -= Job2.gpus  # 32 GPUs still free
    
    # Can we run Job 3 (4 GPUs, 30 min)?
    if Job3.gpus <= free_gpus and Job3.duration < time_until_job1_starts:
        schedule(Job3)  # Yes! Run Job 3 now
        free_gpus -= Job3.gpus  # 28 GPUs still free

# Result:
# - Job 1 still starts in 2 hours (not delayed)
# - Jobs 2 and 3 complete in the meantime (bonus!)
# - GPU utilization: 60 + 8 + 4 = 72/100 = 72% (vs 60% without backfilling)
```

**Backfilling Impact** (Uber's metrics):
- GPU utilization: 85% → 92% (7% improvement from backfilling)
- Short jobs (<1 hour): Average wait time reduced from 30 min to 5 min
- Long jobs: No impact on wait time (still get priority)

---

**CHALLENGES & SOLUTIONS** (Real issues from production):

**Challenge 1: "My job was killed during critical experiment!"**

Problem:
- Engineer running 3-day experiment to hit conference deadline
- Job preempted after 2 days (high-priority job arrived)
- Engineer furious: "Lost 2 days of work!"

Solution (Uber's approach):
```python
# Checkpoint granularity: Save every 15 minutes
def training_loop():
    for epoch in range(100):
        for batch_idx, batch in enumerate(dataloader):
            loss = train_step(batch)
            
            # Checkpoint every 50 batches (~15 minutes)
            if batch_idx % 50 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                    'timestamp': datetime.now(),
                }
                torch.save(checkpoint, f'ckpt_epoch{epoch}_batch{batch_idx}.pt')
                upload_to_s3(checkpoint)  # Async

# On preemption: Lost at most 15 minutes, not 2 days!
```

**Additional Safety**: Priority escalation
```python
# Job close to deadline gets priority boost
if job.deadline - now() < 24_hours:
    job.priority = "critical"  # Cannot be preempted
```

---

**Challenge 2: "Checkpoint saving is too slow! (30 seconds)"**

Problem:
- Large model (7B params = 14GB checkpoint)
- Saving to shared NFS: 30 seconds (blocks training)
- Checkpoint every 15 minutes = 3% overhead (acceptable)
- But preemption often = multiple checkpoints close together = 10%+ overhead

Solution (Meta's approach for Llama):
```python
# Distributed checkpoint saving (FSDP)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint import save

# Each GPU saves only its shard (parallel I/O)
# 8 GPUs, each saves 1.75GB = 8x faster than 1 GPU saving 14GB

with FSDP(model):
    save(
        model.state_dict(),
        checkpoint_id=f"ckpt_{epoch}",
        storage_writer=S3Writer("s3://checkpoints/")
    )
# Time: 30 seconds → 4 seconds (7.5x faster!)
```

**Plus**: Asynchronous saving
```python
# Don't block training during checkpoint save
import threading

def save_checkpoint_async(model, epoch):
    threading.Thread(
        target=torch.save,
        args=(model.state_dict(), f'ckpt_{epoch}.pt')
    ).start()

# Training continues immediately, save happens in background
```

---

**Challenge 3: "How do we prevent gaming the system?"**

Problem:
- Engineer discovers: "If I submit job with priority=high, it starts immediately!"
- Everyone sets priority=high → priority system useless

Solution (Uber's approach):
```python
# Priority is NOT user-specified, it's calculated by system

def calculate_job_priority(job, user, team):
    base_priority = 100
    
    # Factor 1: Team's fair-share (used less = higher priority)
    fair_share_bonus = (avg_team_usage / team_usage[team]) * 50
    
    # Factor 2: Deadline proximity
    if job.deadline:
        hours_until_deadline = (job.deadline - now()).hours
        deadline_bonus = max(0, 100 - hours_until_deadline)  # Closer deadline = higher priority
    else:
        deadline_bonus = 0
    
    # Factor 3: User seniority (slight boost for senior engineers)
    seniority_bonus = user.years_at_company * 2
    
    # Factor 4: Job type
    if job.type == "production_model":
        type_bonus = 100  # Production always high priority
    elif job.type == "experimentation":
        type_bonus = 0
    else:
        type_bonus = 50  # Research: medium priority
    
    priority = base_priority + fair_share_bonus + deadline_bonus + seniority_bonus + type_bonus
    
    # Manager override (for emergencies)
    if job.manager_override:
        priority += 200
        log_override(job, user, reason=job.override_reason)
    
    return priority
```

**Result**:
- Gaming prevented: Priority is calculated, not user-specified
- Fair: Multiple factors considered (not just first-come-first-served)
- Transparent: Engineers can see priority calculation
- Auditable: Manager overrides logged (prevent abuse)

---

**Challenge 4: "My training crashes randomly after preemption!"**

Problem:
- Job preempted, checkpointed, re-queued
- Job restarts from checkpoint
- Sometimes crashes immediately: "RuntimeError: CUDA out of memory"

**Root cause** (from debugging):
- Checkpoint saves model + optimizer state
- On resume, optimizer state loaded incorrectly
- Optimizer state has accumulated gradients (large memory)
- GPU memory: training state + optimizer state = OOM!

Solution:
```python
# Checkpoint saving: Include ALL state, not just model
checkpoint = {
    'epoch': epoch,
    'batch_idx': batch_idx,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),  # For mixed precision
    'scheduler': scheduler.state_dict(),  # Learning rate schedule
    'rng_state': torch.get_rng_state(),  # Random number generator
    'cuda_rng_state': torch.cuda.get_rng_state(),  # CUDA RNG
    'dataloader_state': dataloader.state_dict(),  # Data position
}

# Checkpoint loading: Restore ALL state
def load_checkpoint(path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    scheduler.load_state_dict(ckpt['scheduler'])
    torch.set_rng_state(ckpt['rng_state'])
    torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
    dataloader.load_state_dict(ckpt['dataloader_state'])
    
    # Explicitly clear GPU cache before resuming
    torch.cuda.empty_cache()
    
    return ckpt['epoch'], ckpt['batch_idx']
```

**Additional safety**: Test checkpoint resume in CI
```python
# tests/test_checkpoint_resume.py
def test_checkpoint_resume():
    # Train for 10 steps
    model, optimizer = setup()
    for i in range(10):
        loss = train_step(model, optimizer)
    
    # Save checkpoint
    save_checkpoint('test_ckpt.pt', model, optimizer, epoch=0, step=10)
    
    # Load checkpoint and continue
    model2, optimizer2 = setup()
    load_checkpoint('test_ckpt.pt', model2, optimizer2)
    
    # Continue training for 10 more steps
    for i in range(10):
        loss = train_step(model2, optimizer2)
    
    # Should not crash (if crashes, checkpoint system is broken)
    assert True  # Test passes if no crash
```

---

**LESSONS FROM REAL PRODUCTION** (Uber's retrospective blog post):

**What worked**:
1. ✅ Dynamic pooling: 45% → 85% utilization (massive win)
2. ✅ Automatic low-utilization killing: Engineers now optimize data pipelines
3. ✅ Preemption: High-priority work never blocked
4. ✅ Backfilling: Squeezed 7% more utilization
5. ✅ Fair-share scheduling: No team monopolizes resources

**What didn't work initially**:
1. ❌ Too aggressive preemption: Initially preempted every hour → too disruptive
   - Fix: Only preempt if high-priority job waiting >30 minutes
2. ❌ Checkpoint overhead: 30s save time too long
   - Fix: Distributed checkpointing (8x faster)
3. ❌ Engineers gaming the system: Fake deadlines, inflated priority
   - Fix: System-calculated priority, not user-specified
4. ❌ Lack of transparency: Engineers didn't understand why jobs killed
   - Fix: Detailed dashboard showing utilization, priority, queue position
5. ❌ No accountability: Low-utilization jobs killed without warning
   - Fix: 5-minute warning via Slack before killing

**Cultural Change Required**:
- Before: "My GPUs, my rules" (entitlement mentality)
- After: "Shared resource, be responsible" (community mindset)
- How: Weekly reports showing each team's utilization + waste
  - Top team: 95% utilization (praised publicly)
  - Bottom team: 40% utilization (manager follow-up)

**Timeline to Full Adoption**:
- Month 1: Pilot with 2 teams (100 GPUs)
- Month 2: Expand to 5 teams (300 GPUs)
- Month 3-4: Debug edge cases (checkpoint issues, priority gaming)
- Month 5: Full rollout (1000 GPUs)
- Month 6: System stable, engineer satisfaction high

**ROI Calculation**:
- Engineering cost: 2 engineers × 6 months × $200k/year = $200k
- Ongoing maintenance: 0.5 engineer × $200k/year = $100k/year
- Savings: $1.2M/month × 12 = $14.4M/year
- ROI: $14.4M / $200k = **72x return in first year**

---

**IMPLEMENTATION (Kubernetes + Custom Controller)**:

```yaml
# Custom Resource Definition for ML jobs
apiVersion: ml.uber.com/v1
kind: TrainingJob
metadata:
  name: llm-training
spec:
  priority: high  # Calculated by controller, not user-specified
  gpus: 8
  preemptible: false
  checkpoint_interval: 15m
  max_duration: 7d
  team: ml-research
  user: alice@uber.com
  
  # Resource requirements
  resources:
    gpu_type: A100
    gpu_memory: 40GB
    cpu_cores: 64
    memory: 512GB
  
  # Training command
  command: ["python", "train.py"]
  args: ["--config", "config.yaml"]
  
  # Checkpoint configuration
  checkpoint:
    path: s3://uber-ml/checkpoints/llm-training/
    interval: 15m
    async: true

---

#### **Checkpoint Sharding for Large Models**

**Problem**: 7B model checkpoint = 14GB (FP16). Saving to disk = 30+ seconds (blocks training)

**Solution (Meta's approach for Llama)**:
```python
# Distributed checkpoint saving (PyTorch FSDP)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint import save_state_dict

# Each rank saves only its shard
save_state_dict(
    state_dict=model.state_dict(),
    storage_writer=FileSystemWriter("checkpoints/"),
    planner=DefaultSavePlanner(),
)
# Total time: 3 seconds (10x faster than centralized)
```

**Benefits**:
- Parallel I/O (8 GPUs → 8x faster)
- No single-node bottleneck
- Scales to 1000s of GPUs

---

#### **Training Interruption Handling (Production-Grade)**

**Kubernetes SIGTERM Handler**:
```python
import signal
import sys
import os

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/mnt/checkpoints")
GRACEFUL_SHUTDOWN_TIMEOUT = 100  # seconds (K8s gives 120s)

def graceful_shutdown(signum, frame):
    print(f"[SIGTERM] Received signal {signum}. Saving checkpoint...")
    
    # Save checkpoint (fast!)
    checkpoint_path = f"{CHECKPOINT_DIR}/epoch_{epoch}_batch_{batch_idx}.pt"
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    # Upload to S3 (async)
    upload_to_s3_async(checkpoint_path)
    
    print(f"[SIGTERM] Checkpoint saved to {checkpoint_path}. Exiting gracefully.")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGTERM, graceful_shutdown)

# Training loop continues normally
for epoch in range(100):
    for batch_idx, batch in enumerate(data_loader):
        loss = train_step(model, batch)
```

**Key Points**:
- Kubernetes sends SIGTERM **2 minutes before killing pod**
- Must save checkpoint in <100 seconds (leave buffer)
- Job auto-restarts from checkpoint (pod controller)

---

#### **Heterogeneous GPU Training (Mixed GPU Types)**

**Problem**: Team has 4x V100 + 4x A100. Can we train on both simultaneously?

**Solution (NVIDIA's approach)**:
```python
# DeepSpeed heterogeneous pipeline parallelism
import deepspeed

# Stage 0 (embedding): V100 (less compute-intensive)
# Stage 1-6 (transformer): A100 (compute-intensive)
# Stage 7 (output): V100

pipe = PipelineModule(layers=[...], num_stages=8)
pipe.to_device([
    torch.device('cuda:0'),  # V100
    torch.device('cuda:4'),  # A100
    torch.device('cuda:5'),  # A100
    torch.device('cuda:6'),  # A100
    torch.device('cuda:7'),  # A100
    torch.device('cuda:8'),  # A100
    torch.device('cuda:9'),  # A100
    torch.device('cuda:1'),  # V100
])
```

**Caveat**: Slower GPU becomes bottleneck. Only useful if:
- Pipeline parallelism (no synchronous AllReduce)
- Workload balanced (A100 does 2x work of V100)

---

### 12.1B Real-World Experiment Tracking Tricks

#### **Experiment Reproducibility Checklist**

**What to Log** (Uber's standard):
```yaml
Experiment Metadata:
  - git_commit_hash: "a3f4d2c"
  - git_branch: "main"
  - dirty_working_tree: false  # Uncommitted changes?
  - dependencies:
      - torch: "2.0.1"
      - transformers: "4.30.0"
      - cuda: "11.8"
  - hardware:
      - gpu_type: "A100-40GB"
      - num_gpus: 4
      - cpu_cores: 64
      - ram_gb: 512
  - random_seeds:
      - python: 42
      - numpy: 42
      - torch: 42
      - cuda: 42
  - dataset:
      - name: "wikitext-103"
      - version: "v2.1"
      - hash: "sha256:abc123..."
  - hyperparameters:
      - learning_rate: 0.001
      - batch_size: 32
      - epochs: 100
  - training_time_seconds: 14400
  - cost_usd: 123.45
```

**Why This Matters**:
- Reproduce exact results months later
- Debug regressions (what changed?)
- Compliance audits (prove model was trained on authorized data)

---

#### **Automatic Hyperparameter Tracking**

**Problem**: Engineers forget to log hyperparameters manually

**Solution (W&B auto-logging)**:
```python
import wandb

# Auto-log everything in config
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "model": "bert-base",
}

wandb.init(project="llm-training", config=config)

# W&B automatically logs:
# - Git commit, branch, diff
# - System info (GPU, CPU, RAM)
# - Python version, dependencies
# - Command line arguments
# - Environment variables (filtered)

# Training loop
for epoch in range(config["epochs"]):
    loss = train_epoch()
    wandb.log({"loss": loss, "epoch": epoch})
```

**Production Tip**: Use **W&B Artifacts** for dataset versioning:
```python
# Save dataset as artifact
artifact = wandb.Artifact("training-data", type="dataset")
artifact.add_file("train.csv")
artifact.add_file("val.csv")
wandb.log_artifact(artifact)

# Later, load exact same dataset
artifact = wandb.use_artifact("training-data:v3")
artifact_dir = artifact.download()
```

---

#### **Comparing 100s of Experiments (Netflix's approach)**

**Problem**: After 200 hyperparameter trials, how do we find the best?

**Solution (W&B Sweeps + Parallel Coordinates)**:
```python
# Define sweep
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
        'batch_size': {'values': [16, 32, 64, 128]},
        'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
    }
}

sweep_id = wandb.sweep(sweep_config, project="llm-tuning")
wandb.agent(sweep_id, function=train, count=200)

# W&B automatically creates:
# 1. Parallel coordinates plot (see correlations)
# 2. Hyperparameter importance (which params matter?)
# 3. Best run (highest val_accuracy)
```

**Visualization**: Parallel coordinates shows:
- Learning rate = 0.0003 is optimal (not 0.001)
- Batch size doesn't matter much (flat line)
- Dropout > 0.3 hurts performance (downward trend)

---

### 12.2A Advanced CI/CD Patterns

#### **Shadow Mode Testing (Netflix's pattern)**

**Problem**: Want to test new model without affecting users

**Solution**:
```python
# Production API (Flask)
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    
    # Primary model (affects user)
    prediction_v1 = model_v1.predict(input_data)
    
    # Shadow model (logged only, no user impact)
    if SHADOW_MODE_ENABLED:
        asyncio.create_task(shadow_predict(model_v2, input_data, prediction_v1))
    
    return jsonify({"prediction": prediction_v1})

async def shadow_predict(shadow_model, input_data, baseline_prediction):
    prediction_v2 = shadow_model.predict(input_data)
    
    # Log comparison
    logger.info({
        "shadow_test": True,
        "baseline_prediction": baseline_prediction,
        "shadow_prediction": prediction_v2,
        "agreement": (prediction_v2 == baseline_prediction),
        "input_hash": hashlib.md5(str(input_data).encode()).hexdigest()
    })
```

**Analysis** (after 24 hours):
```sql
-- Compare shadow model to baseline
SELECT 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN agreement = true THEN 1 ELSE 0 END) as agreements,
    AVG(CASE WHEN agreement = true THEN 1 ELSE 0 END) as agreement_rate
FROM shadow_logs
WHERE timestamp > NOW() - INTERVAL '24 hours';

-- Result: 94.5% agreement → Safe to deploy
```

---

#### **Automatic Rollback Triggers**

**Problem**: Model degrades after deployment. How fast can we rollback?

**Solution (Uber's pattern)**:
```yaml
# Kubernetes deployment with auto-rollback
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-service
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: model
        image: model:v2.4
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
# Prometheus alert (trigger rollback)
- alert: ModelAccuracyDrop
  expr: model_accuracy < 0.85
  for: 5m
  annotations:
    summary: "Model accuracy dropped below 85%"
  # Webhook triggers: kubectl rollout undo deployment/model-service
```

**Automatic Rollback Script**:
```bash
#!/bin/bash
# Called by Prometheus webhook

DEPLOYMENT="model-service"
NAMESPACE="production"

# Check current error rate
ERROR_RATE=$(curl -s http://prometheus:9090/api/v1/query?query=error_rate | jq '.data.result[0].value[1]')

if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
    echo "[ALERT] Error rate $ERROR_RATE > 5%. Rolling back..."
    kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE
    kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE
    
    # Notify team
    curl -X POST https://slack.com/api/chat.postMessage \
        -H "Authorization: Bearer $SLACK_TOKEN" \
        -d "channel=#ml-alerts" \
        -d "text=🚨 Auto-rollback triggered for $DEPLOYMENT. Error rate: $ERROR_RATE"
fi
```

**Result**: Rollback in **<2 minutes** (vs 30 minutes manual)

---

#### **Feature Store Integration Pattern**

**Problem**: Training uses batch features. Serving needs real-time features. Features computed differently → **Training-serving skew**

**Solution (Tecton / Feast pattern)**:
```python
# Feature definitions (shared between training and serving)
from feast import Entity, Feature, FeatureView, Field
from feast.types import Float32, Int64

user = Entity(name="user_id", join_keys=["user_id"])

user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="avg_purchase_amount", dtype=Float32),
        Field(name="num_purchases_30d", dtype=Int64),
    ],
    source=BigQuerySource(...)  # Batch source
)

# Training (batch)
training_data = feast_client.get_historical_features(
    entity_df=user_ids,
    features=["user_features:age", "user_features:avg_purchase_amount"]
)

# Serving (online)
online_features = feast_client.get_online_features(
    features=["user_features:age", "user_features:avg_purchase_amount"],
    entity_rows=[{"user_id": 12345}]
)
```

**Key Benefits**:
- **Same code path** for training and serving (no skew!)
- **Point-in-time correctness** (training uses features as of event time, not current time)
- **Low latency** (<10ms for online features)

**Production Example (DoorDash)**:
- 500+ features in Feast
- Training-serving skew eliminated (was #1 cause of model failures)
- Feature serving latency: P99 < 15ms

---

### 12.3A Advanced Governance Patterns

#### **Model Risk Tiering (Banking standard)**

**Federal Reserve SR 11-7 Compliance**:
```yaml
Model Risk Tiers:

Tier 1 (High Risk):
  - Use cases: Credit decisions, fraud detection, AML
  - Validation: Full independent validation required
  - Frequency: Annual
  - Documentation: Comprehensive model card + 50-page validation report
  - Approvals: Model Risk Management + Board of Directors
  - Testing: Stress testing, adversarial testing, bias testing
  
Tier 2 (Medium Risk):
  - Use cases: Marketing models, churn prediction
  - Validation: Peer review by senior DS
  - Frequency: Biannual
  - Documentation: Standard model card
  - Approvals: Model Risk Management
  
Tier 3 (Low Risk):
  - Use cases: Internal analytics, A/B test prioritization
  - Validation: Self-assessment
  - Frequency: Annual
  - Documentation: Lightweight model card
  - Approvals: Team lead
```

**Interview Tip**: If asked about model governance at banks, mention **SR 11-7** (Federal Reserve guidance)

---

#### **Automated Bias Testing (Airbnb's pattern)**

**Problem**: Manual bias testing is slow. Need to test every model update.

**Solution (Automated CI pipeline)**:
```python
# tests/test_fairness.py
import pytest
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def test_demographic_parity():
    # Load test set with protected attributes
    test_data = load_test_data_with_demographics()
    
    predictions = model.predict(test_data)
    
    # Test for gender bias
    metric = ClassificationMetric(
        test_data, predictions,
        unprivileged_groups=[{'gender': 'F'}],
        privileged_groups=[{'gender': 'M'}]
    )
    
    disparate_impact = metric.disparate_impact()
    
    # 80% rule (EEOC standard)
    assert 0.8 <= disparate_impact <= 1.25, \
        f"Disparate impact {disparate_impact} violates 80% rule"

def test_equalized_odds():
    # Test TPR and FPR parity
    tpr_diff = metric.true_positive_rate_difference()
    fpr_diff = metric.false_positive_rate_difference()
    
    assert abs(tpr_diff) < 0.05, f"TPR difference {tpr_diff} > 5%"
    assert abs(fpr_diff) < 0.05, f"FPR difference {fpr_diff} > 5%"

# Run in CI/CD
pytest tests/test_fairness.py --junitxml=fairness_report.xml
```

**Production Tip**: Use **AI Fairness 360** (IBM) or **Fairlearn** (Microsoft) for pre-built metrics

---

#### **Explainability at Scale (SHAP for 1M predictions/day)**

**Problem**: SHAP is slow (10ms per prediction). Can't run on all predictions.

**Solution (Sampling strategy)**:
```python
# Explain 1% of predictions (10K/day)
import shap

explainer = shap.TreeExplainer(model)

def predict_with_explanation(input_data, user_id):
    prediction = model.predict(input_data)
    
    # Explain 1% of predictions (hash-based sampling)
    if int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100 < 1:
        shap_values = explainer.shap_values(input_data)
        
        # Log to database for analysis
        log_explanation({
            "user_id": user_id,
            "prediction": prediction,
            "shap_values": shap_values.tolist(),
            "timestamp": datetime.now()
        })
    
    return prediction
```

**For High-Stakes Decisions** (credit, loans):
```python
# Explain ALL decisions (required by FCRA)
# But pre-compute explanations asynchronously
def async_explain(user_id, input_data, prediction):
    shap_values = explainer.shap_values(input_data)
    
    # Store in Redis (fast retrieval)
    redis_client.setex(
        f"explanation:{user_id}:{prediction_id}",
        86400,  # 24 hour TTL
        json.dumps(shap_values.tolist())
    )

# User requests explanation
@app.route('/explain/<prediction_id>')
def explain_prediction(prediction_id):
    explanation = redis_client.get(f"explanation:{user_id}:{prediction_id}")
    return jsonify({"explanation": json.loads(explanation)})
```

---

### 12.4A Advanced Cost Optimization Patterns

#### **GPU Time-Sharing (Run:ai pattern)**

**Problem**: Training jobs don't use GPU 100% (data loading gaps). Waste 20-40% GPU cycles.

**Solution (Run:ai GPU fractions)**:
```yaml
# Kubernetes GPU sharing (NVIDIA MIG or time-slicing)
apiVersion: v1
kind: Pod
metadata:
  name: training-job-1
spec:
  containers:
  - name: trainer
    image: training:v1
    resources:
      limits:
        nvidia.com/gpu: 0.5  # Half GPU
---
apiVersion: v1
kind: Pod
metadata:
  name: training-job-2
spec:
  containers:
  - name: trainer
    image: training:v2
    resources:
      limits:
        nvidia.com/gpu: 0.5  # Other half GPU
```

**Mechanisms**:
1. **MIG (Multi-Instance GPU)**: Hardware partitioning (A100 only)
   - 1x A100 → 7x MIG instances
   - Each instance: isolated memory, compute
2. **Time-slicing**: Context switching (all GPUs)
   - Alternate between jobs every 50ms
   - Works but slower (context switch overhead)

**Savings**: 2x GPU utilization → 50% cost reduction

---

#### **Preemptible Training with Straggler Mitigation**

**Problem**: In distributed training (8 GPUs), 1 preemption kills entire job

**Solution (Elastic training - PyTorch)**:
```python
import torch.distributed.elastic as elastic

# Elastic training (supports dynamic membership)
@elastic.train(min_workers=4, max_workers=8)
def train_elastic():
    # Training continues with 4-8 workers
    # If worker dies, training adapts (rebalances)
    for epoch in range(100):
        for batch in data_loader:
            loss = train_step(batch)
            loss.backward()
            
            # Elastic all-reduce (tolerates failures)
            dist.all_reduce(loss)
```

**Benefits**:
- 1 GPU preempted → Training continues with 7 GPUs (slower, but doesn't fail)
- Cost: Mix spot (cheap) + on-demand (reliable) → Best of both

**Meta's approach (RoBERTa training)**:
- 70% spot instances, 30% on-demand
- Elastic training handles spot interruptions
- Total cost: 60% cheaper than pure on-demand

---

#### **Model Compilation for Inference Cost Reduction**

**Problem**: PyTorch inference is slow (Python overhead, unoptimized kernels)

**Solution (TorchScript + TensorRT)**:
```python
# Step 1: TorchScript (JIT compilation)
model = load_pytorch_model()
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Step 2: TensorRT (NVIDIA optimization)
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    scripted_model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16},  # FP16
)

# Result: 5x faster inference (20ms → 4ms)
```

**Cost Impact**:
- Baseline: 100 req/sec on 5x A100 = $164/hour
- Optimized: 500 req/sec on 5x A100 (same hardware!)
- Or: 100 req/sec on 1x A100 = $33/hour
- **Savings**: $131/hour × 24 × 30 = $94,320/month

---

#### **Inference Request Batching (Hidden Cost Saver)**

**Problem**: Processing 1 request at a time wastes GPU (10% utilization)

**Solution (Dynamic batching - vLLM style)**:
```python
from collections import deque
import asyncio

class BatchedPredictor:
    def __init__(self, model, max_batch_size=32, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.processing = False
    
    async def predict(self, input_data):
        future = asyncio.Future()
        self.queue.append((input_data, future))
        
        # Start batch processing if not already running
        if not self.processing:
            asyncio.create_task(self.process_batch())
        
        return await future
    
    async def process_batch(self):
        self.processing = True
        await asyncio.sleep(self.max_wait_ms / 1000)  # Wait for more requests
        
        # Collect batch
        batch = []
        futures = []
        while len(batch) < self.max_batch_size and self.queue:
            input_data, future = self.queue.popleft()
            batch.append(input_data)
            futures.append(future)
        
        # Batch prediction
        if batch:
            predictions = self.model.predict_batch(batch)
            for future, prediction in zip(futures, predictions):
                future.set_result(prediction)
        
        self.processing = False
        
        # Continue if more requests
        if self.queue:
            asyncio.create_task(self.process_batch())

# Usage
predictor = BatchedPredictor(model)
prediction = await predictor.predict(input_data)
```

**Impact**:
- Throughput: 10 req/sec → 200 req/sec (20x improvement)
- Cost: 20x less infrastructure needed

---

### 12.5A Advanced Observability Patterns

#### **ML-Specific Metrics (Not Just Infra Metrics)**

**Standard metrics** (CPU, memory) don't catch ML issues. Need **model-specific metrics**:

```python
# Custom Prometheus metrics for ML
from prometheus_client import Histogram, Counter, Gauge

# Prediction distribution (detect drift)
prediction_distribution = Histogram(
    'model_prediction_distribution',
    'Distribution of model predictions',
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Feature statistics (detect feature drift)
feature_mean = Gauge('feature_mean', 'Feature mean value', ['feature_name'])
feature_std = Gauge('feature_std', 'Feature std value', ['feature_name'])

# Model confidence (low confidence = uncertain)
prediction_confidence = Histogram(
    'model_confidence',
    'Model confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# Log every prediction
def predict(input_data):
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data).max()
    
    prediction_distribution.observe(prediction)
    prediction_confidence.observe(confidence)
    
    # Track feature statistics
    for feature_name, feature_value in input_data.items():
        feature_mean.labels(feature_name=feature_name).set(feature_value)
    
    return prediction
```

**Alerting on ML Metrics**:
```yaml
# Alert on prediction drift
- alert: PredictionDrift
  expr: |
    abs(
      avg_over_time(model_prediction_distribution[1h]) -
      avg_over_time(model_prediction_distribution[24h] offset 7d)
    ) > 0.1
  annotations:
    summary: "Prediction distribution shifted by >10%"
    
# Alert on low confidence
- alert: LowConfidence
  expr: histogram_quantile(0.5, model_confidence) < 0.7
  for: 1h
  annotations:
    summary: "Median model confidence <70% for 1 hour"
```

**Production Example (Spotify)**:
- Monitors 50+ ML-specific metrics per model
- Caught data drift 2 days before user-facing issues
- Automatic model retraining triggered

---

#### **Distributed Tracing for Multi-Model Pipelines**

**Problem**: Prediction requires 3 models in sequence (embedding → ranking → reranking). Which model is slow?

**Solution (OpenTelemetry context propagation)**:
```python
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

tracer = trace.get_tracer(__name__)

# Service 1: Embedding model
@app.route('/embed', methods=['POST'])
def embed():
    # Extract trace context from request
    ctx = TraceContextTextMapPropagator().extract(request.headers)
    
    with tracer.start_as_current_span("embedding_model", context=ctx):
        input_text = request.json['text']
        embedding = embedding_model.encode(input_text)
        
        # Inject trace context into response
        propagator = TraceContextTextMapPropagator()
        headers = {}
        propagator.inject(headers)
        
        return jsonify({"embedding": embedding.tolist()}), 200, headers

# Service 2: Ranking model
@app.route('/rank', methods=['POST'])
def rank():
    ctx = TraceContextTextMapPropagator().extract(request.headers)
    
    with tracer.start_as_current_span("ranking_model", context=ctx):
        embedding = request.json['embedding']
        scores = ranking_model.predict(embedding)
        return jsonify({"scores": scores.tolist()})
```

**Visualization (Jaeger)**:
```
Trace ID: abc123
Total Duration: 245ms

├─ API Gateway (5ms)
├─ embedding_model (80ms) ← Bottleneck!
│  ├─ tokenization (10ms)
│  ├─ forward_pass (65ms)
│  └─ post_processing (5ms)
├─ ranking_model (50ms)
│  ├─ feature_engineering (15ms)
│  └─ inference (35ms)
└─ reranking_model (30ms)
```

**Action**: Optimize embedding model (quantize, reduce sequence length)

---

#### **Synthetic Monitoring (Canary Requests)**

**Problem**: Real users hit edge cases. We don't know until production breaks.

**Solution (Datadog Synthetic Tests)**:
```python
# Synthetic test runner (runs every 5 minutes)
import requests
import random

ENDPOINT = "https://api.company.com/predict"
TEST_CASES = [
    {"input": "normal case", "expected_class": 0},
    {"input": "edge case 1", "expected_class": 1},
    {"input": "edge case 2", "expected_class": 1},
]

def run_synthetic_test():
    for test_case in TEST_CASES:
        response = requests.post(ENDPOINT, json=test_case)
        
        # Check response
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        prediction = response.json()['prediction']
        assert prediction == test_case['expected_class'], \
            f"Expected {test_case['expected_class']}, got {prediction}"
        
        # Check latency
        latency_ms = response.elapsed.total_seconds() * 1000
        assert latency_ms < 500, f"Latency {latency_ms}ms > 500ms"
    
    print("✅ All synthetic tests passed")

# Schedule every 5 minutes (cron or Kubernetes CronJob)
```

**Benefits**:
- Catch issues BEFORE users do
- Test edge cases (that real traffic might not hit)
- Alert if synthetic tests fail (proactive monitoring)

---

## COMMON INTERVIEW TRAPS & CORRECTIONS

### Trap 1: "We do blue-green deployment"
**Interviewer Follow-up**: "Doesn't that double your infrastructure cost?"

**Correct Answer**:
- Yes, during deployment (2x cost for 5-30 minutes)
- After cutover, old environment terminated (cost back to 1x)
- Total extra cost: 5-30 minutes × daily deployments = ~1 hour/day = 4% overhead
- Alternative: **Canary deployment** (cheaper, gradual rollout)

---

### Trap 2: "We log everything"
**Interviewer Follow-up**: "What's your daily log volume and cost?"

**Correct Answer**:
- Example: 1000 req/sec × 5 log lines × 1KB = 5MB/sec = 432GB/day
- Cost: CloudWatch = $216/day × 30 = $6,480/month
- **Solution**: Log sampling (10% of success, 100% of errors) → $1,000/month
- **Better**: Structured logging + Prometheus metrics (cheaper, queryable)

---

### Trap 3: "We retrain models when accuracy drops"
**Interviewer Follow-up**: "How do you measure accuracy in production?"

**Correct Answer**:
- **Problem**: Ground truth labels arrive late (days/weeks)
- **Solutions**:
  1. **Proxy metrics**: Click-through rate, engagement time, user retention
  2. **Human labeling**: Sample 1% of predictions, label within 24 hours
  3. **Delayed evaluation**: Evaluate accuracy 1 week delayed (when labels arrive)
  4. **Drift detection**: Input distribution shift (no labels needed)

---

### Trap 4: "We use SHAP for explainability"
**Interviewer Follow-up**: "SHAP is slow. How do you scale to 1M predictions/day?"

**Correct Answer**:
- **Problem**: SHAP = 10-50ms per prediction (too slow for production)
- **Solutions**:
  1. **Sampling**: Explain 1% of predictions (hash-based)
  2. **Pre-computation**: Compute SHAP offline, store in Redis
  3. **Approximations**: FastSHAP, Kernel SHAP (faster but less accurate)
  4. **Async**: Compute SHAP asynchronously (return prediction immediately)
  5. **Mandatory for high-stakes**: FCRA requires explanations → Must compute for all (even if slow)

---

### Trap 5: "We use Kubernetes for everything"
**Interviewer Follow-up**: "Why not Slurm for training?"

**Correct Answer**:
- **Kubernetes**: Good for full ML lifecycle (train + serve + monitor)
- **Slurm**: Better for pure training (lower overhead, simpler)
- **Trade-off**: Complexity vs flexibility
- **When Slurm wins**: Research labs, long-running multi-node jobs, on-premise HPC
- **When K8s wins**: Cloud-native, microservices, need auto-scaling

---

## RAPID-FIRE PRODUCTION WINS (30-SECOND ANSWERS)

**Q: How did you reduce training costs by 50%?**
A: Switched 70% of training to spot instances (70% cheaper), implemented auto-checkpointing every 15 minutes, interruption rate was 5%, total savings $2M/year.

**Q: How do you handle model staleness?**
A: Monitor prediction distribution weekly, trigger retraining if KS test p-value < 0.01 (significant drift), automated retraining pipeline (Airflow), deployment via canary rollout.

**Q: Biggest incident and how you fixed it?**
A: Model OOM crash (batch size 1024 → 32GB memory). Fixed: Rolled back in 2 minutes (Kubernetes), added memory profiling to CI/CD, implemented load testing before deployment. Prevented recurrence.

**Q: How do you ensure fairness?**
A: Automated fairness tests in CI/CD (demographic parity, equalized odds), 80% rule for disparate impact, quarterly third-party audits, threshold optimization per demographic group, human review for borderline cases.

**Q: How do you debug a 20% accuracy drop?**
A: Step 1: Rollback (minimize impact). Step 2: Check data (schema, distribution). Step 3: Test offline (holdout set). Step 4: Review changes (deployment, code). Step 5: Root cause (feature drift, training-serving skew). Step 6: Document (incident report).

---

## Summary

**Added Content**:
- 🔢 **Rapid-fire numbers** to memorize (costs, times, percentages)
- 🏭 **Real production patterns** from Meta, Netflix, Uber, Spotify, Airbnb
- 🔧 **Advanced techniques**: GPU pooling, checkpoint sharding, heterogeneous training, elastic training
- 📊 **ML-specific metrics**: Prediction drift, feature drift, confidence monitoring
- 🚨 **Interview traps**: Common mistakes and correct answers
- ⚡ **30-second answers**: Rapid-fire responses for common questions
- 🛠️ **Missing patterns**: Shadow mode, auto-rollback, synthetic monitoring, feature stores

**Total Content**:
- Original notes: ~15,000 words
- Added content: ~8,000 words
- Total: **~23,000 words** of interview-ready material

---

**Next Steps**: Practice rapid-fire responses (30-60 seconds), memorize key numbers, review production case studies
