# Feature Representations in Graph Neural Networks: A Systematic Study on E-Commerce Product Classification

An in-depth investigation of how node feature engineering impacts Graph Convolutional Network performance, demonstrating the fundamental trade-offs between transductive and inductive learning on large-scale product networks.

## Overview

Graph neural networks have emerged as a powerful tool for learning on relational data, but a critical question remains: **how do we represent nodes when their features are unavailable or insufficient?** This project tackles this question through a rigorous experimental study, comparing seven distinct feature engineering approaches on the Amazon Product Co-Purchasing Network.

Using a unified 3-layer Graph Convolutional Network architecture, I systematically evaluated each approach on 7,650 products connected by 238,162 co-purchasing relationships. The results reveal important insights about the balance between model expressiveness and generalization capability.

**Key Finding:** Identity-based features (one-hot encoding) achieved 92.68% test accuracy but cannot generalize to new products, while fully generalizable features reached 87.98% accuracy: a 4.7 percentage point gap that represents the fundamental cost of inductive learning in graph neural networks.

## Why Graph Neural Networks?

Traditional neural networks struggle with graph-structured data for three critical reasons:

**1. Fixed Input Size Problem**

Standard neural networks expect fixed-dimensional inputs. If we naively concatenate a graph's adjacency matrix **A** (size n×n) with node features **X** (size n×d), each node is represented by n+d features. This creates two problems:
- The model can only work on graphs of exactly the same size (transductive setting)
- We have n+d features but only n training samples, leading to severe overfitting

**2. Node Ordering Sensitivity**

Graphs have no natural ordering of nodes. If we simply feed the adjacency matrix to a neural network, permuting the rows and columns (reordering nodes) produces different outputs—even though the graph structure is identical. The model incorrectly treats node position as meaningful information.

**3. Scalability Issues**

For a graph with 7,650 nodes and 745-dimensional features (like our dataset), the naive approach would create 8,395-dimensional input vectors. With only 7,650 training samples, we have more features than data points, making effective learning nearly impossible.

## The GCN Solution: Permutation Invariance and Equivariance

Graph Convolutional Networks solve these problems by building two mathematical properties into their architecture:

### Permutation Equivariance (Node-Level Tasks)

For node classification, we need a function **f** that satisfies:

```
P × f(A, X) = f(P×A×P^T, P×X)
```

This means: if we permute the graph's nodes, the outputs are permuted in exactly the same way. The model treats node identity as arbitrary—what matters is the graph structure and features, not the ordering.

**Example:** The operation `AX` (multiply adjacency by features) is permutation equivariant. Each node's new representation is the sum of its neighbors' features, which is independent of how we label the nodes.

### Permutation Invariance (Graph-Level Tasks)

For graph classification, we need:

```
f(A, X) = f(P×A×P^T, P×X)
```

The output is completely unchanged by node reordering. This is typically achieved through global pooling operations like summing or averaging all node representations.

### How GCN Achieves This

A GCN layer updates each node's representation through message passing:

```
h_i^(l+1) = σ( W^(l) × h_i^(l) + W^(l) × (1/k_i) × Σ_{j∈neighbors(i)} h_j^(l) )
```

Breaking this down:
1. Each node transforms its own representation via learnable weights **W**
2. Each node aggregates its neighbors' representations (averaged by degree **k_i**)
3. The two contributions are summed and passed through a nonlinear activation **σ**

This is permutation equivariant because the neighbor aggregation doesn't depend on node labels—only on the graph structure. In matrix form:

```
H' = σ(H×W^T + D^(-1)×A×H×W^T)
```

where **D** is the diagonal degree matrix and **A** is the adjacency matrix.

## Transductive vs. Inductive Learning: A Fundamental Trade-off

Understanding these two learning paradigms is essential for interpreting the results:

### Transductive Setting

**Definition:** The model has access to all nodes (train, validation, and test) during training, but only sees labels for the training nodes. The test nodes are *fixed and known* at training time.

**Key Characteristic:** Features can be node-specific. For example, one-hot encoding assigns each node a unique ID vector. During training, the model learns which ID corresponds to which class.

**Limitation:** Cannot classify new nodes that weren't present during training. If a new product is added to the network, the model cannot make predictions.

**When to use:** Citation networks (all papers known upfront), social network analysis (fixed user base), molecule property prediction (fixed molecular graph).

### Inductive Setting

**Definition:** The model must generalize to completely unseen nodes not present during training. Features must be *computable for any node* without knowing the full graph in advance.

**Key Characteristic:** Features are derived from graph structure (degree, centrality) or are randomly generated in a consistent way. No node-specific identity information is used.

**Advantage:** Can classify new nodes. When new products are added, the model computes their features and makes predictions immediately.

**When to use:** Recommendation systems (new users/products arrive constantly), fraud detection (new accounts), drug discovery (new molecules).

**The Trade-off:** Transductive models typically achieve higher accuracy because they can memorize node identities. Inductive models sacrifice some accuracy for the ability to generalize.

## Problem Statement

Product categorization in e-commerce presents unique challenges:
- Products form complex relational structures through co-purchasing and co-viewing patterns
- Raw features (bag-of-words descriptions) may not capture the full relational context
- New products arrive constantly, requiring inductive capability
- The graph structure itself carries classification signal (similar products cluster)

**Research Question:** How do different node feature representations affect GCN classification performance, and what is the cost of inductive generalization?

## Dataset: Amazon Product Co-Purchasing Network (Photo Category)

**Graph Statistics:**
- **Nodes:** 7,650 products in the "Photo" category
- **Edges:** 238,162 directed co-purchasing relationships
- **Node Features:** 745-dimensional bag-of-words vectors (original, not used in this study)
- **Target:** 8 product categories
- **Degree Distribution:** Power-law with mean degree 31.13, max degree 1,434

**Data Split (Stratified by Class):**
- **Training:** 5,355 nodes (70%)
- **Validation:** 1,147 nodes (15%)
- **Test:** 1,148 nodes (15%)

**Class Distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| 0 | 369 | 4.8% |
| 1 | 1,686 | 22.0% |
| 2 | 703 | 9.2% |
| 3 | 915 | 12.0% |
| 4 | 882 | 11.5% |
| 5 | 823 | 10.8% |
| 6 | 1,941 | 25.4% |
| 7 | 331 | 4.3% |

The dataset exhibits significant class imbalance, with the two largest categories containing 47.4% of all products. Stratified splitting ensures proportional representation across train/validation/test sets.

**Why This Dataset?**

Co-purchasing networks capture implicit product similarity—if customers frequently buy products A and B together, they likely belong to related categories or serve complementary purposes. The graph structure provides rich classification signal beyond individual product descriptions.

## Methodology

### Model Architecture: 3-Layer Graph Convolutional Network

All experiments used an identical architecture to ensure fair comparison:

```
Input: Node Features (dimension varies by experiment)
    ↓
GCN Layer 1: Input → 128 units
    - Message passing with degree-normalized aggregation
    - ReLU activation
    - Dropout (p=0.5)
    ↓
GCN Layer 2: 128 → 128 units
    - Message passing with degree-normalized aggregation
    - ReLU activation
    - Dropout (p=0.5)
    ↓
GCN Layer 3: 128 → 8 units (output classes)
    - Message passing with degree-normalized aggregation
    - No activation (raw logits)
    ↓
Output: 8-dimensional class logits → Softmax → Predictions
```

**Key Design Choices:**

**Why 3 layers?**
Each GCN layer allows information to propagate one hop through the graph. With 3 layers, each node aggregates information from neighbors up to 3 hops away. In a graph with mean degree 31, a 3-hop neighborhood covers thousands of nodes, providing rich context while avoiding over-smoothing (where deeper networks cause all node representations to become identical).

**Why 128 hidden units?**
This provides sufficient capacity to learn complex patterns without overfitting. Combined with 50% dropout, it balances expressiveness and regularization.

**Why dropout?**
Graph neural networks can overfit to graph structure, especially in transductive settings. Dropout randomly zeroes 50% of activations during training, forcing the model to learn robust features that don't depend on specific neurons.

### Feature Engineering Experiments

I tested seven fundamentally different approaches to node representation:

#### 1. One-Hot Encoding (Transductive)
- **Dimension:** 7,650 features (one per node)
- **Description:** Each node has a unique identity vector with a single 1 and rest 0s
- **Inductive?** No—requires knowing all nodes at training time
- **Intuition:** Maximum expressiveness; the model can memorize each node's class

#### 2. Laplacian Eigenvectors (Transductive)
- **Dimension:** 64 features
- **Description:** Eigenvectors of the graph Laplacian matrix, capturing spectral properties
- **Inductive?** No—eigenvectors are computed from the full graph
- **Intuition:** Spectral graph theory provides global structural features; nodes with similar spectral coordinates tend to be structurally similar

#### 3. Random Features (Transductive)
- **Dimension:** 128 features
- **Description:** Random Gaussian vectors assigned once to each node, then fixed
- **Inductive?** No—each node has a unique random seed
- **Intuition:** Breaks symmetry without being as expressive as one-hot; tests if random node IDs are sufficient

#### 4. Node Centrality Features (Hybrid)
- **Dimension:** 5 features
- **Description:** Degree, betweenness, closeness, PageRank, clustering coefficient
- **Inductive?** Partially—can compute for new nodes, but some metrics (betweenness, closeness) require full graph
- **Intuition:** Captures a node's importance and role in the network; high-degree nodes (hubs) may belong to popular categories

#### 5. Node Degree Features (Inductive)
- **Dimension:** 2 features
- **Description:** Raw degree and log-transformed degree
- **Inductive?** Yes—degree is computed from immediate neighbors
- **Intuition:** Simplest structural feature; tests if local connectivity alone predicts category

#### 6. Constant Features (Fully Inductive)
- **Dimension:** 4 features
- **Description:** Same constant vector for all nodes
- **Inductive?** Yes—no node-specific information
- **Intuition:** Forces the model to learn purely from graph structure via message passing; tests the "baseline" signal in the topology

#### 7. Random Features (Inductive)
- **Dimension:** 128 features
- **Description:** Random Gaussian vectors regenerated each forward pass
- **Inductive?** Yes—uses a shared random generator, no node-specific storage
- **Intuition:** Tests if freshly generated random features provide any learning signal

### Training Configuration

- **Loss Function:** Cross-Entropy Loss (combines LogSoftmax + Negative Log-Likelihood)
- **Optimizer:** Adam with default learning rate (0.001)
- **Epochs:** 200
- **Regularization:** 50% dropout between all layers
- **Early Stopping:** Model checkpoint saved at best validation accuracy
- **Hardware:** GPU-accelerated training via PyTorch Geometric
- **Reproducibility:** Fixed random seeds for consistent results

## Results

### Performance Comparison (Epoch 200)

| Feature Type | Test Accuracy | Train Accuracy | Loss | Transductive? |
|-------------|---------------|----------------|------|---------------|
| **One-Hot Encoding** | **92.68%** | 95.39% | 0.187 | Yes |
| **Laplacian Eigenvectors** | **91.64%** | 92.12% | 0.236 | Yes |
| **Random (Transductive)** | **88.33%** | 90.92% | 0.342 | Yes |
| **Node Centrality** | **84.23%** | 83.66% | 0.494 | Hybrid |
| **Node Degree** | **64.55%** | 65.27% | 0.926 | Yes (Inductive) |
| **Constant Features** | **39.72%** | 40.21% | 1.565 | Yes (Inductive) |
| **Random (Inductive)** | **37.11%** | 36.69% | 1.739 | Yes (Inductive) |

*Random guessing baseline: 12.5% (8 classes)*

### Best Model Performance (with Early Stopping)

| Feature Type | Best Val Epoch | Final Test Accuracy |
|-------------|----------------|---------------------|
| One-Hot Encoding | 181 | 92.07% |
| Laplacian Eigenvectors | 195 | 92.07% |
| Random (Transductive) | 184 | 87.98% |
| Node Centrality | 180 | 84.06% |
| Node Degree | 173 | 66.03% |
| Constant Features | 136 | 40.16% |
| Random (Inductive) | 89 | 37.11% |

**Note:** One-hot and Laplacian eigenvectors both achieved identical final test accuracy (92.07%) at their respective best validation epochs, making them statistically tied for best performance.

## Key Insights

### 1. The Transductive-Inductive Performance Gap

The three best-performing approaches are all transductive:
- One-hot encoding: 92.68%
- Laplacian eigenvectors: 91.64%
- Random transductive: 88.33%

The best purely inductive approach (node centrality at 84.23%) lags behind by **4.1 to 8.4 percentage points**. This quantifies the cost of inductive generalization: you sacrifice roughly 5-8% accuracy to gain the ability to classify unseen nodes.

For practical applications, this trade-off must be carefully considered:
- **Use transductive** if the graph is fixed and complete (e.g., classifying all papers in a citation network)
- **Use inductive** if new nodes arrive dynamically (e.g., real-time product categorization)

### 2. Graph Structure Alone Carries Substantial Signal

Even with **constant features** (identical for all nodes), the GCN achieved 39.72% accuracy—**more than 3× better than random guessing** (12.5%). This demonstrates that:
- Co-purchasing patterns are highly informative about product categories
- Graph neural networks can learn meaningful representations purely from topology through multi-hop message passing
- The network structure itself encodes product similarity

This has practical implications: even without product descriptions, metadata, or images, the purchasing graph provides strong classification signal.

### 3. Random Features Outperform Degree Features (If Transductive)

Surprisingly, random transductive features (88.33%) outperformed carefully engineered node centrality features (84.23%) and degree features (64.55%). This suggests:
- **Unique node identifiers** (even random ones) allow the model to memorize node-specific patterns during training
- **Symmetry breaking** is more important than semantic meaning in transductive settings
- Simple structural features like degree are insufficient on their own, as they fail to distinguish between nodes with similar local connectivity

### 4. Spectral Methods Excel

Laplacian eigenvectors (91.64%) performed nearly as well as one-hot encoding (92.68%), while using only 64 features compared to 7,650. Spectral features capture:
- **Global graph structure** beyond local neighborhoods
- **Smooth variations** across the graph (eigenvectors define a coordinate system where connected nodes have similar coordinates)
- **Clustering information** (eigendecomposition reveals community structure)

This makes spectral features a compelling choice when transductive learning is acceptable but memory/computation must be minimized.

### 5. Centrality Metrics Outperform Raw Degree

Node centrality features (84.23%) substantially outperformed degree features (64.55%), showing that **richer structural information improves inductive performance**. Centrality metrics like PageRank and betweenness capture:
- **Global importance** (PageRank identifies "hub" products)
- **Bridge positions** (betweenness finds products linking different communities)
- **Local clustering** (clustering coefficient measures neighborhood density)

This suggests that when designing inductive GNN systems, investing in sophisticated graph feature computation pays off.

### 6. Inductive Random Features Fail Catastrophically

Random inductive features (37.11%) performed worse than constant features (39.72%). This makes sense:
- Constant features allow the model to learn a consistent representation through message passing
- Random inductive features are **regenerated each forward pass**, preventing the model from learning stable patterns
- The noise introduced by constantly changing features overwhelms any learning signal

This highlights the importance of **feature consistency**—even random features must be deterministic for effective learning.

## Technical Implementation Details

### Message Passing Mechanism

Each GCN layer performs degree-normalized neighborhood aggregation:

```python
def gcn_layer(H, A, D, W):
    """
    H: n × d node feature matrix
    A: n × n adjacency matrix
    D: n × n diagonal degree matrix
    W: d × d' learnable weight matrix
    """
    self_contribution = H @ W.T
    neighbor_contribution = (D^(-1) @ A @ H) @ W.T
    return activation(self_contribution + neighbor_contribution)
```

**Why normalize by degree?**
Without normalization, high-degree nodes dominate: their representations explode because they aggregate from many neighbors. Dividing by degree keeps magnitudes stable, treating all nodes fairly regardless of connectivity.

### Stratified Splitting

With significant class imbalance (25.4% vs. 4.3%), random splitting could create unrepresentative train/validation/test sets. I used stratified splitting to maintain class proportions:

```python
Train Class 6: 25.4% of train set
Val Class 6:   25.4% of val set
Test Class 6:  25.3% of test set
```

This ensures the model sees balanced examples of rare classes and evaluation metrics are reliable.

### Multi-Metric Tracking

Rather than relying solely on final test accuracy, I tracked three metrics simultaneously:
- **Train accuracy:** Detects underfitting (if low, model lacks capacity)
- **Validation accuracy:** Guides early stopping
- **Test accuracy:** Final performance measure

Large gaps between train and test accuracy indicate overfitting. For example:
- One-hot encoding: 95.39% train vs. 92.68% test (small gap, well-regularized)
- Node centrality: 83.66% train vs. 84.23% test (test slightly higher—rare but indicates stable learning)

## Practical Applications

### E-Commerce Recommendation Systems
This work directly applies to product categorization, enabling:
- **Automated taxonomy assignment** for newly listed products based on early purchase patterns
- **Cold-start recommendation** by inferring category from co-purchased items before detailed features are available
- **Category refinement** by combining metadata (bag-of-words) with graph features for robust classification

### Cold Start Problem
Inductive features enable classification of:
- New products with minimal purchase history (degree-based features available immediately)
- Cross-platform integration (classify products from different marketplaces using structural similarities)
- Temporal networks (classify new nodes in growing graphs without retraining)

### Network Science Research
This systematic comparison provides guidance for:
- **Feature selection** based on transductive/inductive requirements
- **Benchmark comparisons** (constant features establish topology-only baseline)
- **Ablation studies** (isolate the contribution of graph structure vs. node features)

## Future Directions

### 1. Attention Mechanisms
Graph Attention Networks (GATs) learn **edge importance weights**, allowing the model to focus on relevant neighbors:
```
h_i' = Σ_{j∈neighbors(i)} α_ij × W × h_j
```
where α_ij is learned via attention. This could improve performance on heterogeneous graphs where not all edges are equally informative (e.g., "frequently co-purchased" vs. "occasionally co-purchased").

### 2. Deeper Architectures with Residual Connections
Current model uses 3 layers to avoid over-smoothing (where all nodes converge to identical representations). Residual connections enable deeper models:
```
h_i^(l+1) = GCN_layer(h_i^(l)) + h_i^(l)
```
This preserves information from earlier layers, potentially enabling 5-10 layer networks that capture longer-range dependencies.

### 3. Hybrid Features
Combine multiple feature types:
- Spectral + Centrality: Global structure + local importance
- Original bag-of-words + Graph features: Content + context
- Learned embeddings + Graph structure: Pretrained knowledge + relational patterns

### 4. Dynamic Graph Learning
Extend to temporal networks where edges appear/disappear over time:
- Track how product relationships evolve with seasons, trends
- Predict category changes (products shifting between categories)
- Model cascading effects (how new products influence neighbors)

### 5. Explainability and Interpretability
Analyze which graph patterns drive predictions:
- Visualize attention weights (if using GAT)
- Identify influential neighbors for each prediction
- Understand what structural patterns correspond to each category

## Requirements

```
torch >= 1.9.0
torch-geometric >= 2.0.0
numpy >= 1.19.0
matplotlib >= 3.3.0
scikit-learn >= 0.24.0
networkx >= 2.5
```

## Usage

The notebook is fully self-contained and automated:

```bash
jupyter notebook GNN.ipynb
```

Simply run all cells sequentially. The notebook will:
1. Load the Amazon-Photo dataset from PyTorch Geometric
2. Generate all seven feature representations
3. Train seven GCN models (one per feature type)
4. Generate comparative visualizations
5. Output final accuracy tables

**Expected runtime:** ~10-15 minutes on GPU, ~30-45 minutes on CPU (200 epochs × 7 models).

## Reproducibility

All experiments use fixed random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
```

Results should be reproducible within ±0.5% accuracy across different runs due to non-deterministic GPU operations.

## Acknowledgments

- **Dataset:** Amazon Product Co-Purchasing Network from PyTorch Geometric's built-in datasets
- **Framework:** PyTorch Geometric library for efficient graph neural network implementation
- **Theory:** Based on the seminal GCN paper by Kipf & Welling (2017)

## References

Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.

---

*This project demonstrates rigorous experimental methodology in graph representation learning, with a focus on understanding the practical trade-offs between model expressiveness and generalization capability. The systematic ablation study and comprehensive evaluation provide actionable insights for deploying graph neural networks in production systems.*
