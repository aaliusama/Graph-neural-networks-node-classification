# Technical Guide: Graph Convolutional Networks - Step-by-Step Calculation

This guide walks through the mathematical operations of a Graph Convolutional Network using a concrete example with real numbers. By the end, you'll understand exactly how GCNs transform node features through message passing.

## Example Problem Setup

Let's build a small social network where we want to predict each person's interests based on their friend connections.

### The Graph

```
    Alice (A)
     /  \
    /    \
   Bob  Carol
   (B)   (C)
    \    /
     \  /
    David (D)
      |
    Emma (E)
```

**Adjacency Matrix A (5×5):**

```
      A  B  C  D  E
  A [ 0  1  1  0  0 ]
  B [ 1  0  0  1  0 ]
  C [ 1  0  0  1  0 ]
  D [ 0  1  1  0  1 ]
  E [ 0  0  0  1  0 ]
```

**Degree Matrix D (5×5):**

Each diagonal entry is the degree (number of connections) of that node:

```
      A  B  C  D  E
  A [ 2  0  0  0  0 ]
  B [ 0  2  0  0  0 ]
  C [ 0  0  2  0  0 ]
  D [ 0  0  0  3  0 ]
  E [ 0  0  0  0  1 ]
```

### Initial Node Features

Each person starts with a 2-dimensional feature vector representing their interests:
- Feature 1: Sports interest (0-1 scale)
- Feature 2: Arts interest (0-1 scale)

**Feature Matrix X (5×2):**

```
      Sports  Arts
  A [  0.9    0.1  ]   Alice loves sports
  B [  0.8    0.3  ]   Bob likes sports, some arts
  C [  0.2    0.9  ]   Carol loves arts
  D [  0.3    0.8  ]   David loves arts
  E [  0.1    0.7  ]   Emma loves arts
```

### The Task

Predict whether each person belongs to the "Sports Club" or "Arts Club" based on their features and their friends' interests.

## GCN Layer: Detailed Calculation

A single GCN layer computes:

```
H' = σ(H×W̄^T + D^(-1)×A×H×W^T)
```

Let's break this down step by step, using simplified weight matrices for clarity.

### Step 1: Define Learnable Weights

We need two weight matrices:
- **W̄**: Transforms the node's own features (2×2 matrix, mapping 2 features → 2 hidden units)
- **W**: Transforms the aggregated neighbor features (2×2 matrix)

**Example Weights (initialized randomly, then learned):**

```
W̄ = [ 1.0   0.5 ]
    [ 0.5   1.0 ]

W = [ 0.8  -0.3 ]
    [-0.3   0.9 ]
```

These weights would be learned during training, but we'll use these fixed values to demonstrate the forward pass.

### Step 2: Compute Self-Contribution (H×W̄^T)

Each node transforms its own features:

**X×W̄^T:**

For Alice (A): `[0.9, 0.1] × [[1.0, 0.5], [0.5, 1.0]]^T`

```
[0.9, 0.1] × [[1.0  0.5]     [0.9, 0.1] × [[1.0  0.5]
              [0.5  1.0]]  =              [0.5  1.0]]

= [0.9×1.0 + 0.1×0.5,  0.9×0.5 + 0.1×1.0]
= [0.95, 0.55]
```

**Full self-contribution matrix:**

```
      Dim1   Dim2
  A [ 0.95   0.55 ]
  B [ 0.95   0.70 ]
  C [ 0.65   0.95 ]
  D [ 0.70   0.95 ]
  E [ 0.45   0.75 ]
```

### Step 3: Compute Degree-Normalized Adjacency (D^(-1)×A)

The inverse degree matrix D^(-1) divides by each node's degree:

```
D^(-1) = [ 1/2   0     0     0     0   ]
         [  0   1/2    0     0     0   ]
         [  0    0    1/2    0     0   ]
         [  0    0     0    1/3    0   ]
         [  0    0     0     0     1   ]
```

**D^(-1)×A (normalized adjacency):**

```
      A    B    C    D    E
  A [ 0   1/2  1/2   0    0  ]
  B [1/2   0    0   1/2   0  ]
  C [1/2   0    0   1/2   0  ]
  D [ 0   1/3  1/3   0   1/3 ]
  E [ 0    0    0    1    0  ]
```

**Interpretation:** Each row sums to 1 (or less). This matrix computes the **average** of a node's neighbors' features.

### Step 4: Aggregate Neighbor Features (D^(-1)×A×X)

Now we compute what each node receives from its neighbors:

**For Alice (row A):**
```
[0, 1/2, 1/2, 0, 0] × [[0.9, 0.1],   =  (1/2)×[0.8, 0.3] + (1/2)×[0.2, 0.9]
                       [0.8, 0.3],
                       [0.2, 0.9],   =  [0.5, 0.6]
                       [0.3, 0.8],
                       [0.1, 0.7]]
```

Alice averages her two friends' features: Bob (0.8, 0.3) and Carol (0.2, 0.9).

**Full aggregated neighbor features:**

```
      Sports  Arts
  A [  0.50   0.60 ]   Average of Bob & Carol
  B [  0.60   0.50 ]   Average of Alice & David
  C [  0.60   0.50 ]   Average of Alice & David
  D [  0.43   0.67 ]   Average of Bob, Carol & Emma
  E [  0.30   0.80 ]   Just David (only neighbor)
```

**Observation:** Notice how neighbor aggregation smooths out features across the graph. Emma (0.1, 0.7) gets updated based on David (0.3, 0.8), pulling her closer to his values.

### Step 5: Transform Aggregated Features (D^(-1)×A×X×W^T)

Apply the neighbor weight matrix W to the aggregated features:

**For Alice:**
```
[0.50, 0.60] × [[0.8, -0.3],^T    [0.50, 0.60] × [[0.8  -0.3]
                [-0.3,  0.9]]  =                  [-0.3   0.9]]

= [0.50×0.8 + 0.60×(-0.3),  0.50×(-0.3) + 0.60×0.9]
= [0.22, 0.39]
```

**Full transformed neighbor contributions:**

```
      Dim1   Dim2
  A [ 0.22   0.39 ]
  B [ 0.33   0.27 ]
  C [ 0.33   0.27 ]
  D [ 0.14   0.47 ]
  E [ 0.00   0.63 ]
```

### Step 6: Combine Self and Neighbor Contributions

Add the self-contribution (Step 2) and neighbor contribution (Step 5):

**For Alice:**
```
Self:     [0.95, 0.55]
Neighbor: [0.22, 0.39]
Sum:      [1.17, 0.94]
```

**Full combined matrix:**

```
      Dim1   Dim2
  A [ 1.17   0.94 ]
  B [ 1.28   0.97 ]
  C [ 0.98   1.22 ]
  D [ 0.84   1.42 ]
  E [ 0.45   1.38 ]
```

### Step 7: Apply Non-Linear Activation (ReLU)

ReLU(x) = max(0, x). Since all our values are positive, they remain unchanged:

```
      Dim1   Dim2
  A [ 1.17   0.94 ]
  B [ 1.28   0.97 ]
  C [ 0.98   1.22 ]
  D [ 0.84   1.42 ]
  E [ 0.45   1.38 ]
```

**If we had negative values,** they would be set to zero. For example, if Alice's Dim1 was -0.5, it would become 0.

### Step 8: Interpret the Output

These are the new 2-dimensional embeddings for each node after one GCN layer:

- **Alice [1.17, 0.94]:** Still high on Dim1 (sports-leaning), influenced slightly by Carol's arts interest
- **Bob [1.28, 0.97]:** Highest on Dim1, strongly sports-oriented
- **Carol [0.98, 1.22]:** Shifted towards Dim2 (arts), but influenced by Alice's sports interest
- **David [0.84, 1.42]:** Highest on Dim2, very arts-oriented (influenced by Carol and Emma)
- **Emma [0.45, 1.38]:** Strongly arts-oriented, pulled slightly towards David

**Key Insight:** The GCN layer has **smoothed** the features across the graph. Nodes that were connected have more similar representations than before. This is the essence of message passing.

## Stacking Multiple Layers

In practice, we stack 2-3 GCN layers:

**Layer 1 Output → Layer 2 Input:**

We would take the output from Step 7 and pass it through another GCN layer with different weights:

```
H^(2) = σ(H^(1)×W̄^(2)T + D^(-1)×A×H^(1)×W^(2)T)
```

**Why multiple layers?**
- **Layer 1:** Each node aggregates information from 1-hop neighbors
- **Layer 2:** Each node aggregates information from 2-hop neighbors (friends of friends)
- **Layer 3:** Each node aggregates information from 3-hop neighbors

With 3 layers, information can flow across the entire graph.

## Final Classification Layer

After L layers, we have final node embeddings H^(L). To classify nodes:

**Add a final linear layer:**

```
Logits = H^(L) × W_output
```

If we're classifying into 2 clubs (Sports vs Arts), W_output is 2×2, producing 2 logits per node.

**Apply Softmax:**

```
P(Sports) = exp(logit_sports) / (exp(logit_sports) + exp(logit_arts))
P(Arts) = exp(logit_arts) / (exp(logit_sports) + exp(logit_arts))
```

**Example for Alice:**

Suppose after 3 layers, Alice's embedding is [2.1, 1.3], and:

```
W_output = [[1.0  -0.5]     (maps hidden features to class logits)
            [-0.5  1.0]]
```

Logits:
```
[2.1, 1.3] × [[1.0, -0.5], [-0.5, 1.0]] = [1.45, 0.25]
```

Softmax:
```
P(Sports) = exp(1.45) / (exp(1.45) + exp(0.25)) = 4.26 / (4.26 + 1.28) = 0.77
P(Arts) = exp(0.25) / (exp(1.45) + exp(0.25)) = 1.28 / (4.26 + 1.28) = 0.23
```

**Prediction:** Alice belongs to Sports Club (77% confidence)

## Training: Learning the Weights

During training, we:
1. Forward pass: Compute predictions using current weights
2. Compute loss: Compare predictions to true labels (cross-entropy)
3. Backward pass: Compute gradients of loss with respect to all weights
4. Update weights: Use optimizer (Adam) to adjust weights

**Example Loss Calculation:**

If Alice's true label is "Sports Club" (label = 0):
```
Loss = -log(P(Sports)) = -log(0.77) = 0.26
```

If we predicted Arts Club incorrectly:
```
Loss = -log(P(Sports)) = -log(0.23) = 1.47
```

The loss is lower when predictions are correct. Over many epochs, gradient descent adjusts weights to minimize this loss.

## Real Project Scale: Amazon Product Network

In the actual project:
- **Nodes:** 7,650 products (vs. 5 people in our example)
- **Edges:** 238,162 co-purchases (vs. 5 friendships)
- **Features:** Varies by experiment (7,650 for one-hot, 128 for random, 2 for degree)
- **Hidden Dimension:** 128 units per layer (vs. 2 in our example)
- **Layers:** 3 GCN layers
- **Output:** 8 product categories (vs. 2 clubs)

**Computational Complexity:**

For a single GCN layer:
1. **Self-transformation:** O(n × d × d') where n=nodes, d=input dimension, d'=output dimension
2. **Neighbor aggregation:** O(edges × d) for sparse graphs
3. **Total for 3 layers:** ~millions of operations per forward pass

This is why GPU acceleration is essential—the same matrix operations run in parallel across thousands of cores.

## Matrix Implementation in PyTorch

Here's how the actual code implements this:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        return x  # Return logits (no activation)
```

**Input Format:**
- `x`: Node feature matrix (7650 × d)
- `edge_index`: Sparse edge list [2 × 238162] listing [source, target] for each edge

**Output:**
- `x`: Node logits (7650 × 8) before softmax

## Permutation Equivariance: Why It Matters

Let's verify our example is permutation equivariant. Suppose we relabel nodes:
- Alice → Person 3
- Bob → Person 1
- Carol → Person 5
- David → Person 2
- Emma → Person 4

**Original Order Output:**
```
  A [ 1.17   0.94 ]
  B [ 1.28   0.97 ]
  C [ 0.98   1.22 ]
  D [ 0.84   1.42 ]
  E [ 0.45   1.38 ]
```

**Permuted Input:** Reorder X and A according to the new labels (B→1, D→2, A→3, E→4, C→5)

**Permuted Output:**
```
  1 (Bob)   [ 1.28   0.97 ]
  2 (David) [ 0.84   1.42 ]
  3 (Alice) [ 1.17   0.94 ]
  4 (Emma)  [ 0.45   1.38 ]
  5 (Carol) [ 0.98   1.22 ]
```

**Result:** The output is simply reordered in the same way as the input. Each node's embedding is identical regardless of labeling. This is permutation equivariance.

**Why this is crucial:** Without this property, the model would learn spurious patterns based on node IDs (e.g., "node 0 is always class A"). Permutation equivariance forces the model to learn from actual graph structure, not arbitrary orderings.

## Common Pitfalls and Solutions

### 1. Over-Smoothing
**Problem:** With too many layers (>4), all node representations converge to the same value.

**Why?** Each layer averages neighbors. After many iterations, information spreads everywhere, and distinctions disappear.

**Solution:** Use 2-4 layers max, or add residual connections:
```python
x = self.conv(x, edge_index) + x  # Add skip connection
```

### 2. Degree Imbalance
**Problem:** High-degree nodes (hubs) aggregate from many neighbors, leading to large magnitude features.

**Solution:** Degree normalization (D^(-1)×A) divides by degree, keeping magnitudes stable.

### 3. Vanishing/Exploding Gradients
**Problem:** Gradients become too small or large to update weights effectively.

**Solution:**
- Use ReLU activation (prevents vanishing)
- Apply dropout (prevents overfitting)
- Normalize inputs (keeps values in reasonable range)

### 4. Transductive Overfitting
**Problem:** Model memorizes training nodes when using one-hot features.

**Solution:**
- Use inductive features (degree, centrality) for better generalization
- Add regularization (dropout, L2 penalty)
- Use validation set for early stopping

## Key Takeaways

1. **GCNs aggregate neighbor information** through matrix multiplication (D^(-1)×A×H)
2. **Degree normalization** ensures fairness between high-degree and low-degree nodes
3. **Multiple layers** enable multi-hop information flow (3 layers = 3-hop neighborhoods)
4. **Permutation equivariance** is achieved through symmetric operations (summation over neighbors)
5. **Self-transformation + Neighbor aggregation** combines local and global information
6. **Non-linear activations** (ReLU) enable learning complex patterns
7. **Stacking layers** trades off between expressiveness and over-smoothing

## Further Reading

- **Original GCN Paper:** Kipf & Welling (2017) - "Semi-Supervised Classification with Graph Convolutional Networks"
- **Message Passing Framework:** Gilmer et al. (2017) - "Neural Message Passing for Quantum Chemistry"
- **Graph Attention Networks:** Veličković et al. (2018) - "Graph Attention Networks"
- **Over-Smoothing Analysis:** Li et al. (2018) - "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning"

---

*This guide demonstrates the mathematical foundations of graph neural networks through concrete numerical examples. Understanding these operations at a low level is essential for debugging, optimization, and developing novel architectures.*
