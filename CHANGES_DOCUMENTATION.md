# APGL4SR Code Changes Documentation

## Overview

We implemented **directional graphs** with **distance decay weighting** and **learnable mixing** to improve the APGL4SR sequential recommendation system.

### Key Improvements:
1. **Directional Graphs**: Separate forward (item → future items) and backward (item → past items) graphs
2. **Distance Decay**: Edge weights decay exponentially with distance (`weight = decay_base^distance`)
3. **Learnable Mixing**: Parameters `lambda_fwd` and `lambda_bwd` learn optimal mixing of forward/backward attention biases
4. **CPU Compatibility**: Fallbacks for faiss-cpu and DGL-cpu environments

---

## File 1: `src/models.py`

### Change 1: Faiss GPU Fallback to CPU

**Purpose:** Handle environments without GPU faiss by falling back to CPU faiss.

**Location:** `__init_cluster()` method in `KMeans` class (lines ~55-66)

**Original code:**
```python
def __init_cluster(
    self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
):
    print(" cluster train iterations:", niter)
    clus = faiss.Clustering(hidden_size, self.num_cluster)
    clus.verbose = verbose
    clus.niter = niter
    clus.nredo = nredo
    clus.seed = self.seed
    clus.max_points_per_centroid = max_points_per_centroid
    clus.min_points_per_centroid = min_points_per_centroid

    res = faiss.StandardGpuResources()
    res.noTempMemory()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = self.gpu_id
    index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
    return clus, index
```

**New code:**
```python
def __init_cluster(
    self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
):
    print(" cluster train iterations:", niter)
    clus = faiss.Clustering(hidden_size, self.num_cluster)
    clus.verbose = verbose
    clus.niter = niter
    clus.nredo = nredo
    clus.seed = self.seed
    clus.max_points_per_centroid = max_points_per_centroid
    clus.min_points_per_centroid = min_points_per_centroid

    # Try GPU faiss, fallback to CPU if not available
    try:
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
    except AttributeError:
        print("GPU faiss not available, using CPU faiss")
        index = faiss.IndexFlatL2(hidden_size)
    return clus, index
```

---

### Change 2: Learnable Directional Mixing Weights

**Purpose:** Add learnable parameters `lambda_fwd` and `lambda_bwd` to mix forward and backward graph attention biases.

**Location:** `__init__()` method in `SASRecModel` class (after `self.fuse_layer`)

**Original code:**
```python
if self.args.fuse:
    self.fuse_layer = nn.Linear(args.hidden_size * 2, args.hidden_size)

self.apply(self.init_weights)
```

**New code:**
```python
if self.args.fuse:
    self.fuse_layer = nn.Linear(args.hidden_size * 2, args.hidden_size)

# Directional graph mixing weights (APGL4SR improvement)
if hasattr(args, 'directional') and args.directional:
    self.lambda_fwd = nn.Parameter(torch.tensor(0.5))
    self.lambda_bwd = nn.Parameter(torch.tensor(0.5))

self.apply(self.init_weights)
```

---

### Change 3: Rewritten `global_graph_construction()` Method

**Purpose:** Build separate forward (item → future items) and backward (item → past items) directed graphs with distance decay weighting.

**Original code:**
```python
def global_graph_construction(self, train_data):
    args = train_data.args
    user_seq, n_items, k = train_data.user_seq, args.item_size - 2, args.k

    row, col = [], []
    for item_list in user_seq:
        item_list = item_list[:-2]  # remove valid/test data
        item_list_len = len(item_list)
        if item_list_len >= 1:
            for item_idx in range(item_list_len):
                target_num = min(k, item_list_len - item_idx - 1)
                if target_num > 0:
                    row += [item_list[item_idx]] * target_num
                    col += item_list[item_idx + 1: item_idx + 1 + target_num]

    row = np.array(row) - 1  # remove padding offset
    col = np.array(col) - 1
    data = np.ones(len(row))
    sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items))
    sparse_matrix = sparse_matrix + sparse_matrix.T  # symmetrize
    sparse_matrix = sparse_matrix + sp.eye(n_items)
    
    degree = np.array((sparse_matrix > 0).sum(1)).flatten()
    degree = np.nan_to_num(1 / degree, posinf=0)
    degree = sp.diags(degree)
    norm_adj = (degree @ sparse_matrix + sparse_matrix @ degree).tocoo()
    
    g = dgl.from_scipy(norm_adj)
    g.edata['weight'] = torch.tensor(norm_adj.data)
    self.g = g
    self.norm_adj = torch.sparse_coo_tensor(
        np.row_stack([norm_adj.row, norm_adj.col]),
        norm_adj.data, (n_items, n_items), dtype=torch.float32
    )
```

**New code:**
```python
def global_graph_construction(self, train_data):
    args = train_data.args
    user_seq, n_items, k = train_data.user_seq, args.item_size - 2, args.k
    
    decay_base = getattr(args, 'distance_decay_base', 0.8)
    directional = getattr(args, 'directional', False)
    
    # Build forward graph (A -> B: item at position i connects to items at i+1, i+2, ...)
    fwd_row, fwd_col, fwd_data = [], [], []
    # Build backward graph (B -> A: item at position i connects to items at i-1, i-2, ...)
    bwd_row, bwd_col, bwd_data = [], [], []
    
    for item_list in user_seq:
        item_list = item_list[:-2]  # remove valid/test data
        item_list_len = len(item_list)
        
        for item_idx in range(item_list_len):
            # Forward edges: current item -> future items
            target_num = min(k, item_list_len - item_idx - 1)
            if target_num > 0:
                fwd_row += [item_list[item_idx]] * target_num
                fwd_col += item_list[item_idx + 1: item_idx + 1 + target_num]
                # Distance decay: weight = decay_base^distance
                distances = np.arange(1, 1 + target_num)
                fwd_data.append(np.power(decay_base, distances - 1))  # distance 1 gets weight 1
            
            # Backward edges: current item -> past items
            target_num = min(k, item_idx)
            if target_num > 0:
                bwd_row += [item_list[item_idx]] * target_num
                bwd_col += item_list[item_idx - target_num: item_idx][::-1]  # reverse order (closest first)
                distances = np.arange(1, 1 + target_num)
                bwd_data.append(np.power(decay_base, distances - 1))
    
    def build_normalized_adj(row, col, data, n_items):
        """Build normalized adjacency matrix from COO data."""
        if len(data) == 0:
            return sp.csc_matrix((n_items, n_items)), None
        data = np.concatenate(data)
        row, col = np.array(row) - 1, np.array(col) - 1  # remove padding offset
        sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items))
        sparse_matrix = sparse_matrix + sp.eye(n_items)  # add self-loops
        degree = np.array((sparse_matrix > 0).sum(1)).flatten()
        degree = np.nan_to_num(1 / degree, posinf=0)
        degree = sp.diags(degree)
        norm_adj = (degree @ sparse_matrix + sparse_matrix @ degree).tocoo()
        return sparse_matrix, norm_adj
    
    if directional:
        # Build separate forward and backward normalized adjacencies
        _, fwd_norm_adj = build_normalized_adj(fwd_row, fwd_col, fwd_data, n_items)
        _, bwd_norm_adj = build_normalized_adj(bwd_row, bwd_col, bwd_data, n_items)
        
        self.norm_adj_fwd = torch.sparse_coo_tensor(
            np.row_stack([fwd_norm_adj.row, fwd_norm_adj.col]),
            fwd_norm_adj.data, (n_items, n_items), dtype=torch.float32
        )
        self.norm_adj_bwd = torch.sparse_coo_tensor(
            np.row_stack([bwd_norm_adj.row, bwd_norm_adj.col]),
            bwd_norm_adj.data, (n_items, n_items), dtype=torch.float32
        )
        
        # Save graphs for visualization
        self._save_graphs_for_viz(fwd_norm_adj, bwd_norm_adj, args.data_name)
        
        # Combined graph for GNN (still symmetric for message passing)
        combined_data = fwd_data + bwd_data
        combined_row = fwd_row + bwd_row
        combined_col = fwd_col + bwd_col
        _, combined_norm_adj = build_normalized_adj(combined_row, combined_col, combined_data, n_items)
        
        g = dgl.from_scipy(combined_norm_adj)
        g.edata['weight'] = torch.tensor(combined_norm_adj.data)
        self.g = g
        self.g_cpu = g.clone()
        self.norm_adj = torch.sparse_coo_tensor(
            np.row_stack([combined_norm_adj.row, combined_norm_adj.col]),
            combined_norm_adj.data, (n_items, n_items), dtype=torch.float32
        )
    else:
        # Original behavior: symmetric graph (backward compatible)
        row, col, data = [], [], []
        for item_list in user_seq:
            item_list = item_list[:-2]
            item_list_len = len(item_list)
            for item_idx in range(item_list_len - 1):
                target_num = min(k, item_list_len - item_idx - 1)
                if target_num > 0:
                    row += [item_list[item_idx]] * target_num
                    col += item_list[item_idx + 1: item_idx + 1 + target_num]
                    distances = np.arange(1, 1 + target_num)
                    data.append(np.power(decay_base, distances - 1))
        
        data = np.concatenate(data) if data else np.array([])
        row, col = np.array(row) - 1, np.array(col) - 1
        sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items))
        sparse_matrix = sparse_matrix + sparse_matrix.T + sp.eye(n_items)  # symmetrize
        degree = np.array((sparse_matrix > 0).sum(1)).flatten()
        degree = np.nan_to_num(1 / degree, posinf=0)
        degree = sp.diags(degree)
        norm_adj = (degree @ sparse_matrix + sparse_matrix @ degree).tocoo()
        
        g = dgl.from_scipy(norm_adj)
        g.edata['weight'] = torch.tensor(norm_adj.data)
        self.g = g
        self.g_cpu = g.clone()
        self.norm_adj = torch.sparse_coo_tensor(
            np.row_stack([norm_adj.row, norm_adj.col]),
            norm_adj.data, (n_items, n_items), dtype=torch.float32
        )
```

---

### Change 4: New `_save_graphs_for_viz()` Method

**Purpose:** Save forward and backward graphs for visualization and debugging.

**Location:** Add as new method in `SASRecModel` class (after `global_graph_construction`)

**Original code:** Did not exist

**New code:**
```python
def _save_graphs_for_viz(self, fwd_adj, bwd_adj, data_name):
    """Save forward and backward graphs for visualization."""
    import os
    save_dir = "output/graphs"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as scipy sparse matrices
    sp.save_npz(f"{save_dir}/{data_name}_forward_graph.npz", fwd_adj.tocsr())
    sp.save_npz(f"{save_dir}/{data_name}_backward_graph.npz", bwd_adj.tocsr())
    
    # Save edge lists for easy inspection
    with open(f"{save_dir}/{data_name}_forward_edges.txt", 'w') as f:
        f.write("source,target,weight\n")
        for i, j, w in zip(fwd_adj.row, fwd_adj.col, fwd_adj.data):
            f.write(f"{i},{j},{w:.4f}\n")
    
    with open(f"{save_dir}/{data_name}_backward_edges.txt", 'w') as f:
        f.write("source,target,weight\n")
        for i, j, w in zip(bwd_adj.row, bwd_adj.col, bwd_adj.data):
            f.write(f"{i},{j},{w:.4f}\n")
    
    # Print stats
    print(f"\nGraph Statistics:")
    print(f"  Forward graph: {fwd_adj.nnz} edges")
    print(f"  Backward graph: {bwd_adj.nnz} edges")
    print(f"  Saved to: {save_dir}/")
    
    # Check asymmetry
    diff = np.abs(fwd_adj.toarray() - bwd_adj.toarray()).sum()
    print(f"  Asymmetry (sum of |fwd - bwd|): {diff:.2f}")
```

---

### Change 5: DGL CPU Compatibility in `get_gnn_embeddings()`

**Purpose:** DGL CPU version cannot move graphs to GPU. Keep DGL graph operations on CPU and only move resulting tensors.

**Original code:**
```python
def get_gnn_embeddings(self, device, noise=True):
    self.g = self.g.to(device)
    self.norm_adj = self.norm_adj.to(device)
    if noise:
        US = torch.sparse.mm(self.norm_adj, self.US.weight.T)
        V = torch.sparse.mm(self.norm_adj, self.V.weight.T)
    emb = self.item_embeddings.weight[1:-1]
    emb_list = [emb]
    for idx in range(self.args.gnn_layers):
        if noise:
            if self.args.svd:
                self.real_US, self.real_V = self.real_US.to(device), self.real_V.to(device)
                emb = self.real_US @ (self.real_V.T @ emb)
            else:
                emb = self.graph_conv(self.g, emb) + self.args.graph_noise * US @ (V.T @ emb)
        else:
            emb = self.graph_conv(self.g, emb)
        emb_list.append(emb)
    emb = torch.stack(emb_list, dim=1).mean(1)
    return emb
```

**New code:**
```python
def get_gnn_embeddings(self, device, noise=True):
    # Keep DGL graph on CPU (DGL-CPU compatibility), only move norm_adj to device
    self.norm_adj = self.norm_adj.to(device)
    if noise:
        US = torch.sparse.mm(self.norm_adj, self.US.weight.T)
        V = torch.sparse.mm(self.norm_adj, self.V.weight.T)
    emb = self.item_embeddings.weight[1:-1]
    emb_list = [emb]
    for idx in range(self.args.gnn_layers):
        if noise:
            if self.args.svd:
                self.real_US, self.real_V = self.real_US.to(device), self.real_V.to(device)
                emb = self.real_US @ (self.real_V.T @ emb)
            else:
                # Run graph conv on CPU, then move result to device
                emb_cpu = emb.cpu()
                emb_cpu = self.graph_conv(self.g, emb_cpu)
                emb = emb_cpu.to(device) + self.args.graph_noise * US @ (V.T @ emb)
        else:
            # Run graph conv on CPU, then move result to device
            emb_cpu = emb.cpu()
            emb_cpu = self.graph_conv(self.g, emb_cpu)
            emb = emb_cpu.to(device)
        emb_list.append(emb)
    emb = torch.stack(emb_list, dim=1).mean(1)
    return emb
```

---

### Change 6: DGL CPU Compatibility in `add_position_embedding()`

**Purpose:** Same DGL CPU compatibility fix.

**Original code:**
```python
def add_position_embedding(self, sequence, user_ids=None):
    self.g = self.g.to(sequence.device)
    self.norm_adj = self.norm_adj.to(sequence.device)
```

**New code:**
```python
def add_position_embedding(self, sequence, user_ids=None):
    # Keep DGL graph on CPU (DGL-CPU compatibility), only move norm_adj to device
    self.norm_adj = self.norm_adj.to(sequence.device)
```

---

### Change 7: Directional Attention Bias in `forward()` Method

**Purpose:** When directional mode is enabled, compute separate attention biases from forward and backward graphs, then mix them using learnable weights.

**Original code:**
```python
if self.args.att_bias and user_ids != None:
    row, col = input_ids.repeat_interleave(max_len, dim=-1).flatten(), input_ids.repeat(1, max_len).flatten()
    self.norm_adj = self.norm_adj.to(input_ids.device)
    self.dense_norm_adj = self.dense_norm_adj.to(input_ids.device)
    g = self.dense_norm_adj
    att_bias = g[row - 1, col - 1].reshape(input_ids.shape[0], 1, max_len, max_len)
    
    if self.args.gsl_weight:
        unique_row, inv_row = torch.unique(row - 1, return_inverse=True)
        unique_col, inv_col = torch.unique(col - 1, return_inverse=True)
        g_gsl = self.US.weight.T[unique_row] @ self.V.weight.T[unique_col].T
        g_gsl = g_gsl[inv_row, inv_col].reshape(input_ids.shape[0], 1, max_len, max_len)
        att_bias = att_bias + self.args.graph_noise * g_gsl
    user_weight = self.adaption_layer(self.user_embeddings.weight[user_ids]).unsqueeze(-1).unsqueeze(-1)
    if input_ids.shape[0] == 2 * user_ids.shape[0]: # for aug
        user_weight = user_weight.repeat(2, 1, 1, 1)
    att_bias = self.args.att_bias * user_weight * att_bias
else:
    att_bias = None
```

**New code:**
```python
if self.args.att_bias and user_ids != None:
    row, col = input_ids.repeat_interleave(max_len, dim=-1).flatten(), input_ids.repeat(1, max_len).flatten()
    self.norm_adj = self.norm_adj.to(input_ids.device)
    self.dense_norm_adj = self.dense_norm_adj.to(input_ids.device)
    
    # Check if directional mode is enabled
    directional = getattr(self.args, 'directional', False)
    
    if directional and hasattr(self, 'dense_norm_adj_fwd') and hasattr(self, 'dense_norm_adj_bwd'):
        # Move directional adjacencies to device
        self.dense_norm_adj_fwd = self.dense_norm_adj_fwd.to(input_ids.device)
        self.dense_norm_adj_bwd = self.dense_norm_adj_bwd.to(input_ids.device)
        
        # Compute separate forward and backward attention biases
        g_fwd = self.dense_norm_adj_fwd
        g_bwd = self.dense_norm_adj_bwd
        
        # Forward graph: A→B means A comes before B
        # For attention[B,A] (B attending to A), we want forward edge A→B
        # So lookup g_fwd[col-1, row-1] (transposed from row,col)
        att_bias_fwd = g_fwd[col - 1, row - 1].reshape(input_ids.shape[0], 1, max_len, max_len)
        
        # Backward graph: B→A means B is preceded by A  
        # For attention[B,A], we want backward edge B→A
        # So lookup g_bwd[row-1, col-1] (normal order)
        att_bias_bwd = g_bwd[row - 1, col - 1].reshape(input_ids.shape[0], 1, max_len, max_len)
        
        # Apply learnable mixing weights with softmax normalization
        lambda_fwd = torch.sigmoid(self.lambda_fwd)
        lambda_bwd = torch.sigmoid(self.lambda_bwd)
        # Normalize so they sum to 1
        total = lambda_fwd + lambda_bwd
        lambda_fwd_norm = lambda_fwd / total
        lambda_bwd_norm = lambda_bwd / total
        
        att_bias = lambda_fwd_norm * att_bias_fwd + lambda_bwd_norm * att_bias_bwd
    else:
        # Original behavior (backward compatible)
        g = self.dense_norm_adj
        att_bias = g[row - 1, col - 1].reshape(input_ids.shape[0], 1, max_len, max_len)
    
    if self.args.gsl_weight:
        unique_row, inv_row = torch.unique(row - 1, return_inverse=True)
        unique_col, inv_col = torch.unique(col - 1, return_inverse=True)
        g_gsl = self.US.weight.T[unique_row] @ self.V.weight.T[unique_col].T
        g_gsl = g_gsl[inv_row, inv_col].reshape(input_ids.shape[0], 1, max_len, max_len)
        att_bias = att_bias + self.args.graph_noise * g_gsl
    user_weight = self.adaption_layer(self.user_embeddings.weight[user_ids]).unsqueeze(-1).unsqueeze(-1)
    if input_ids.shape[0] == 2 * user_ids.shape[0]: # for aug
        user_weight = user_weight.repeat(2, 1, 1, 1)
    att_bias = self.args.att_bias * user_weight * att_bias
else:
    att_bias = None
```

---

## File 2: `src/main.py`

### Change 8: New Command Line Arguments

**Purpose:** Add CLI arguments to enable directional graphs and control distance decay.

**Location:** After the `--gsl_weight` argument definition (around line 127)

**Original code:** Did not exist

**New code (add after `--gsl_weight`):**
```python
# Directional graph arguments (APGL4SR improvements)
parser.add_argument("--directional", action="store_true", help="use separate forward/backward directed graphs with learnable mixing")
parser.add_argument("--distance_decay_base", type=float, default=0.8, help="base for exponential distance decay (weight = base^distance)")
```

---

### Change 9: Dense Directional Adjacency Creation

**Purpose:** Create dense versions of forward and backward adjacency matrices for fast GPU lookup.

**Location:** After `model.dense_norm_adj` creation (around line 218)

**Original code:**
```python
model = SASRecModel(args=args)
model.global_graph_construction(train_dataset)
model.dense_norm_adj = model.norm_adj.to_dense()
model.dense_norm_adj = torch.cat([model.dense_norm_adj, torch.zeros(1, model.dense_norm_adj.shape[1])]) # last one is for padding
model.dense_norm_adj = torch.cat([model.dense_norm_adj, torch.zeros(model.dense_norm_adj.shape[0], 1)], dim=-1) # last one is for padding

if args.svd:
    model.get_svd()
```

**New code:**
```python
model = SASRecModel(args=args)
model.global_graph_construction(train_dataset)
model.dense_norm_adj = model.norm_adj.to_dense()
model.dense_norm_adj = torch.cat([model.dense_norm_adj, torch.zeros(1, model.dense_norm_adj.shape[1])]) # last one is for padding
model.dense_norm_adj = torch.cat([model.dense_norm_adj, torch.zeros(model.dense_norm_adj.shape[0], 1)], dim=-1) # last one is for padding

# Create dense directional adjacencies if directional mode is enabled
if hasattr(args, 'directional') and args.directional:
    model.dense_norm_adj_fwd = model.norm_adj_fwd.to_dense()
    model.dense_norm_adj_fwd = torch.cat([model.dense_norm_adj_fwd, torch.zeros(1, model.dense_norm_adj_fwd.shape[1])])
    model.dense_norm_adj_fwd = torch.cat([model.dense_norm_adj_fwd, torch.zeros(model.dense_norm_adj_fwd.shape[0], 1)], dim=-1)
    
    model.dense_norm_adj_bwd = model.norm_adj_bwd.to_dense()
    model.dense_norm_adj_bwd = torch.cat([model.dense_norm_adj_bwd, torch.zeros(1, model.dense_norm_adj_bwd.shape[1])])
    model.dense_norm_adj_bwd = torch.cat([model.dense_norm_adj_bwd, torch.zeros(model.dense_norm_adj_bwd.shape[0], 1)], dim=-1)
    print(f"Directional mode enabled: forward graph shape {model.dense_norm_adj_fwd.shape}, backward graph shape {model.dense_norm_adj_bwd.shape}")

if args.svd:
    model.get_svd()
```

---

## File 3: `src/scripts/visualize_graphs.py` (NEW FILE)

**Purpose:** Visualization script to analyze and compare forward vs backward graphs.

**This is a completely new file. Create it with the following content:**

```python
"""
Visualize forward and backward graphs from APGL4SR directional graph construction.
"""
import argparse
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def load_graphs(data_name, graph_dir="output/graphs"):
    """Load saved forward and backward graphs."""
    fwd_path = f"{graph_dir}/{data_name}_forward_graph.npz"
    bwd_path = f"{graph_dir}/{data_name}_backward_graph.npz"
    
    fwd_graph = sp.load_npz(fwd_path)
    bwd_graph = sp.load_npz(bwd_path)
    
    return fwd_graph, bwd_graph


def plot_adjacency_heatmap(adj_matrix, title, ax, sample_size=100):
    """Plot a heatmap of adjacency matrix (sampled for large graphs)."""
    dense = adj_matrix.toarray()
    n = dense.shape[0]
    
    if n > sample_size:
        # Sample random indices
        indices = np.random.choice(n, sample_size, replace=False)
        indices = np.sort(indices)
        dense = dense[np.ix_(indices, indices)]
    
    im = ax.imshow(dense, cmap='Blues', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Target Item')
    ax.set_ylabel('Source Item')
    return im


def plot_degree_distribution(adj_matrix, title, ax):
    """Plot in-degree and out-degree distributions."""
    out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    in_degree = np.array(adj_matrix.sum(axis=0)).flatten()
    
    ax.hist(out_degree, bins=50, alpha=0.5, label='Out-degree', color='blue')
    ax.hist(in_degree, bins=50, alpha=0.5, label='In-degree', color='red')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')


def print_graph_stats(fwd_graph, bwd_graph):
    """Print statistics about the graphs."""
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    
    print(f"\nForward Graph (item -> future items):")
    print(f"  Shape: {fwd_graph.shape}")
    print(f"  Non-zero edges: {fwd_graph.nnz}")
    print(f"  Density: {fwd_graph.nnz / (fwd_graph.shape[0] * fwd_graph.shape[1]):.6f}")
    
    fwd_out = np.array(fwd_graph.sum(axis=1)).flatten()
    fwd_in = np.array(fwd_graph.sum(axis=0)).flatten()
    print(f"  Avg out-degree: {fwd_out.mean():.2f}")
    print(f"  Avg in-degree: {fwd_in.mean():.2f}")
    
    print(f"\nBackward Graph (item -> past items):")
    print(f"  Shape: {bwd_graph.shape}")
    print(f"  Non-zero edges: {bwd_graph.nnz}")
    print(f"  Density: {bwd_graph.nnz / (bwd_graph.shape[0] * bwd_graph.shape[1]):.6f}")
    
    bwd_out = np.array(bwd_graph.sum(axis=1)).flatten()
    bwd_in = np.array(bwd_graph.sum(axis=0)).flatten()
    print(f"  Avg out-degree: {bwd_out.mean():.2f}")
    print(f"  Avg in-degree: {bwd_in.mean():.2f}")
    
    # Check asymmetry
    diff = np.abs(fwd_graph.toarray() - bwd_graph.toarray()).sum()
    print(f"\nAsymmetry (sum of |fwd - bwd|): {diff:.2f}")
    
    # Check if graphs are transposes
    fwd_dense = fwd_graph.toarray()
    bwd_dense = bwd_graph.toarray()
    transpose_diff = np.abs(fwd_dense - bwd_dense.T).sum()
    print(f"Transpose difference (|fwd - bwd.T|): {transpose_diff:.2f}")


def print_top_edges(adj_matrix, title, top_k=20):
    """Print top weighted edges."""
    coo = adj_matrix.tocoo()
    
    # Sort by weight
    sorted_idx = np.argsort(coo.data)[::-1][:top_k]
    
    print(f"\nTop {top_k} edges in {title}:")
    print("-" * 40)
    for idx in sorted_idx:
        src, tgt, weight = coo.row[idx], coo.col[idx], coo.data[idx]
        print(f"  Item {src} -> Item {tgt}: {weight:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize APGL4SR directional graphs')
    parser.add_argument('--dataset', type=str, default='Beauty', help='Dataset name')
    parser.add_argument('--graph_dir', type=str, default='output/graphs', help='Directory with saved graphs')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size for heatmap visualization')
    parser.add_argument('--save_fig', type=str, default=None, help='Path to save figure')
    args = parser.parse_args()
    
    print(f"Loading graphs for {args.dataset}...")
    fwd_graph, bwd_graph = load_graphs(args.dataset, args.graph_dir)
    
    # Print statistics
    print_graph_stats(fwd_graph, bwd_graph)
    
    # Print top edges
    print_top_edges(fwd_graph, "Forward Graph")
    print_top_edges(bwd_graph, "Backward Graph")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Heatmaps
    plot_adjacency_heatmap(fwd_graph, f'Forward Graph (sampled {args.sample_size}x{args.sample_size})', 
                           axes[0, 0], args.sample_size)
    plot_adjacency_heatmap(bwd_graph, f'Backward Graph (sampled {args.sample_size}x{args.sample_size})', 
                           axes[0, 1], args.sample_size)
    
    # Degree distributions
    plot_degree_distribution(fwd_graph, 'Forward Graph Degree Distribution', axes[1, 0])
    plot_degree_distribution(bwd_graph, 'Backward Graph Degree Distribution', axes[1, 1])
    
    plt.suptitle(f'{args.dataset} - Directional Graph Analysis', fontsize=14)
    plt.tight_layout()
    
    if args.save_fig:
        plt.savefig(args.save_fig, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {args.save_fig}")
    
    plt.show()


if __name__ == '__main__':
    main()
```

---

## Summary Table

| # | File | Method/Location | Change Description |
|---|------|----------------|-------------------|
| 1 | models.py | `KMeans.__init_cluster()` | GPU faiss → CPU fallback |
| 2 | models.py | `SASRecModel.__init__()` | Add `lambda_fwd`, `lambda_bwd` learnable params |
| 3 | models.py | `global_graph_construction()` | Complete rewrite for directional + decay |
| 4 | models.py | `_save_graphs_for_viz()` | New method for saving graphs |
| 5 | models.py | `get_gnn_embeddings()` | DGL CPU compatibility |
| 6 | models.py | `add_position_embedding()` | DGL CPU compatibility |
| 7 | models.py | `forward()` | Directional attention bias mixing |
| 8 | main.py | CLI arguments | Add `--directional`, `--distance_decay_base` |
| 9 | main.py | After model construction | Create dense directional adjacencies |
| 10 | scripts/ | visualize_graphs.py | New visualization script |

---

## Usage Examples

### Original command (baseline):
```bash
python main.py --data_name Beauty --cf_weight 0.1 --gcl_weight 0.1 --k 2
```

### New command (with directional graphs and decay):
```bash
python main.py --data_name Beauty --cf_weight 0.1 --gcl_weight 0.1 --k 2 --directional --distance_decay_base 0.8
```

### Visualize the graphs:
```bash
python scripts/visualize_graphs.py --dataset Beauty --sample_size 50
```

---

## Technical Explanation

### Why Directional Graphs?

In sequential recommendation, the order of items matters:
- **Forward graph** (A→B): Captures "items that follow A" - useful for predicting what comes next
- **Backward graph** (B→A): Captures "items that precede B" - useful for understanding context

The original APGL4SR used a **symmetric** graph (A↔B), which loses directional information.

### Why Distance Decay?

Items closer in the sequence are more related:
- Immediate next item (distance=1): weight = 1.0
- Second next item (distance=2): weight = 0.8
- Third next item (distance=3): weight = 0.64

Formula: `weight = decay_base^(distance-1)`

### Why Learnable Mixing?

Different datasets may benefit from different emphasis on forward vs backward information:
- `lambda_fwd`: learned weight for forward graph attention
- `lambda_bwd`: learned weight for backward graph attention

These are normalized via softmax so they sum to 1.
