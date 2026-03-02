# Directional Graph Attention for Sequential Recommendation: Extending APGL4SR with Temporal Direction and Distance Decay

**Atalia Solash and Shir Mashiah**
Department of Information Systems Engineering, Ben-Gurion University of the Negev, Israel

This repository extends the original [APGL4SR (CIKM 2023)](https://dl.acm.org/doi/abs/10.1145/3583780.3614781) paper with two targeted modifications to its global item graph component.

---

## Overview

APGL4SR augments a contrastive Transformer backbone with an adaptive global co-occurrence graph, using graph neural network propagation and a graph-conditioned attention bias to inject collaborative item-transition signals into the per-user sequence encoder. We replicate APGL4SR and propose two modifications to its graph component:

1. **Directional Graphs with Learnable Mixing** — Replace the symmetric co-occurrence graph with two directed matrices (forward and backward transitions) and combine their attention biases via learnable mixing scalars.
2. **Distance Decay Weighting** — Introduce exponential distance decay `w(d) = β^(d-1)` to down-weight distant co-occurrences, enabling a wider context window beyond the original k=2.

---

## Motivation

APGL4SR's global item graph is symmetric by construction — this collapses directed temporal relationships into undirected ones. Consider a product catalog containing iPhones and phone cases: users almost always buy a phone case *after* buying an iPhone, rarely the reverse. When the graph is symmetrized, this asymmetry is lost, and both query-to-key and key-to-query attention receive the same bias, even though they carry different predictive implications.

Additionally, APGL4SR fixes k=2 for the co-occurrence window. With flat edge weights, a larger window floods the graph with equally weighted but weakly related distant pairs. Distance decay resolves this tension: it down-weights distant pairs exponentially, so expanding to k ∈ {3, 4, 5} adds longer-range context without the flat-weight noise.

---

## Proposed Changes

### Change 1: Directional Graph Construction and Attention Bias

Instead of one symmetrized adjacency matrix A, we build two directed matrices:

- **A⁺ (forward)**: accumulates weight each time item `i` appears *before* item `j`
- **A⁻ (backward)**: accumulates weight each time item `i` appears *after* item `j`

Both matrices are row-normalized and stored as separate look-up tables. For LightGCN propagation, they are summed (direction doesn't affect neighbourhood aggregation). For the attention bias, they are kept separate:

```
B⁺_q,k = Ã⁺_vk,vq   (how often vk appeared before vq)
B⁻_q,k = Ã⁻_vq,vk   (how often vq appeared after vk)
```

**Learnable mixing weights** λ⁺ and λ⁻ are normalized via sigmoid so they always sum to 1:

```
λ̃⁺ = σ(λ⁺) / (σ(λ⁺) + σ(λ⁻))
λ̃⁻ = 1 - λ̃⁺
```

The two biases enter the Transformer self-attention as:

```
Attention(Q, K, V) = softmax(QK^T/√d + α·wu·λ̃⁺·B⁺ + α·wu·λ̃⁻·B⁻) V
```

Both scalars are initialized to produce equal mixing (λ̃⁺ = λ̃⁻ = 0.5) and shift during training toward the direction most informative for the dataset.

### Change 2: Distance Decay Weighting with Expanded Window

The original APGL4SR assigns a flat weight of `1/k` to all co-occurring pairs within the window. We replace this with an exponentially decaying weight:

```
w(d) = β^(d-1)
```

At distance d=1 the weight is 1.0; at d=2 it is β; at d=3 it is β², and so on. This preserves strong evidence from adjacent transitions while progressively discounting distant co-occurrences. The window size k can then safely be increased to {3, 4, 5} without flooding the graph with undifferentiated noise.

We use β=0.9 as the default value. Note that β=1 recovers the original flat weighting.

---

## Running the Code

### Requirements

```
Python >= 3.9
PyTorch >= 2.0
tqdm
faiss-cpu (or faiss-gpu == 1.7.1)
dgl
nni == 2.10
```

### Datasets

Four prepared datasets are included in the `data` folder.

### Train & Eval — Original APGL4SR

```bash
cd src
python main.py --data_name Beauty --cf_weight 0.1 --gcl_weight 0.1
```

Or use the provided scripts:

```bash
cd src
chmod +x ./scripts/run_<DATASET>.sh
./scripts/run_<DATASET>.sh
```

### Train & Eval — With Directional + Decay Modifications

```bash
cd src
python main.py --data_name Beauty --cf_weight 0.1 --gcl_weight 0.1 \
  --directional --distance_decay_base 0.9 --k 3
```

Key arguments:

| Argument | Description | Default |
|---|---|---|
| `--directional` | Enable separate forward/backward directed graphs | `False` |
| `--distance_decay_base` | Exponential decay base β; set to 1.0 for flat weights | `0.8` |
| `--k` | Co-occurrence window size | `2` |

---

## Original APGL4SR

Source code for the original paper:
[APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation (CIKM'2023)](https://dl.acm.org/doi/abs/10.1145/3583780.3614781)

```
@inproceedings{yin2023apgl4sr,
  title={APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation},
  author={Yin, Mingjia and Wang, Hao and Xu, Xiang and Wu, Likang and Zhao, Sirui and Guo, Wei and Liu, Yong and Tang, Ruiming and Lian, Defu and Chen, Enhong},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={3009--3019},
  year={2023}
}
```

---

## Acknowledgment

- Transformer and training pipeline are implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec) and [ICLRec](https://github.com/salesforce/ICLRec).
- Original APGL4SR code: [Graph-Team/APGL4SR](https://github.com/Graph-Team/APGL4SR).
