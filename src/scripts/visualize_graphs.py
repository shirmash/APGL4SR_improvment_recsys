"""
Visualize Forward vs Backward Graphs from APGL4SR

Usage:
    python scripts/visualize_graphs.py --dataset Beauty --sample_size 50
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import argparse
import os


def load_graphs(dataset):
    """Load saved graphs."""
    graph_dir = "output/graphs"
    fwd = sp.load_npz(f"{graph_dir}/{dataset}_forward_graph.npz")
    bwd = sp.load_npz(f"{graph_dir}/{dataset}_backward_graph.npz")
    return fwd, bwd


def visualize_adjacency_sample(fwd, bwd, sample_size=50, save_path=None):
    """Visualize a sample of the adjacency matrices side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sample a submatrix
    fwd_sample = fwd[:sample_size, :sample_size].toarray()
    bwd_sample = bwd[:sample_size, :sample_size].toarray()
    diff_sample = np.abs(fwd_sample - bwd_sample)
    
    # Forward graph
    im1 = axes[0].imshow(fwd_sample, cmap='Blues', aspect='auto')
    axes[0].set_title(f'Forward Graph (item→future)\n{sample_size}x{sample_size} sample')
    axes[0].set_xlabel('Target Item')
    axes[0].set_ylabel('Source Item')
    plt.colorbar(im1, ax=axes[0])
    
    # Backward graph
    im2 = axes[1].imshow(bwd_sample, cmap='Oranges', aspect='auto')
    axes[1].set_title(f'Backward Graph (item→past)\n{sample_size}x{sample_size} sample')
    axes[1].set_xlabel('Target Item')
    axes[1].set_ylabel('Source Item')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    im3 = axes[2].imshow(diff_sample, cmap='Reds', aspect='auto')
    axes[2].set_title(f'Absolute Difference\nSum: {diff_sample.sum():.2f}')
    axes[2].set_xlabel('Target Item')
    axes[2].set_ylabel('Source Item')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def visualize_degree_distribution(fwd, bwd, save_path=None):
    """Compare degree distributions of forward vs backward graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Out-degree (row sums)
    fwd_out = np.array(fwd.sum(axis=1)).flatten()
    bwd_out = np.array(bwd.sum(axis=1)).flatten()
    
    axes[0].hist(fwd_out, bins=50, alpha=0.7, label='Forward', color='blue')
    axes[0].hist(bwd_out, bins=50, alpha=0.7, label='Backward', color='orange')
    axes[0].set_xlabel('Out-Degree (weighted)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Out-Degree Distribution')
    axes[0].legend()
    
    # In-degree (column sums)
    fwd_in = np.array(fwd.sum(axis=0)).flatten()
    bwd_in = np.array(bwd.sum(axis=0)).flatten()
    
    axes[1].hist(fwd_in, bins=50, alpha=0.7, label='Forward', color='blue')
    axes[1].hist(bwd_in, bins=50, alpha=0.7, label='Backward', color='orange')
    axes[1].set_xlabel('In-Degree (weighted)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('In-Degree Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def print_sample_edges(dataset, num_edges=20):
    """Print sample edges from both graphs."""
    graph_dir = "output/graphs"
    
    print("\n=== Sample Forward Edges (item → future items) ===")
    with open(f"{graph_dir}/{dataset}_forward_edges.txt", 'r') as f:
        for i, line in enumerate(f):
            if i > num_edges:
                break
            print(line.strip())
    
    print("\n=== Sample Backward Edges (item → past items) ===")
    with open(f"{graph_dir}/{dataset}_backward_edges.txt", 'r') as f:
        for i, line in enumerate(f):
            if i > num_edges:
                break
            print(line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--sample_size", type=int, default=50)
    args = parser.parse_args()
    
    print(f"Loading graphs for {args.dataset}...")
    fwd, bwd = load_graphs(args.dataset)
    
    print(f"\nGraph shapes: Forward {fwd.shape}, Backward {bwd.shape}")
    print(f"Forward edges: {fwd.nnz}, Backward edges: {bwd.nnz}")
    
    # Check asymmetry
    total_diff = np.abs(fwd.toarray() - bwd.toarray()).sum()
    print(f"Total asymmetry: {total_diff:.2f}")
    
    # Print sample edges
    print_sample_edges(args.dataset)
    
    # Visualize
    save_dir = "output/graphs"
    visualize_adjacency_sample(fwd, bwd, args.sample_size, 
                               f"{save_dir}/{args.dataset}_adjacency_comparison.png")
    visualize_degree_distribution(fwd, bwd,
                                   f"{save_dir}/{args.dataset}_degree_distribution.png")


if __name__ == "__main__":
    main()
