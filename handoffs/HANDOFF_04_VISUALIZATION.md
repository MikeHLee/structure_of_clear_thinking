# Handoff 04: Embedding Space Visualization

**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Estimated Effort**: 1 day  
**Dependencies**: Handoff 01 or 03 (trained embeddings)

---

## 1. Problem Statement

The 2.71× separation ratio is a single number. Reviewers and readers need **visual confirmation** that:
1. Types cluster by ontology membership
2. Semantically similar types are close despite ontology boundaries
3. Valid transitions form interpretable manifolds
4. Invalid transitions are visibly separated

Currently, no visualization exists for the embedding space.

---

## 2. Success Criteria

| Deliverable | Description |
|-------------|-------------|
| t-SNE plot | 2D projection colored by ontology |
| UMAP plot | 2D projection with better global structure |
| Transition heatmap | Valid vs. invalid transition distances |
| Interactive notebook | Jupyter notebook for exploration |
| Paper figure | Publication-ready figure (300 DPI, vector) |

---

## 3. Technical Specification

### 3.1 Visualization Components

1. **Type Embedding Scatter Plot**: t-SNE/UMAP projection colored by ontology
2. **Relation Embedding Scatter Plot**: Same for relation embeddings
3. **Transition Distance Heatmap**: Matrix showing pairwise distances
4. **Path Visualization**: Graph showing valid composition paths
5. **Anomaly Detection**: Highlight types that cluster "wrong"

### 3.2 Implementation Plan

#### Step 1: Load Trained Embeddings
```python
import torch
import numpy as np
import json
from pathlib import Path

def load_embeddings(model_path: str, config_path: str = None):
    """
    Load trained OlogEmbedding model and extract embeddings.
    """
    state_dict = torch.load(model_path, map_location='cpu')
    
    type_embeddings = state_dict.get('type_embeddings.weight', 
                                     state_dict.get('type_embedding.weight'))
    relation_embeddings = state_dict.get('relation_embeddings.weight',
                                         state_dict.get('relation_embedding.weight'))
    
    return {
        "type_embeddings": type_embeddings.numpy(),
        "relation_embeddings": relation_embeddings.numpy() if relation_embeddings is not None else None
    }


def load_metadata(ontologies: dict):
    """
    Create metadata for visualization labels and colors.
    """
    type_to_ontology = {}
    type_labels = []
    ontology_labels = []
    
    for ont_name, ont in ontologies.items():
        for t in ont["types"]:
            type_to_ontology[t] = ont_name
            type_labels.append(t)
            ontology_labels.append(ont_name)
    
    return {
        "type_labels": type_labels,
        "ontology_labels": ontology_labels,
        "type_to_ontology": type_to_ontology,
        "ontology_names": list(ontologies.keys())
    }
```

#### Step 2: Dimensionality Reduction
```python
from sklearn.manifold import TSNE
import umap

def compute_projections(embeddings: np.ndarray, perplexity: int = 30, n_neighbors: int = 15):
    """
    Compute t-SNE and UMAP projections.
    """
    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=42,
        n_iter=1000
    )
    tsne_coords = tsne.fit_transform(embeddings)
    
    # UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        min_dist=0.1,
        random_state=42
    )
    umap_coords = reducer.fit_transform(embeddings)
    
    return {
        "tsne": tsne_coords,
        "umap": umap_coords
    }
```

#### Step 3: Main Scatter Plot
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

def plot_embedding_scatter(
    coords: np.ndarray,
    ontology_labels: list,
    type_labels: list,
    title: str = "Ontological Embeddings",
    method: str = "t-SNE",
    save_path: str = None
):
    """
    Create publication-quality scatter plot of embeddings.
    """
    # Color palette
    unique_ontologies = list(set(ontology_labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_ontologies)))
    ont_to_color = {ont: colors[i] for i, ont in enumerate(unique_ontologies)}
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # Plot points
    for ont in unique_ontologies:
        mask = [o == ont for o in ontology_labels]
        points = coords[mask]
        labels = [type_labels[i] for i, m in enumerate(mask) if m]
        
        ax.scatter(
            points[:, 0], points[:, 1],
            c=[ont_to_color[ont]],
            label=ont,
            s=150,
            alpha=0.7,
            edgecolors='white',
            linewidth=1.5
        )
        
        # Add labels
        for (x, y), label in zip(points, labels):
            ax.annotate(
                label,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    
    ax.set_xlabel(f"{method} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{method} Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title="Ontology", loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax
```

#### Step 4: Transition Distance Heatmap
```python
import seaborn as sns

def plot_transition_heatmap(
    embeddings: np.ndarray,
    type_labels: list,
    ontologies: dict,
    save_path: str = None
):
    """
    Plot heatmap of pairwise distances with valid/invalid annotations.
    """
    n_types = len(type_labels)
    distances = np.zeros((n_types, n_types))
    
    for i in range(n_types):
        for j in range(n_types):
            distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
    
    # Create validity mask
    valid_mask = np.zeros((n_types, n_types), dtype=bool)
    type_to_idx = {t: i for i, t in enumerate(type_labels)}
    
    for ont in ontologies.values():
        for src, tgt, rel in ont["aspects"]:
            if src in type_to_idx and tgt in type_to_idx:
                valid_mask[type_to_idx[src], type_to_idx[tgt]] = True
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Full distance heatmap
    sns.heatmap(
        distances,
        ax=axes[0],
        xticklabels=type_labels,
        yticklabels=type_labels,
        cmap='viridis',
        square=True
    )
    axes[0].set_title("Pairwise Type Distances", fontweight='bold')
    axes[0].tick_params(labelsize=6)
    
    # Valid vs Invalid comparison
    valid_dists = distances[valid_mask]
    invalid_dists = distances[~valid_mask & ~np.eye(n_types, dtype=bool)]
    
    axes[1].hist(valid_dists, bins=20, alpha=0.7, label=f'Valid (n={len(valid_dists)})', color='green')
    axes[1].hist(invalid_dists, bins=20, alpha=0.7, label=f'Invalid (n={len(invalid_dists)})', color='red')
    axes[1].axvline(np.mean(valid_dists), color='green', linestyle='--', linewidth=2, label=f'Valid mean: {np.mean(valid_dists):.2f}')
    axes[1].axvline(np.mean(invalid_dists), color='red', linestyle='--', linewidth=2, label=f'Invalid mean: {np.mean(invalid_dists):.2f}')
    axes[1].set_xlabel("Distance", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Valid vs Invalid Transition Distances", fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
```

#### Step 5: Composition Path Graph
```python
import networkx as nx

def plot_ontology_graph(
    ontologies: dict,
    embeddings: np.ndarray,
    type_labels: list,
    save_path: str = None
):
    """
    Plot ontology graph with edge weights from embedding distances.
    """
    type_to_idx = {t: i for i, t in enumerate(type_labels)}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for ax_idx, (ont_name, ont) in enumerate(ontologies.items()):
        if ax_idx >= 4:
            break
        
        G = nx.DiGraph()
        
        # Add nodes
        for t in ont["types"]:
            G.add_node(t)
        
        # Add edges with distances as weights
        for src, tgt, rel in ont["aspects"]:
            if src in type_to_idx and tgt in type_to_idx:
                dist = np.linalg.norm(
                    embeddings[type_to_idx[src]] - embeddings[type_to_idx[tgt]]
                )
                G.add_edge(src, tgt, label=rel, weight=dist)
        
        pos = nx.spring_layout(G, seed=42)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, ax=axes[ax_idx], node_size=800, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, ax=axes[ax_idx], font_size=9)
        
        # Edge widths based on inverse distance (closer = thicker)
        edge_widths = [3.0 / (G[u][v]['weight'] + 0.1) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=axes[ax_idx], width=edge_widths, 
                               arrows=True, arrowsize=15, alpha=0.7)
        
        # Edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=axes[ax_idx], font_size=7)
        
        axes[ax_idx].set_title(f"{ont_name.title()} Ontology", fontweight='bold')
        axes[ax_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
```

#### Step 6: Main Visualization Script
```python
def main():
    """Generate all visualizations."""
    # Load embeddings
    embeddings_data = load_embeddings("results/olog_embeddings.pt")
    type_embeddings = embeddings_data["type_embeddings"]
    
    # Load ontology metadata
    from scripts.modal_olog_training import ONTOLOGIES
    metadata = load_metadata(ONTOLOGIES)
    
    # Compute projections
    projections = compute_projections(type_embeddings)
    
    # Output directory
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_embedding_scatter(
        projections["tsne"],
        metadata["ontology_labels"],
        metadata["type_labels"],
        title="Ontological Type Embeddings (t-SNE)",
        method="t-SNE",
        save_path=output_dir / "tsne_embeddings.png"
    )
    
    plot_embedding_scatter(
        projections["umap"],
        metadata["ontology_labels"],
        metadata["type_labels"],
        title="Ontological Type Embeddings (UMAP)",
        method="UMAP",
        save_path=output_dir / "umap_embeddings.png"
    )
    
    plot_transition_heatmap(
        type_embeddings,
        metadata["type_labels"],
        ONTOLOGIES,
        save_path=output_dir / "transition_heatmap.png"
    )
    
    plot_ontology_graph(
        ONTOLOGIES,
        type_embeddings,
        metadata["type_labels"],
        save_path=output_dir / "ontology_graphs.png"
    )
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

---

## 4. Files to Create

| File | Purpose |
|------|---------|
| `scripts/visualize_embeddings.py` | Main visualization script |
| `notebooks/embedding_explorer.ipynb` | Interactive Jupyter notebook |
| `results/visualizations/` | Output directory |
| `results/visualizations/tsne_embeddings.png` | t-SNE plot |
| `results/visualizations/umap_embeddings.png` | UMAP plot |
| `results/visualizations/transition_heatmap.png` | Distance heatmap |
| `results/visualizations/ontology_graphs.png` | Graph visualizations |

---

## 5. Execution Commands

```bash
cd /Users/Michaellee/Documents/Runes/ai_research/topics/ontological_induction_sequence_modeling

# Install visualization dependencies
pip install umap-learn seaborn matplotlib networkx

# Generate visualizations (requires trained model)
python scripts/visualize_embeddings.py

# Or run in notebook
jupyter notebook notebooks/embedding_explorer.ipynb
```

---

## 6. Validation Checklist

- [ ] t-SNE plot shows ontology clustering
- [ ] UMAP plot preserves global structure
- [ ] Heatmap distinguishes valid/invalid distances
- [ ] Histogram shows clear separation
- [ ] Graph edge widths correlate with embedding similarity
- [ ] All plots are publication-ready (300 DPI, proper labels)

---

## 7. Expected Visual Results

**Good t-SNE/UMAP Result**:
- 4 distinct clusters (one per ontology)
- Semantically similar types (Customer/User) are adjacent across clusters
- Clear separation between dissimilar types

**Good Heatmap Result**:
- Block diagonal structure (intra-ontology similarity)
- Off-diagonal blocks lighter (inter-ontology distance)
- Valid transitions in histogram shifted left (smaller distances)

---

## 8. Handoff Prompt

```
Execute HANDOFF_04: Embedding Space Visualization

Current state: No visualization of the embedding space exists.
Task: Create t-SNE, UMAP, heatmap, and graph visualizations.

Requirements:
- Trained model at results/olog_embeddings.pt
- pip install umap-learn seaborn matplotlib networkx

Steps:
1. Create scripts/visualize_embeddings.py
2. Load trained embeddings
3. Compute t-SNE and UMAP projections
4. Generate 4 visualization types:
   - Scatter plot colored by ontology
   - Distance heatmap
   - Valid vs. invalid histogram
   - Ontology graphs with weighted edges
5. Save to results/visualizations/
6. Create Jupyter notebook for exploration

Success: Clear visual clustering by ontology, valid/invalid separation visible
```
