# Handoff 05: Depth Ablation on Hierarchical Ontologies

**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Estimated Effort**: 2-3 days  
**Dependencies**: Handoff 03 (scaled training infrastructure)

---

## 1. Problem Statement

The current ontologies are **flat** (depth ≤ 2). We don't know if the 2.71× separation ratio holds for:
- Deep hierarchies (e.g., Gene Ontology with 10+ levels)
- Fine-grained leaf-node transitions vs. coarse domain-level transitions
- Ontologies with varying branching factors

**Research Question**: Does separation ratio degrade with ontology depth? Is it easier to separate high-level domains than fine-grained subtypes?

---

## 2. Success Criteria

| Metric | Flat (Current) | Shallow (3-4) | Deep (5+) |
|--------|----------------|---------------|-----------|
| Separation ratio | 2.71× | ? | ? |
| Training convergence | Fast | ? | ? |
| Invalid detection | 100% | ? | ? |

**Hypothesis**: Separation ratio decreases with depth because fine-grained types are semantically closer.

---

## 3. Technical Specification

### 3.1 Hierarchical Ontology Sources

| Source | Depth | Types | Description |
|--------|-------|-------|-------------|
| WordNet | 10+ | 117,000 | Lexical hierarchy (hypernym/hyponym) |
| Gene Ontology | 12+ | 45,000 | Biological process/function/component |
| Schema.org | 5-6 | ~800 | Web content types |
| DBpedia Ontology | 4-5 | ~760 | Wikipedia-derived |

**Recommended**: Start with **Schema.org** (manageable size, clear hierarchy).

### 3.2 Synthetic Hierarchical Ontology Generator

```python
import random
from typing import List, Tuple, Dict

def generate_hierarchical_ontology(
    depth: int,
    branching_factor: int = 3,
    base_name: str = "Type"
) -> Dict:
    """
    Generate a synthetic hierarchical ontology.
    
    Args:
        depth: Maximum depth of hierarchy
        branching_factor: Children per node
        base_name: Prefix for type names
    
    Returns:
        Ontology dict with types, aspects, and depth metadata
    """
    types = []
    aspects = []
    type_depths = {}
    
    def generate_subtree(parent: str, current_depth: int, path: str):
        if current_depth >= depth:
            return
        
        for i in range(branching_factor):
            child = f"{path}_{i}"
            types.append(child)
            type_depths[child] = current_depth + 1
            
            # Add "is_a" relation (standard for hierarchies)
            aspects.append((child, parent, "is_a"))
            
            # Add sibling relations at some probability
            if i > 0 and random.random() < 0.3:
                sibling = f"{path}_{i-1}"
                aspects.append((child, sibling, "related_to"))
            
            generate_subtree(child, current_depth + 1, child)
    
    # Root
    root = f"{base_name}_root"
    types.append(root)
    type_depths[root] = 0
    
    generate_subtree(root, 0, root)
    
    return {
        "types": types,
        "aspects": aspects,
        "type_depths": type_depths,
        "max_depth": depth,
        "branching_factor": branching_factor,
        "n_types": len(types),
        "n_aspects": len(aspects),
    }


def generate_depth_ablation_suite():
    """Generate ontologies at multiple depths for ablation."""
    return {
        f"depth_{d}": generate_hierarchical_ontology(depth=d, branching_factor=3)
        for d in [2, 3, 4, 5, 6]
    }
```

### 3.3 Depth-Stratified Evaluation

```python
def evaluate_by_depth(
    model,
    ontology: Dict,
    type_to_idx: Dict[str, int]
) -> Dict[str, float]:
    """
    Evaluate separation ratio stratified by type depth.
    """
    type_depths = ontology["type_depths"]
    type_embs = model.type_embeddings.weight.detach().cpu().numpy()
    
    results = {}
    max_depth = ontology["max_depth"]
    
    for d in range(max_depth + 1):
        # Types at this depth
        types_at_depth = [t for t, depth in type_depths.items() if depth == d]
        
        if len(types_at_depth) < 2:
            continue
        
        # Intra-depth distances (same level, potentially siblings)
        intra_dists = []
        for i, t1 in enumerate(types_at_depth):
            for j, t2 in enumerate(types_at_depth):
                if i < j:
                    idx1 = type_to_idx.get(t1)
                    idx2 = type_to_idx.get(t2)
                    if idx1 is not None and idx2 is not None:
                        dist = np.linalg.norm(type_embs[idx1] - type_embs[idx2])
                        intra_dists.append(dist)
        
        # Cross-depth distances (different levels)
        other_types = [t for t, depth in type_depths.items() if depth != d]
        inter_dists = []
        for t1 in types_at_depth[:10]:  # Sample
            for t2 in random.sample(other_types, min(10, len(other_types))):
                idx1 = type_to_idx.get(t1)
                idx2 = type_to_idx.get(t2)
                if idx1 is not None and idx2 is not None:
                    dist = np.linalg.norm(type_embs[idx1] - type_embs[idx2])
                    inter_dists.append(dist)
        
        results[f"depth_{d}"] = {
            "n_types": len(types_at_depth),
            "intra_dist_mean": np.mean(intra_dists) if intra_dists else 0,
            "inter_dist_mean": np.mean(inter_dists) if inter_dists else 0,
            "separation_ratio": (
                np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8)
                if intra_dists else 0
            ),
        }
    
    return results


def evaluate_parent_child_separation(
    model,
    ontology: Dict,
    type_to_idx: Dict[str, int]
) -> Dict:
    """
    Evaluate how well parent-child relations are captured.
    
    Good embedding: parent-child distance < sibling distance < cousin distance
    """
    type_embs = model.type_embeddings.weight.detach().cpu().numpy()
    
    parent_child_dists = []
    sibling_dists = []
    
    # Build parent map
    parent_map = {}
    for child, parent, rel in ontology["aspects"]:
        if rel == "is_a":
            parent_map[child] = parent
    
    # Compute distances
    for child, parent in parent_map.items():
        idx_c = type_to_idx.get(child)
        idx_p = type_to_idx.get(parent)
        if idx_c is not None and idx_p is not None:
            parent_child_dists.append(np.linalg.norm(type_embs[idx_c] - type_embs[idx_p]))
    
    # Sibling distances (same parent)
    children_by_parent = {}
    for child, parent in parent_map.items():
        if parent not in children_by_parent:
            children_by_parent[parent] = []
        children_by_parent[parent].append(child)
    
    for parent, children in children_by_parent.items():
        for i, c1 in enumerate(children):
            for j, c2 in enumerate(children):
                if i < j:
                    idx1 = type_to_idx.get(c1)
                    idx2 = type_to_idx.get(c2)
                    if idx1 is not None and idx2 is not None:
                        sibling_dists.append(np.linalg.norm(type_embs[idx1] - type_embs[idx2]))
    
    return {
        "parent_child_dist_mean": np.mean(parent_child_dists) if parent_child_dists else 0,
        "sibling_dist_mean": np.mean(sibling_dists) if sibling_dists else 0,
        "hierarchy_respected": np.mean(parent_child_dists) < np.mean(sibling_dists) if sibling_dists else None,
    }
```

### 3.4 Real Ontology Loader (Schema.org)

```python
import requests
import json

def load_schema_org() -> Dict:
    """
    Load Schema.org ontology from their JSON-LD endpoint.
    """
    url = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
    response = requests.get(url)
    data = response.json()
    
    types = set()
    aspects = []
    type_depths = {}
    
    # Parse @graph
    for item in data.get("@graph", []):
        item_type = item.get("@type")
        item_id = item.get("@id", "").replace("schema:", "")
        
        if item_type == "rdfs:Class":
            types.add(item_id)
            
            # Get parent (subClassOf)
            parent = item.get("rdfs:subClassOf", {})
            if isinstance(parent, dict):
                parent_id = parent.get("@id", "").replace("schema:", "")
                if parent_id:
                    aspects.append((item_id, parent_id, "subClassOf"))
            elif isinstance(parent, list):
                for p in parent:
                    parent_id = p.get("@id", "").replace("schema:", "")
                    if parent_id:
                        aspects.append((item_id, parent_id, "subClassOf"))
    
    # Compute depths via BFS from "Thing"
    adj = {t: [] for t in types}
    for child, parent, rel in aspects:
        if parent in adj:
            adj[parent].append(child)
    
    type_depths = {"Thing": 0}
    queue = ["Thing"]
    while queue:
        current = queue.pop(0)
        for child in adj.get(current, []):
            if child not in type_depths:
                type_depths[child] = type_depths[current] + 1
                queue.append(child)
    
    return {
        "types": list(types),
        "aspects": aspects,
        "type_depths": type_depths,
        "max_depth": max(type_depths.values()) if type_depths else 0,
        "n_types": len(types),
        "n_aspects": len(aspects),
    }
```

---

## 4. Experiment Design

### 4.1 Ablation Matrix

| Ontology | Depth | Types | Branching | Expected Separation |
|----------|-------|-------|-----------|---------------------|
| Synthetic-D2 | 2 | ~13 | 3 | ~2.5× (baseline) |
| Synthetic-D3 | 3 | ~40 | 3 | ~2.0× |
| Synthetic-D4 | 4 | ~121 | 3 | ~1.7× |
| Synthetic-D5 | 5 | ~364 | 3 | ~1.5× |
| Schema.org | 5-6 | ~800 | varies | ~1.3× |

### 4.2 Metrics to Report

1. **Overall separation ratio** at each depth
2. **Depth-stratified separation** (leaf vs. root nodes)
3. **Parent-child distance** vs. **sibling distance**
4. **Hierarchy preservation score**: Does embedding respect tree structure?
5. **Invalid detection rate** at each depth level

---

## 5. Files to Create

| File | Purpose |
|------|---------|
| `scripts/depth_ablation.py` | Main ablation experiment |
| `scripts/modal_depth_ablation.py` | Modal cloud version |
| `src/hierarchical_ontology.py` | Ontology generator and loaders |
| `results/depth_ablation_results.json` | Results |
| `results/visualizations/depth_ablation_plot.png` | Visualization |

---

## 6. Execution Commands

```bash
cd /Users/Michaellee/Documents/Runes/ai_research/topics/ontological_induction_sequence_modeling

# Generate synthetic ontologies
python -c "from src.hierarchical_ontology import *; print(generate_depth_ablation_suite())"

# Run ablation (local test)
python scripts/depth_ablation.py --depths 2,3,4 --epochs 50

# Full Modal run
modal run scripts/modal_depth_ablation.py
```

---

## 7. Validation Checklist

- [ ] Synthetic generator produces valid hierarchies
- [ ] Schema.org loader parses correctly
- [ ] Training works on deep ontologies (no gradient issues)
- [ ] Depth-stratified metrics computed correctly
- [ ] Parent-child < sibling distance holds
- [ ] Results table and plot generated

---

## 8. Expected Insights

**If separation degrades with depth**:
- Narrative: Fine-grained distinctions are inherently harder
- Mitigation: Use hierarchical loss (penalize mistakes more at coarse levels)

**If separation holds across depths**:
- Strong result: Model captures hierarchical structure
- Claim: Ontological embeddings scale to real-world hierarchies

---

## 9. Handoff Prompt

```
Execute HANDOFF_05: Depth Ablation Study

Current state: Only tested on flat ontologies (depth ≤ 2).
Task: Evaluate separation ratio across hierarchical depths.

Steps:
1. Create src/hierarchical_ontology.py with:
   - generate_hierarchical_ontology(depth, branching_factor)
   - load_schema_org()
2. Create scripts/depth_ablation.py with:
   - evaluate_by_depth()
   - evaluate_parent_child_separation()
3. Generate synthetic ontologies at depths 2,3,4,5,6
4. Train embedding model on each
5. Compute depth-stratified separation ratios
6. Optionally: Run on Schema.org (~800 types)
7. Generate ablation plot: depth vs. separation ratio
8. Update paper with findings

Success: Clear trend line showing relationship between depth and separation
```
