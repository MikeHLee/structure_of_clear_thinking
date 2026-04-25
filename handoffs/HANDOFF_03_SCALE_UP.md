# Handoff 03: Scale Up to Text2KGBench

**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Estimated Effort**: 2-3 days  
**Dependencies**: Handoff 01 (hard negatives), Handoff 02 (baselines)

---

## 1. Problem Statement

The current experiments use **4 toy ontologies with 24 total types**. This is insufficient for publication-quality claims. Real knowledge graph benchmarks operate at much larger scales:

| Dataset | Entities | Relations | Triples |
|---------|----------|-----------|---------|
| **Our current** | 24 | 24 | ~24 |
| FB15k-237 | 14,541 | 237 | 310,116 |
| WN18RR | 40,943 | 11 | 93,003 |
| Text2KGBench | ~1,000+ | 430 | 7,943 |

We already have **Text2KGBench** data (`training_data/olog_training.jsonl` with 7,943 examples) that should be used for proper evaluation.

---

## 2. Success Criteria

| Metric | Current (Toy) | Target (Text2KGBench) |
|--------|---------------|----------------------|
| Types/Entities | 24 | 331+ |
| Relations | 24 | 430 |
| Training examples | ~2,500 | 7,943 |
| Separation ratio | 2.71× | >1.5× (harder at scale) |
| Training time | ~5 min | <2 hours (Modal GPU) |

---

## 3. Technical Specification

### 3.1 Data Source Analysis

The existing training data is in `training_data/olog_training.jsonl`:

```json
{
  "text": "The customer places an order...",
  "amr": "...",
  "olog": {
    "types": ["Customer", "Order", ...],
    "aspects": [["Customer", "Order", "places"], ...]
  },
  "source": "text2kgbench"
}
```

### 3.2 Implementation Plan

#### Step 1: Data Loader for Text2KGBench
```python
import json
from collections import defaultdict
from pathlib import Path

def load_text2kgbench(jsonl_path: str):
    """
    Load and aggregate ontologies from Text2KGBench training data.
    
    Returns:
        all_types: Set of all unique types
        all_relations: Set of all unique relations
        all_triples: List of (source, target, relation) tuples
        ontology_membership: Dict mapping type -> ontology_id
    """
    all_types = set()
    all_relations = set()
    all_triples = []
    type_to_ontology = {}
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f):
            example = json.loads(line)
            olog = example.get("olog", {})
            ont_id = example.get("source", f"ont_{line_num}")
            
            for t in olog.get("types", []):
                all_types.add(t)
                if t not in type_to_ontology:
                    type_to_ontology[t] = ont_id
            
            for aspect in olog.get("aspects", []):
                if len(aspect) == 3:
                    src, tgt, rel = aspect
                    all_relations.add(rel)
                    all_triples.append((src, tgt, rel))
    
    return {
        "types": list(all_types),
        "relations": list(all_relations),
        "triples": all_triples,
        "type_to_ontology": type_to_ontology,
        "n_types": len(all_types),
        "n_relations": len(all_relations),
        "n_triples": len(all_triples),
    }


def create_type_index(data: dict):
    """Create efficient index structures for training."""
    type_to_idx = {t: i for i, t in enumerate(data["types"])}
    rel_to_idx = {r: i for i, r in enumerate(data["relations"])}
    
    # Build adjacency for reachability
    adjacency = defaultdict(set)
    for src, tgt, rel in data["triples"]:
        adjacency[src].add(tgt)
    
    return type_to_idx, rel_to_idx, adjacency
```

#### Step 2: Scaled Embedding Model
```python
class ScaledOlogEmbedding(nn.Module):
    """
    Embedding model scaled for larger ontologies.
    
    Changes from toy version:
    - Batch processing for efficiency
    - Sparse reachability computation
    - Memory-efficient negative sampling
    """
    def __init__(self, n_types: int, n_relations: int, embed_dim: int = 128):
        super().__init__()
        self.n_types = n_types
        self.n_relations = n_relations
        self.embed_dim = embed_dim
        
        # Embeddings
        self.type_embeddings = nn.Embedding(n_types, embed_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embed_dim)
        
        # Composition layer
        self.compose = nn.Sequential(
            nn.Linear(3 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        
        # Initialize
        nn.init.xavier_uniform_(self.type_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, src_idx, rel_idx, tgt_idx):
        """
        Batch forward pass.
        
        Args:
            src_idx: (batch_size,) tensor of source type indices
            rel_idx: (batch_size,) tensor of relation indices
            tgt_idx: (batch_size,) tensor of target type indices
        
        Returns:
            (batch_size, embed_dim) composed embeddings
        """
        src = self.type_embeddings(src_idx)
        rel = self.relation_embeddings(rel_idx)
        tgt = self.type_embeddings(tgt_idx)
        
        combined = torch.cat([src, rel, tgt], dim=-1)
        return self.compose(combined)
    
    def score_triple(self, src_idx, rel_idx, tgt_idx):
        """
        Score a triple (lower = more valid for TransE-style).
        """
        src = self.type_embeddings(src_idx)
        rel = self.relation_embeddings(rel_idx)
        tgt = self.type_embeddings(tgt_idx)
        
        # TransE-style: src + rel ≈ tgt
        return torch.norm(src + rel - tgt, dim=-1)
```

#### Step 3: Efficient Batch Training
```python
class Text2KGDataset(Dataset):
    def __init__(self, triples, type_to_idx, rel_to_idx, negative_ratio=5):
        self.triples = triples
        self.type_to_idx = type_to_idx
        self.rel_to_idx = rel_to_idx
        self.negative_ratio = negative_ratio
        self.n_types = len(type_to_idx)
    
    def __len__(self):
        return len(self.triples) * (1 + self.negative_ratio)
    
    def __getitem__(self, idx):
        triple_idx = idx % len(self.triples)
        is_negative = idx >= len(self.triples)
        
        src, tgt, rel = self.triples[triple_idx]
        src_idx = self.type_to_idx.get(src, 0)
        tgt_idx = self.type_to_idx.get(tgt, 0)
        rel_idx = self.rel_to_idx.get(rel, 0)
        
        if is_negative:
            # Corrupt head or tail
            if random.random() < 0.5:
                src_idx = random.randint(0, self.n_types - 1)
            else:
                tgt_idx = random.randint(0, self.n_types - 1)
            label = 0
        else:
            label = 1
        
        return {
            "src": src_idx,
            "rel": rel_idx,
            "tgt": tgt_idx,
            "label": label
        }


def train_scaled(model, dataloader, epochs=100, lr=0.001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            src = batch["src"].to(device)
            rel = batch["rel"].to(device)
            tgt = batch["tgt"].to(device)
            label = batch["label"].float().to(device)
            
            scores = model.score_triple(src, rel, tgt)
            
            # Margin ranking loss
            pos_mask = label == 1
            neg_mask = label == 0
            
            if pos_mask.any() and neg_mask.any():
                pos_scores = scores[pos_mask].mean()
                neg_scores = scores[neg_mask].mean()
                loss = F.relu(pos_scores - neg_scores + 0.5)
            else:
                loss = scores.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
```

#### Step 4: Scaled Evaluation
```python
def evaluate_scaled(model, data, type_to_idx, device):
    """
    Evaluate on full Text2KGBench scale.
    """
    model.eval()
    type_embs = model.type_embeddings.weight.detach().cpu().numpy()
    
    # Sample subset for distance computation (full matrix too large)
    n_sample = min(500, len(data["types"]))
    sampled_types = random.sample(data["types"], n_sample)
    
    intra_dists = []
    inter_dists = []
    
    for i, t1 in enumerate(sampled_types):
        for j, t2 in enumerate(sampled_types):
            if i < j:
                idx1 = type_to_idx.get(t1)
                idx2 = type_to_idx.get(t2)
                if idx1 is None or idx2 is None:
                    continue
                
                dist = np.linalg.norm(type_embs[idx1] - type_embs[idx2])
                
                ont1 = data["type_to_ontology"].get(t1)
                ont2 = data["type_to_ontology"].get(t2)
                
                if ont1 == ont2:
                    intra_dists.append(dist)
                else:
                    inter_dists.append(dist)
    
    return {
        "n_types": data["n_types"],
        "n_relations": data["n_relations"],
        "n_triples": data["n_triples"],
        "sampled_pairs": len(intra_dists) + len(inter_dists),
        "intra_dist_mean": np.mean(intra_dists) if intra_dists else 0,
        "inter_dist_mean": np.mean(inter_dists) if inter_dists else 0,
        "separation_ratio": (
            np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8)
            if intra_dists else 0
        ),
    }
```

---

## 4. Files to Create/Modify

| File | Purpose |
|------|---------|
| `scripts/scaled_training.py` | New: Text2KGBench training script |
| `scripts/modal_scaled_training.py` | New: Modal cloud version |
| `src/data_loader.py` | New: Data loading utilities |
| `results/scaled_results.json` | New: Scaled experiment results |
| `docs/PAPER_DRAFT_V1.md` | Update: Section 7 with scaled results |

---

## 5. Execution Commands

```bash
cd /Users/Michaellee/Documents/Runes/ai_research/topics/ontological_induction_sequence_modeling

# Verify data exists
wc -l training_data/olog_training.jsonl
# Expected: 7943

# Local test (small subset)
python scripts/scaled_training.py --max-examples 1000 --epochs 10

# Full Modal run
modal run scripts/modal_scaled_training.py --epochs 100
```

---

## 6. Validation Checklist

- [ ] Data loader correctly parses all 7,943 examples
- [ ] No OOM errors with full dataset on T4 GPU
- [ ] Training converges within 100 epochs
- [ ] Separation ratio computed on sampled subset
- [ ] Results include scale metrics (n_types, n_relations, n_triples)
- [ ] Comparison table: Toy scale vs. Text2KGBench scale

---

## 7. Expected Outcomes

| Metric | Toy (Current) | Text2KGBench (Expected) |
|--------|---------------|------------------------|
| Types | 24 | ~331 |
| Relations | 24 | ~430 |
| Triples | ~24 | ~7,943 |
| Separation ratio | 2.71× | 1.5-2.5× |
| Training time | 5 min | 30-60 min |

**Note**: Separation ratio may decrease at scale due to:
- More semantic overlap between types
- More diverse ontology structures
- Harder negative sampling (more candidates)

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OOM on large type count | Use gradient checkpointing, reduce batch size |
| Slow reachability computation | Precompute and cache, use sparse matrices |
| Ontology membership unclear | Use source field or clustering |
| Separation ratio collapses | Increase embedding dim (128 or 256) |

---

## 9. Handoff Prompt

```
Execute HANDOFF_03: Scale Up to Text2KGBench

Current state: Experiments use 24 types from 4 toy ontologies.
Task: Scale to Text2KGBench (331 types, 430 relations, 7,943 examples).

Data: training_data/olog_training.jsonl

Steps:
1. Create data loader for olog_training.jsonl
2. Build ScaledOlogEmbedding model (embed_dim=128)
3. Implement efficient batch training with DataLoader
4. Train on full dataset (100 epochs, Modal T4 GPU)
5. Evaluate separation ratio on sampled subset
6. Compare to toy-scale results
7. Update paper with scaled results

Success: Training completes without OOM, separation ratio >1.5×
```
