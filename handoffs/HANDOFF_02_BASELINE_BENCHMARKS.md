# Handoff 02: Baseline Benchmarks Against TransE/RotatE

**Status**: NOT STARTED  
**Priority**: HIGH  
**Estimated Effort**: 2-3 days  
**Dependencies**: None (can run in parallel with Handoff 01)

---

## 1. Problem Statement

The current 2.71× separation ratio has no external baseline comparison. Without comparing to established knowledge graph embedding methods (TransE, RotatE, DistMult, ComplEx), reviewers cannot assess whether this result is competitive.

**Missing comparisons**:
- TransE (Bordes et al., 2013) - Translation-based
- RotatE (Sun et al., 2019) - Rotation-based with self-adversarial sampling
- DistMult (Yang et al., 2015) - Bilinear diagonal
- ComplEx (Trouillon et al., 2016) - Complex-valued embeddings

---

## 2. Success Criteria

| Metric | Our Model | TransE | RotatE | DistMult |
|--------|-----------|--------|--------|----------|
| Separation ratio | 2.71× | ? | ? | ? |
| MRR (link prediction) | ? | ? | ? | ? |
| Hits@1 | ? | ? | ? | ? |
| Hits@10 | ? | ? | ? | ? |
| Invalid transition detection | 100% | ? | ? | ? |

Goal: Show our method is competitive OR argue why separation ratio is a better metric for hallucination prevention than MRR.

---

## 3. Technical Specification

### 3.1 Unified Evaluation Framework

Create a benchmark harness that:
1. Trains each model on the same ontology data
2. Evaluates using identical test triples
3. Reports both standard KG metrics AND separation ratio

### 3.2 Implementation Plan

#### Step 1: Install KG Embedding Libraries
```bash
pip install pykeen  # PyKEEN: Knowledge graph embedding library
# or
pip install torch-geometric  # For custom implementations
```

#### Step 2: Data Converter
```python
# Convert our Olog format to PyKEEN TriplesFactory

from pykeen.triples import TriplesFactory
import numpy as np

def olog_to_triples(ontologies: dict) -> np.ndarray:
    """
    Convert our ontology format to (head, relation, tail) triples.
    """
    triples = []
    for ont_name, ont in ontologies.items():
        for src, tgt, rel in ont["aspects"]:
            # Prefix with ontology name for uniqueness
            triples.append([
                f"{ont_name}:{src}",
                rel,
                f"{ont_name}:{tgt}"
            ])
    return np.array(triples)

def create_pykeen_factory(ontologies: dict):
    """Create PyKEEN TriplesFactory from our ontologies."""
    triples = olog_to_triples(ontologies)
    return TriplesFactory.from_labeled_triples(triples)
```

#### Step 3: Baseline Training
```python
from pykeen.pipeline import pipeline

def train_baseline(model_name: str, triples_factory: TriplesFactory):
    """
    Train a baseline KG embedding model.
    
    Args:
        model_name: "TransE", "RotatE", "DistMult", "ComplEx"
    """
    result = pipeline(
        model=model_name,
        training=triples_factory,
        testing=triples_factory,  # Use same for small datasets
        model_kwargs={
            "embedding_dim": 64,  # Match our embedding dim
        },
        training_kwargs={
            "num_epochs": 100,
        },
        evaluation_kwargs={
            "batch_size": 32,
        },
    )
    return result

# Run all baselines
baselines = {}
for model in ["TransE", "RotatE", "DistMult", "ComplEx"]:
    baselines[model] = train_baseline(model, factory)
```

#### Step 4: Unified Separation Ratio Computation
```python
def compute_separation_ratio_pykeen(model, triples_factory, ontologies):
    """
    Compute separation ratio using PyKEEN model embeddings.
    """
    # Get entity embeddings
    entity_embeddings = model.entity_representations[0]()
    entity_to_idx = triples_factory.entity_to_id
    
    # Compute distances
    intra_dists = []
    inter_dists = []
    
    for ont_name, ont in ontologies.items():
        types = ont["types"]
        for i, t1 in enumerate(types):
            for j, t2 in enumerate(types):
                if i < j:
                    e1 = f"{ont_name}:{t1}"
                    e2 = f"{ont_name}:{t2}"
                    if e1 in entity_to_idx and e2 in entity_to_idx:
                        idx1 = entity_to_idx[e1]
                        idx2 = entity_to_idx[e2]
                        dist = torch.norm(
                            entity_embeddings[idx1] - entity_embeddings[idx2]
                        ).item()
                        intra_dists.append(dist)
    
    # Cross-ontology distances
    ont_names = list(ontologies.keys())
    for i, ont1 in enumerate(ont_names):
        for j, ont2 in enumerate(ont_names):
            if i < j:
                for t1 in ontologies[ont1]["types"]:
                    for t2 in ontologies[ont2]["types"]:
                        e1 = f"{ont1}:{t1}"
                        e2 = f"{ont2}:{t2}"
                        if e1 in entity_to_idx and e2 in entity_to_idx:
                            idx1 = entity_to_idx[e1]
                            idx2 = entity_to_idx[e2]
                            dist = torch.norm(
                                entity_embeddings[idx1] - entity_embeddings[idx2]
                            ).item()
                            inter_dists.append(dist)
    
    return {
        "intra_mean": np.mean(intra_dists),
        "inter_mean": np.mean(inter_dists),
        "separation_ratio": np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8)
    }
```

#### Step 5: Link Prediction Evaluation
```python
from pykeen.evaluation import RankBasedEvaluator

def evaluate_link_prediction(result):
    """Extract standard KG metrics."""
    metrics = result.metric_results.to_dict()
    return {
        "MRR": metrics.get("both.realistic.inverse_harmonic_mean_rank"),
        "Hits@1": metrics.get("both.realistic.hits_at_1"),
        "Hits@3": metrics.get("both.realistic.hits_at_3"),
        "Hits@10": metrics.get("both.realistic.hits_at_10"),
    }
```

#### Step 6: Invalid Transition Detection Test
```python
def evaluate_invalid_detection(model, triples_factory, ontologies):
    """
    Test if model correctly scores invalid transitions lower than valid ones.
    """
    valid_scores = []
    invalid_scores = []
    
    for ont_name, ont in ontologies.items():
        # Valid triples
        for src, tgt, rel in ont["aspects"]:
            score = model.score_hrt(
                torch.tensor([[
                    triples_factory.entity_to_id[f"{ont_name}:{src}"],
                    triples_factory.relation_to_id[rel],
                    triples_factory.entity_to_id[f"{ont_name}:{tgt}"]
                ]])
            ).item()
            valid_scores.append(score)
        
        # Invalid triples (reversed)
        for src, tgt, rel in ont["aspects"]:
            try:
                score = model.score_hrt(
                    torch.tensor([[
                        triples_factory.entity_to_id[f"{ont_name}:{tgt}"],
                        triples_factory.relation_to_id[rel],
                        triples_factory.entity_to_id[f"{ont_name}:{src}"]
                    ]])
                ).item()
                invalid_scores.append(score)
            except KeyError:
                pass
    
    # Detection rate: invalid scores should be lower (for distance-based) 
    # or higher (for energy-based)
    # For TransE (distance-based): lower = more valid
    detection_rate = sum(
        1 for v, i in zip(valid_scores, invalid_scores) if v < i
    ) / len(valid_scores)
    
    return {
        "valid_score_mean": np.mean(valid_scores),
        "invalid_score_mean": np.mean(invalid_scores),
        "detection_rate": detection_rate
    }
```

---

## 4. Files to Create/Modify

| File | Purpose |
|------|---------|
| `scripts/baseline_benchmarks.py` | New: Main benchmark script |
| `scripts/modal_baseline_benchmarks.py` | New: Modal cloud version |
| `results/baseline_comparison.json` | New: Comparison results |
| `docs/PAPER_DRAFT_V1.md` | Update: Add Section 7.6 Baseline Comparison |

---

## 5. Execution Commands

```bash
cd /Users/Michaellee/Documents/Runes/ai_research/topics/ontological_induction_sequence_modeling

# Install dependencies
pip install pykeen

# Local run
python scripts/baseline_benchmarks.py

# Modal cloud run (GPU)
modal run scripts/modal_baseline_benchmarks.py
```

---

## 6. Expected Results Table

```markdown
| Model | Separation Ratio | MRR | Hits@1 | Hits@10 | Invalid Detection |
|-------|------------------|-----|--------|---------|-------------------|
| **Ours (OlogEmbed)** | **2.71×** | ? | ? | ? | **100%** |
| TransE | ? | ? | ? | ? | ? |
| RotatE | ? | ? | ? | ? | ? |
| DistMult | ? | ? | ? | ? | ? |
| ComplEx | ? | ? | ? | ? | ? |
```

---

## 7. Interpretation Guide

**If our separation ratio is higher than baselines**:
- Claim: Ontological structure improves clustering for hallucination prevention
- Narrative: Baselines optimize for link prediction; we optimize for type separation

**If our separation ratio is lower**:
- Pivot: Emphasize invalid detection rate (100% is the key metric)
- Narrative: Separation ratio alone doesn't capture compositional validity

**If MRR is lower than baselines**:
- Expected: We optimize for a different objective
- Narrative: MRR measures ranking; we measure hard rejection of invalid claims

---

## 8. Validation Checklist

- [ ] PyKEEN installed and working
- [ ] Data converter produces valid triples
- [ ] All 4 baselines train successfully
- [ ] Separation ratio computed for all models
- [ ] Link prediction metrics extracted
- [ ] Invalid detection test implemented
- [ ] Results table generated
- [ ] Paper draft updated

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| PyKEEN incompatible with small datasets | Use custom implementations with same loss functions |
| Baselines outperform on all metrics | Reframe as "different objectives" - hallucination vs. link prediction |
| Training time too long | Use Modal GPU, reduce epochs to 50 |

---

## 10. Handoff Prompt

```
Execute HANDOFF_02: Baseline Benchmarks

Current state: No comparison to established KG embedding methods.
Task: Benchmark our OlogEmbed against TransE, RotatE, DistMult, ComplEx.

Key files to create:
- scripts/baseline_benchmarks.py

Steps:
1. Install PyKEEN: pip install pykeen
2. Create olog_to_triples() converter
3. Train all 4 baselines with embed_dim=64, epochs=100
4. Compute separation ratio for each baseline
5. Extract MRR, Hits@1/10 metrics
6. Test invalid transition detection for each
7. Generate comparison table
8. Update paper Section 7.6

Success: Complete comparison table with interpretation
```
