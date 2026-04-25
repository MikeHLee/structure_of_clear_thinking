# Handoff 01: Hard Negative Sampling for Ontological Embeddings

**Status**: NOT STARTED  
**Priority**: HIGH  
**Estimated Effort**: 1-2 days  
**Dependencies**: None (can start immediately)

---

## 1. Problem Statement

The current 2.71× separation ratio is computed using **easy negatives**—cross-ontology type pairs that are trivially distinguishable (e.g., "Customer" from business vs. "Doctor" from healthcare). This inflates the metric and doesn't demonstrate the model's ability to discriminate subtle ontological violations.

**Current Implementation** (`scripts/modal_olog_training.py:180-214`):
```python
# Current: Cross-ontology pairs as negatives
if type_to_ontology.get(t1) == type_to_ontology.get(t2):
    positive_type_pairs.append(...)  # Same ontology
else:
    negative_type_pairs.append(...)  # Different ontology (EASY)
```

**Goal**: Implement hard negative sampling where negatives are ontologically "close" but invalid—types that are 1-hop away but in the wrong direction, or semantically similar but violating composition rules.

---

## 2. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Negative type | Cross-ontology (easy) | Same-ontology invalid paths (hard) |
| Separation ratio baseline | 2.71× (easy) | Report both easy and hard |
| Hard negative separation | N/A | > 1.5× |
| Invalid transition detection | 100% (STRICT) | Maintain 100% |

---

## 3. Technical Specification

### 3.1 Hard Negative Categories

Define three levels of negative difficulty:

| Level | Name | Definition | Example |
|-------|------|------------|---------|
| L0 | Easy | Cross-ontology pairs | Customer (business) vs. Doctor (healthcare) |
| L1 | Medium | Same ontology, >2 hops apart | Customer → Shipment (valid path exists but long) |
| L2 | Hard | Same ontology, 1-hop but wrong direction | Order → Customer (reverse of valid "places") |
| L3 | Adversarial | Semantically similar, different ontology | Customer (business) vs. User (ecommerce) |

### 3.2 Implementation Plan

#### Step 1: Compute Directed Reachability with Distance
```python
def compute_reachability_with_distance(ontology):
    """
    Returns dict: (src, tgt) -> shortest_path_length
    -1 if unreachable, 0 if same node
    """
    types = ontology["types"]
    aspects = ontology["aspects"]
    
    # Build adjacency
    adj = {t: [] for t in types}
    for src, tgt, rel in aspects:
        adj[src].append(tgt)
    
    # BFS from each node
    distances = {}
    for start in types:
        visited = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        for t in types:
            distances[(start, t)] = visited.get(t, -1)
    
    return distances
```

#### Step 2: Hard Negative Sampler
```python
def sample_hard_negatives(ontology, distances, n_samples, level="L2"):
    """
    Sample hard negative pairs based on difficulty level.
    """
    types = ontology["types"]
    hard_negatives = []
    
    if level == "L2":  # Wrong direction (hardest within ontology)
        for src, tgt, rel in ontology["aspects"]:
            # Reverse edge is hard negative
            if distances.get((tgt, src), -1) == -1:
                hard_negatives.append((tgt, src))
    
    elif level == "L1":  # Distant but reachable
        for t1 in types:
            for t2 in types:
                d = distances.get((t1, t2), -1)
                if d > 2:  # More than 2 hops
                    hard_negatives.append((t1, t2))
    
    return random.sample(hard_negatives, min(n_samples, len(hard_negatives)))
```

#### Step 3: Modified Contrastive Loss
```python
def hard_contrastive_loss(model, anchor, positive, hard_negative, margin=0.5):
    """
    Triplet loss with hard negatives.
    """
    e_anchor = model.get_type_embedding(anchor)
    e_pos = model.get_type_embedding(positive)
    e_neg = model.get_type_embedding(hard_negative)
    
    pos_dist = F.pairwise_distance(e_anchor, e_pos)
    neg_dist = F.pairwise_distance(e_anchor, e_neg)
    
    return F.relu(pos_dist - neg_dist + margin)
```

#### Step 4: Evaluation with Difficulty Stratification
```python
def evaluate_separation_by_difficulty(model, ontologies):
    """
    Report separation ratio at each difficulty level.
    """
    results = {}
    for level in ["L0", "L1", "L2", "L3"]:
        pos_dists = []
        neg_dists = []
        
        for ont in ontologies:
            distances = compute_reachability_with_distance(ont)
            positives = get_valid_pairs(ont)
            negatives = sample_hard_negatives(ont, distances, level=level)
            
            for p1, p2 in positives:
                pos_dists.append(compute_distance(model, p1, p2))
            for n1, n2 in negatives:
                neg_dists.append(compute_distance(model, n1, n2))
        
        results[level] = {
            "pos_mean": np.mean(pos_dists),
            "neg_mean": np.mean(neg_dists),
            "separation_ratio": np.mean(neg_dists) / (np.mean(pos_dists) + 1e-8)
        }
    
    return results
```

---

## 4. Files to Modify

| File | Changes |
|------|---------|
| `scripts/modal_olog_training.py` | Add hard negative sampling functions, modify `create_training_pairs()` |
| `results/embedding_results.json` | Add stratified separation ratios |
| `docs/PAPER_DRAFT_V1.md` | Update Section 7.3 with hard negative results |

---

## 5. Execution Commands

```bash
# Run with hard negatives (after implementation)
cd /Users/Michaellee/Documents/Runes/ai_research/topics/ontological_induction_sequence_modeling

# Local test
python -c "from scripts.modal_olog_training import *; print('imports ok')"

# Modal cloud run
modal run scripts/modal_olog_training.py --experiment embeddings
```

---

## 6. Validation Checklist

- [ ] `compute_reachability_with_distance()` returns correct distances
- [ ] L0 negatives match current implementation (baseline)
- [ ] L2 negatives are all invalid transitions
- [ ] L3 negatives span ontology boundaries but are semantically similar
- [ ] Separation ratio reported for all levels
- [ ] Paper draft updated with new results table

---

## 7. Expected Outcomes

| Level | Expected Separation | Interpretation |
|-------|---------------------|----------------|
| L0 (Easy) | ~2.7× | Baseline, easy to achieve |
| L1 (Medium) | ~2.0× | Moderate difficulty |
| L2 (Hard) | ~1.5× | Good if >1.5×, shows real discrimination |
| L3 (Adversarial) | ~1.2× | Hard to achieve, publishable if >1.3× |

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Hard negatives collapse separation ratio | Increase margin in triplet loss, add curriculum learning |
| Not enough hard negatives in small ontologies | Expand ontologies or use data augmentation |
| Training instability with hard negatives | Mix easy and hard negatives (70/30 ratio) |

---

## 9. References

- **FaceNet**: Schroff et al. (2015) - Hard negative mining for face recognition
- **Curriculum Learning**: Bengio et al. (2009) - Start easy, increase difficulty
- **RotatE**: Sun et al. (2019) - Self-adversarial negative sampling

---

## 10. Handoff Prompt

When ready to execute, use this prompt:

```
Execute HANDOFF_01: Hard Negative Sampling

Current state: The embedding experiment uses easy (cross-ontology) negatives.
Task: Implement hard negative sampling with difficulty levels L0-L3.

Key files:
- scripts/modal_olog_training.py (main implementation)
- results/embedding_results.json (output)

Steps:
1. Add compute_reachability_with_distance() function
2. Add sample_hard_negatives() with level parameter
3. Modify training loop to use mixed negatives
4. Add evaluate_separation_by_difficulty() for stratified reporting
5. Run experiment and report results at all levels
6. Update paper draft Section 7.3

Success: Hard (L2) separation ratio >1.5×
```
