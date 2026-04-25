# Ontological Embedding Research: Master Plan

**Project**: Ontological Induction Sequence Modeling  
**Location**: `/Users/Michaellee/Documents/Runes/ai_research/topics/ontological_induction_sequence_modeling/`  
**Last Updated**: 2026-02-26  
**Paper Target**: ICML 2026 / NeurIPS 2026

---

## Executive Summary

The 2.71× separation ratio between valid and invalid type transitions is a promising initial result, but requires substantial experimental reinforcement for publication. This plan outlines 5 research tasks to strengthen the empirical foundation.

---

## Current State

| Metric | Value | Status |
|--------|-------|--------|
| Separation ratio | 2.71× | ⚠️ Easy negatives only |
| Baseline comparison | None | ❌ Missing |
| Scale | 24 types | ❌ Toy scale |
| Visualization | None | ❌ Missing |
| Depth ablation | None | ❌ Flat ontologies only |
| Invalid detection (STRICT) | 100% | ✅ Strong |
| Invalid attn weight | 0.0 | ✅ Perfect |

---

## Research Tasks

### Task Dependency Graph

```
HANDOFF_01 (Hard Negatives) ──┐
                              ├──> HANDOFF_04 (Visualization)
HANDOFF_02 (Baselines) ───────┤
                              │
HANDOFF_03 (Scale Up) ────────┴──> HANDOFF_05 (Depth Ablation)
```

**Parallel tracks**:
- Track A: HANDOFF_01 + HANDOFF_02 (can run simultaneously)
- Track B: HANDOFF_03 (requires data infrastructure)
- Track C: HANDOFF_04 + HANDOFF_05 (requires trained models from A or B)

---

## Task Status Tracker

| ID | Task | Priority | Status | Est. Days | Actual Days |
|----|------|----------|--------|-----------|-------------|
| 01 | Hard Negative Sampling | HIGH | NOT STARTED | 1-2 | - |
| 02 | Baseline Benchmarks | HIGH | NOT STARTED | 2-3 | - |
| 03 | Scale to Text2KGBench | MEDIUM | NOT STARTED | 2-3 | - |
| 04 | Visualization | MEDIUM | NOT STARTED | 1 | - |
| 05 | Depth Ablation | MEDIUM | NOT STARTED | 2-3 | - |
| 08 | Hierarchical + Semantic Tokenization | **CRITICAL** | DESIGN | 6 weeks | - |

**Total Estimated**: 8-12 days (tasks 01–05) + 6 weeks (task 08, gates NeurIPS/EMNLP scale-up)

See [HANDOFF_08_HIERARCHICAL_SEMANTIC_TOKENIZATION.md](HANDOFF_08_HIERARCHICAL_SEMANTIC_TOKENIZATION.md) — combines hierarchical slot-tokens (type/content/modality/provenance via GHRR bind) with a neural merge scorer, and specifies how to reuse pretrained encoders/tokenizers/decoders rather than train from scratch. This is now the critical path for learned (not hand-specified) ontologies.

---

## Execution Sequence

### Phase 1: Foundation (Days 1-3)
Run HANDOFF_01 and HANDOFF_02 in parallel.

```
Day 1-2: HANDOFF_01 (Hard Negatives)
Day 1-3: HANDOFF_02 (Baselines)
```

**Deliverables**:
- [ ] Stratified separation ratios (L0, L1, L2, L3)
- [ ] Baseline comparison table (TransE, RotatE, DistMult, ComplEx)

### Phase 2: Scale (Days 4-6)
Run HANDOFF_03 after data infrastructure verified.

```
Day 4-6: HANDOFF_03 (Scale Up)
```

**Deliverables**:
- [ ] Training on 7,943 examples
- [ ] Separation ratio at 331+ types

### Phase 3: Analysis (Days 7-10)
Run HANDOFF_04 and HANDOFF_05 with trained models.

```
Day 7: HANDOFF_04 (Visualization)
Day 8-10: HANDOFF_05 (Depth Ablation)
```

**Deliverables**:
- [ ] t-SNE/UMAP plots
- [ ] Depth vs. separation ratio curve

---

## Prompt Templates

### Start Next Task

```
I'm ready to execute the next task in the Ontological Embedding research plan.

Current state: [copy from status tracker above]

Please execute: HANDOFF_[XX]

Reference: handoffs/HANDOFF_[XX]_[NAME].md
```

### Check Progress

```
Please check the status of the Ontological Embedding research plan.

Read: handoffs/EMBEDDING_RESEARCH_MASTER_PLAN.md

Report:
1. Which tasks are complete?
2. What are the key results so far?
3. What should I execute next?
```

### Update Results

```
Task HANDOFF_[XX] is complete.

Results:
- [Metric 1]: [Value]
- [Metric 2]: [Value]

Please update:
1. handoffs/EMBEDDING_RESEARCH_MASTER_PLAN.md (status tracker)
2. docs/PAPER_DRAFT_V1.md (Section 7)
```

---

## Results Log

### HANDOFF_01: Hard Negative Sampling
**Status**: NOT STARTED

| Level | Separation Ratio | Notes |
|-------|------------------|-------|
| L0 (Easy) | - | Baseline (cross-ontology) |
| L1 (Medium) | - | Same ontology, >2 hops |
| L2 (Hard) | - | Same ontology, wrong direction |
| L3 (Adversarial) | - | Semantically similar, cross-ontology |

---

### HANDOFF_02: Baseline Benchmarks
**Status**: NOT STARTED

| Model | Separation | MRR | Hits@1 | Hits@10 | Invalid Det. |
|-------|------------|-----|--------|---------|--------------|
| OlogEmbed (Ours) | 2.71× | - | - | - | 100% |
| TransE | - | - | - | - | - |
| RotatE | - | - | - | - | - |
| DistMult | - | - | - | - | - |
| ComplEx | - | - | - | - | - |

---

### HANDOFF_03: Scale Up
**Status**: NOT STARTED

| Metric | Toy (Current) | Text2KGBench |
|--------|---------------|--------------|
| Types | 24 | - |
| Relations | 24 | - |
| Triples | ~24 | - |
| Separation | 2.71× | - |
| Training time | ~5 min | - |

---

### HANDOFF_04: Visualization
**Status**: NOT STARTED

| Plot | File | Status |
|------|------|--------|
| t-SNE | results/visualizations/tsne_embeddings.png | - |
| UMAP | results/visualizations/umap_embeddings.png | - |
| Heatmap | results/visualizations/transition_heatmap.png | - |
| Graphs | results/visualizations/ontology_graphs.png | - |

---

### HANDOFF_05: Depth Ablation
**Status**: NOT STARTED

| Depth | Types | Separation | Parent-Child | Sibling |
|-------|-------|------------|--------------|---------|
| 2 | ~13 | - | - | - |
| 3 | ~40 | - | - | - |
| 4 | ~121 | - | - | - |
| 5 | ~364 | - | - | - |
| Schema.org | ~800 | - | - | - |

---

## Paper Update Checklist

After completing all tasks, update these paper sections:

- [ ] **Section 7.3** (Embedding Quality): Add hard negative results, baseline comparison
- [ ] **Section 7.4** (Attention Ablation): No changes needed
- [ ] **Section 7.5** (Pipeline Validation): No changes needed
- [ ] **Section 7.6** (NEW): Baseline Comparison table
- [ ] **Section 7.7** (NEW): Scale-up results
- [ ] **Section 7.8** (NEW): Depth ablation analysis
- [ ] **Figures**: Add t-SNE/UMAP visualization, depth ablation plot

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `handoffs/HANDOFF_01_HARD_NEGATIVE_SAMPLING.md` | Hard negative implementation |
| `handoffs/HANDOFF_02_BASELINE_BENCHMARKS.md` | PyKEEN baseline comparison |
| `handoffs/HANDOFF_03_SCALE_UP.md` | Text2KGBench scaling |
| `handoffs/HANDOFF_04_VISUALIZATION.md` | t-SNE/UMAP/heatmap |
| `handoffs/HANDOFF_05_DEPTH_ABLATION.md` | Hierarchical ontology tests |
| `scripts/modal_olog_training.py` | Current training script |
| `results/embedding_results.json` | Embedding experiment results |
| `docs/PAPER_DRAFT_V1.md` | Paper draft to update |

---

## Success Criteria (Publication Ready)

| Criterion | Threshold | Current |
|-----------|-----------|---------|
| Hard negative separation (L2) | >1.5× | - |
| Competitive with RotatE | Within 20% MRR | - |
| Scale test passes | >300 types | 24 |
| Visualization confirms clustering | Visual | - |
| Depth ablation shows trend | Clear curve | - |
| All integration tests | 18/18 | 18/18 ✅ |

---

## Notes

- Use **Modal GPU cloud** for all training runs
- Prefer T4 for cost efficiency; use A10G for large-scale experiments
- Save all results to `results/` with timestamps
- Commit checkpoints to git after each handoff completion

---

## Quick Start

To begin the research plan:

```
Execute HANDOFF_01: Hard Negative Sampling

Current state: Separation ratio 2.71× uses easy (cross-ontology) negatives.
Task: Implement hard negative sampling with difficulty levels L0-L3.

Reference: handoffs/HANDOFF_01_HARD_NEGATIVE_SAMPLING.md
```
