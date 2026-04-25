# HANDOFF 07: Full Architecture & Research Review

**Date**: March 21, 2026  
**Status**: Pre-Training Review  
**Next Phase**: Modal GPU Training at Scale

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Foundations](#2-research-foundations)
3. [System Architecture](#3-system-architecture)
4. [Implementation Status](#4-implementation-status)
5. [Evaluation Results](#5-evaluation-results)
6. [Novel Contributions](#6-novel-contributions)
7. [Modal GPU Training Plan](#7-modal-gpu-training-plan)
8. [Publication Strategy](#8-publication-strategy)
9. [Open Questions](#9-open-questions)

---

## 1. Executive Summary

### What We Built

A three-layer system for **auditable AI** that combines:

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Algebraic** | Hyperdimensional Computing (GHRR) | Direction-preserving embeddings |
| **Topological** | Sheaf Cohomology (H⁰/H¹) | Consistency detection |
| **Logical** | Proof Objects | Verifiable reasoning chains |

### Core Thesis

> **Hallucinations are detectable as topological obstructions (H¹ ≠ 0) in the sheaf of local knowledge claims.**

When an AI makes inconsistent claims across documents/contexts, these appear as non-trivial cycles in the first cohomology group.

### Key Result

Injecting 76 conflicting triples into a knowledge graph increased H¹ from 5 to 58—a **10.6× amplification** that provides a clear signal for inconsistency detection.

---

## 2. Research Foundations

### 2.1 Hyperdimensional Computing (HDC)

**Reference**: Kanerva (2009), "Hyperdimensional Computing: An Introduction"

HDC represents symbols as high-dimensional vectors (~4096 dims) with three operations:

| Operation | Symbol | Property |
|-----------|--------|----------|
| **Bundling** | ⊕ | Superposition (OR-like) |
| **Binding** | ⊛ | Association (AND-like) |
| **Permutation** | ρ | Sequence encoding |

**Our Extension**: We use **Generalized Holographic Reduced Representations (GHRR)** with non-commutative binding to preserve directional relationships:

```
bind(A, B) ≠ bind(B, A)
```

This is critical for ontological relations where `parent_of(X, Y) ≠ parent_of(Y, X)`.

### 2.2 Sheaf Theory

**References**:
- Robinson (2014), "Topological Signal Processing"
- Bodnar et al. (2022), "Neural Sheaf Diffusion"
- Hansen & Ghrist (2019), "Learning by Examples of Consistent Sections"

A **sheaf** assigns data to open sets with consistency requirements:

```
          F(U)
         /    \
    ρ_UV       ρ_UW
       ↓        ↓
      F(V)    F(W)
         \    /
          ↘  ↙
          F(V∩W)
```

**Restriction maps** ρ must satisfy the **gluing axiom**: if local sections agree on overlaps, they glue to a global section.

### 2.3 Cohomology Groups

| Group | Interpretation | Detection |
|-------|----------------|-----------|
| **H⁰** | Global sections (consistent beliefs) | dim(ker(∂₀)) |
| **H¹** | Obstructions (inconsistencies/cycles) | dim(ker(∂₁))/im(∂₀) |

**Key insight**: H¹ ≠ 0 implies there exist local claims that cannot be consistently globalized—exactly what happens during hallucination.

### 2.4 Sheaf Laplacian & Diffusion

The **Sheaf Laplacian** L generalizes the graph Laplacian to include restriction map structure:

```
L = D - A_sheaf
```

**Diffusion** on L propagates "heat" through the ontology:

```
x(t) = exp(-tL) · x(0)
```

This enables **topological queries**: start heat at query nodes, see where it spreads based on sheaf structure.

### 2.5 Proof Objects (Curry-Howard)

**Reference**: Wadler (2015), "Propositions as Types"

The **Curry-Howard correspondence** identifies:
- Types ↔ Propositions
- Programs ↔ Proofs
- Type checking ↔ Proof verification

We extend this to natural language:
- **Claims** are propositions
- **Proof objects** are morphism paths in the Olog
- **Validation** is path existence checking

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Documents → Text2KG Extraction → (subject, predicate, object)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ALGEBRAIC LAYER (HDC)                        │
├─────────────────────────────────────────────────────────────────┤
│  GHRREncoder                                                     │
│  ├── encode_type(entity) → hypervector ∈ ℝ^4096                 │
│  ├── encode_morphism(src, tgt, rel) → hypervector               │
│  ├── bind(A, B) → non-commutative composition                   │
│  └── similarity(A, B) → cosine distance                         │
│                                                                  │
│  Properties:                                                     │
│  • Non-commutativity: bind(A,B) ≠ bind(B,A) (asymmetry ~1.0)    │
│  • Quasi-orthogonality: random vectors ~orthogonal              │
│  • Superposition: bundle preserves constituents                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TOPOLOGICAL LAYER (SHEAF)                      │
├─────────────────────────────────────────────────────────────────┤
│  OntologySheaf                                                   │
│  ├── add_local_section(source_id, triples)                      │
│  ├── build_sheaf_laplacian() → L ∈ ℝ^(n×n)                      │
│  ├── compute_boundary_operators() → ∂₀, ∂₁                      │
│  └── compute_cohomology() → (H⁰, H¹, consistency_score)         │
│                                                                  │
│  Outputs:                                                        │
│  • dim(H⁰): Number of connected consistent components            │
│  • dim(H¹): Number of irreconcilable cycles (obstructions)      │
│  • Consistency score: 1 - (conflicts / total_edges)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LOGICAL LAYER (PROOFS)                      │
├─────────────────────────────────────────────────────────────────┤
│  ProofEngine                                                     │
│  ├── prove(claim) → ProofObject                                 │
│  ├── modes: STRICT | COMPOSITIONAL | REACHABILITY               │
│  └── audit_response(text) → List[ProofObject]                   │
│                                                                  │
│  TopologicalQueryEngine                                          │
│  ├── query_diffusion(types, time) → activated_types             │
│  ├── query_bfs(types) → paths (baseline)                        │
│  ├── query_hybrid(types) → validated_paths + proofs             │
│  └── benchmark(strategies) → latency comparison                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  • Validated claims with proof objects                          │
│  • Consistency reports (H⁰/H¹ dimensions)                       │
│  • Ontological gaps (contradictions, missing links)             │
│  • Auditable reasoning chains                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Example

```
Input: "The president of France is Emmanuel Macron"
        ↓
Text2KG: (France, president, Emmanuel_Macron)
        ↓
HDC:    hv_France = encode_type("France")
        hv_Macron = encode_type("Emmanuel_Macron")
        hv_edge = encode_morphism("France", "Emmanuel_Macron", "president")
        ↓
Sheaf:  add_local_section("doc_1", [(France, Emmanuel_Macron, president)])
        L = build_sheaf_laplacian()
        (H⁰=1, H¹=0) → consistent
        ↓
Query:  query({"France"}, diffusion_time=1.5)
        → activated: [France, Emmanuel_Macron, Europe, ...]
        ↓
Proof:  prove("France has president Emmanuel_Macron")
        → ProofObject(status=VALID, path=[president])
```

### 3.3 Component Dependencies

```
benchmark_datasets.py
        │
        ▼
hdc_sheaf_pipeline.py ──────────────────┐
        │                                │
        ├──► ghrr_encoder.py            │
        │         │                      │
        │         ▼                      │
        ├──► ontology_sheaf.py          │
        │         │                      │
        │         ▼                      │
        ├──► topological_query.py ◄─────┤
        │         │                      │
        │         ▼                      │
        └──► proof_objects.py ◄── olog_core.py
```

---

## 4. Implementation Status

### 4.1 Completed Modules

| Module | File | Lines | Status |
|--------|------|-------|--------|
| GHRR Encoder | `ghrr_encoder.py` | 333 | ✅ Complete |
| Ontology Sheaf | `ontology_sheaf.py` | 583 | ✅ Complete |
| Topological Query | `topological_query.py` | 650 | ✅ Complete |
| Proof Objects | `proof_objects.py` | 726 | ✅ Complete |
| Olog Core | `olog_core.py` | 206 | ✅ Complete |
| Benchmark Datasets | `benchmark_datasets.py` | 420 | ✅ Complete |
| Pipeline | `hdc_sheaf_pipeline.py` | 510 | ✅ Complete |

**Total**: ~3,400 lines of production code

### 4.2 Test Coverage

| Test | Command | Status |
|------|---------|--------|
| GHRR demo | `python ghrr_encoder.py` | ✅ Pass |
| Sheaf demo | `python ontology_sheaf.py` | ✅ Pass |
| Query demo | `python topological_query.py` | ✅ Pass |
| Pipeline eval | `python hdc_sheaf_pipeline.py` | ✅ Pass |

### 4.3 Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
networkx>=3.0
pydantic>=2.0
```

All dependencies are standard scientific Python—no exotic requirements.

---

## 5. Evaluation Results

### 5.1 Dataset Summary

| Dataset | Triples | Entities | Relations | Source |
|---------|---------|----------|-----------|--------|
| Text2KGBench | 7,943 | ~4,000 | 29 | ISWC 2023 |
| FB15K-237 | 272,115 | 14,541 | 237 | Freebase |
| WN18RR | 86,835 | 40,943 | 11 | WordNet |

### 5.2 Cohomology Results

| Dataset | Subset | H⁰ | H¹ | Consistency |
|---------|--------|----|----|-------------|
| Text2KG | 2,000 | 857 | 6 | 1.000 |
| FB15K-237 | 5,000 | 784 | 350 | 0.999 |
| WN18RR | 5,000 | 2,993 | 49 | 0.999 |

**Interpretation**:
- High H⁰ reflects sampling (many small components)
- FB15K-237's high H¹ reflects complex multi-hop structure
- Near-perfect consistency indicates clean benchmark data

### 5.3 Conflict Detection

```
Baseline (500 triples):     H¹ = 5
After 76 conflicts:         H¹ = 58
────────────────────────────────────
H¹ increase:                +53 (10.6×)
Consistency drop:           0.009
```

**This is the key result**: H¹ provides a **quantitative signal** for inconsistency magnitude.

### 5.4 Query Latency

| Strategy | Latency | Complexity |
|----------|---------|------------|
| naive_bfs | 0.5ms | O(V + E) |
| hdc_similarity | 6ms | O(n·d) |
| diffusion | 140ms | O(n³) |
| hybrid | 134ms | O(n³) |

**Bottleneck**: Matrix exponential exp(-tL) dominates diffusion time.

### 5.5 HDC Properties Verified

| Property | Measured | Expected |
|----------|----------|----------|
| Non-commutativity | 0.997 | ~1.0 |
| Quasi-orthogonality | ~0.0 | ~0.0 |
| Dimension | 4,096 | 4,096 |

---

## 6. Novel Contributions

### 6.1 Academic Novelty

| Contribution | Prior Work | Our Extension |
|--------------|------------|---------------|
| HDC for KGs | Scalar binding | **Non-commutative GHRR** preserves direction |
| Sheaf consistency | Image alignment | **Ontology cohomology** for text claims |
| Topological queries | Graph diffusion | **Sheaf Laplacian diffusion** with proof validation |
| Hallucination detection | Classifier-based | **Cohomological obstruction** (H¹ ≠ 0) |

### 6.2 Key Insight

Traditional hallucination detection treats inconsistency as a **classification problem** (is this claim true/false?).

We reframe it as a **topological problem**: inconsistent claims form **non-trivial cycles** in the knowledge sheaf, detected by H¹ ≠ 0.

This provides:
1. **Quantitative severity**: dim(H¹) measures inconsistency magnitude
2. **Localization**: Obstruction analysis identifies conflicting claims
3. **Compositionality**: Local consistency + gluing → global consistency

### 6.3 Practical Value

| Use Case | How It Helps |
|----------|--------------|
| RAG systems | Detect conflicting retrievals before generation |
| Multi-agent | Identify disagreement between agent beliefs |
| Fine-tuning | Filter training data with H¹ > 0 |
| Auditing | Provide proof objects for every claim |

---

## 7. Modal GPU Training Plan

### 7.1 Training Objectives

| Task | Input | Output | Loss |
|------|-------|--------|------|
| **Link prediction** | (h, r, ?) | t | Cross-entropy |
| **Consistency scoring** | Sheaf section | H¹ estimate | MSE |
| **Proof path ranking** | Query types | Valid paths | Contrastive |

### 7.2 Model Architecture

```
┌─────────────────────────────────────────────────┐
│              HDCSheafTransformer                │
├─────────────────────────────────────────────────┤
│  Encoder:                                       │
│  ├── EntityEmbedding(vocab_size, 4096)         │
│  ├── RelationEmbedding(num_rels, 4096)         │
│  ├── GHRRBindingLayer(4096 → 4096)             │
│  └── TransformerEncoder(6 layers, 8 heads)      │
│                                                 │
│  Sheaf Head:                                    │
│  ├── LaplacianApproximator(sparse)              │
│  ├── CohomologyPredictor(H⁰, H¹)               │
│  └── DiffusionLayer(Chebyshev approx)           │
│                                                 │
│  Output Heads:                                  │
│  ├── LinkPredictor(4096 → vocab_size)          │
│  └── ConsistencyScorer(4096 → 1)               │
└─────────────────────────────────────────────────┘
```

### 7.3 Modal Deployment

```python
# modal_hdc_training.py (to be created)

import modal

app = modal.App("hdc-sheaf-training")

@app.function(
    gpu="A100",
    timeout=3600,
    volumes={"/data": modal.Volume.from_name("hdc-sheaf-data")}
)
def train_hdc_sheaf(
    dataset: str = "FB15K-237",
    epochs: int = 100,
    batch_size: int = 512,
):
    from hdc_sheaf_pipeline import HDCSheafPipeline
    from benchmark_datasets import BenchmarkSuite
    
    # Load full dataset
    suite = BenchmarkSuite()
    data = suite.load_fb15k237()
    
    # Train
    pipeline = HDCSheafPipeline()
    pipeline.train(data.train, epochs=epochs)
    
    # Evaluate
    results = pipeline.evaluate(data.test)
    return results
```

### 7.4 Estimated Compute

| Dataset | Triples | GPU | Est. Time | Cost |
|---------|---------|-----|-----------|------|
| FB15K-237 | 272K | A100 | ~2 hours | ~$8 |
| WN18RR | 87K | A100 | ~1 hour | ~$4 |
| Full pipeline | 360K | A100 | ~4 hours | ~$16 |

### 7.5 Optimization Targets

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Diffusion latency | 140ms | <10ms | Chebyshev approximation |
| H¹ computation | 100ms | <20ms | Sparse SVD |
| Memory (5K types) | ~500MB | ~100MB | Sparse Laplacian |

---

## 8. Publication Strategy

### 8.1 Target Venues

| Venue | Deadline | Fit | Priority |
|-------|----------|-----|----------|
| **NeurIPS 2026** | May 15 | Excellent | 🥇 Primary |
| ICLR 2027 | July 15 | Excellent | 🥈 Backup |
| ACL 2026 | Already passed | Good | ❌ |
| EMNLP 2026 | ~June | Good | 🥉 Alternative |

### 8.2 Paper Outline

```
Title: Topological Hallucination Detection via Sheaf Cohomology

Abstract: We present a method for detecting LLM hallucinations as
topological obstructions in knowledge sheaves...

1. Introduction
   - Hallucination problem in LLMs
   - Limitations of classifier-based detection
   - Our approach: cohomological obstruction theory

2. Background
   - Hyperdimensional computing
   - Sheaf theory basics
   - Cohomology groups

3. Method
   - GHRR encoding for directed relations
   - Ontology sheaf construction
   - Sheaf Laplacian and diffusion
   - H¹ as inconsistency measure

4. Experiments
   - Datasets: FB15K-237, WN18RR, Text2KGBench
   - Conflict injection study
   - Comparison to baseline methods

5. Results
   - H¹ amplification from conflicts
   - Query latency analysis
   - Ablation studies

6. Discussion
   - Computational complexity
   - Limitations
   - Future work: neural sheaf diffusion

7. Conclusion
```

### 8.3 Differentiation from Prior Work

| Paper | Their Approach | Our Difference |
|-------|----------------|----------------|
| Neural Sheaf Diffusion (Bodnar 2022) | GNN message passing | Explicit cohomology computation |
| SelfCheckGPT (Manakul 2023) | Sampling consistency | Topological rather than statistical |
| G-Eval (Liu 2023) | GPT-4 as judge | No LLM dependency, purely geometric |

---

## 9. Open Questions

### 9.1 Theoretical

1. **Optimal diffusion time**: How to automatically select t for topological_query?
2. **H¹ interpretation**: Can we decompose H¹ into conflict types (contradiction vs. incompleteness)?
3. **Spectral gap**: Does the Sheaf Laplacian spectral gap predict query performance?

### 9.2 Practical

1. **Sparse diffusion**: Can Chebyshev polynomials approximate exp(-tL) efficiently?
2. **Incremental cohomology**: Can we update H¹ incrementally as triples added?
3. **Neural integration**: How to backprop through cohomology computation?

### 9.3 Evaluation

1. **Ground truth**: Need human-labeled hallucination dataset for precision/recall
2. **Baseline comparison**: SelfCheckGPT, G-Eval on same data
3. **Real LLM outputs**: Test on actual GPT-4/Claude outputs, not synthetic conflicts

---

## 10. File Manifest

### Core Implementation

| File | Purpose | Lines |
|------|---------|-------|
| `ghrr_encoder.py` | Non-commutative HDC binding | 333 |
| `ontology_sheaf.py` | Sheaf construction & cohomology | 583 |
| `topological_query.py` | Diffusion + proof queries | 650 |
| `proof_objects.py` | Proof engine | 726 |
| `olog_core.py` | Olog graph structure | 206 |

### Data & Evaluation

| File | Purpose |
|------|---------|
| `benchmark_datasets.py` | FB15K-237, WN18RR, Text2KG loaders |
| `hdc_sheaf_pipeline.py` | End-to-end pipeline |
| `prepare_training_data.py` | Text2KG preparation |

### Training Data

| Path | Size | Source |
|------|------|--------|
| `training_data/FB15K-237/` | 272K triples | Downloaded |
| `training_data/WN18RR/` | 87K triples | Downloaded |
| `training_data/Text2KGBench/` | 7.9K samples | Cloned |
| `training_data/olog_training.jsonl` | 7.4MB | Prepared |

### Documentation

| File | Content |
|------|---------|
| `handoffs/HANDOFF_06_HDC_SHEAF_INTEGRATION.md` | 4-week plan |
| `handoffs/HANDOFF_07_ARCHITECTURE_REVIEW.md` | This document |
| `results/week4_evaluation_report.md` | Evaluation results |

---

## Appendix A: Mathematical Definitions

### A.1 GHRR Binding

For hypervectors x, y ∈ ℂ^d:
```
bind(x, y) = IFFT(FFT(x) ⊙ FFT(y))
```

Non-commutativity achieved via permutation:
```
bind_nc(x, y) = bind(x, ρ(y))  where ρ is cyclic shift
```

### A.2 Sheaf Laplacian

For sheaf F over graph G = (V, E):
```
L_F = Σ_{e=(u,v)} (ρ_e ⊗ ρ_e^T - I ⊗ I)
```

Where ρ_e is the restriction map for edge e.

### A.3 Cohomology Computation

```
H^0 = ker(∂_0)
H^1 = ker(∂_1) / im(∂_0)
```

Computed via SVD:
```
∂_0 = U Σ V^T
dim(H^0) = nullity(∂_0)
dim(H^1) = nullity(∂_1) - rank(∂_0)
```

---

## Appendix B: Quick Start

```bash
# Activate environment
cd /Users/Michaellee/Documents/Runes/ai_research/topics/ontological_induction_sequence_modeling
source venv/bin/activate

# Run demos
python ghrr_encoder.py       # HDC demo
python ontology_sheaf.py     # Sheaf demo
python topological_query.py  # Query demo
python hdc_sheaf_pipeline.py # Full evaluation

# Download datasets (if not already)
python benchmark_datasets.py
```

---

*Review complete. Ready for Modal GPU training phase.*
