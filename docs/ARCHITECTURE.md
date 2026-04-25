# Neuro-Symbolic Schema Induction Architecture

## Overview

This document describes the architecture for a **provably grounded** language model that replaces statistical token prediction with **ontologically constrained composition**. The core insight is that hallucinations occur when models generate relations that don't exist in the underlying semantic structure.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ONTOLOGICAL INDUCTION ENGINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Raw Text ──▶ [AMR Parser] ──▶ Semantic Graph ──▶ [LLM Refiner]    │
│                                       │                              │
│                                       ▼                              │
│                              ┌─────────────────┐                     │
│                              │   OlogGraph     │                     │
│                              │  (Types+Aspects)│                     │
│                              └────────┬────────┘                     │
│                                       │                              │
│   ┌───────────────────────────────────┼───────────────────────────┐ │
│   │                                   │                           │ │
│   ▼                                   ▼                           ▼ │
│ ┌──────────────┐            ┌─────────────────┐         ┌──────────┐│
│ │ Ontological  │            │   Ontological   │         │  Proof   ││
│ │ Embeddings   │◀──────────▶│    Attention    │────────▶│  Engine  ││
│ │ (Hydration)  │            │ (Type-Constrained)│        │ (Verify) ││
│ └──────────────┘            └─────────────────┘         └──────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## The Hallucination Problem

Standard LLMs suffer from **semantic hallucination** - generating plausible-sounding but factually incorrect relations:

| Claim | Standard LLM | Our System |
|-------|--------------|------------|
| "Customer places Order" | ✓ (correct) | ✓ VALID - direct edge exists |
| "Payment places Customer" | ✓ (sounds plausible) | ✗ HALLUCINATION - no such edge |
| "Einstein invented the iPhone" | ✓ (grammatical) | ✗ HALLUCINATION - no proof path |

The problem: **reachability ≠ semantic validity**. Just because two concepts are connected doesn't mean any relation between them is valid.

## Core Components

### 1. Ontological Types (Objects in Category)

Types are the **nouns** of our ontology - entities that exist:

```python
class OlogNode:
    name: str           # "Customer", "Order", "Invoice"
    description: str    # Natural language description
    type_embedding: np.ndarray  # Learned vector representation
```

**Key Insight**: Types form the **objects** of a category. They constrain what can be composed.

### 2. Ontological Aspects (Morphisms in Category)

Aspects are the **verbs** - relations between types:

```python
class OlogMorphism:
    source: str      # Domain type
    target: str      # Codomain type  
    label: str       # Relation name ("places", "generates", "requires")
    morphism_embedding: np.ndarray  # Learned vector representation
```

**Key Insight**: Morphisms form the **arrows** of a category. They can only compose when types match:
- `f: A → B` and `g: B → C` compose to `g∘f: A → C`
- `f: A → B` and `h: D → E` **cannot compose** (type mismatch)

### 3. Hydration Manifests (Selective Materialization)

Unlike dense embeddings that encode everything, Hydration Manifests specify **which subgraph to materialize**:

```python
class HydrationManifest:
    root_type: str              # Starting point
    included_types: Set[str]    # Types to include
    included_aspects: Set[Tuple[str, str, str]]  # Edges to include
    depth: int                  # Walk depth
    strategy: HydrationStrategy # BFS, DFS, TYPED, SEMANTIC
```

**Why Hydration?**
1. **Sparsity**: Only materialize relevant structure
2. **Composability**: Manifests can be merged/intersected
3. **Interpretability**: Know exactly what's in the embedding
4. **Efficiency**: O(k) instead of O(n) for k << n relevant nodes

### 4. Ontological Attention (Type-Constrained Composition)

Standard attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

**Problem**: Any query can attend to any key, even if semantically invalid.

**Ontological Attention** adds **type constraints**:

```
OntologicalAttention(Q, K, V, G) = softmax(QK^T / √d ⊙ M_G) V

where M_G[i,j] = {
    1  if ∃ morphism from type(q_i) to type(k_j) in graph G
    0  otherwise
}
```

The mask `M_G` **prevents attention across invalid type boundaries**.

#### Type Matching

For attention to flow from token `q` to token `k`, there must exist a valid morphism path:

```
q has type T_q
k has type T_k
Attention allowed iff ∃ path T_q →* T_k in Olog
```

This is the **categorical constraint**: composition is only defined when types align.

### 5. Proof Objects (Constructive Verification)

Every generated claim must be accompanied by a **proof object**:

```python
class ProofObject:
    claim: str              # "Customer places Order"
    status: ProofStatus     # VALID, INVALID, PARTIAL, TIMEOUT
    root: ProofNode         # Derivation tree
    failure_reason: str     # Why proof failed (hallucination detection)
```

**Proof Modes** (strictness spectrum):

| Mode | Rule | Catches |
|------|------|---------|
| STRICT | Direct edge with exact label | Relation fabrication |
| COMPOSITIONAL | Relation must appear in path | Wrong compositions |
| REACHABILITY | Any path suffices | Only disconnected claims |

**STRICT mode** is the default for production - it catches semantic hallucinations.

## Mathematical Foundation

### Category Theory View

The Olog forms a **category** C where:
- Objects: Types (Customer, Order, Invoice, ...)
- Morphisms: Aspects (places, generates, requires, ...)
- Composition: Path concatenation with type checking
- Identity: id_A : A → A for each type A

**Commutative diagrams** encode semantic equivalences:
```
If path A --f--> B --g--> C equals path A --h--> D --k--> C
Then g∘f = k∘h (semantic equivalence)
```

### Sheaf Theory View

The ontology forms a **presheaf** on the Olog category:
- Each type A has a "stalk" F(A) of possible instances
- Each morphism f: A → B has a restriction map F(f): F(B) → F(A)
- **Gluing**: Local sections that agree on overlaps extend to global sections

**H¹ cohomology** measures **inconsistency**:
- H¹ = 0: All local data can be glued consistently
- H¹ ≠ 0: There exist cyclic inconsistencies (potential hallucinations)

### Curry-Howard Correspondence

Proofs are programs, propositions are types:
- A proof of "Customer places Order" is a term of type `Hom(Customer, Order)`
- Composition of proofs corresponds to morphism composition
- **No proof = hallucination** (the claim is unprovable)

## Training Architecture

### Phase 1: Ontology Embedding Pre-training

Train type and morphism embeddings on existing ontologies:

```python
# Contrastive loss: similar types/relations should have similar embeddings
loss = ContrastiveLoss(
    positive_pairs=[(Customer, Client), (places, orders)],
    negative_pairs=[(Customer, Invoice), (places, triggers)]
)
```

Data sources:
- Text2KGBench (29 ontologies, 331 types, 430 relations)
- DBpedia/Wikidata ontology schemas
- Domain-specific ontologies (FHIR for healthcare, FIBO for finance)

### Phase 2: Olog Generation Fine-tuning

Train LLM to generate valid Ologs from text:

```
Input:  "A customer places an order which generates an invoice"
Output: {
    "types": ["Customer", "Order", "Invoice"],
    "aspects": [
        {"source": "Customer", "target": "Order", "label": "places"},
        {"source": "Order", "target": "Invoice", "label": "generates"}
    ]
}
```

### Phase 3: Attention Constraint Learning

Train the attention mask to respect type constraints:
1. Tokenize text with ontological grounding
2. Assign type labels to tokens
3. Mask attention based on Olog structure
4. Fine-tune with proof-verified outputs only

### Phase 4: Proof-Guided Generation

Use proof engine during inference:
1. Generate candidate output
2. Extract claims from output
3. Attempt proof for each claim
4. Reject/regenerate claims without valid proofs

## Inference Pipeline

```
Input Text
    │
    ▼
┌─────────────────┐
│  AMR Parsing    │  (Structural grounding)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Olog Induction │  (Extract types + aspects)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hydration       │  (Selective subgraph)
│ Manifest        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ontological     │  (Type-constrained)
│ Attention       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Candidate       │
│ Generation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Proof Engine    │  (Verify all claims)
│ (STRICT mode)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 VALID     INVALID
 Output    (Regenerate)
```

## Comparison to Existing Approaches

| Approach | Grounding | Verifiable | Composable | Interpretable |
|----------|-----------|------------|------------|---------------|
| Standard LLM | ✗ | ✗ | ✗ | ✗ |
| RAG | Partial | ✗ | ✗ | Partial |
| Knowledge Graphs | ✓ | ✗ | ✗ | ✓ |
| Neuro-Symbolic | Partial | Partial | ✗ | Partial |
| **This Work** | ✓ | ✓ | ✓ | ✓ |

Key differentiators:
1. **Categorical structure** ensures only valid compositions
2. **Proof objects** make verification constructive
3. **Hydration manifests** enable efficient sparse computation
4. **Type-constrained attention** prevents cross-boundary hallucination

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Core Olog Engine | ✓ Complete | `olog_core.py` |
| Hybrid Encoder (AMR+LLM) | ✓ Complete | `hybrid_encoder.py` |
| Semantic Analysis | ✓ Complete | `semantic_analysis.py` |
| RDF/OWL Export | ✓ Complete | `rdf_export.py` |
| Ontological Embeddings | ✓ Complete | `ontological_embeddings.py` |
| Proof Objects | ✓ Complete | `proof_objects.py` |
| Fine-tuning Pipeline | ✓ Complete | `train_olog_model.py` |
| **Ontological Attention** | 🔄 Design | (this doc) |
| Integration Tests | ⏳ Pending | - |

## Next Steps

1. **Implement OntologicalAttention layer** in PyTorch/JAX
2. **Train type/relation embeddings** on Text2KGBench
3. **Benchmark hallucination detection** accuracy
4. **Write research paper** formalizing the theoretical framework
5. **Create blog post** with visualizations and demos

## References

- Spivak, D. I. (2014). *Category Theory for the Sciences*
- Curry, J. (2014). *Sheaves, Cosheaves and Applications*
- Baez, J. & Stay, M. (2011). *Physics, Topology, Logic and Computation: A Rosetta Stone*
- Robinson, M. (2014). *Topological Signal Processing*
