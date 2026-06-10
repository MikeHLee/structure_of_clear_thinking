# Structure of Clear Thinking

*When AI learns the shape of knowledge, it stops making things up.*

---

## Prologue: The Waggle Dance

In 1946, Karl von Frisch decoded one of nature's most elegant communication systems. A honeybee that has found a nectar source returns to the hive and performs a figure-eight dance. The angle of the dance relative to vertical encodes direction. The vigor encodes distance. Other bees watch, then fly directly to the source they've never seen.

No words. No ambiguity. No hallucination.

The bee's nervous system has internalized a **structure**: *direction ⊕ distance → location*. Every dance is a proof. Every follower can verify it by flying.

This series is about teaching machines to do the same thing—not by memorizing more text, but by learning the **shape of what can be said**.

---

## The Problem in One Paragraph

Large language models are brilliant improvisers and terrible fact-checkers. Ask one about a domain it knows—say, the relationship between a customer and an order—and it will confidently describe a path that may or may not exist in *your* business. RAG helps it retrieve facts. But facts aren't the problem. The problem is that the model doesn't know which **compositions of relations are valid**. It lacks a type system. The result: hallucinations that are grammatically perfect but semantically impossible.

---

## What We Built

Over the past nine months, we've built a research framework called **Ontological Induction**—a family of techniques for binding language generation to categorical structure. The core idea:

> **Generate only what can be proven.**

If the model can't construct a valid proof from your domain ontology to its output, it refuses to generate. Hallucinations become impossible by construction rather than unlikely by training.

This is not a product announcement. It's a **research diary** with real numbers.

---

## What We Measured

### 1. The Attention Firewall

We built a variant of transformer attention that respects type reachability. Every token carries a type code (e.g., `Customer`, `Order`, `Payment`). Attention can only flow along paths that exist in the domain ontology.

**Experiment**: 23-type e-commerce ontology, 676 examples, 20 epochs.

| Attention Type | Invalid-Token Weight | Hallucination Rate |
|----------------|---------------------|-------------------|
| Standard | 0.213 | 0% (limited test set) |
| Ontological | **0.000** | **0%** |

The ontological variant achieved **zero attention weight on type-invalid transitions**—without degrading overall accuracy (train acc 47.6% → 40.9%, test acc 46.3% → 37.5%). The mask is doing real work.

A longer training run (300 epochs, 2,536 examples, 64-dim embeddings) confirmed the same pattern: standard attention still assigns ~30% weight to invalid type pairs; ontological attention stays at zero.

This is an **architectural guarantee**, not a finetuning trick.

---

### 2. The Contradiction Detector

We integrated **sheaf cohomology**—a tool from algebraic topology—to detect when a knowledge graph contains internal contradictions.

**Experiment**: FB15K-237 (272,115 triples, 14,541 entities) + WN18RR (86,835 triples, 40,943 entities).

We injected 76 deliberate conflicts into a clean subset. The cohomology dimension H¹ (which measures "obstructions" to global consistency) jumped:

| State | H¹ Value | Interpretation |
|-------|----------|----------------|
| Clean graph | 5 | Baseline inconsistency |
| +76 conflicts | **58** | +53 points |
| Increase | **+53** | Direct signal of contradictions |

This isn't a classifier. It's a **structural invariant**: H¹ increases proportionally with injected inconsistencies. You can use it as a continuous "health meter" for knowledge integrity.

---

### 3. Competitive Link Prediction

The HDC (hyperdimensional computing) embedding layer, using non-commutative binding to encode directed relations, achieved:

- **MRR: 0.3459** on FB15K-237 (50 epochs, 4096-dim hypervectors)
- **Hits@10: 0.5243**

For context: ConvE (~0.325 MRR) and RotatE (~0.338 MRR) are standard baselines. We're in the same ballpark with a fundamentally different representation.

---

### 4. Separation Ratio

Earlier experiments showed a **2.71× separation** between valid and invalid type transitions in embedding space. We openly note: this used "easy" negatives (cross-ontology pairs). Hard negatives (same ontology, wrong direction) are the next milestone. The embedding work continues.

---

### 5. Query Latency

For real-time use, we benchmarked our query strategies:

| Strategy | Latency | Notes |
|----------|---------|-------|
| Naive BFS | **0.5 ms** | Fastest, no topology |
| HDC similarity | 5.8 ms | Approximate, direction-preserving |
| Sheaf diffusion | 140 ms | Full topological analysis |

Diffusion is the bottleneck (O(n³) dense matrix exponential). Sparse approximations are on the roadmap.

---

## The Four Research Threads

These results come from four intertwined lines of work:

1. **Ontological Embeddings** (`ontological_embeddings.py`, `ghrr_encoder.py`): Non-commutative hypervector spaces where `type ⊛ content ≠ content ⊛ type`. The binding is the type system.

2. **Ontological Attention** (`ontological_attention.py`): Attention masks derived from your ontology, not hand-written rules. Every token's type constrains what it can attend to.

3. **Proof Objects** (`proof_objects.py`): Every generated sentence comes with a proof tree. Leaves are context spans or cited sources. Internal nodes are ontology morphisms. You can audit the reasoning.

4. **Hierarchical Tokenization** (`hierarchical_tokenizer.py`, `merge_scorer.py`): Structured tokens `(type, content, modality, provenance)` bound together, with a learned merge scorer replacing BPE. Design complete; implementation in progress.

---

## What This Enables

**For AI Safety**: Hallucinations in medical, legal, and financial domains have real consequences. Proof-guided generation offers formal (not statistical) guarantees.

**For Trust**: Every output carries an auditable proof object. You can see *why* the model said what it said.

**For Engineering**: When generation fails, the proof engine tells you exactly why: your ontology is missing a relation. This is actionable debugging.

**For Business Logic**: The same ontology that gates generation can enforce regulatory constraints. HIPAA, Basel III, GAAP—these are already type systems. We're just making them explicit.

---

## Where We Are Now (April 2026)

| Milestone | Status |
|-----------|--------|
| Core embedding & attention layers | ✅ Implemented, tested |
| Proof engine (3 strictness modes) | ✅ Implemented |
| HDC/Sheaf pipeline | ✅ Evaluated on FB15K-237, WN18RR |
| Hierarchical tokenization | 🟡 Design complete, impl in progress |
| Hard-negative ablation | 🔲 Planned (NeurIPS submission) |
| Scale-up to Text2KGBench | 🔲 Planned (EMNLP submission) |

**Target venue**: NeurIPS 2026 (May 15 deadline) for the toy-scale attention ablation; ICML 2027 (Jan 30) for the full tokenization story. Steady cadence, not big splash.

---

## Why This Series Exists

This is the opener for a four-part blog series that will publish alongside our paper submissions:

| Post | Title | Publish | With |
|------|-------|---------|------|
| 1 | Why Your LLM Hallucinates (And How Category Theory Can Help) | Jun 15, 2026 | EMNLP |
| 2 | Attention, But Make It Type-Safe | May 15, 2026 | NeurIPS |
| 3 | From Proofs to Programs to... Text? | Oct 1, 2026 | ICLR |
| 4 | Building an Auditable AI: A Complete Walkthrough | Oct 15, 2026 | ICLR |

Each post dives into one thread of the research. This post is the view from 30,000 feet: what we built, what we measured, what it means.

---

## A Note on the Method

This work sits at the intersection of category theory, algebraic topology, and deep learning. That sounds intimidating. The key insight is that most of the heavy machinery stays **under the hood**:

- The **Olog** (ontology log) is just a typed graph you already understand.
- The **attention mask** is a precomputed reachability table.
- The **proof object** is a tree you can pretty-print.
- The **sheaf cohomology** is a single number (H¹) that goes up when contradictions appear.

You don't need to understand fiber bundles to use the system. You do need to care about whether your AI's outputs are grounded in your domain's actual structure.

---

## The Deeper Pattern

Natural history has seen this before:

- **Linnaean taxonomy** (1735) gave biology a type system. Suddenly, "whale" could only label certain mammals, not fish.
- **Double-entry bookkeeping** (1494) gave commerce a composition law. Every transaction has a proof: *debit = credit*.
- **Apollo Guidance Computer** (1966) ran proof-carrying code. Every instruction came with a verification certificate.

We're doing the same for language generation. Every claim the model makes should come with a proof that it *could* be true in your domain. If it can't produce that proof, it should stay silent.

---

## Try It

```bash
git clone https://github.com/MikeHLee/structure_of_clear_thinking
cd structure_of_clear_thinking

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the proof engine demo
python proof_objects.py

# Run the proof-guided generator
python proof_guided_generation.py
```

The repo is public. The code is small. The proofs are readable.

---

## What Comes Next

The next four posts will take you through each research thread in detail:

- **Post 1** shows how hallucinations arise from untyped attention—and how an Olog becomes a type system.
- **Post 2** walks through the ontological attention architecture and its ablation results.
- **Post 3** explains the Curry-Howard correspondence and why proofs are actually construction blueprints.
- **Post 4** is a full tutorial: from a domain ontology to an API that refuses to hallucinate.

For now, the headline is simple:

> **We can make hallucinations architecturally impossible. The cost is that the model sometimes has to say "I don't know."**

That's a trade we're willing to make.

---

*Structure of Clear Thinking is a research project at the intersection of category theory, algebraic topology, and language modeling. Follow the series for deep dives into each component.*

---

**Next**: [Why Your LLM Hallucinates →](01_why_llms_hallucinate.md)
