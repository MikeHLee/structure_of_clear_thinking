# The Structure of Clear Thinking

A research series on **neurosymbolic architectures for hallucination-free, auditable language models** — gating attention and generation by categorical ontologies (Ologs), detecting knowledge-graph inconsistency with sheaf cohomology, and compiling typed transition systems directly into transformers with machine-checkable audit certificates.

*When AI learns the shape of knowledge, it stops making things up.*

## Research Tracks

| Track | Directory | Status | Key Result |
|-------|-----------|--------|------------|
| **1 — Ontological Induction Engine** (typed attention + proof engine) | `src/` | Core implementation + ablations complete | Typed attention drives invalid-pair attention mass from 0.295 → **0.000** with test accuracy at parity (47.4% vs 45.7%; 23-type ontology, 300 epochs) |
| **2 — HDC/Sheaf Pipeline** (hyperdimensional encoding + cohomology) | `src/` (`ghrr_encoder`, `ontology_sheaf`, …) | Benchmarks complete (Modal A100) | FB15K-237 MRR **0.346** (competitive with ConvE ~0.325, RotatE ~0.338); H¹ cohomology detects injected conflicts (+53 dims for 76 conflicts) |
| **3 — TLTS-Compilation** (type-safe & verifiable transformers) | `Percepta_Transformer_VM/`, `papers/` | **NeSy 2026 full-paper package ready** | Constrained decoding: **100% trajectory soundness by construction** vs 4–48% unconstrained; reachability masking alone proven insufficient; JSON audit certificates re-checkable without model weights |
| **4 — Epistemic-Status Bridge** (sourced vs unfalsifiable claims) | `src/epistemic_status.py`, `docs/` | Implemented, tests passing | SOURCED / FALSIFIABLE_UNSOURCED / UNFALSIFIABLE / UNKNOWABLE gate over (modality, provenance) — formalizes "reframe, don't assert" |
| **Blog series** | `blog/` | Posts 0–5 drafted, publish schedule synced to venue windows | "Structure of Clear Thinking" — hallucination, type-safe attention, proofs-to-text, auditable AI, compiled attention |

## Core Idea

Hallucination is the symptom of an **untyped generation process**: every token can attend to every other token and emit any relation, whether or not that relation exists in the domain. This project treats a domain ontology as a categorical structure — an *Olog* (Spivak): objects are types, morphisms are admissible relations — and enforces it at three levels:

- **Attention**: mask attention by morphism reachability in the Olog (`src/ontological_attention.py`)
- **Generation**: only emit claims that carry a proof object — a composable morphism path — in the category (`src/proof_objects.py`, `src/proof_guided_generation.py`)
- **Audit**: emit a per-trajectory JSON certificate that any third party can re-check against the transition-system spec, with no model weights required (`Percepta_Transformer_VM/verification_certificate.py`)

Two supporting layers make this scale: **hyperdimensional computing** (GHRR, non-commutative binding) for fast approximate retrieval over the ontology, and **sheaf cohomology** (H⁰/H¹ over the knowledge graph) as a principled inconsistency detector.

## Experimental Progress & Results

### Track 1 — Typed attention ablation

23-type multi-domain ontology, 2,536 samples, 300 epochs ([results/attention_ablation_v2.json](results/attention_ablation_v2.json)):

| Model | Test Acc | Invalid-pair attention mass | Attention entropy |
|-------|----------|-----------------------------|-------------------|
| Standard attention | 47.4% | 0.295 | 0.695 |
| **Ontological attention** | 45.7% | **0.000** | 0.440 |

The type-derived mask zeroes attention to ontologically invalid pairs **by construction**, at no meaningful accuracy cost. Reproduce with `scripts/attention_ablation_experiment.py`.

### Track 2 — HDC/Sheaf pipeline

Benchmarked on FB15K-237 (272K triples) and WN18RR (87K triples); details in [docs/EVALUATION_SUMMARY_HDC_SHEAF.md](docs/EVALUATION_SUMMARY_HDC_SHEAF.md) and [results/week4_evaluation_report.md](results/week4_evaluation_report.md):

- **Link prediction** (A100, 50 epochs, 4096-dim hypervectors): MRR 0.346, Hits@10 0.524 — non-commutative HDC binding encodes directed relations at scale (~167M parameters)
- **Hyperparameter sweep** (10 configs): 1024–2048 dims converge fastest at 30 epochs; diffusion time is performance-invariant
- **Cohomology at scale**: exact O(n+m) sparse H⁰/H¹ (replacing O(n³) eigendecomposition); topological consistency scores match dataset structure (WN18RR 0.63 tree-like vs FB15K-237 0.29 cycle-dense)
- **Conflict detection**: injecting 76 contradictions raises H¹ from 5 → 58 — cohomology dimension tracks inconsistency
- **Query latency**: BFS 0.5ms · HDC similarity 5.8ms · sheaf diffusion 140ms per query

### Track 3 — TLTS-compilation (NeSy 2026 submission)

Synthetic-prior harness over e-commerce Ologs, N=1000 trajectories per condition; full numbers in [Percepta_Transformer_VM/experiment_results.md](Percepta_Transformer_VM/experiment_results.md):

| Variant | Soundness (good prior) | Soundness (bad prior) |
|---------|------------------------|----------------------|
| (A) Unconstrained sampling | 48.1% | 4.2% |
| (B′) Reachability masking | 61.5% | 4.3% |
| (C) In-FFN gates / (D) logit masks | **100%** | **100%** |

Findings that survived stress-testing:

- **Soundness-by-construction is real and its cost is conditional**: enforcement is free under an aligned prior (logP actually *improves*) and costs ~5.5 nats/trajectory only when the prior is misaligned — KL/step is a monitorable drift signal
- **Reachability masking is not enough** (the paper's non-obvious claim): it admits destinations without requiring the path step to be a real edge; on cyclic Ologs it collapses to the unconstrained baseline (0–8% soundness). Confirmed against the production attention layer: 66.7% of forward-pass attention mass lands on reachable-but-not-admissible pairs
- **Deployment heuristic**: in-FFN gates (C) beat pure logit masks (D) on latency when the functional fragment exceeds ~20% of non-terminal types (crossover measured in a topology sweep)
- **Audit certificates work**: per-trajectory JSON certificates detect tampering and spec drift via TLTS fingerprints + per-step δ-checks, verifiable with no model access

Submission package (9-page paper + supplementary) in [papers/nesy_submission/](papers/nesy_submission/); extended working draft in [papers/tlts_compilation.md](papers/tlts_compilation.md).

### Track 4 — Epistemic-status bridge

Frontier models already *softly* partition claims into sourced vs unfalsifiable halves; this track makes that a **hard, gateable axis**. `src/epistemic_status.py` derives SOURCED / FALSIFIABLE_UNSOURCED / UNFALSIFIABLE / UNKNOWABLE / UNVERIFIED from the tokenizer's (modality, provenance) slots and maps each to an emission gate (EMIT / DOWNGRADE / ABSTAIN). Design note: [docs/EPISTEMIC_STATUS_BRIDGE.md](docs/EPISTEMIC_STATUS_BRIDGE.md).

## Directory Structure

```
structure_of_clear_thinking/
├── README.md
├── requirements.txt
│
├── src/                          # Core library
│   ├── olog_core.py              #   Olog (categorical ontology) data structures
│   ├── ontological_attention.py  #   Type-constrained attention (reachability masks)
│   ├── proof_objects.py          #   Proof engine: morphism-path proof objects
│   ├── proof_guided_generation.py#   Generation constrained to provable claims
│   ├── ghrr_encoder.py           #   HDC encoding (non-commutative binding)
│   ├── ontology_sheaf.py         #   Sheaf construction + H⁰/H¹ cohomology
│   ├── topological_query.py      #   Diffusion queries with proof validation
│   ├── hdc_sheaf_pipeline.py     #   End-to-end HDC/Sheaf pipeline
│   ├── epistemic_status.py       #   Epistemic-status axis + emission gates
│   ├── hierarchical_tokenizer.py #   Typed tokens w/ modality + provenance slots
│   └── …                         #   Embeddings, benchmarks, loaders, RDF export
│
├── Percepta_Transformer_VM/      # Track 3: TLTS-compilation experiments
│   ├── experiment_loci_comparison.py   # (A)/(B′)/(C)/(D) enforcement loci
│   ├── experiment_cyclic_stress.py     # Reachability collapse on cyclic Ologs
│   ├── experiment_real_attention_b.py  # Production attention-layer mask audit
│   ├── verification_certificate.py     # JSON audit certificates + verifier
│   └── experiment_results.md           # Consolidated results
│
├── papers/                       # Publications
│   ├── tlts_compilation.md       #   Extended working draft
│   └── nesy_submission/          #   NeSy 2026 package (tex + supplementary)
│
├── scripts/                      # Runners
│   ├── attention_ablation_experiment.py
│   ├── modal_hdc_training.py     #   Modal GPU: HDC/Sheaf training
│   ├── modal_olog_training.py    #   Modal GPU: attention experiments
│   └── modal_olog_finetune.py    #   Modal GPU: olog-generation fine-tuning
│
├── tests/                        # Unit + behavioral tests
├── results/                      # Headline artifacts (ablation JSON/figures, reports)
├── docs/                         # Architecture, evaluation, strategy, design notes
├── handoffs/                     # Collaboration handoff documents (01–08 + early)
├── blog/                         # "Structure of Clear Thinking" series (posts 0–5)
├── blog_handoff_oasis_cloud/     # Publish-synced copy + schedule
├── eval/                         # Evaluation memos
└── prior_explorations/           # Earlier related work (chunking, structured output)
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the test suite
python tests/test_epistemic_status.py
python tests/test_hallucination_detection.py

# Reproduce the typed-attention ablation (CPU, ~10 min)
python scripts/attention_ablation_experiment.py --plot

# Run the TLTS enforcement-loci comparison (CPU, seconds)
python Percepta_Transformer_VM/experiment_loci_comparison.py

# Emit + verify an audit certificate
python Percepta_Transformer_VM/verification_certificate.py
```

GPU experiments (HDC/Sheaf training, fine-tuning) run on [Modal](https://modal.com): `modal run scripts/modal_hdc_training.py`.

## Writing

- **Paper**: *TLTS-Compilation: A Neurosymbolic Framework for Type-Safe and Verifiable Transformers* — NeSy 2026 (see `papers/nesy_submission/SUBMISSION_METADATA.md`)
- **Blog series** (`blog/`): 0 — Structure of Clear Thinking · 1 — Why Your LLM Hallucinates · 2 — Attention, But Make It Type-Safe · 3 — From Proofs to Programs to… Text? · 4 — Building an Auditable AI · 5 — Compiling Programs into Attention
- **Key docs**: [ARCHITECTURE.md](docs/ARCHITECTURE.md) · [PUBLICATION_STRATEGY.md](docs/PUBLICATION_STRATEGY.md) · [EPISTEMIC_STATUS_BRIDGE.md](docs/EPISTEMIC_STATUS_BRIDGE.md) · [FINETUNING_STRATEGY.md](docs/FINETUNING_STRATEGY.md)

## Related Series

- [The Shape of Good Behavior](https://github.com/MikeHLee/shape_of_good_behavior) — topological and geometric methods for alignment (Hodge decomposition of preferences, conformal safety metrics, peer-consistency sheaves)

## Citation

```bibtex
@inproceedings{lee2026tlts,
  title={TLTS-Compilation: A Neurosymbolic Framework for Type-Safe and Verifiable Transformers},
  author={Lee, Michael},
  booktitle={Proceedings of the 20th International Conference on Neurosymbolic Learning and Reasoning (NeSy)},
  year={2026}
}
```
