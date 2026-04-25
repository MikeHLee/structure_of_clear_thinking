# Handoff 08 — Hierarchical + Semantic Ontological Tokenization

**Date**: 2026-04-24
**Owner**: TBD
**Status**: Design stage
**Depends on**: `ontological_embeddings.py`, `ontological_attention.py`, `ghrr_encoder.py`, `proof_objects.py`
**Blocks**: Track 2 (NeurIPS 2026) scale-up; Track 3 (ICLR 2027) training.

---

## 1. End goal (do not lose sight of this)

A generative sequence transformer that operates like a modern LLM, but:

1. Every generated statement is bound to a **context-typed datum** (the witness).
2. No claim exceeds what that witness supports — e.g. "what time is it" can only resolve to the **most recent valid `TimeOfDay` token** in context; anything else is rejected at decode.
3. The audit trail is a **proof object** (Track 3) whose leaves are witnesses and whose internal nodes are Olog morphisms.

For this to scale beyond toy ontologies, the vocabulary itself must be **learned**, not enumerated. This handoff specifies how.

---

## 2. The combined tokenization scheme

We merge the user's two proposals into a single pipeline. Neither alone is sufficient:

- Hierarchical alone → handcrafted token layers; does not generalize to unseen domains.
- Semantic alone → learns coherent clusters but loses the type/content separation the Olog attention mask exploits.

### 2.1 Token layout (hierarchical)

Each ontological token is a structured tuple of **slots**, not a single integer:

```
τ = ( type_code,     # which Olog object / morphism class
      content_code,  # the specific filler (entity, value, span)
      modality_code, # {assertion, hypothesis, question, citation, ...}
      provenance_code) # witness id → links to a context span or external source
```

Each slot has its own embedding table; the final token embedding is the **GHRR bind** of the four (non-commutative, so `type⊛content ≠ content⊛type`). This is a direct extension of `ghrr_encoder.py`; no new primitive needed.

**Why this fits prove-then-generate**: the type slot is what the Olog attention mask gates on; the content slot is what the decoder is free to choose *within* that type; provenance is what the proof object points to. The three concerns decouple cleanly.

### 2.2 Merge scoring (semantic)

Replace BPE with a **neural merge scorer** `s_θ(τ_i, τ_j) → ℝ` trained to predict:

- **Coherence**: does `τ_i ⊛ τ_j` correspond to a valid Olog morphism composition? (supervised by `olog_core.py` during training; self-supervised at inference)
- **Accuracy**: does merging preserve downstream next-token prediction on a held-out corpus?
- **Compression**: does the merged token reduce sequence length without increasing proof-object size?

Loss:
```
L = L_coherence + λ_acc · L_LM + λ_comp · L_length − λ_H · H(merge_dist)
```
The entropy term prevents collapse to a single mega-token. This is the piece inspired by Anthropic-style learned tokenization — the scorer itself is a small transformer, trained jointly with the main model but frozen during inference for determinism.

Merges are applied greedily at encode time but the *merge graph* is retained — the proof object can point back to pre-merge leaves, so provenance is not lost when tokens are compacted.

---

## 3. How this is learned, not specified (the hard question)

Three learning signals, in order of decreasing supervision:

1. **Text2KGBench + FB15K-237 + WN18RR** → weakly-supervised type-slot labels from triples. Already in repo ([training_data/Text2KGBench](../training_data/Text2KGBench)).
2. **Self-distillation from a pretrained LLM** (see §4): run a frozen teacher over a corpus, cluster attention patterns, and use cluster IDs as pseudo-types. Sinha et al. (arXiv 2509.07122) frames this as the "attention-as-relational-structure" pattern; it fits our Q=Rules / K=Graphs / V=Objects reading of attention in `ontological_attention.py`.
3. **Cycle / H¹ feedback** from `ontology_sheaf.py`: when the merge scorer produces a tokenization that creates H¹ > 0 on a coherent corpus, that is a negative signal — the sheaf disagrees with the tokenizer. This is a **free, unsupervised training signal** we already compute.

The Sheth/Roy/Gaur framing (arXiv 2305.00813) — *perception maps sensory inputs to symbols; cognition maps symbols to environment knowledge* — maps onto our two stages: the merge scorer is perception (text → hierarchical ontological tokens), the Olog-masked transformer is cognition (symbols → consequences under type constraints).

---

## 4. Leveraging pretrained models (do not rebuild from scratch)

This is where we get to avoid 99% of the compute cost. Concrete plan:

| Layer | Source | How we reuse it |
|---|---|---|
| **Content embeddings** | A frozen pretrained text encoder (e.g. `bge-m3`, `nomic-embed-text-v2`, or a Llama-3 mid-layer) | Initialize `content_code` slot embeddings by projecting pretrained vectors into our GHRR space via a learned linear map. Do NOT retrain the encoder. |
| **Subword fallback** | Pretrained BPE tokenizer (tiktoken / SentencePiece) | For any span the merge scorer cannot type-label with confidence > τ, fall back to the pretrained BPE tokens with `type_code = UNTYPED`. Graceful degradation: model still works, just without hallucination guarantees for those spans. |
| **Teacher attention** | Frozen Llama-3-8B or Qwen-3-14B | Used once to mine pseudo-type clusters (§3.2). Not used at inference. |
| **Decoder backbone** | LoRA adapters on a pretrained 7B–8B (DeepSeek-R1-Distill-Qwen-7B is already in the memory notes) | Only the attention mask and output head are swapped; the FFN weights come free from pretraining. Training cost drops ~10×. |

**Key design rule**: pretrained weights are the prior; Olog structure is the likelihood. We never fight the pretrained model — we constrain it.

---

## 5. Forward-pass proof scheme (revised)

Decode loop, per step t:

1. Current context = sequence of hierarchical tokens τ_1..τ_{t−1}, each with a provenance pointer.
2. Proof state `π` (from `proof_objects.py`) enumerates the set of type slots that are legal next.
3. Ontological attention masks K,V to tokens whose type appears as a source in some morphism toward a legal target type.
4. Decoder proposes a distribution over `(type_code, content_code)` pairs.
5. For `content_code`, we **restrict to witnesses present in context** whose type matches. This is the "only the most recent `TimeOfDay` is valid" rule, generalized: every content slot must either (a) copy from context, or (b) be a Skolem term justified by a morphism chain back to context data.
6. Provenance is set to the witness ID; proof node is appended.

If step 5 has an empty feasible set, the model outputs a `CANNOT_ANSWER` token. This is the hallucination firewall — architecturally, not heuristically.

---

## 6. Deliverables & sequencing (6 weeks → NeurIPS May 15)

| Week | Task | Output |
|---|---|---|
| W1 (Apr 27–May 3) | Implement `hierarchical_tokenizer.py`: slot schema, GHRR bind of 4 slots | Unit tests on FB15K-237 triples |
| W2 (May 4–10) | Implement `merge_scorer.py` + joint loss | Scorer checkpoint, merge-graph export |
| W3 (May 11–17) | Wire into `ontological_attention.py`; ablate slots (type-only vs type+modality vs full) | NeurIPS submission version |
| W4 | Pretrained-encoder projection head | Frozen bge-m3 → GHRR map trained |
| W5 | Context-restricted decoding (§5.5) + `CANNOT_ANSWER` firewall | Track 3 prototype |
| W6 | End-to-end eval on Text2KGBench + a synthetic time-of-day task | Track 1 EMNLP numbers |

---

## 7. Open questions (flag before coding)

1. Is the GHRR space large enough to hold 4-slot bindings without interference at our target vocab size (≥50K types × ≥100K contents)? Needs dimension sweep.
2. Does the merge scorer stay stable when trained jointly with the Olog mask, or do we need two-stage training (scorer first, then freeze)?
3. Pretrained-encoder projection: linear enough, or do we need a small MLP? Linear preserves the "no retraining" story.
4. Fallback UNTYPED tokens: do they pollute the proof audit trail? Proposal: mark their spans as "unverified" in the output, not "verified".

---

## 8. References mined this round

- Sinha et al., *Neuro-Symbolic Frameworks: A Conceptual Overview*, arXiv **2509.07122** — taxonomy; confirms attention-as-constraint is a recognized integration pattern.
- Sheth, Roy, Gaur, *Neurosymbolic AI: Why, What, and How*, arXiv **2305.00813** — perception/cognition split; motivates our two-stage pipeline.
- PMC9166567 — neurosymbolic survey; four directions, including KB-augmented deep learning (maps to our "pretrained as prior, Olog as likelihood" framing).
- IEEE 11192262 and Artificial Intelligence S0004370224002091 — **paywalled, could not fetch**. TODO: pull PDFs through institutional access or author pages before EMNLP submission.

---

*Cross-reference: this handoff supersedes the tokenization section of [HANDOFF_07_ARCHITECTURE_REVIEW.md](HANDOFF_07_ARCHITECTURE_REVIEW.md).*
