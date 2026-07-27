# Correspondence: Procedural-Ontology VM ⊆ Type-Safe Attention Paradigm

> What you asked me to crystallize: the shape of the correspondence between
> Percepta-style WASM-in-a-transformer and your type-safe / proof-guided
> attention work. I'm not claiming a formal equivalence — I'm claiming a
> **subsumption** with an explicit mapping. The procedural case is a strict
> specialization of the ontological case.

## The shared abstract object: a Typed Labeled Transition System (TLTS)

Define a TLTS as a tuple

```
M = (T, L, δ, t₀)
```

where

- `T` — a set of **types** (interpretation: structural categories of state)
- `L` — a set of **labels** (interpretation: tokens carrying a type tag)
- `δ ⊆ T × L × T` — a **typed transition relation**
- `t₀ ∈ T` — initial type

Write `t₁ —[ℓ]→ t₂` for `(t₁, ℓ, t₂) ∈ δ`. A **trajectory** is a sequence
`t₀ —[ℓ₁]→ t₁ —[ℓ₂]→ t₂ —[ℓ₃]→ …` that respects δ at every step.

**Two specializations matter:**

- **Functional TLTS (`TLTS_func`)**: δ is a partial function `T × L ⇀ T`. At
  most one successor per `(type, label)`. Trajectories are deterministic given
  a label sequence.
- **Relational TLTS (`TLTS_rel`)**: δ is a relation. Multiple successors are
  allowed. Trajectories branch.

`TLTS_func ⊊ TLTS_rel` (every function is a relation; not every relation is
single-valued).

## Compilation pattern: TLTS → Transformer

Both your work and the Percepta description compile a TLTS into transformer
mechanics via the same three-part recipe.

| Step              | Compilation rule                                                                 |
|-------------------|-----------------------------------------------------------------------------------|
| **Embed types**   | Each `t ∈ T` gets a type vector `e(t)`; token embedding adds `e(type(token))`.   |
| **Mask attention**| `mask(i, j) = 1` iff there exists `ℓ` such that `type(qᵢ) —[ℓ]→ type(kⱼ) ∈ δ`.   |
| **Apply δ in FFN**| FFN row for label `ℓ` activates only on `(state, ℓ)` matching some `δ` entry.    |

Call a transformer **TLTS-compiled** if its embeddings, attention mask, and
FFN obey this recipe for some TLTS `M`. Write this class `Comp(M)`.

The **soundness condition** (parallels your `proof_guided_generation.py:683`):

> If a transformer in `Comp(M)` produces a token sequence `ℓ₁ ℓ₂ … ℓₙ`, then
> there exists a trajectory in `M` whose labels match that sequence.

For functional TLTS this is determinism. For relational TLTS this is your
"prove-then-generate" property: every emission corresponds to *some* admissible
ontological path.

## The mapping

### Your system → TLTS_rel

From `ontological_attention.py`:

| TLTS element     | In your code                                                                 |
|------------------|------------------------------------------------------------------------------|
| `T`              | `OlogGraph.graph.nodes()` — Olog types                                       |
| `L`              | Olog edge labels (morphism names) ∪ entity tokens                            |
| `δ`              | `(t₁, ℓ, t₂) ∈ δ` ⇔ edge `t₁ —ℓ→ t₂` exists, plus transitive closure to `max_path_length` |
| Mask rule        | `OntologicalAttention._compute_reachability` → `M_G[i,j] = 1 iff t_j ∈ reach(t_i)` |
| Token embedding  | `embed_tokens`: `base + type_emb + relation_emb`                             |
| Soundness        | `ProofGuidedGenerator` + `ConstrainedDecoder` enforcing valid morphism paths |

Your δ is *relational*: a Customer reaches an Order which generates an Invoice
*or* contains a Product — both are valid successors. Generation branches; the
proof object selects one branch.

### Percepta's system → TLTS_func

| TLTS element     | In the WASM-VM construction                                                  |
|------------------|------------------------------------------------------------------------------|
| `T`              | WASM machine states modulo stack/register-shape type signatures              |
| `L`              | WASM opcodes ∪ immediates                                                    |
| `δ`              | WASM operational semantics (deterministic, single-valued)                    |
| Mask rule        | Hull-backed KV cache: query `[1, 0]` over `(step, value)` keys retrieves the unique most-recent matching write |
| FFN dispatch     | `output = gate(state) · transition(state)` — one row per opcode, sharp gating |
| Soundness        | Determinism: the trajectory is unique given input tokens                     |

Percepta's δ is *functional*: each `(state, opcode)` pair has exactly one
successor. The hull trick is a *retrieval* optimization for this functional
case — when you know the answer is unique, you can index it geometrically
instead of soft-attending.

## The subsumption claim

**Claim.** Percepta-style WASM-in-transformer is an instance of `Comp(M)` where
`M ∈ TLTS_func`. Your Olog/proof-guided system is an instance of `Comp(M)`
where `M ∈ TLTS_rel`. Since `TLTS_func ⊊ TLTS_rel`, the procedural case is a
strict specialization of the ontological case.

**What that gives you.** Three things, in increasing order of usefulness.

1. **Vocabulary.** "Procedural ontology" and "semantic ontology" are both
   TLTS-compilations; they differ in whether δ is functional or relational.
2. **Architectural transferability.** Hull-backed retrieval is a δ-functional
   trick. It does not lift to your setting in general because your δ branches.
   But it lifts to **deterministic sub-fragments** of your Olog (e.g.,
   chain-shaped paths with single outgoing edges per node). Those fragments
   could use hull-indexed cache; the rest of the graph falls back to standard
   masked attention.
3. **A precise place to attempt equivalence.** A formal equivalence would say:
   when restricted to functional δ, your TLTS-compilation and Percepta's
   coincide up to representation choice. That's a tractable theorem to attempt;
   the obstacles are concrete (encoding choice for `e(t)`, sharpness of FFN
   gates, hull-update semantics) rather than conceptual.

## What this is *not*

- **Not** a proof that running WASM in a transformer is equivalent to running
  proof-guided generation over an Olog. It isn't. The state spaces are
  different, the labels are different, the trajectories are different.
- **Not** a claim that Percepta's specific compilation is novel relative to
  Tracr-style work. The novelty (if any) is in the WASM-as-target choice and
  the hull primitive — not in TLTS-compilation as a category.
- **Not** a claim that your system can do what Percepta's does. You'd need to
  add an FFN-encoded δ-application step (currently your δ is applied via
  proof search at generation time, not in-weight). That's the missing rung.

## The missing rung — what would make the equivalence tight

If you wanted to bring the two systems into the same architectural surface
(not just the same abstract category), the move is:

> Compile a deterministic sub-Olog into FFN rows, the same way Percepta compiles
> WASM opcodes. Within that sub-Olog, generation becomes one forward pass per
> step, no proof search. Outside it, fall back to your proof-guided constrained
> decoder.

That's a hybrid that uses Percepta's primitive where it applies (functional δ,
hull retrieval) and your primitive everywhere else (relational δ, proof search).
The architecture would have a precise type signature: input residual stream is
either in the "compiled" subspace (where deterministic forward pass advances
state) or in the "search" subspace (where the proof engine drives generation).

This is the natural next experiment if you wanted to operationalize the
correspondence rather than just observe it.

## Honest caveats

- I have not verified that Percepta's paper exists or matches the handoff
  summary. If the paper diverges on the construction, this mapping needs to be
  redone against the actual artifact.
- "TLTS-compilation" as a category is something I'm using as an organizing
  abstraction here; it isn't a published framework I'm citing. If you want to
  publish along these lines, the literature to engage with is Tracr, RASP, and
  the "attention-as-Turing-complete" line — they cover adjacent territory.
- The formal subsumption claim is at the level of "both fit the same recipe."
  Promoting it to a theorem requires fixing a precise definition of `Comp(M)`,
  which in turn requires fixing how `e(t)`, the mask, and the FFN encoding are
  parameterized. I left those open here because the right parameterization
  depends on which architectural commitments you want to keep.
