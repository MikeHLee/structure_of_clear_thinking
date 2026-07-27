# TLTS-Compilation: A Categorical Framework for Type-Safe Transformer Design and Post-hoc Verification of LLM Computation

**Working draft · May 2026 · Structure of Clear Thinking · target venues: NeSy 2026, NeurIPS 2026, ICLR 2027, ACT 2027**

---

## Abstract

We propose **TLTS-compilation** as a unifying framework for two recently visible
research threads: (i) "type-safe attention" mechanisms that gate transformer
attention by morphism reachability in a domain ontology, and (ii) "transformer
as virtual machine" constructions that compile deterministic programs (such as
WebAssembly bytecode) directly into transformer weights. We show that both
threads are instances of compiling a **typed labeled transition system**
(TLTS) into a transformer via a structure-preserving functor, with the
procedural (VM-style) case arising as a strict specialization of the
ontological case. The framework yields three deliverables. First, a clean
categorical statement of soundness — what it means for a compiled transformer
to faithfully implement a transition system. Second, a concrete experimental
design that operationalizes the correspondence: compiling a chain-shaped
sub-Olog into FFN rows à la the procedural construction, and benchmarking
against reachability-masked attention and unconstrained baselines. Third, a
**post-hoc verification protocol**: a procedure for checking, given a forward
pass and a claimed transition system, whether the trajectory is admissible
under the system. We argue that TLTS-compilation supplies what the broader
"verifiable AI" agenda has lacked — a typed audit artifact that is generated
*as a byproduct of the forward pass*, not retrofitted afterward.

---

## 1. Introduction

Two parallel lines of work have recently converged on the same
architectural intuition from opposite ends of the symbolic / neural axis.

From the **symbolic** side, researchers (including the present authors;
[`ontological_attention.py`](../src/ontological_attention.py),
[`proof_guided_generation.py`](../proof_guided_generation.py)) have argued
that hallucination in language models is the symptom of an
**untyped attention mechanism**: every token can attend to every other
token, regardless of whether the implicit semantic relation between them is
admissible in the target domain. The remedy is to gate attention by a
domain ontology — a categorical structure (an *Olog*, in Spivak's sense [^Spivak2014]
of a category whose objects are types and whose morphisms are
admissible relations) — and to constrain generation to outputs that admit
a proof object in that category. Empirically, the invalid-token attention
weight drops from 0.21–0.30 to 0.000 on a 23-type e-commerce ontology
under this regime, with no degradation of test-set accuracy
([`results/attention_ablation.json`](../results/)).

From the **neural** side, researchers have shown that transformers are
not merely Turing-complete in principle but can be *analytically compiled*
to execute specified programs in practice. The Tracr line [^Lindner2023]
compiles RASP programs into transformer weights. More aggressively, recent
work claims to compile a complete WebAssembly virtual machine into a
PyTorch transformer such that one forward pass implements one VM step, with
a hull-backed key-value cache providing logarithmic-time memory retrieval
in place of the linear scan of standard attention [^Percepta2026].
The transformer becomes a deterministic computer: same input, same output,
no hallucination by construction.

These threads look unrelated — one is about typed knowledge graphs, the
other about machine code — but they share a load-bearing pattern. **Both
compile a discrete, typed transition system into transformer mechanics
such that the forward pass implements the system's semantics.** The
question this paper answers is: what is the right level of abstraction at
which to state that pattern, and what does it buy us?

We claim the right abstraction is the **typed labeled transition system
(TLTS)** and that the right buy is post-hoc verifiability. Our
contributions are:

1. **A categorical formalization** of TLTS-compilation as a functor from
   the path category of a transition system to a category of compiled
   transformer behaviors (§2).
2. **A subsumption result** showing that procedural (VM-style)
   compilation is the deterministic-δ specialization of ontological
   (Olog-style) compilation (§3).
3. **An experimental design** that operationalizes the correspondence by
   compiling a chain-shaped sub-Olog into FFN rows, with the experiment
   ready to instantiate against the existing ontological attention
   codebase (§4).
4. **A post-hoc verification protocol** that turns the compiled
   transformer's forward pass into an audit artifact admitting independent
   re-execution and trajectory-validity checks (§5).
5. **An honest accounting** of what is and isn't verifiable in published
   "transformers as computers" claims, including reproducibility risks
   that arise from how the compilation is reported (§5.4).

---

## 2. Categorical formalization

### 2.1 Typed labeled transition systems

A **typed labeled transition system** (TLTS) is a tuple
$M = (T, L, \delta, t_0)$ where $T$ is a set of *types* (state-shape
classes), $L$ is a set of *labels* (typed tokens), $\delta \subseteq T
\times L \times T$ is a *typed transition relation*, and $t_0 \in T$ is
an initial type.

Two specializations matter:

- $M$ is **functional** if $\delta$ is a partial function
  $T \times L \rightharpoonup T$. At most one successor per
  $(\text{type}, \text{label})$.
- $M$ is **relational** otherwise.

Write $\mathsf{TLTS}_{\mathrm{func}}$ and $\mathsf{TLTS}_{\mathrm{rel}}$
for these classes. We have
$\mathsf{TLTS}_{\mathrm{func}} \subsetneq \mathsf{TLTS}_{\mathrm{rel}}$.

### 2.2 The path category

Every TLTS $M$ induces a category $\mathcal{C}_M$:

- **Objects**: types $t \in T$.
- **Morphisms**: $\mathrm{Hom}(t_1, t_2)$ is the set of finite label
  sequences $(\ell_1, \ldots, \ell_n) \in L^*$ such that there exist
  $t^{(1)}, \ldots, t^{(n-1)}$ with $(t_1, \ell_1, t^{(1)}), (t^{(1)},
  \ell_2, t^{(2)}), \ldots, (t^{(n-1)}, \ell_n, t_2) \in \delta$.
- **Composition**: concatenation of label sequences.
- **Identity**: the empty sequence at each $t$.

This is the free category on the directed multigraph of $\delta$, or
equivalently the *path category*.

When $M$ is functional, $\mathcal{C}_M$ is **thin** (at most one
morphism per ordered pair of objects), because deterministic execution
traces a unique path. When $M$ is relational, parallel morphisms exist:
multiple label sequences can connect the same pair of types.

For a domain Olog $\mathcal{O}$ (a typed graph plus equational
identifications between paths), $\mathcal{C}_\mathcal{O}$ is the
*classifying category* of the Olog — Spivak's standard framing. So our
$\mathcal{C}_M$ generalizes the Olog construction to the case where the
"morphisms" are operational steps (opcodes, generation moves) rather
than purely declarative relations.

### 2.3 Compiled transformers as a category

Fix a transformer architecture (number of layers, attention heads, FFN
width, hidden dimension $d$). A **compiled transformer specification**
is a triple $(E, A, \Phi)$ where:

- $E: T \to \mathbb{R}^d$ is an *embedding* assigning each type a
  residual-stream subspace representative.
- $A \in \{0, 1\}^{T \times T}$ is an *attention admissibility mask*
  predicate over query/key type pairs (lifted from a function
  $T \times T \to \{0,1\}$).
- $\Phi$ is a per-label *FFN gate-and-transition* assignment: for each
  $\ell \in L$, a pair $(g_\ell, \tau_\ell)$ of functions
  $\mathbb{R}^d \to \mathbb{R}$ (gate) and
  $\mathbb{R}^d \to \mathbb{R}^d$ (transition), realizable as
  rows of an FFN layer.

Specifications form a category $\mathsf{CompT}$ where morphisms are
embedding-preserving simulations between specifications (we omit the
formal definition; it is the standard simulation category for
parameterized transformers).

### 2.4 The compilation functor

A **TLTS-compilation** of $M = (T, L, \delta, t_0)$ is a functor

$$
F_M: \mathcal{C}_M \longrightarrow \mathsf{CompT}
$$

obtained by the following recipe:

1. (Type embedding) Choose $E$ injective on $T$ with images sufficiently
   separated under the chosen residual-stream metric to avoid bleed.
2. (Attention mask) Set $A(t_q, t_k) = 1$ iff there exist
   $\ell$ and a chain of length $\leq k$ in $\delta$ from $t_q$ to $t_k$
   for some bound $k$ (we take $k = \infty$ for full reachability).
3. (Transition enforcement) Realize $\delta$ at one or more of three
   enforcement loci (§2.5). The functor $F_M$ does not specify which
   locus; it specifies only that $\delta$ is honored. Different choices
   give different cost profiles and different architectural implications,
   but the same soundness.

A compiled transformer **realizes** $F_M$ if its forward pass on a token
sequence $\ell_1, \ell_2, \ldots, \ell_n$ produces residuals
$E(t_0), E(t_1), \ldots, E(t_n)$ where each $(t_{i-1}, \ell_i, t_i) \in
\delta$.

### 2.5 Three enforcement loci for $\delta$

The transition relation can be enforced at three distinct points in
the inference pipeline. They are not exclusive — production systems
will combine them — but they have different cost regimes and admit
different architectural commitments.

| Locus              | Where $\delta$ is checked                             | Per-step cost          | Architectural cost                          |
|--------------------|-------------------------------------------------------|------------------------|---------------------------------------------|
| **In-FFN**         | During forward pass; FFN row $\ell$ fires only when admissible | $O(1)$ in the FFN row count | Requires recompiling FFN weights            |
| **Pre-decoder**    | At sampling; logit mask restricts token distribution to admissible labels | $O(\|L\|)$ mask construction per step | Drop-in for any trained model               |
| **Post-hoc**       | After generation; verifier audits the trace against $\delta$ | $O(n)$ for length-$n$ trajectory | None on the model; external checker         |

The **in-FFN** locus is what the WASM-VM construction described in
§3.1 uses; the gates *are* the enforcement. This is the most aggressive
choice — once compiled, the model cannot violate $\delta$ regardless
of what tokens it is fed — at the cost of requiring the weights to be
constructed (or fine-tuned) for the specific TLTS.

The **pre-decoder** locus is the natural relational-$\delta$ primitive
and corresponds to the well-established literature on constrained
decoding [^Outlines] [^Geng2023]. At each generation step, the
admissible label set for the current type-state is computed from
$\delta$ and the logits over $L$ are masked accordingly. The model
itself remains unconstrained; the enforcement is at the
sampling boundary. This is what
[`proof_guided_generation.ConstrainedDecoder`](../proof_guided_generation.py)
already implements: the proof object dictates the admissible set per
step, the decoder masks logits to that set.

The **post-hoc** locus does not enforce $\delta$ at all — it audits
that $\delta$ was respected. Used alone, it implies a verify-and-retry
loop that is wasteful: every invalid generation is wasted compute, and
some inputs may exhibit no admissible completion under the model's
distribution, causing the loop to hang. Used in combination with one
of the first two loci, the verifier becomes a monitoring artifact
rather than a flow-control gate (§5.3).

The **architectural prediction** of the framework, then, is: use
in-FFN compilation for functional sub-fragments (where the choice of
successor is unique and FFN compilation is cheap), pre-decoder
constraints for relational fragments (where multiple successors are
admissible and the model must choose), and post-hoc verification as a
shipped audit certificate. The experiment of §4 is designed to compare
all three.

### 2.6 Soundness

**Definition (Trajectory-soundness).** A compiled transformer $C$ is
*trajectory-sound* with respect to $M$ if for every input token sequence
$\ell_1, \ldots, \ell_n$ such that $C$ produces a corresponding
residual trace, that trace decodes to a valid trajectory of $M$.

**Proposition 2.1.** *Any compiled transformer realizing $F_M$ is
trajectory-sound with respect to $M$, provided $E$ is injective and $g_\ell$
gates are sharp* (i.e., the per-label dispatch is exclusive).

*Proof sketch.* The attention mask $A$ blocks any cross-token mixing
that violates type reachability; the FFN gates ensure that only the
unique $\ell$-row fires per step; injectivity of $E$ guarantees that a
recovered residual decodes to a single type. Composition along the
forward pass therefore traces an admissible path in $\delta$. ∎

This is the formal counterpart of the soundness theorem stated
informally in [`proof_guided_generation.py:683`](../proof_guided_generation.py)
("if T is generated via proof P against Olog O, all claims in T are
valid in O"). The categorical statement makes precise *what kind* of
correctness this is: it is faithfulness of the functor $F_M$ on
morphisms, not on objects, and it does not say anything about
*completeness* (whether all valid trajectories are realizable).

### 2.7 Why category theory rather than just sets

The categorical setup buys three things over a purely set-theoretic
statement:

1. **Composition.** Multi-step trajectories are morphism composition, not
   ad-hoc loop unrolling. Verification of an n-step trajectory factors
   through verification of each step.
2. **Functoriality.** The same theorem statement covers small toy
   ontologies and large compiled VMs. Changing the underlying $M$ requires
   re-instantiating $E$, $A$, $\Phi$ — not re-stating the soundness theorem.
3. **Equational reasoning.** When the underlying Olog has equations
   (e.g., commutative diagrams encoding "two routes lead to the same
   conclusion"), the categorical statement extends naturally; the
   classifying category quotients by these equations and we ask whether
   $F_M$ respects them.

We do not pursue the equational case in this paper but flag it as the
natural next mathematical frontier.

---

## 3. Subsumption: procedural compilation is functional TLTS-compilation

### 3.1 Mapping the WASM-VM construction

Construe a deterministic WebAssembly machine as a TLTS:

- $T$ = WASM machine state types (stack-shape × register-set tags).
- $L$ = WASM opcodes ∪ immediate values.
- $\delta$ = WASM operational semantics (single-valued).

The construction described in [^Percepta2026] instantiates the
compilation functor as follows:

| Recipe step      | Procedural instantiation                                                  |
|------------------|---------------------------------------------------------------------------|
| Type embedding   | Stack/register encoding in residual stream                                |
| Attention mask   | Hull-backed key/value cache; query $[1,0]$ retrieves most-recent write    |
| FFN gate         | Sharp opcode classifier: $g_\ell(\text{state}) \approx [\text{op}=\ell]$  |
| FFN transition   | Hard-coded $\tau_\ell$: WASM small-step operational semantics             |

The hull trick is a *retrieval optimization* available because $\delta$
is functional: when you know the answer is unique, you can index for it
geometrically rather than soft-attending over all keys. Formally, the
attention mask can be replaced by a deterministic argmax over a 2D point
set; for any fixed query direction the answer lies on the upper convex
hull, giving $O(\log h)$ retrieval where $h$ is the hull size.

### 3.2 Mapping the Olog / proof-guided construction

| Recipe step      | Ontological instantiation                                                 |
|------------------|---------------------------------------------------------------------------|
| Type embedding   | `embed_tokens()` adds Olog-type embedding to base ([`ontological_attention.py:266`](../src/ontological_attention.py:266)) |
| Attention mask   | Reachability mask $M_G$ from Olog transitive closure ([`ontological_attention.py:166`](../src/ontological_attention.py:166)) |
| FFN gate         | (Currently external) Proof-search dispatches to morphism rules            |
| FFN transition   | (Currently external) `ProofGuidedGenerator` applies morphism step         |

Note the asymmetry: in the procedural case, $\Phi$ is fully baked into
FFN weights. In the current ontological case, $\Phi$ lives outside the
forward pass — proof search runs as a separate module. This is the
"missing rung" identified in our prior correspondence note: the
ontological side has not yet pushed $\Phi$ down into FFN rows.

### 3.3 The subsumption claim

**Theorem 3.1 (informal).** *Let $\mathsf{Comp}(M)$ be the class of
compiled transformers realizing $F_M$. Then:*

- *Procedural (WASM-style) compilation $\subseteq \mathsf{Comp}(M)$ for
  $M \in \mathsf{TLTS}_{\mathrm{func}}$.*
- *Ontological (Olog-style) compilation $\subseteq \mathsf{Comp}(M)$ for
  $M \in \mathsf{TLTS}_{\mathrm{rel}}$.*
- *Since $\mathsf{TLTS}_{\mathrm{func}} \subsetneq \mathsf{TLTS}_{\mathrm{rel}}$,
  the procedural case is a strict specialization of the ontological case.*

A formal version requires fixing a parameterization of $E$, $A$, $\Phi$
(precision, gate sharpness, residual-stream geometry); we treat that as
the natural next theoretical task.

### 3.4 What the subsumption is *not*

The subsumption does not say that running WASM in a transformer is
equivalent to running proof-guided generation over an Olog. The two
underlying TLTSs differ. What it says is that the *compilation pattern*
is shared, that the procedural primitive (FFN-encoded $\delta$) is
applicable to the deterministic sub-fragments of an Olog, and that the
ontological mask construction generalizes to the relational case where
the procedural hull-trick does not apply.

This is the architectural prediction of the framework: hybrid
compilation is possible. Functional sub-fragments of an Olog
(chain-shaped paths, single-outgoing-edge nodes) admit procedural-style
FFN compilation; the general graph requires the ontological mask. We
operationalize this prediction in §4.

---

## 4. Operationalizing the correspondence: experimental design

### 4.1 Hypothesis

**H1.** Compiling the deterministic sub-Olog of a domain ontology into
FFN rows achieves the same trajectory-soundness as proof-guided
generation, with substantially lower per-step latency.

**H2.** Hybrid compilation (FFN where $\delta$ is functional, masked
attention where $\delta$ branches) Pareto-dominates either pure approach
on the latency × soundness frontier.

### 4.2 Setup

We use the existing e-commerce Olog from
[`ontological_attention.py:720`](../src/ontological_attention.py:720)
(Customer → Cart → Checkout → Payment → Order → Delivery, with three
side branches). We identify the **functional fragment** as the longest
chain with single outgoing edges per node (Customer → Cart →
Checkout → Payment → Order). Four system variants, one per
enforcement-locus combination:

- **(A) Standard transformer** — no mask, no compilation, no decoder
  constraints. Baseline.
- **(B) Reachability-masked attention** — current
  [`OntologicalAttention`](../src/ontological_attention.py); enforces $\delta$
  at the attention layer (a relaxation of the in-FFN locus).
- **(C) Compiled-FFN hybrid** — functional fragment compiled into FFN
  rows à la §2.5 in-FFN; relational remainder uses the (B) mask.
- **(D) Pre-decoder constrained decoding** — model unchanged from (A);
  logits masked at sample-time to the admissible-label set computed
  from $\delta$. Built on
  [`ConstrainedDecoder`](../proof_guided_generation.py:475).

(D) is the cheapest off-the-shelf option: zero architectural change,
just a logit mask. (C) is the most aggressive: deterministic forward
pass on the functional fragment. (B) sits in between. (A) is the
floor.

### 4.3 Compilation procedure for (C)

For each morphism $\ell: t_1 \to t_2$ in the functional fragment:

1. Allocate an FFN row indexed by $\ell$.
2. Set the input gate $g_\ell$ to a learned (or analytically
   constructed) classifier firing iff the residual encodes $E(t_1)$ AND
   the current token is $\ell$.
3. Set the output transition $\tau_\ell$ such that
   $\tau_\ell(E(t_1)) = E(t_2)$, with off-target inputs producing zero.

Concretely, with one-hot type embeddings of dimension $|T|$, this is a
projection matrix per morphism. A scaffold implementation is in
[`Percepta_Transformer_VM/experiment_compiled_subolog.py`](../Percepta_Transformer_VM/experiment_compiled_subolog.py).

### 4.4 Metrics

For each variant on a held-out evaluation set of 1000 sequences:

- **Soundness rate** — fraction of generations whose trajectory is
  admissible under the Olog (auto-checked via the verification protocol
  of §5).
- **Per-step latency** — wall time for one forward step on a fixed CPU
  reference (Apple M-series, single-thread).
- **Hallucination rate** — using the existing
  [`results/attention_ablation`](../results/) protocol.
- **Trajectory length** — distribution of generations to detect
  degenerate-short outputs.

### 4.5 Predictions

- (B) vs (A): replicate the existing 0.21 → 0.000 invalid-token weight
  drop reported in [`00_structure_of_clear_thinking.md:46`](../blog_handoff_oasis_cloud/00_structure_of_clear_thinking.md).
- (D) vs (A): match (B) on soundness with no architectural change to
  the underlying transformer; expected fluency penalty on highly
  branching parts of the Olog (cf. §7.1).
- (C) vs (B): same soundness, lower latency on the functional-fragment
  fraction of trajectories.
- (C) vs (D): comparable soundness on the functional fragment; (C)
  wins on per-step cost, (D) wins on architectural simplicity.
- All three (B), (C), (D) jointly dominate (A) on soundness ×
  hallucination rate; the choice among them is a cost/locus tradeoff.

### 4.6 Preliminary results: synthetic-prior baseline

Pending the full trained-model experiment, we report results from a
synthetic-prior harness that isolates constraint mechanics from model
quality. The harness simulates an unconstrained "model" as a hand-
crafted distribution $p_M(\ell \mid t)$ over labels per type; variants
(A), (C), (D) are sampling policies that consume this prior. Variant
(B) requires a real attention forward pass and is deferred. Full
methodology and reading guide:
[`Percepta_Transformer_VM/experiment_results.md`](../Percepta_Transformer_VM/experiment_results.md).

We test two priors. **GOOD** places 80% of mass on admissible labels
per type (a model that broadly knows the domain). **BAD** places 70%
of mass on a randomly chosen "favorite" label per type, which is
typically inadmissible (a model with strong but wrong priors).

| Prior | Variant | Soundness | logP/traj | KL/step | µs/step |
|-------|---------|-----------|-----------|---------|---------|
| GOOD  | A       |   48.1%   |  −2.54    | 0.22    | 11.0    |
| GOOD  | C       |  100.0%   |  −1.48    | 0.22    |  6.0    |
| GOOD  | D       |  100.0%   |  −1.47    | 0.22    |  8.5    |
| BAD   | A       |    4.2%   |  −1.95    | 1.12    |  7.9    |
| BAD   | C       |  100.0%   |  −7.42    | 1.92    |  5.5    |
| BAD   | D       |  100.0%   |  −7.33    | 1.91    |  8.8    |

Three findings:

1. **Soundness is by construction.** (C) and (D) achieve 100%
   regardless of prior quality; (A) lucks into valid trajectories at
   a rate proportional to prior alignment.
2. **Fluency cost scales with prior–$\delta$ misalignment.** Under
   GOOD prior the constraint is approximately free (logP barely
   changes); under BAD prior the realized trajectories are ~5.5 nats
   less likely under the model's own distribution. This is the
   precision–fluency frontier (§7.1) made quantitative.
3. **(C) wins on latency by ~30%** over (D) on this Olog because
   deterministic steps on functional types skip the sampling pipeline.
   Magnitude depends on the functional/relational ratio of the
   target ontology.

The KL-per-step diagnostic emerges as a deployment-friendly signal of
over-constraint risk: low KL means the prior is already concentrating
on admissible labels and the constraint is doing little; high KL
means substantial reshaping is happening. A production deployment
can monitor this in real time.

The synthetic harness validates the framework's qualitative
predictions but does not substitute for the trained-model experiment.
Variant (B), Olog topology sweeps, and constraint-aware fine-tuning
are flagged in the results document as the next concrete experiments.

### 4.7 A theoretical finding from the harness: (B) is insufficient on its own

Adding a sampling-time projection of attention-layer reachability
masking — call it (B′), which admits any label whose target type is
reachable from the current state — produces a striking result:

| Prior | (A)   | (B′)  | (C)   | (D)   |
|-------|-------|-------|-------|-------|
| GOOD  | 48.1% | 61.5% | 100%  | 100%  |
| BAD   |  4.2% |  4.3% | 100%  | 100%  |

(B′) is barely better than (A). Reachability admits the *destination*
without requiring the *step* to exist as a δ-edge: from a current
state $t$, (B′) admits a label $\ell$ if some reachable type is
targeted by $\ell$ — but $\ell$ may have no edge originating at $t$.
The trajectory then breaks at the next step.

The implication for the framework: **attention-layer reachability
masking is necessary but not sufficient for trajectory soundness**.
It restricts information flow inside the model, which is the right
job for the attention mask, but it does not constrain the output
distribution to admissible-from-current-state labels. The
production
[`OntologicalAttention`](../src/ontological_attention.py) implements (B);
the production system inherits soundness not from this mask but from
[`ConstrainedDecoder`](../proof_guided_generation.py:475), which
implements (D) at the decoder. The two operate at different layers
and serve different purposes.

This was easy to miss before the framework drew the distinction
explicitly. Reachability is for information flow; direct-edge
admissibility is for output validity. Production systems aiming for
soundness need both.

**Cyclic-Olog confirmation.** Adding a cycle to the e-commerce TLTS
(Delivery → Customer) drives reachability to 100% across all type
pairs. (B′) soundness collapses from 65% to 8% under GOOD prior and
from 3.9% to 0.0% under BAD; (D) remains 100% on both
([`experiment_cyclic_stress.py`](../Percepta_Transformer_VM/experiment_cyclic_stress.py)).
The framework predicts this limit: as Olog connectivity grows,
reachability admits more and the constraint does less work, until
under near-universal reachability the mask is doing nothing at all.

**Validation against production code.** A structural integration
test ([`experiment_real_attention_b.py`](../Percepta_Transformer_VM/experiment_real_attention_b.py))
bridges the cyclic TLTS into an `OlogGraph` and runs the production
[`OntologicalAttention`](../src/ontological_attention.py) forward pass.
The mask admits 24 reachable-only (query, key) pairs against only 6
δ-direct-edge pairs in a 6-token trajectory window — a 4× leakage
ratio. Forward-pass attention mass under random init lands 66.7% on
reachable-only pairs and only 16.4% on δ-direct-edge pairs. The
(B′) sampling-time projection is faithful to what the production
layer is actually doing.

### 4.8 Topology sweep: where the (C)/(D) tradeoff crosses

A parameterized TLTS family (length-5 chain plus
$k \in \{0, \ldots, 5\}$ optional side-branches) lets us trace the
(C) vs (D) latency tradeoff as a function of functional ratio. The
latency ratio (C)/(D) under GOOD prior:

| k    | functional ratio | (C)/(D) latency ratio |
|------|-----------------:|----------------------:|
| 0    | 1.00             | **0.39**              |
| 1    | 0.80             | 0.67                  |
| 2    | 0.60             | 0.80                  |
| 3    | 0.40             | 0.91                  |
| 4    | 0.20             | 0.99                  |
| 5    | 0.00             | 1.19                  |

Crossover at functional ratio ≈ 0.2. Below that, (C)'s
deterministic-step shortcut on functional types pays for its
bookkeeping overhead and (C) is up to ~2.5× faster than (D). Above
that, the bookkeeping overhead costs more than it saves.

**Deployment heuristic.** Use (C) when the functional fragment
exceeds ~20% of the non-terminal type set; otherwise use (D). The
threshold is implementation-specific and should be re-derived per
codebase, but the qualitative pattern is robust.

### 4.9 What success and failure look like

If (C) matches (B) on soundness but loses on latency, the procedural
primitive is strictly worse for ontological tasks and we should report
that honestly. If (C) gains on latency but loses on soundness, the
gates aren't sharp enough — a precision/numerical issue, not a
framework issue. If (C) wins on both, the hybrid claim is supported.

For (D), the diagnostic is fluency: if pre-decoder constraints match
the others on soundness but produce stilted or off-topic text relative
to (A), the precision-fluency frontier (§7.1) is biting. Either the
ontology is too restrictive for the task, or constraint-aware
fine-tuning (§7.1) is required to close the gap.

---

## 5. Post-hoc verification: turning the forward pass into an audit artifact

This section is the contribution most distinctive to the framework, and
the one most directly responsive to the question of whether claims like
the WASM-in-a-transformer construction are *checkable after the fact*.

### 5.1 The verification problem

Given:

- A compiled transformer $C$ with claimed compilation specification
  $(E, A, \Phi)$ for some TLTS $M$,
- An input sequence $x$,
- The forward-pass trace $h_0, h_1, \ldots, h_n$ produced by $C$ on $x$,

decide whether the trace decodes to a valid trajectory of $M$.

### 5.2 Protocol

The verification protocol has four steps. Each is decoupled from the
others — a verifier need not understand transformer internals to run them.

1. **Decode.** Apply $E^{-1}$ (or its nearest-neighbor approximation) to
   each $h_i$ to recover a candidate type $\hat{t}_i$. Reject if any
   $h_i$ falls outside the codomain of $E$ by more than a threshold
   $\epsilon$ (chosen at compile time).
2. **Recover labels.** From the input sequence and the model's
   token-emission log, extract the label sequence $\ell_1, \ldots,
   \ell_n$.
3. **Check transitions.** For each $i$, verify $(\hat{t}_{i-1}, \ell_i,
   \hat{t}_i) \in \delta$. This is a $|T|^2 |L|$-sized table lookup.
4. **Check mask coherence.** Inspect attention weight matrices: confirm
   that no mass exceeds threshold on $(q, k)$ pairs where
   $A(\hat{t}_q, \hat{t}_k) = 0$.

Pass requires steps 1, 3, 4 all succeed. Step 2 is a precondition.

### 5.3 What this verifies, and how pre-gating changes its role

The protocol verifies **trajectory-soundness on this specific input** —
it does not certify the transformer is sound on all inputs. It is the
analog of unit testing a compiler: a single input/output pair audited
against a reference semantics.

A subtle but important point: the verifier's *operational role* depends
on whether $\delta$ is also enforced upstream (in-FFN or at the
decoder).

- **Without upstream enforcement** (post-hoc only): the verifier is a
  flow-control gate. Generations that fail must be retried, regenerated,
  or discarded. This is wasteful and can hang on inputs that admit no
  admissible completion under the model's distribution. We do not
  recommend this configuration in production.
- **With upstream enforcement** (in-FFN or pre-decoder): the verifier
  should never fail except on numerical-precision drift. It becomes a
  monitoring artifact — a regression test that catches bugs in the
  upstream enforcement, and a public audit certificate that ships with
  the artifact. This is the configuration we recommend.

The same code runs in both cases; only the operational interpretation
changes.

For a stronger guarantee, one needs **whitebox certification**:
inspecting the actual weight matrices of $C$ to verify the gate
sharpness, mask injectivity, and embedding separation conditions of
Proposition 2.1. This is more expensive but provides a model-level
soundness certificate, not an input-level one.

The two together — input-level checks routinely + model-level
certification once at compile time — form a complete audit trail.

### 5.4 Reference implementation: the audit certificate

A reference implementation of the four-step verifier ships JSON audit
certificates ([`verification_certificate.py`](../Percepta_Transformer_VM/verification_certificate.py)).
A certificate contains:

- **Schema and version** for forward compatibility.
- **TLTS fingerprint** (sha256-prefix over sorted $T$, $L$, $\delta$)
  for detecting drift between the TLTS used at generation time and
  the one held by the verifier at audit time.
- **Per-step record**: $(t_{\text{in}}, \ell, t_{\text{out}})$ plus
  diagnostics (`in_delta`, `prior_argmax`, `forced`, `masked_kl`).
- **Summary**: length, `all_steps_in_delta`, forced-step count, mean
  KL/step, $\log p$ under prior.

A separate `verify_certificate(cert_json, tlts)` function consumes
the JSON and the TLTS, and returns pass/fail with reasons. It does
not need the model, the weights, or any deep-learning framework.

The reference implementation passes a battery of adversarial checks:
direct tampering with a state_out (e.g., to a type not in $T$) is
caught by the per-step δ lookup; TLTS drift (verifier holds a
modified $\delta$) is caught by both the fingerprint mismatch and
per-step failures.

This is what we propose as the publication norm: ship the JSON
certificate alongside outputs; ship the TLTS spec alongside the
weights; require the verifier to be runnable without the model.
Doing so closes the verifiability gap that currently makes
"transformer as computer" claims hard to evaluate.

### 5.5 What is and isn't verifiable in published claims

**Verifiable, given the artifact:**

- The compiled weights themselves (run the protocol).
- Reproducibility on declared inputs (deterministic forward pass).
- Concordance with a reference implementation of $M$ on overlapping
  inputs (run WASM separately, compare traces).

**Not verifiable from a paper alone:**

- Performance numbers (require hardware reproduction).
- Sound-on-all-inputs claims (require model-level certification, which
  requires the weights *and* a precision argument that may or may not
  be provided).
- Claims of zero hallucination "by construction" (require gate
  sharpness analysis that is rarely provided).

We argue this last gap is what makes published "transformers as
computers" claims hard to evaluate. A paper that reports the
construction without the per-row gate sharpness analysis describes a
compilation that *should* work but provides no certificate that the
specific weights *do* work. The verification protocol of §5.2 is what
closes that gap — it should be considered a publication norm for
TLTS-compilation work.

### 5.6 Application to existing claims

For the WASM-in-a-transformer construction summarized in
[^Percepta2026]: the technical primitives (FFN-as-dispatch,
hull-as-indexed-memory) admit verification under §5.2 *if* the
authors release weights, declared $\delta$ (the WASM subset), and a
declared $E$. Without these, the construction is plausible but
unverifiable.

For our own ontological attention work: the attention mask is
analytically derivable from the Olog ([`ontological_attention.py:166`](../src/ontological_attention.py:166)),
so step 4 is fully auditable today. Steps 1 and 3 require
$E^{-1}$ and a $\delta$-table; we have both. The protocol is therefore
implementable against our existing artifact and produces a per-input
audit log. We treat building this verifier as the immediate next
deliverable.

---

## 6. Related work

**Compiled transformers.** Tracr [^Lindner2023] compiles RASP programs.
The ALTA framework [^Shaw2024] generalizes RASP. Earlier theoretical
work [^Pérez2021] [^Bhattamishra2020] establishes Turing-completeness of
attention-based architectures. The construction summarized in
[^Percepta2026] (if the artifact exists as described) is a more
aggressive instance of this lineage.

**Type-safe attention and ontology-guided generation.** Work on
constrained decoding (Outlines, GBNF in llama.cpp, Picard for SQL) is a
weaker, syntactic version of what an Olog mask achieves semantically.
The closest categorical-DL antecedent is Spivak's Olog program
[^Spivak2014]. Our prior work [`ontological_attention.py`,
`proof_guided_generation.py`] sits in this tradition.

**Verification.** ProofWriter and related symbolic-NLI lines audit
generated text against a knowledge base, but post-hoc, not via forward-pass
trace inspection. Mechanistic interpretability work
[^Conmy2023] [^Nanda2023] inspects circuits in trained models; we
inspect the forward pass against a declared compilation specification,
which is a different problem.

**The novel claim.** The novelty is not any of the three components in
isolation but the unification: stating Olog-attention and VM-compilation
as instances of one functor; deriving the verification protocol from the
functoriality; and proposing hybrid compilation as the natural
architectural consequence.

---

## 7. Discussion: over-constraint, fluency, and model size

The framework as stated promises soundness. It is silent on a question
practitioners will reasonably ask first: **does enforcing $\delta$ hurt
output quality, and if so, how much?** This section treats two
intertwined sub-questions: when over-constraint costs fluency, and
whether TLTS-compilation enables smaller models or requires larger
ones.

### 7.1 The precision–fluency frontier

Constraining a generative model's output distribution is a
precision–recall tradeoff in another guise. Let $p_M(\cdot \mid \text{ctx})$
be the unconstrained model distribution and $\mathcal{A} \subseteq L$
be the admissible set computed from $\delta$ at a given step. The
constrained distribution is

$$
p_M^{\mathcal{A}}(\ell \mid \text{ctx}) = \frac{p_M(\ell \mid \text{ctx}) \cdot \mathbf{1}[\ell \in \mathcal{A}]}{\sum_{\ell' \in \mathcal{A}} p_M(\ell' \mid \text{ctx})}
$$

Five well-documented pathologies follow when $p_M$ and $\mathcal{A}$
are misaligned:

1. **Mass-redistribution drift.** When the model's preferred token is
   inadmissible, mass redistributes to a less-preferred but admissible
   one. Each downstream step conditions on having taken the
   redistribution, compounding KL divergence from natural generation.
2. **Premature lock-in.** The locally-best admissible token at step $t$
   may close off paths that would have produced better completions at
   $t + k$. Greedy constrained decoding is especially vulnerable.
3. **Confidence collapse.** When no admissible token has substantial
   $p_M$ mass, the within-$\mathcal{A}$ distribution flattens toward
   uniform. Output becomes effectively random within the constraint.
4. **Dead-end states.** Some Olog topologies admit reachable type-states
   from which no admissible continuation has acceptable likelihood.
   Without backoff, the system stalls.
5. **Topic drift suppression.** Tight constraints prevent the model
   from following its prior to off-Olog material that would have been
   coherent and useful but is judged inadmissible.

These are not framework bugs; they are the tradeoff. Acknowledging
them, six mitigation strategies are available, ordered by intervention
strength:

| Strategy                         | Mechanism                                                                       | Cost               | Preserves soundness? |
|----------------------------------|---------------------------------------------------------------------------------|--------------------|---------------------|
| Olog enrichment                  | Add morphisms / equational identifications so $\mathcal{A}$ is larger           | Ontology design    | Yes                 |
| Beam search with constraint      | Track $K$ partial trajectories; let constraints shape the future                | $K\times$ inference | Yes                 |
| Multi-proof generation + rerank  | Find $K$ proofs; verbalize each; pick highest unconstrained $p_M$               | $K\times$ proof search | Yes                 |
| Hierarchical constraint staging  | Coarse type-level constraints early, fine morphism-level constraints late       | Mild               | Yes                 |
| Backoff / abstention             | When admissible-set entropy too low or top-$\mathcal{A}$ probability below $\theta$, emit "I don't know" | None | Yes — by trading recall for precision |
| Soft constraints                 | Replace $-\infty$ with $-\alpha$ on inadmissible logits                         | None               | **No** — relaxes hard guarantee |
| Constraint-aware fine-tuning     | Train on TLTS-constrained outputs to align $p_M$ with $\mathcal{A}$             | Training run       | Yes (if soundness loss is included) |

We propose constraint-aware fine-tuning as the strongest fix, and
backoff as the cheapest. The former closes the KL gap between $p_M$
and the constraint geometry; the latter trades coverage for quality
without violating any guarantees. Soft constraints are the seductive
trap: they look free but cost soundness, and we recommend against
them in domains where soundness is the reason TLTS-compilation was
chosen.

**Is over-constraint avoidable?** Not entirely. The frontier exists.
Tight $\delta$ → strong soundness, weak fluency; loose $\delta$ →
strong fluency, weak soundness. The right framing is not to escape
the frontier but to choose the right point on it for each
application. Medical, legal, financial: the soundness corner. Open-
domain chat: the fluency corner. Most enterprise applications:
somewhere along the curve, with constraint-aware fine-tuning closing
as much of the gap as possible.

### 7.2 Implications for model size

A second question: does the framework enable smaller models, or
require larger ones?

The honest answer factorizes: **smaller models become viable for
domain tasks; general-purpose tasks still scale with capacity.**

**Why smaller becomes viable.**

1. *Capacity offload.* Standard transformers spend a substantial
   fraction of capacity learning what *not* to say in domain context —
   distributional regularities about which compositions of types and
   relations are unlikely. TLTS-compilation moves that knowledge into
   the Olog. The model can spend its capacity on linguistic fluency
   and within-$\mathcal{A}$ selection rather than on memorizing domain
   structure.
2. *Tracr-scale evidence.* Procedural compilation succeeds at toy
   model sizes; the WASM-VM construction works because the skill is in
   the compilation, not in scale. Functional-$\delta$ tasks need
   precision, not capacity.
3. *Constrained decoding empirics.* JSON-mode and grammar-constrained
   decoding work near-perfectly with very small models on structured
   tasks. The constraint replaces what would otherwise need to be
   learned.
4. *Distillation path.* Constraint-aware fine-tuning on outputs from a
   larger model gives a small model that operates only within the
   admissible region of the larger model's distribution. We expect
   substantial size reduction for domain-specific deployment via this
   route.

**Where larger remains necessary.**

1. *Within-constraint discrimination.* For relational $\delta$ with
   many admissible successors at each step, the model still has to
   choose well among them. Discrimination quality scales with
   capacity even when the admissible set is small.
2. *Open-domain fluency.* The Olog cannot capture everything. General
   chat, creative writing, and open-domain reasoning still benefit
   from scale because no realistic ontology covers the surface of
   what users will ask.
3. *Out-of-Olog robustness.* When the input strays from the ontology's
   coverage, the model must degrade gracefully. Larger models do this
   better.

**Net architectural prediction.** TLTS-compilation enables a regime
shift in *domain-specific* deployment: ontology-rich, weights-thin
models replacing ontology-thin, weights-heavy models for tasks where
the domain is well-specified. For *general-purpose* deployment, the
scaling curve is unchanged but the framework provides a cleaner
factoring of what the weights are for. The "we need bigger models"
argument loses force for enterprise applications in roughly the same
proportion as the relevant domain admits a clean Olog.

This is not the only research direction the framework opens, but it
is the most economically consequential one: TLTS-compilation is a
candidate substitute for scale in any domain where structure is
already known.

### 7.3 Limitations

1. *The functor account is informal.* Promoting Theorem 3.1 to a
   formal theorem requires fixing residual-stream geometry,
   gate-sharpness precision, and mask precision parameters. This is
   real work; we have not done it.
2. *The empirical predictions are moderate.* Hybrid compilation is
   likely to win modestly on latency for the functional fragment of
   an Olog and not at all for fully relational graphs. We expect (C)
   to beat (B) by 1.5–3× on latency over the functional fragment,
   less on whole-task throughput. (D) is likely to match (B)/(C) on
   soundness with a fluency penalty whose magnitude depends on the
   ontology.
3. *Verification scales with $|T|^2|L|$.* For a 23-type Olog this is
   trivial; for a million-type knowledge graph the transition table
   is the bottleneck, and lazy evaluation against actually-emitted
   labels is required.
4. *Constraint-aware fine-tuning is unstudied here.* §7.1 names it as
   the strongest mitigation, but we do not run it. Doing so is the
   next paper.

### 7.4 What the framework predicts that prior work doesn't

1. **The missing rung** — that ontological generation can be brought
   into the forward pass via FFN compilation of functional
   sub-fragments.
2. **Three loci, one functor** — that in-FFN compilation, constrained
   decoding, and post-hoc verification are not competing approaches
   but three enforcement points within the same framework, with
   complementary cost regimes.
3. **Reachability ≠ admissibility for output.** Attention-layer
   reachability masking (B) is necessary but not sufficient for
   trajectory soundness; direct-edge admissibility (D) is also
   required at the decoder. The framework makes this distinction
   explicit; in the synthetic harness it produces a 38-percentage-
   point soundness gap on a chain Olog and collapses (B′) to within
   1 point of (A) on a cyclic Olog. Validation against production
   code confirms: 4:1 reachable-only-to-direct-edge mask admittance
   ratio, 66.7% of forward-pass attention mass on reachable-only
   pairs. Production systems that enforce δ only at the attention
   layer are unsound.
4. **The verification protocol** — that a TLTS-compiled transformer
   admits a four-step audit independent of model internals, and that
   pre-gating turns this verifier from a flow-control gate into a
   monitoring artifact. The certificate is shippable JSON;
   verification needs only the TLTS spec.
5. **Reproducibility norm** — that publishing a "transformer as
   computer" claim should include the gate-sharpness analysis and
   compilation specification without which the construction is
   unverifiable.
6. **Capacity offload as size leverage** — that domain-specific
   deployment can move structure from weights into the Olog,
   substituting ontology engineering for scale in well-specified
   domains.
7. **Topology-aware locus selection** — that the (C)/(D) latency
   tradeoff has a measurable crossover (≈ 0.2 functional ratio in
   our experiments) usable as a deployment heuristic.

---

## 8. Conclusion

TLTS-compilation is a unifying lens for two threads — type-safe
attention and procedural transformer compilation — that have so far
been treated as unrelated. The lens predicts a hybrid architecture, a
verification protocol, and a publication norm. Each is concrete enough
to attempt; none requires waiting for new mathematics. The immediate
next steps are to instantiate the experiment of §4 and the verifier of
§5 against the existing ontological attention codebase.

---

## References (placeholder; to be made canonical for submission)

[^Spivak2014]: Spivak, D.I. (2014). *Category Theory for the Sciences.* MIT Press. Chapter on Ologs.
[^Lindner2023]: Lindner, D., Kramár, J., et al. (2023). Tracr: Compiled Transformers as a Laboratory for Interpretability. NeurIPS.
[^Pérez2021]: Pérez, J., Marinković, J., Barceló, P. (2021). On the Turing Completeness of Modern Neural Network Architectures. ICLR.
[^Bhattamishra2020]: Bhattamishra, S., Patel, A., Goyal, N. (2020). On the Computational Power of Transformers and Its Implications in Sequence Modeling. CoNLL.
[^Shaw2024]: Shaw, P., et al. (2024). ALTA: Compiler-Based Analysis of Transformers. (placeholder citation; verify before submission)
[^Conmy2023]: Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS.
[^Nanda2023]: Nanda, N., et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability. ICLR.
[^Percepta2026]: "Can LLMs Be Computers?" (claimed March 2026 publication, percepta.ai). **Note for submission: verify this paper exists and update citation; if it does not exist or differs from our summary, reframe §3.1 against the actual primary literature on compiled transformers (Tracr, ALTA) without the Percepta-specific framing.**
[^Outlines]: Willard, B.T., Louf, R. (2023). Efficient Guided Generation for Large Language Models. arXiv:2307.09702.
[^Geng2023]: Geng, S., et al. (2023). Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. EMNLP.

---

## Appendix A — Submission strategy

| Venue          | Deadline (2026)   | Fit                                                                  | Action                                                                 |
|----------------|-------------------|----------------------------------------------------------------------|------------------------------------------------------------------------|
| ACT 2026       | Mar 23 (passed)   | ⭐⭐⭐⭐⭐ formalism                                                       | Aim ACT 2027 with formalized Theorem 3.1                                |
| COLM 2026      | Mar 31 (passed)   | ⭐⭐⭐⭐⭐ LLM angle                                                       | Skip, target COLM 2027                                                  |
| NeSy 2026      | ~May              | ⭐⭐⭐⭐⭐ neuro-symbolic                                                  | **Primary near-term target** if deadline open                           |
| NeurIPS 2026   | ~May 15           | ⭐⭐⭐ general ML                                                        | Possible if §4 experiment runs cleanly by deadline                      |
| ICLR 2027      | Sep/Oct 2026      | ⭐⭐⭐⭐                                                                  | **Primary medium-term target**: full empirical story + verifier         |
| ACT 2027       | ~Mar 2027         | ⭐⭐⭐⭐⭐                                                                 | Companion theory paper with formalized soundness + functoriality        |

Recommendation: split the work. NeSy/NeurIPS for the empirical
hybrid-compilation paper (this draft, §4 + §5 results). ACT 2027 for the
formal categorical paper (this draft, §2 + §3 fully formalized).
