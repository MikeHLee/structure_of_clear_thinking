---
marp: true
theme: default
backgroundColor: #1a1a2e
color: #ffffff
style: |
  section {
    font-family: 'Arial', sans-serif;
    display: flex;
    flex-direction: column;
    justify-content: center;
    font-size: 30px;
  }
  h1 {
    color: #4361ee;
    font-size: 60px;
    margin-bottom: 20px;
  }
  h2 {
    color: #4361ee;
    font-size: 48px;
    margin-bottom: 20px;
  }
  h3 {
    color: #a0a0a0;
    font-size: 36px;
  }
  p, li {
    line-height: 1.5;
  }
  strong {
    color: #4361ee;
  }
  code {
    background-color: #0d0d18;
    color: #a5b4fc;
    border-radius: 5px;
    padding: 2px 5px;
  }
  pre {
    background-color: #0d0d18;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #4361ee;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
  }
  th {
    color: #4361ee;
    border-bottom: 2px solid #4361ee;
    padding: 10px;
    text-align: left;
  }
  td {
    border-bottom: 1px solid #2a2a3e;
    padding: 10px;
  }
---

<!-- _class: lead -->
# From Statistical Patterns to Categorical Proofs
### Bridging Neural and Symbolic AI through Ontological Attention

**Mike Lee**
*February 2026*

---

# The Problem

* **LLMs Hallucinate**: They lack grounded semantics and "guess" based on statistics.
* **Symbolic AI is Brittle**: Hand-crafted rules don't scale or learn flexibly.
* **The Gap**: We have powerful learners (Neural) and reliable reasoners (Symbolic), but they don't talk to each other.

> **Neither alone is sufficient.**

---

# The Insight

**Attention mechanisms implicitly implement categorical semantics.**

* **Q (Queries)** = Rules ("What do I need?")
* **K (Keys)** = Graph Structure ("How are things connected?")
* **V (Values)** = Objects ("What is the data?")

*Transformers are accidental **graph neural networks** learning implicit ontologies.*

---

# Our Solution

**Make the implicit explicit.**

1. **Explicit Ontologies**: Define the "rules of the road" using Ologs (Ontology Logs).
2. **Proof-Guided Generation**: Use proof objects as blueprints for generation.
3. **Correct by Construction**: If a path cannot be proven, it cannot be generated.

---

# The Architecture

```
┌────────────────────────────────────────────────────────┐
│             ONTOLOGICAL INDUCTION ENGINE               │
├────────────────────────────────────────────────────────┤
│  ┌────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ Olog Core  │───▶│ Proof Engine │───▶│ Generator  │  │
│  └────────────┘    └──────────────┘    └────────────┘  │
│        │                  │                   │        │
│        ▼                  ▼                   ▼        │
│  ┌────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ Ontological│    │ Hallucination│    │   Proof    │  │
│  │ Attention  │    │  Detection   │    │   Traces   │  │
│  └────────────┘    └──────────────┘    └────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

# Demo: E-Commerce Ontology

**Goal**: Generate a valid order flow.

**Graph**:
`Customer` → `Cart` → `Checkout` → `Payment` → `Order` → `Delivery`

**Process**:
1. **User Request**: "Process order for User A"
2. **Proof Engine**: Finds valid path through `Cart` → `Payment`.
3. **Generator**: Fills in text *only* along the proven path.
4. **Result**: Guaranteed valid sequence. No "Free Gift" hallucination.

---

# Tested Ontologies
### Simple Domain Models for Embedding Validation

We validated our embedding methods on four distinct domain ontologies:

```
BUSINESS:   Customer -> Order -> Product -> Invoice -> Payment
ACADEMIC:   Student -> Course -> Professor -> Department -> Grade
HEALTHCARE: Patient -> Doctor -> Diagnosis -> Treatment -> Insurance
ECOMMERCE:  User -> Cart -> Item -> Checkout -> Payment -> Delivery
```

* **Goal**: Measure separation ratio between intra-domain and inter-domain types.
* **Result**: Types from the same domain cluster tightly, while distinct domains are pushed apart in the latent space.

---

# Results

| Metric | Result |
|--------|--------|
| **Embedding Separation** | **2.71** (Clear distinction) |
| **Invalid Transitions** | **100%** (42/42 blocked) |
| **STRICT Mode Accuracy** | **100%** (No hallucinations) |
| **Integration Tests** | **18/18** Passing |

---

# The Math

* **Category Theory**: Objects (Types) + Morphisms (Relations).
* **Curry-Howard Correspondence**:
  * **Proofs** ↔ **Programs**
  * We extend this to: **Proofs** ↔ **Generation Traces**
* **Attention Mask**:
  $M_{ij} = 0$ if $Type_i 	o Type_j$ is valid, else $-\infty$.

---

# Prior Art & Differentiation

* **Vaswani et al. (2017)**: Noted attention heads learn "semantic structure".
* **DeLong et al. (2024)**: Surveyed Neurosymbolic AI.
* **Our Novelty**:
  * We don't just *add* rules to embeddings.
  * We use **Proof Objects** as the *primary* generation control structure.

---

# Roadmap

1. ✅ **Core Engine**: Complete and tested.
2. ✅ **GPU Training**: Validated on Modal.
3. 🔄 **Real-World Experiments**: In progress.
4. 📝 **Publication**: 3 Papers, 4 Blog posts planned.

---

# Business Applications

1. **Auditable AI**: Every response has a derivation tree.
2. **Domain-Specific LLMs**: Inject expert ontologies (Legal, Medical).
3. **Compliance**: Provable correctness for regulated industries.

---

# Call to Action

* **Seeking**: Domain ontologies for validation.
* **Next**: Production pilots with real data.
* **Goal**: Trustworthy AI that **cannot** hallucinate.

