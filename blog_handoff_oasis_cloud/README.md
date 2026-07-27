# Oasis-X Dev Blog: Structure of Clear Thinking Series

**Publishing Schedule**: Aligned with NeurIPS / EMNLP / ICLR submission windows

---

## Blog Posts

| # | Title | Publish Date | Status |
|---|-------|--------------|--------|
| 0 | Structure of Clear Thinking | **Series Opener** | Draft |
| 1 | Why Your LLM Hallucinates (And How Category Theory Can Help) | **Jun 15, 2026** (w/ EMNLP) | Ready |
| 2 | Attention, But Make It Type-Safe | **May 15, 2026** (w/ NeurIPS) | Ready |
| 3 | From Proofs to Programs to... Text? | **Oct 1, 2026** (w/ ICLR) | Ready |
| 4 | Building an Auditable AI: A Complete Walkthrough | **Oct 15, 2026** (w/ ICLR) | Ready |

---

## Deployment Instructions

1. Copy markdown files to `oasis-cloud/src/blog/posts/` (create directory if needed)
2. Or place in root of blog for direct serving

```bash
# From ai_research directory:
cp -r topics/structure_of_clear_thinking/blog_handoff_oasis_cloud/*.md \
    /path/to/oasis-cloud/src/blog/posts/
```

---

## Series Overview

This 5-part dev blog series introduces **Ontological Induction**—a framework for eliminating LLM hallucinations using category theory, proof-guided generation, and type-constrained attention.

**Post 0** is the series opener: what we built, what we measured, and what it means. It's written for a lay audience and anchors on concrete experimental results.

Posts 1–4 dive into each research thread in technical detail, aligned with our paper submission deadlines.

### Target Audience
- ML engineers building production LLM systems
- AI safety researchers
- Developers interested in formal methods for AI
- Technical leaders evaluating hallucination-mitigation strategies

### Key Themes
1. **Post 0**: Series introduction + experimental results in plain language
2. **Post 1**: The hallucination problem + category theory intro
3. **Post 2**: Type-constrained attention implementation
4. **Post 3**: Curry-Howard correspondence extended to NLG
5. **Post 4**: Full tutorial from ontology to deployment

### Call to Action
Each post links to the open-source implementation on GitHub.

---

## Social Media Hooks

**Post 0 (Series Opener)**:
> "We taught an LLM to prove before it speaks. Invalid-token attention weight dropped to zero—not by training, by construction. Read the research diary: https://..."

**Blog 1 (June 15, w/ EMNLP)**:
> "Your LLM hallucinates because it lacks a type system for your domain. Category theory provides one. New dev blog on provably-grounded generation 🧵"

**Blog 2 (May 15, w/ NeurIPS)**:
> "What if transformer attention could only flow along valid semantic paths? We built type-constrained attention that blocks hallucinations architecturally. Here's how 🔒"

**Blog 3 (October 1, w/ ICLR)**:
> "Proofs aren't just for verification—they're construction blueprints. Extending Curry-Howard to natural language generation 📐"

**Blog 4 (October 15, w/ ICLR)**:
> "Full tutorial: Build an auditable AI that can't hallucinate. Ontology → Proof Engine → Constrained Generator → API. Code included 🛠️"

---

## File Manifest

```
blog_handoff_oasis_cloud/
├── README.md                           # This file
├── 00_structure_of_clear_thinking.md    # Series opener (new)
├── 01_why_llms_hallucinate.md          # Post 1 (w/ EMNLP)
├── 02_type_safe_attention.md           # Post 2 (w/ NeurIPS)
├── 03_proofs_to_text.md                # Post 3 (w/ ICLR)
└── 04_building_auditable_ai.md         # Post 4 (w/ ICLR)
```

---

## Key Results Cited in Post 0

| Result | Metric | Source |
|--------|--------|--------|
| Ontological attention firewall | Invalid-token weight: 0.213 → 0.000 | `results/attention_ablation.json` |
| Attention v2 (300 epochs) | Invalid-token weight: 0.295 → 0.000 | `results/attention_ablation_v2.json` |
| Contradiction detection (H¹) | +53 with 76 injected conflicts | `results/week4_evaluation_report.md` |
| Link prediction MRR | 0.3459 on FB15K-237 | `docs/EVALUATION_SUMMARY_HDC_SHEAF.md` |
| Separation ratio | 2.71× (easy negatives) | `handoffs/EMBEDDING_RESEARCH_MASTER_PLAN.md` |

---

*Prepared for Oasis-X Dev Blog · April 2026*
