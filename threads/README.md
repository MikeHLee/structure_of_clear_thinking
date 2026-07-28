# X/Twitter Thread Series

Public-communication track for the Structure of Clear Thinking results.
Conferences are deprioritized as a venue; distribution is **X threads +
blog posts (Ghost, mirrored to Substack)**, with arXiv for citable anchors.

Each thread directory contains:

- `thread.md` — the tweet-by-tweet copy, with alt text for every image and
  posting notes
- `generate_figures.py` — reproducible figure generation (numbers sourced
  from experiment result files, never hand-entered twice)
- `figures/` — the rendered PNGs, designed to be self-interpretable
  (takeaway in the title, every mark directly labeled)

## Planned series

| # | Thread | Source result | Status |
|---|--------|---------------|--------|
| 01 | Reachability masking is not enough | `Percepta_Transformer_VM/experiment_results.md` (loci comparison, cyclic stress, mask audit) | **Draft ready** |
| 02 | Receipts for AI: audit certificates you can verify without the model | `verification_certificate.py`, `sample_audit_certificate.json` | Planned |
| 03 | Teaching the model the rulebook: constraint-aware fine-tuning | Fine-tuning phase (upcoming — SOGB-style scale-up) | Blocked on training runs |
| 04 | Finding contradictions in a knowledge graph with topology (H¹) | `docs/EVALUATION_SUMMARY_HDC_SHEAF.md`, `results/week4_evaluation_report.md` | Planned |

## Blog mirroring

- Canonical posts live in `blog/` and publish to the oasis Ghost CMS.
  **Note**: dev and prod share one Ghost instance — publishing surfaces the
  post publicly immediately; there is no staging.
- **Substack mirror**: Substack has no publishing API — mirror by pasting the
  markdown (or importing the Ghost RSS feed once at setup:
  Substack → Settings → Import). Keep titles identical so links are
  swappable; canonical URL should point at the Ghost post.
- Each thread links to its paired blog post; each blog post embeds the same
  figures from the thread's `figures/` directory.

## Figure style

Figures follow the repo's dataviz conventions: light surface `#fcfcfb`,
categorical palette blue `#2a78d6` / orange `#eb6834` / aqua `#1baf7a`
(CVD-validated in this order), direct value labels on every mark, takeaway
stated in the title, source + repo URL in the footer.
