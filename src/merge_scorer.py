"""
Merge Scorer — HANDOFF_08 §2.2 / W2

Replaces BPE merge rules with a neural scorer  s_θ(τ_i, τ_j) → ℝ  that
decides whether two adjacent HierarchicalTokens should be bound into one.

Architecture
------------
A lightweight 2-layer MLP with a bilinear interaction head:

    h_i   = proj(τ_i.embedding)          # (hidden,)
    h_j   = proj(τ_j.embedding)          # (hidden,)
    score = W_out · tanh(W_bi(h_i ⊗ h_j) + W_cat[h_i; h_j] + b)

The bilinear term captures token-order asymmetry (τ_i left, τ_j right) and
the concat term adds additive interactions.  Total parameters: ~3 M for
hidden=512, input=4096.

Loss (§2.2)
-----------
    L = L_coherence + λ_acc · L_LM + λ_comp · L_length − λ_H · H(merge_dist)

- L_coherence : BCE — does merge(τ_i, τ_j) correspond to a valid Olog
                morphism composition? (labels from CoherenceOracle)
- L_LM        : placeholder MSE to a downstream LM signal; caller provides
                lm_target per pair when available.
- L_length    : mean probability of merging — penalises too-short sequences.
- H(merge_dist): entropy of merge probabilities — penalises collapse to one
                  mega-token (all merges) or full no-merge.

Merge graph
-----------
MergeGraph records every merge decision as a binary tree.  Leaves are the
original pre-merge HierarchicalTokens; internal nodes are merged tokens.
ProofObject can walk back to leaves to recover provenance.

Inference
---------
GreedyMerger applies score greedily in a single left-to-right pass, merging
adjacent pairs whose score > threshold.  The scorer is frozen at inference
(set scorer.eval(); no grad).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ghrr_encoder import GHRREncoder, HypervectorConfig
from hierarchical_tokenizer import (
    HierarchicalToken,
    HierarchicalTokenizer,
    Modality,
    NULL_PROVENANCE,
    UNTYPED_TYPE,
    TokenSlots,
)


# ---------------------------------------------------------------------------
# Coherence oracle — checks Olog for valid morphism composition
# ---------------------------------------------------------------------------

class CoherenceOracle:
    """
    Provides binary coherence labels for (τ_i, τ_j) pairs.

    A merge is *coherent* iff:
      1. τ_i.type_code is the source of some Olog morphism whose target is
         τ_j.type_code (direct composition), OR
      2. Both tokens are UNTYPED (we cannot judge — label = 0.5, soft).

    Supervised signal comes from OlogGraph during training; at inference the
    scorer's own θ is trusted.
    """

    def __init__(self, olog=None) -> None:
        # olog: OlogGraph | None.  None → all pairs get soft label 0.5.
        self._olog = olog
        self._edge_set: set[Tuple[str, str]] = set()
        if olog is not None:
            for u, v, _k in olog.graph.edges(keys=True):
                self._edge_set.add((u, v))

    def label(self, ti: HierarchicalToken, tj: HierarchicalToken) -> float:
        """
        Return a coherence label in [0, 1].

        1.0 → valid composition  |  0.0 → invalid  |  0.5 → unknown
        """
        if self._olog is None:
            return 0.5
        if ti.slots.type_code == UNTYPED_TYPE or tj.slots.type_code == UNTYPED_TYPE:
            return 0.5
        if (ti.slots.type_code, tj.slots.type_code) in self._edge_set:
            return 1.0
        return 0.0

    def batch_labels(
        self, pairs: List[Tuple[HierarchicalToken, HierarchicalToken]]
    ) -> torch.Tensor:
        """Return (N,) float tensor of coherence labels."""
        return torch.tensor([self.label(a, b) for a, b in pairs], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Merge graph
# ---------------------------------------------------------------------------

@dataclass
class MergeNode:
    """A node in the merge graph: either a leaf or a merged pair."""
    token:    HierarchicalToken
    left:     Optional["MergeNode"] = None   # None → leaf
    right:    Optional["MergeNode"] = None
    depth:    int = 0
    merge_id: int = -1   # sequential merge index (−1 = leaf)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def leaves(self) -> List[HierarchicalToken]:
        """Return all leaf tokens reachable from this node."""
        if self.is_leaf:
            return [self.token]
        result: List[HierarchicalToken] = []
        if self.left:
            result.extend(self.left.leaves())
        if self.right:
            result.extend(self.right.leaves())
        return result

    def to_dict(self) -> dict:
        d: dict = {
            "merge_id": self.merge_id,
            "depth": self.depth,
            "token_type": self.token.slots.type_code,
            "token_content": self.token.slots.content_code,
            "provenance": self.token.slots.provenance_code,
            "is_leaf": self.is_leaf,
        }
        if self.left:
            d["left"] = self.left.to_dict()
        if self.right:
            d["right"] = self.right.to_dict()
        return d


@dataclass
class MergeGraph:
    """
    Records the full merge history for a tokenized sequence.

    Roots are the final (post-merge) token nodes.
    Leaves are the original pre-merge tokens.
    """
    roots:   List[MergeNode] = field(default_factory=list)
    _counter: int = field(default=0, repr=False)

    def add_leaf(self, token: HierarchicalToken) -> MergeNode:
        node = MergeNode(token=token, depth=0, merge_id=-1)
        self.roots.append(node)
        return node

    def merge(
        self,
        left: MergeNode,
        right: MergeNode,
        merged_token: HierarchicalToken,
    ) -> MergeNode:
        """Record a merge and update roots list."""
        node = MergeNode(
            token=merged_token,
            left=left,
            right=right,
            depth=max(left.depth, right.depth) + 1,
            merge_id=self._counter,
        )
        self._counter += 1
        # Replace left and right in roots with the merged node
        li = self.roots.index(left)
        ri = self.roots.index(right)
        assert ri == li + 1, "Can only merge adjacent nodes"
        self.roots[li] = node
        self.roots.pop(ri)
        return node

    def leaf_tokens(self) -> List[HierarchicalToken]:
        """All original pre-merge tokens, in order."""
        result: List[HierarchicalToken] = []
        for r in self.roots:
            result.extend(r.leaves())
        return result

    def final_tokens(self) -> List[HierarchicalToken]:
        """Post-merge token sequence."""
        return [r.token for r in self.roots]

    def export(self) -> dict:
        return {
            "n_merges": self._counter,
            "n_final": len(self.roots),
            "n_leaves": len(self.leaf_tokens()),
            "tree": [r.to_dict() for r in self.roots],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.export(), f, indent=2)


# ---------------------------------------------------------------------------
# Neural scorer
# ---------------------------------------------------------------------------

class MergeScorerNet(nn.Module):
    """
    s_θ(τ_i, τ_j) → ℝ  (positive = merge, negative = keep separate).

    Architecture:
        proj     : Linear(input_dim, hidden)          × 2 (shared weights)
        bilinear : Linear(hidden, hidden) applied to hadamard(h_i, h_j)
        cat_proj : Linear(2*hidden, hidden)
        score    : Linear(hidden, 1)
    """

    def __init__(self, input_dim: int = 4096, hidden: int = 512) -> None:
        super().__init__()
        self.proj     = nn.Linear(input_dim, hidden, bias=False)
        self.bilinear = nn.Linear(hidden, hidden)
        self.cat_proj = nn.Linear(2 * hidden, hidden)
        self.score    = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        emb_i: torch.Tensor,   # (B, input_dim) or (input_dim,)
        emb_j: torch.Tensor,
    ) -> torch.Tensor:          # (B,) or scalar
        batched = emb_i.dim() == 2
        if not batched:
            emb_i = emb_i.unsqueeze(0)
            emb_j = emb_j.unsqueeze(0)

        h_i = self.proj(emb_i)                           # (B, hidden)
        h_j = self.proj(emb_j)

        bilinear = self.bilinear(h_i * h_j)              # element-wise interaction
        cat      = self.cat_proj(torch.cat([h_i, h_j], dim=-1))
        combined = torch.tanh(bilinear + cat)
        logit    = self.score(combined).squeeze(-1)       # (B,)

        return logit if batched else logit.squeeze(0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@dataclass
class MergeLossWeights:
    coherence: float = 1.0
    lm:        float = 0.1     # λ_acc
    length:    float = 0.05    # λ_comp
    entropy:   float = 0.2     # λ_H


def merge_loss(
    logits:        torch.Tensor,           # (N,)  raw scorer outputs
    coherence_labels: torch.Tensor,        # (N,)  ∈ [0,1]
    lm_targets:    Optional[torch.Tensor], # (N,) LM signal; None → skip
    weights:       MergeLossWeights = MergeLossWeights(),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the 4-component merge loss.

    Returns (total_loss, component_dict).

    L = L_coherence + λ_acc·L_LM + λ_comp·L_length − λ_H·H(merge_dist)
    """
    probs = torch.sigmoid(logits)          # merge probabilities

    # --- L_coherence: skip soft labels (0.5) to avoid noise ---
    hard_mask = (coherence_labels != 0.5)
    if hard_mask.any():
        l_coh = F.binary_cross_entropy_with_logits(
            logits[hard_mask], coherence_labels[hard_mask]
        )
    else:
        l_coh = torch.zeros(1, device=logits.device)

    # --- L_LM: MSE with downstream LM target (if provided) ---
    if lm_targets is not None:
        l_lm = F.mse_loss(probs, lm_targets.clamp(0, 1))
    else:
        l_lm = torch.zeros(1, device=logits.device)

    # --- L_length: mean merge probability (penalise over-merging) ---
    l_length = probs.mean()

    # --- H(merge_dist): entropy of p — prevent distribution collapse ---
    eps = 1e-7
    h_entropy = -(
        probs * (probs + eps).log() +
        (1 - probs) * (1 - probs + eps).log()
    ).mean()

    total = (
        weights.coherence * l_coh
        + weights.lm       * l_lm
        + weights.length   * l_length
        - weights.entropy  * h_entropy
    )

    components = {
        "coherence": l_coh.item(),
        "lm":        l_lm.item(),
        "length":    l_length.item(),
        "entropy":   h_entropy.item(),
        "total":     total.item(),
    }
    return total, components


# ---------------------------------------------------------------------------
# MergeScorer — wraps net + oracle + tokenizer utilities
# ---------------------------------------------------------------------------

class MergeScorer:
    """
    High-level interface combining the neural net, oracle, and merge logic.

    Parameters
    ----------
    input_dim : int
        Dimension of HierarchicalToken embeddings (= GHRREncoder.dim).
    hidden : int
        Hidden dimension of MergeScorerNet.
    threshold : float
        Score > threshold → merge.  At inference, greedy left-to-right.
    device : str
        'cpu', 'mps', or 'cuda'.
    olog : OlogGraph | None
        Passed to CoherenceOracle.  None → soft labels only.
    weights : MergeLossWeights
        Loss component weights.
    """

    def __init__(
        self,
        input_dim:  int = 4096,
        hidden:     int = 512,
        threshold:  float = 0.5,
        device:     str = "cpu",
        olog=None,
        weights:    MergeLossWeights = MergeLossWeights(),
    ) -> None:
        self.threshold = threshold
        self.weights   = weights
        self.device    = torch.device(device)
        self.net       = MergeScorerNet(input_dim, hidden).to(self.device)
        self.oracle    = CoherenceOracle(olog)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def score(
        self,
        ti: HierarchicalToken,
        tj: HierarchicalToken,
    ) -> float:
        """Score a single adjacent pair.  Returns raw logit (no sigmoid)."""
        self.net.eval()
        with torch.no_grad():
            ei = torch.tensor(ti.embedding, dtype=torch.float32, device=self.device)
            ej = torch.tensor(tj.embedding, dtype=torch.float32, device=self.device)
            return float(self.net(ei, ej).item())

    def score_batch(
        self,
        pairs: List[Tuple[HierarchicalToken, HierarchicalToken]],
    ) -> torch.Tensor:
        """Score a list of adjacent pairs. Returns (N,) logit tensor."""
        ei = torch.stack([
            torch.tensor(ti.embedding, dtype=torch.float32) for ti, _ in pairs
        ]).to(self.device)
        ej = torch.stack([
            torch.tensor(tj.embedding, dtype=torch.float32) for _, tj in pairs
        ]).to(self.device)
        with torch.no_grad():
            self.net.eval()
            return self.net(ei, ej)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        pairs:      List[Tuple[HierarchicalToken, HierarchicalToken]],
        optimizer:  torch.optim.Optimizer,
        lm_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        One gradient step.  Returns component dict.

        Coherence labels are computed from the oracle for every pair.
        lm_targets, if provided, must be shape (N,).
        """
        self.net.train()
        optimizer.zero_grad()

        ei = torch.stack([
            torch.tensor(ti.embedding, dtype=torch.float32) for ti, _ in pairs
        ]).to(self.device)
        ej = torch.stack([
            torch.tensor(tj.embedding, dtype=torch.float32) for _, tj in pairs
        ]).to(self.device)

        logits = self.net(ei, ej)
        coh_labels = self.oracle.batch_labels(pairs).to(self.device)

        loss, components = merge_loss(logits, coh_labels, lm_targets, self.weights)
        loss.backward()
        optimizer.step()
        return components

    # ------------------------------------------------------------------
    # Greedy merge
    # ------------------------------------------------------------------

    def greedy_merge(
        self,
        tokens:     List[HierarchicalToken],
        tokenizer:  HierarchicalTokenizer,
    ) -> Tuple[List[HierarchicalToken], MergeGraph]:
        """
        Apply greedy left-to-right merging.

        A pair (τ_i, τ_{i+1}) is merged iff score > threshold.
        The merged token is produced by re-binding the GHRR of τ_i and τ_{i+1}:

            merged_emb = ghrr.bind(τ_i.embedding, τ_{i+1}.embedding)

        Type slot of the merged token = τ_j.type_code (right token is the
        target — consistent with directed morphism convention).
        Provenance of merged token = τ_i.provenance (left / source witness).
        Modality is promoted: ASSERTION wins over others.

        Returns (final_token_sequence, MergeGraph).
        """
        self.net.eval()

        # Build initial leaf nodes
        mg = MergeGraph()
        nodes: List[MergeNode] = [mg.add_leaf(t) for t in tokens]

        changed = True
        while changed and len(nodes) > 1:
            changed = False
            i = 0
            while i < len(nodes) - 1:
                left_node  = nodes[i]
                right_node = nodes[i + 1]
                s = self.score(left_node.token, right_node.token)
                if s > self.threshold:
                    merged = _merge_tokens(
                        left_node.token,
                        right_node.token,
                        tokenizer,
                        position=left_node.token.position,
                    )
                    # Patch merge graph roots in-place
                    merged_node = MergeNode(
                        token=merged,
                        left=left_node,
                        right=right_node,
                        depth=max(left_node.depth, right_node.depth) + 1,
                        merge_id=mg._counter,
                    )
                    mg._counter += 1
                    nodes[i] = merged_node
                    nodes.pop(i + 1)
                    # Sync mg.roots to match nodes
                    mg.roots = nodes[:]
                    changed = True
                    # Don't advance i — re-evaluate new pair at i
                else:
                    i += 1

        mg.roots = nodes[:]
        return [n.token for n in nodes], mg

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net_state_dict": self.net.state_dict(),
                "threshold":      self.threshold,
                "weights":        vars(self.weights),
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path:      Path,
        input_dim: int = 4096,
        hidden:    int = 512,
        device:    str = "cpu",
        olog=None,
    ) -> "MergeScorer":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        scorer = cls(
            input_dim=input_dim,
            hidden=hidden,
            threshold=ckpt["threshold"],
            device=device,
            olog=olog,
            weights=MergeLossWeights(**ckpt["weights"]),
        )
        scorer.net.load_state_dict(ckpt["net_state_dict"])
        return scorer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_tokens(
    ti:        HierarchicalToken,
    tj:        HierarchicalToken,
    tokenizer: HierarchicalTokenizer,
    position:  int = -1,
) -> HierarchicalToken:
    """
    Produce a merged token from two adjacent tokens.

    Slot assignment:
      type_code       = tj.type_code   (right = morphism target)
      content_code    = f"{ti.content}⊛{tj.content}"
      modality_code   = max(ti, tj) under ASSERTION > CITATION > HYPOTHESIS > QUESTION > UNTYPED
      provenance_code = ti.provenance  (left = witness / source)
      embedding       = ghrr.bind(ti.embedding, tj.embedding)
    """
    _MOD_PRIORITY = {
        Modality.ASSERTION:  5,
        Modality.CITATION:   4,
        Modality.HYPOTHESIS: 3,
        Modality.QUESTION:   2,
        Modality.UNTYPED:    1,
    }
    mod = max(
        ti.slots.modality_code,
        tj.slots.modality_code,
        key=lambda m: _MOD_PRIORITY[m],
    )

    type_hv  = tokenizer.type_vocab.encode(tj.slots.type_code)
    cont_str = f"{ti.slots.content_code}\u229b{tj.slots.content_code}"
    cont_hv  = tokenizer.content_vocab.encode(cont_str)
    mod_hv   = tokenizer.modality_vocab.encode(mod.value)
    prov_hv  = tokenizer.provenance_vocab.encode(ti.slots.provenance_code)

    emb = tokenizer.bind_slots(type_hv, cont_hv, mod_hv, prov_hv)

    return HierarchicalToken(
        slots=TokenSlots(
            type_code=tj.slots.type_code,
            content_code=cont_str,
            modality_code=mod,
            provenance_code=ti.slots.provenance_code,
        ),
        embedding=emb,
        text=f"{ti.text}⊛{tj.text}",
        position=position,
        _type_hv=type_hv,
        _content_hv=cont_hv,
        _modality_hv=mod_hv,
        _provenance_hv=prov_hv,
    )


# ---------------------------------------------------------------------------
# Smoke test / demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    print("=" * 64)
    print("MergeScorer — W2 smoke test")
    print("=" * 64)

    # ---- Setup ----
    ghrr      = GHRREncoder(HypervectorConfig(dim=4096, seed=42))
    tokenizer = HierarchicalTokenizer(ghrr)

    # Use the e-commerce ontology from the existing demos
    from olog_core import OlogGraph
    olog = OlogGraph("ECommerce")
    for t in ("Customer", "Cart", "Order", "Payment", "Delivery"):
        olog.add_type(t)
    olog.add_aspect("Customer", "Cart",     "creates")
    olog.add_aspect("Cart",     "Order",    "becomes")
    olog.add_aspect("Order",    "Payment",  "requires")
    olog.add_aspect("Payment",  "Delivery", "triggers")

    device  = "mps" if torch.backends.mps.is_available() else "cpu"
    scorer  = MergeScorer(input_dim=4096, hidden=512, threshold=0.0, device=device, olog=olog)
    oracle  = scorer.oracle

    # ---- Tokenize a sequence ----
    sequence = [
        tokenizer.encode_token("Alice",    "Customer", "Alice",    Modality.ASSERTION, position=0),
        tokenizer.encode_token("creates",  "Customer", "creates",  Modality.ASSERTION, position=1),
        tokenizer.encode_token("my_cart",  "Cart",     "my_cart",  Modality.ASSERTION, position=2),
        tokenizer.encode_token("becomes",  "Cart",     "becomes",  Modality.ASSERTION, position=3),
        tokenizer.encode_token("order_42", "Order",    "order_42", Modality.ASSERTION, position=4),
    ]

    print(f"\n[1] Input sequence ({len(sequence)} tokens)")
    for t in sequence:
        print(f"  {t}")

    # ---- Oracle labels ----
    adjacent_pairs = list(zip(sequence, sequence[1:]))
    print("\n[2] Coherence oracle labels")
    for (ti, tj), lbl in zip(adjacent_pairs, oracle.batch_labels(adjacent_pairs)):
        flag = "COHERENT" if lbl > 0.5 else "soft" if lbl == 0.5 else "INCOHERENT"
        print(f"  {ti.slots.type_code} → {tj.slots.type_code} : {lbl:.1f} ({flag})")

    # ---- Scorer scores (untrained — random init) ----
    print("\n[3] Scorer logits (untrained)")
    for (ti, tj) in adjacent_pairs:
        s = scorer.score(ti, tj)
        print(f"  {ti.slots.content_code} ⊛ {tj.slots.content_code} : {s:+.4f}")

    # ---- Training loop (5 steps) ----
    print("\n[4] Training (5 steps)")
    opt = torch.optim.Adam(scorer.net.parameters(), lr=1e-3)
    for step in range(5):
        comps = scorer.train_step(adjacent_pairs, opt)
        print(
            f"  step {step+1}: total={comps['total']:+.4f} "
            f"coh={comps['coherence']:.4f} "
            f"len={comps['length']:.4f} "
            f"H={comps['entropy']:.4f}"
        )

    # ---- Greedy merge ----
    print("\n[5] Greedy merge (threshold=0.0)")
    scorer.threshold = 0.0
    merged_tokens, mg = scorer.greedy_merge(sequence, tokenizer)
    print(f"  {len(sequence)} → {len(merged_tokens)} tokens after merge")
    for t in merged_tokens:
        print(f"  {t}")
    print(f"\n  MergeGraph: {mg._counter} merges, {len(mg.leaf_tokens())} leaves")

    # ---- Checkpoint round-trip ----
    ckpt_path = Path("/tmp/merge_scorer_demo.pt")
    scorer.save_checkpoint(ckpt_path)
    scorer2 = MergeScorer.load_checkpoint(ckpt_path, input_dim=4096, hidden=512, device=device)
    s1 = scorer.score(sequence[0], sequence[1])
    s2 = scorer2.score(sequence[0], sequence[1])
    print(f"\n[6] Checkpoint round-trip: score before={s1:+.4f}, after={s2:+.4f} "
          f"(match={'YES' if abs(s1 - s2) < 1e-5 else 'NO'})")

    # ---- Merge graph export ----
    mg_path = Path("/tmp/merge_graph_demo.json")
    mg.save(mg_path)
    print(f"\n[7] Merge graph saved → {mg_path}")
    with open(mg_path) as f:
        summary = json.load(f)
    print(f"  n_merges={summary['n_merges']} n_final={summary['n_final']} n_leaves={summary['n_leaves']}")

    print("\n" + "=" * 64)
    print("Smoke test complete.")
    print("=" * 64)


if __name__ == "__main__":
    import json
    _demo()
