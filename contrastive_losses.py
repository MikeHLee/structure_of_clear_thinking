# -*- coding: utf-8 -*-
"""
Contrastive Loss Functions for Token-Based Ontological Embeddings

This module provides two loss functions optimized for small-batch training
with LLM-encoded ontological statements:

1. InfoNCELoss: Temperature-scaled contrastive loss with hard negative injection
2. AdaptiveMarginTripletLoss: Dynamic margin triplet loss for hierarchical negatives

Architecture Context:
- Inputs are [CLS] pooled embeddings from LLM encoders (e.g., BERT, RoBERTa)
- Tensor shapes: [batch_size, hidden_dim] (e.g., [32, 768])
- Small batch sizes due to VRAM constraints require explicit hard negative mining

Reference:
- InfoNCE: Oord et al. "Representation Learning with Contrastive Predictive Coding" (2018)
- Triplet Loss: Schroff et al. "FaceNet: A Unified Embedding for Face Recognition" (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class InfoNCELoss(nn.Module):
    """
    Temperature-Scaled Contrastive Loss (InfoNCE) for Ontological Embeddings.
    
    Loss = -log( exp(sim(anchor, positive) / τ) / Σ exp(sim(anchor, negative_j) / τ) )
    
    Key Features:
    - Low temperature (τ=0.05) for sharp distributions with small batches
    - Support for both cosine similarity and negated L2 distance
    - Explicit hard negative injection to augment in-batch negatives
    
    Args:
        temperature: Softmax temperature τ. Lower = sharper distribution.
                     Default 0.05 is aggressive for small batches.
        similarity: "cosine" or "l2_neg" (negated L2 for geometric preservation)
        reduction: "mean" or "sum" over batch
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        similarity: Literal["cosine", "l2_neg"] = "cosine",
        reduction: Literal["mean", "sum"] = "mean"
    ):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
        self.reduction = reduction
        
        # Validate temperature - too low causes numerical instability
        if temperature < 0.01:
            raise ValueError(f"Temperature {temperature} too low; risk of overflow. Use >= 0.01")
    
    def compute_similarity(
        self,
        anchors: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise similarity between anchors and targets.
        
        Args:
            anchors: [batch_size, hidden_dim]
            targets: [num_targets, hidden_dim]
        
        Returns:
            similarities: [batch_size, num_targets]
        """
        if self.similarity == "cosine":
            # Normalize to unit sphere for cosine similarity
            anchors_norm = F.normalize(anchors, p=2, dim=-1)  # [B, D]
            targets_norm = F.normalize(targets, p=2, dim=-1)  # [N, D]
            # Cosine similarity via dot product of normalized vectors
            return torch.mm(anchors_norm, targets_norm.t())  # [B, N]
        
        elif self.similarity == "l2_neg":
            # Negated L2 distance: higher = more similar (closer in space)
            # Using cdist for efficient pairwise L2 computation
            distances = torch.cdist(anchors, targets, p=2)  # [B, N]
            # Negate so that smaller distance = higher similarity
            return -distances
        
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")
    
    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        in_batch_negatives: Optional[torch.Tensor] = None,
        hard_negatives: Optional[torch.Tensor] = None,
        hard_negative_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with optional hard negative injection.
        
        Args:
            anchors: Anchor embeddings [batch_size, hidden_dim]
                     e.g., "[CLS] Entity A has type B [SEP]"
            
            positives: Positive embeddings [batch_size, hidden_dim]
                       e.g., valid transitions for each anchor
            
            in_batch_negatives: Optional in-batch negatives [batch_size, hidden_dim]
                                If None, uses all non-matching positives as negatives
                                (standard in-batch negative mining)
            
            hard_negatives: Explicitly mined hard negatives [num_hard, hidden_dim]
                            These are appended to the denominator for ALL anchors.
                            Critical for small batches where in-batch diversity is low.
            
            hard_negative_weights: Optional weights for hard negatives [num_hard]
                                   Allows weighting by ontological distance tier (L0-L3)
        
        Returns:
            loss: Scalar tensor (reduced) or [batch_size] if reduction="none"
        
        Example:
            # In-batch negatives only (each anchor's positive is negative for others)
            loss = infonce(anchors, positives)
            
            # With explicit hard negatives (mined from memory bank or Tier 3 violations)
            loss = infonce(anchors, positives, hard_negatives=hard_negs)
        """
        batch_size = anchors.size(0)
        device = anchors.device
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: Compute anchor-positive similarities (numerator)
        # ═══════════════════════════════════════════════════════════════════
        # Shape: [batch_size, batch_size] where diagonal = anchor_i with positive_i
        pos_sim_matrix = self.compute_similarity(anchors, positives)  # [B, B]
        
        # Extract diagonal: each anchor's similarity with its own positive
        # Shape: [batch_size]
        pos_similarities = torch.diag(pos_sim_matrix)
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 2: Build negative pool for denominator
        # ═══════════════════════════════════════════════════════════════════
        
        # Start with in-batch negatives: all positives are negatives for non-matching anchors
        # The full pos_sim_matrix already contains anchor_i -> positive_j similarities
        # We'll mask out the diagonal (true positive) later
        all_neg_similarities = pos_sim_matrix  # [B, B]
        
        # If explicit in-batch negatives provided (e.g., from different ontologies)
        if in_batch_negatives is not None:
            explicit_neg_sim = self.compute_similarity(anchors, in_batch_negatives)  # [B, B_neg]
            all_neg_similarities = torch.cat([all_neg_similarities, explicit_neg_sim], dim=1)
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 3: Inject hard negatives (CRITICAL for small batches)
        # ═══════════════════════════════════════════════════════════════════
        # Hard negatives are shared across all anchors in the batch
        if hard_negatives is not None:
            # Compute similarity of each anchor to all hard negatives
            # Shape: [batch_size, num_hard_negatives]
            hard_neg_sim = self.compute_similarity(anchors, hard_negatives)
            
            # Optional: weight hard negatives by ontological distance tier
            # Higher weight for Tier 3 (adversarial) negatives
            if hard_negative_weights is not None:
                # Broadcast weights: [num_hard] -> [1, num_hard] -> [B, num_hard]
                hard_neg_sim = hard_neg_sim * hard_negative_weights.unsqueeze(0)
            
            # Concatenate to negative pool
            # Shape: [batch_size, B + num_hard] or [B, B + B_neg + num_hard]
            all_neg_similarities = torch.cat([all_neg_similarities, hard_neg_sim], dim=1)
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 4: Apply temperature scaling
        # ═══════════════════════════════════════════════════════════════════
        # Low temperature (0.05) creates sharp distribution:
        # - Hard negatives dominate gradient
        # - Easy negatives contribute minimally
        pos_logits = pos_similarities / self.temperature  # [B]
        neg_logits = all_neg_similarities / self.temperature  # [B, N_total]
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 5: Compute InfoNCE loss via log-softmax trick
        # ═══════════════════════════════════════════════════════════════════
        # For numerical stability, we use the log-sum-exp trick:
        # log(exp(pos) / sum(exp(neg))) = pos - logsumexp(all)
        
        # Create mask to exclude true positive from denominator
        # The first B columns correspond to pos_sim_matrix where diagonal is true positive
        mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        # Expand mask to cover all negative columns (hard negs don't need masking)
        n_extra_negs = neg_logits.size(1) - batch_size
        if n_extra_negs > 0:
            extra_mask = torch.zeros(batch_size, n_extra_negs, device=device, dtype=torch.bool)
            mask = torch.cat([mask, extra_mask], dim=1)
        
        # Set masked positions to large negative value (excluded from softmax)
        neg_logits_masked = neg_logits.masked_fill(mask, float('-inf'))
        
        # Denominator: logsumexp over all negatives (excluding true positive)
        # Shape: [batch_size]
        log_denominator = torch.logsumexp(neg_logits_masked, dim=1)
        
        # InfoNCE loss: -log(positive / sum(negatives))
        # = -(pos_logit - log_denominator)
        loss_per_sample = -pos_logits + log_denominator  # [batch_size]
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 6: Reduction
        # ═══════════════════════════════════════════════════════════════════
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class AdaptiveMarginTripletLoss(nn.Module):
    """
    Adaptive Margin Triplet Loss for Hierarchical Ontological Negatives.
    
    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    Unlike standard triplet loss with fixed margin, this accepts a per-sample
    margin tensor based on the ontological distance tier of the negative:
    
    - L0 (same type): margin = 0.1 (should be very close to positive)
    - L1 (sibling type): margin = 0.3
    - L2 (cousin type): margin = 0.5
    - L3 (adversarial): margin = 1.0 (hard boundary enforcement)
    
    This allows strict ontological boundary enforcement even with small batches,
    where you can explicitly sample negatives from each tier.
    
    Args:
        default_margin: Fallback margin if dynamic margins not provided
        distance: "l2" (Euclidean) or "cosine" (1 - cosine_sim)
        reduction: "mean" or "sum" over batch
    """
    
    def __init__(
        self,
        default_margin: float = 0.5,
        distance: Literal["l2", "cosine"] = "l2",
        reduction: Literal["mean", "sum"] = "mean"
    ):
        super().__init__()
        self.default_margin = default_margin
        self.distance = distance
        self.reduction = reduction
    
    def compute_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise distance between corresponding pairs.
        
        Args:
            x: [batch_size, hidden_dim]
            y: [batch_size, hidden_dim]
        
        Returns:
            distances: [batch_size]
        """
        if self.distance == "l2":
            # Euclidean distance (preserves geometric space)
            return F.pairwise_distance(x, y, p=2)
        
        elif self.distance == "cosine":
            # Cosine distance = 1 - cosine_similarity
            x_norm = F.normalize(x, p=2, dim=-1)
            y_norm = F.normalize(y, p=2, dim=-1)
            cosine_sim = (x_norm * y_norm).sum(dim=-1)
            return 1.0 - cosine_sim
        
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
    
    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        margins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute adaptive margin triplet loss.
        
        Args:
            anchors: Anchor embeddings [batch_size, hidden_dim]
            positives: Positive embeddings [batch_size, hidden_dim]
            negatives: Negative embeddings [batch_size, hidden_dim]
            margins: Per-sample margins [batch_size]
                     Based on ontological distance tier (L0-L3)
                     If None, uses default_margin for all samples
        
        Returns:
            loss: Scalar tensor (reduced)
        
        Example:
            # Tier-based margins
            tiers = torch.tensor([0, 1, 2, 3, ...])  # L0, L1, L2, L3
            tier_to_margin = {0: 0.1, 1: 0.3, 2: 0.5, 3: 1.0}
            margins = torch.tensor([tier_to_margin[t.item()] for t in tiers])
            
            loss = triplet_loss(anchors, positives, negatives, margins=margins)
        """
        batch_size = anchors.size(0)
        device = anchors.device
        
        # Use default margin if not provided
        if margins is None:
            margins = torch.full((batch_size,), self.default_margin, device=device)
        
        # Ensure margins has correct shape
        assert margins.size(0) == batch_size, \
            f"Margins shape {margins.shape} doesn't match batch size {batch_size}"
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: Compute distances
        # ═══════════════════════════════════════════════════════════════════
        d_pos = self.compute_distance(anchors, positives)  # [batch_size]
        d_neg = self.compute_distance(anchors, negatives)  # [batch_size]
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 2: Compute triplet loss with adaptive margins
        # ═══════════════════════════════════════════════════════════════════
        # loss = max(0, d_pos - d_neg + margin)
        # We want: d_neg > d_pos + margin (negative farther than positive by margin)
        loss_per_sample = F.relu(d_pos - d_neg + margins)  # [batch_size]
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 3: Reduction
        # ═══════════════════════════════════════════════════════════════════
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: Tier-Based Margin Mapping for Ontological Hierarchies
# ═══════════════════════════════════════════════════════════════════════════════

def get_ontological_margins(
    tiers: torch.Tensor,
    tier_margins: Optional[dict] = None
) -> torch.Tensor:
    """
    Map ontological distance tiers (L0-L3) to margin values.
    
    Default tier definitions:
    - L0: Same type (very easy negative) → margin 0.1
    - L1: Sibling types (share parent) → margin 0.3
    - L2: Cousin types (share grandparent) → margin 0.5
    - L3: Adversarial (ontologically distant but syntactically plausible) → margin 1.0
    
    Args:
        tiers: Tensor of tier labels [batch_size], values in {0, 1, 2, 3}
        tier_margins: Optional custom mapping {tier: margin}
    
    Returns:
        margins: [batch_size] margin values
    """
    if tier_margins is None:
        tier_margins = {
            0: 0.1,   # L0: Same type
            1: 0.3,   # L1: Sibling
            2: 0.5,   # L2: Cousin
            3: 1.0,   # L3: Adversarial
        }
    
    margins = torch.zeros_like(tiers, dtype=torch.float32)
    for tier, margin in tier_margins.items():
        margins[tiers == tier] = margin
    
    return margins


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION: Forward Pass with Dummy Data
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO: Token-Based Ontological Embedding Loss Functions")
    print("=" * 70)
    
    # Configuration matching LLM encoder output
    batch_size = 32
    hidden_dim = 768  # BERT/RoBERTa hidden size
    num_hard_negatives = 64  # Mined from memory bank or Tier 3 pool
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Hard negatives: {num_hard_negatives}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Simulate [CLS] embeddings from LLM encoder
    # ═══════════════════════════════════════════════════════════════════════
    # In practice, these come from: encoder(tokenized_ontology_statements).last_hidden_state[:, 0, :]
    
    torch.manual_seed(42)
    
    # Anchors: "[CLS] Entity A has ontological type B [SEP]"
    anchors = torch.randn(batch_size, hidden_dim, device=device)
    
    # Positives: Valid transitions for each anchor
    # Simulated as anchors + small noise (should be close in embedding space)
    positives = anchors + torch.randn(batch_size, hidden_dim, device=device) * 0.1
    
    # In-batch negatives: Other positives serve as negatives (standard contrastive)
    # This is handled automatically by InfoNCE using the positives tensor
    
    # Hard negatives: Explicitly mined Tier 3 (adversarial) negatives
    # e.g., "Cart places Order" - syntactically plausible but ontologically invalid
    hard_negatives = torch.randn(num_hard_negatives, hidden_dim, device=device)
    
    # ═══════════════════════════════════════════════════════════════════════
    # DEMO 1: InfoNCE Loss
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("DEMO 1: InfoNCE Loss (Temperature-Scaled Contrastive)")
    print("─" * 70)
    
    # Initialize with low temperature for sharp distribution
    infonce_loss = InfoNCELoss(
        temperature=0.05,      # Sharp distribution for small batches
        similarity="cosine",   # Use cosine similarity
        reduction="mean"
    )
    
    # Forward pass: In-batch negatives only
    loss_inbatch = infonce_loss(anchors, positives)
    print(f"\n[In-batch negatives only]")
    print(f"  Negative pool size: {batch_size - 1} (other positives)")
    print(f"  Loss: {loss_inbatch.item():.4f}")
    
    # Forward pass: With hard negative injection (RECOMMENDED)
    loss_with_hard = infonce_loss(
        anchors=anchors,
        positives=positives,
        hard_negatives=hard_negatives
    )
    print(f"\n[With hard negative injection]")
    print(f"  Negative pool size: {batch_size - 1 + num_hard_negatives}")
    print(f"  Loss: {loss_with_hard.item():.4f}")
    print(f"  Loss increase: {(loss_with_hard - loss_inbatch).item():.4f}")
    print(f"  → Hard negatives increase loss (good! more gradient signal)")
    
    # Forward pass: With weighted hard negatives (Tier-based)
    hard_weights = torch.ones(num_hard_negatives, device=device)
    hard_weights[:16] = 0.5   # First 16: L2 (cousin) - lower weight
    hard_weights[16:] = 2.0   # Rest: L3 (adversarial) - higher weight
    
    loss_weighted = infonce_loss(
        anchors=anchors,
        positives=positives,
        hard_negatives=hard_negatives,
        hard_negative_weights=hard_weights
    )
    print(f"\n[With weighted hard negatives (Tier-based)]")
    print(f"  L2 negatives (16): weight 0.5")
    print(f"  L3 negatives (48): weight 2.0")
    print(f"  Loss: {loss_weighted.item():.4f}")
    
    # Compare L2 vs cosine similarity
    infonce_l2 = InfoNCELoss(temperature=0.05, similarity="l2_neg")
    loss_l2 = infonce_l2(anchors, positives, hard_negatives=hard_negatives)
    print(f"\n[L2 distance mode (preserves geometric space)]")
    print(f"  Loss: {loss_l2.item():.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # DEMO 2: Adaptive Margin Triplet Loss
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("DEMO 2: Adaptive Margin Triplet Loss (Hierarchical Negatives)")
    print("─" * 70)
    
    # Generate negatives with tier labels
    negatives = torch.randn(batch_size, hidden_dim, device=device)
    
    # Assign random tiers (in practice, from ontological distance computation)
    tiers = torch.randint(0, 4, (batch_size,), device=device)
    
    # Map tiers to margins
    margins = get_ontological_margins(tiers)
    
    # Initialize adaptive triplet loss
    adaptive_triplet = AdaptiveMarginTripletLoss(
        default_margin=0.5,
        distance="l2",      # Preserve geometric space
        reduction="mean"
    )
    
    # Forward pass with fixed margin (baseline)
    loss_fixed = adaptive_triplet(anchors, positives, negatives)
    print(f"\n[Fixed margin (default=0.5)]")
    print(f"  Loss: {loss_fixed.item():.4f}")
    
    # Forward pass with adaptive margins
    loss_adaptive = adaptive_triplet(anchors, positives, negatives, margins=margins)
    print(f"\n[Adaptive margins (tier-based)]")
    print(f"  Tier distribution: L0={int((tiers==0).sum())}, L1={int((tiers==1).sum())}, "
          f"L2={int((tiers==2).sum())}, L3={int((tiers==3).sum())}")
    print(f"  Margin range: [{margins.min():.1f}, {margins.max():.1f}]")
    print(f"  Loss: {loss_adaptive.item():.4f}")
    
    # Show per-tier losses
    print(f"\n[Per-tier analysis]")
    for tier in range(4):
        tier_mask = (tiers == tier)
        if tier_mask.any():
            tier_anchors = anchors[tier_mask]
            tier_positives = positives[tier_mask]
            tier_negatives = negatives[tier_mask]
            tier_margins = margins[tier_mask]
            
            tier_loss = adaptive_triplet(
                tier_anchors, tier_positives, tier_negatives, margins=tier_margins
            )
            print(f"  L{tier}: margin={tier_margins[0]:.1f}, loss={tier_loss.item():.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: Recommended Usage")
    print("=" * 70)
    print("""
    PRIMARY (InfoNCE):
    - Use for main training with small batches
    - Always inject hard negatives from memory bank or Tier 3 pool
    - Low temperature (0.05) ensures hard negatives dominate gradient
    
    FALLBACK (Adaptive Triplet):
    - Use when explicit negative tiers are available
    - Enforces strict ontological boundaries via dynamic margins
    - Better interpretability for debugging
    
    NEXT STEPS:
    1. Memory Bank (MoCo-style): Decouple batch size from negative pool
    2. Prompt Engineering: Optimize textual ontology statements
    """)
    
    print("\n✓ All demos completed successfully!")
