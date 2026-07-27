# -*- coding: utf-8 -*-
"""
Memory Bank for Ontological Contrastive Learning

This module implements a MoCo-style memory bank that decouples batch size
from negative sample pool size. Critical for small-batch LLM-based training.

The Problem:
    - LLM encoders (BERT, RoBERTa) require significant VRAM per sample
    - Typical batch size: 16-64 samples
    - In-batch negatives: batch_size - 1 = 15-63 negatives
    - Probability of sampling hard negatives (L2, L3) is low
    
The Solution:
    - Maintain a large queue of past embeddings (8K-65K)
    - Each embedding is tagged with its ontological tier
    - Sample hard negatives from queue, not just current batch
    - Result: batch_size=32 can access 8000+ negatives

Reference:
    He et al. "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo, 2020)
    
Ontological Extension:
    - Tier-aware sampling: preferentially sample L2/L3 (hard) negatives
    - Ontology-aware eviction: maintain tier balance in queue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import IntEnum


class OntologicalTier(IntEnum):
    """
    Ontological distance tiers for negative classification.
    
    The tier determines how "hard" a negative is:
    - L0: Same type (trivial negative, basically positive)
    - L1: Sibling type (shares parent in ontology tree)
    - L2: Cousin type (shares grandparent, harder to distinguish)
    - L3: Adversarial (different domain but syntactically plausible)
    
    Example for "Customer places Order":
    - L0: "Customer places Order" (same - not useful)
    - L1: "Client places Order" (sibling: Client ≈ Customer)
    - L2: "Customer places Request" (cousin: Request similar to Order)
    - L3: "Cart places Order" (adversarial: syntactically OK, ontologically invalid)
    """
    L0_SAME = 0
    L1_SIBLING = 1
    L2_COUSIN = 2
    L3_ADVERSARIAL = 3


@dataclass
class MemoryBankConfig:
    """Configuration for the ontological memory bank."""
    bank_size: int = 8192           # Total capacity (power of 2 for efficiency)
    hidden_dim: int = 768           # Embedding dimension (match encoder)
    momentum: float = 0.999         # EMA momentum for key encoder (if used)
    tier_sampling_weights: Dict[int, float] = None  # Sampling probability per tier
    
    def __post_init__(self):
        if self.tier_sampling_weights is None:
            # Default: heavily weight hard negatives
            self.tier_sampling_weights = {
                OntologicalTier.L0_SAME: 0.0,       # Never sample L0
                OntologicalTier.L1_SIBLING: 0.1,   # Rarely sample L1
                OntologicalTier.L2_COUSIN: 0.3,    # Sometimes sample L2
                OntologicalTier.L3_ADVERSARIAL: 0.6  # Prefer L3
            }


class OntologyMemoryBank(nn.Module):
    """
    MoCo-style memory bank with ontological tier awareness.
    
    Architecture:
    ┌────────────────────────────────────────────────────────────┐
    │                    MEMORY BANK                              │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ queue: [bank_size, hidden_dim] - embedding storage   │  │
    │  │ tiers: [bank_size] - ontological tier per embedding  │  │
    │  │ ontologies: [bank_size] - source ontology ID         │  │
    │  │ ptr: int - circular buffer write pointer             │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                                                            │
    │  Operations:                                               │
    │  - enqueue(embeddings, tiers): add new embeddings          │
    │  - sample(n, target_tiers): get n hard negatives          │
    │  - get_tier_distribution(): stats for monitoring          │
    └────────────────────────────────────────────────────────────┘
    
    Usage:
        bank = OntologyMemoryBank(config)
        
        # Training loop
        for batch in dataloader:
            anchors, positives, negatives, tiers = encode(batch)
            
            # Sample hard negatives from bank
            hard_negs, hard_tiers = bank.sample(
                n=64, 
                target_tiers=[OntologicalTier.L2_COUSIN, OntologicalTier.L3_ADVERSARIAL]
            )
            
            # Compute loss with augmented negatives
            loss = infonce(anchors, positives, hard_negatives=hard_negs)
            
            # Update bank with current batch's negatives
            bank.enqueue(negatives.detach(), tiers)
    """
    
    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config
        
        # Main storage buffers (registered as buffers, not parameters)
        # queue: stores the actual embeddings
        self.register_buffer(
            "queue", 
            torch.randn(config.bank_size, config.hidden_dim)
        )
        # Normalize initial random embeddings
        self.queue = F.normalize(self.queue, p=2, dim=1)
        
        # tiers: ontological tier for each embedding
        self.register_buffer(
            "tiers",
            torch.full((config.bank_size,), OntologicalTier.L3_ADVERSARIAL, dtype=torch.long)
        )
        
        # ontology_ids: which ontology each embedding came from (for cross-ontology sampling)
        self.register_buffer(
            "ontology_ids",
            torch.zeros(config.bank_size, dtype=torch.long)
        )
        
        # valid_mask: which slots have been filled (vs. random init)
        self.register_buffer(
            "valid_mask",
            torch.zeros(config.bank_size, dtype=torch.bool)
        )
        
        # Circular buffer pointer
        self.register_buffer(
            "ptr",
            torch.zeros(1, dtype=torch.long)
        )
        
        # Count of valid entries
        self.register_buffer(
            "count",
            torch.zeros(1, dtype=torch.long)
        )
    
    @property
    def current_size(self) -> int:
        """Number of valid embeddings in the bank."""
        return min(int(self.count.item()), self.config.bank_size)
    
    @property
    def is_full(self) -> bool:
        """Whether the bank has wrapped around at least once."""
        return self.count.item() >= self.config.bank_size
    
    @torch.no_grad()
    def enqueue(
        self,
        embeddings: torch.Tensor,
        tiers: torch.Tensor,
        ontology_ids: Optional[torch.Tensor] = None
    ) -> None:
        """
        Add new embeddings to the memory bank.
        
        Uses a circular buffer: when full, oldest embeddings are evicted.
        This ensures the bank always contains recent, relevant negatives.
        
        Args:
            embeddings: [batch_size, hidden_dim] - embeddings to store
            tiers: [batch_size] - ontological tier for each embedding
            ontology_ids: [batch_size] - source ontology ID (optional)
        """
        batch_size = embeddings.size(0)
        ptr = int(self.ptr.item())
        
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Default ontology IDs to 0 if not provided
        if ontology_ids is None:
            ontology_ids = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)
        
        # Handle wrap-around for circular buffer
        if ptr + batch_size <= self.config.bank_size:
            # Simple case: no wrap
            self.queue[ptr:ptr + batch_size] = embeddings
            self.tiers[ptr:ptr + batch_size] = tiers
            self.ontology_ids[ptr:ptr + batch_size] = ontology_ids
            self.valid_mask[ptr:ptr + batch_size] = True
        else:
            # Wrap around
            first_part = self.config.bank_size - ptr
            second_part = batch_size - first_part
            
            self.queue[ptr:] = embeddings[:first_part]
            self.queue[:second_part] = embeddings[first_part:]
            
            self.tiers[ptr:] = tiers[:first_part]
            self.tiers[:second_part] = tiers[first_part:]
            
            self.ontology_ids[ptr:] = ontology_ids[:first_part]
            self.ontology_ids[:second_part] = ontology_ids[first_part:]
            
            self.valid_mask[ptr:] = True
            self.valid_mask[:second_part] = True
        
        # Update pointer and count
        self.ptr[0] = (ptr + batch_size) % self.config.bank_size
        self.count[0] = self.count[0] + batch_size
    
    @torch.no_grad()
    def sample(
        self,
        n: int,
        target_tiers: Optional[List[int]] = None,
        exclude_ontology: Optional[int] = None,
        weighted: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negatives from the memory bank.
        
        Args:
            n: Number of negatives to sample
            target_tiers: List of tiers to sample from (default: L2, L3)
            exclude_ontology: Ontology ID to exclude (for cross-ontology negatives)
            weighted: Use tier-based sampling weights from config
        
        Returns:
            embeddings: [n, hidden_dim] sampled negative embeddings
            tiers: [n] tier labels for sampled embeddings
        """
        device = self.queue.device
        
        # Default to hard negatives only
        if target_tiers is None:
            target_tiers = [OntologicalTier.L2_COUSIN, OntologicalTier.L3_ADVERSARIAL]
        
        # Build mask of valid candidates
        candidate_mask = self.valid_mask.clone()
        
        # Filter by tier
        tier_mask = torch.zeros(self.config.bank_size, dtype=torch.bool, device=device)
        for tier in target_tiers:
            tier_mask |= (self.tiers == tier)
        candidate_mask &= tier_mask
        
        # Optionally exclude specific ontology
        if exclude_ontology is not None:
            candidate_mask &= (self.ontology_ids != exclude_ontology)
        
        # Get valid indices
        valid_indices = candidate_mask.nonzero(as_tuple=True)[0]
        num_valid = len(valid_indices)
        
        if num_valid == 0:
            # Fallback: return random embeddings if bank is empty/filtered
            return (
                torch.randn(n, self.config.hidden_dim, device=device),
                torch.full((n,), OntologicalTier.L3_ADVERSARIAL, dtype=torch.long, device=device)
            )
        
        # Sample with or without tier weighting
        if weighted and num_valid >= n:
            # Compute sampling weights based on tier
            weights = torch.zeros(num_valid, device=device)
            for i, idx in enumerate(valid_indices):
                tier = int(self.tiers[idx].item())
                weights[i] = self.config.tier_sampling_weights.get(tier, 0.5)
            
            # Normalize to probability distribution
            weights = weights / weights.sum()
            
            # Sample without replacement
            sample_indices = torch.multinomial(weights, min(n, num_valid), replacement=False)
            selected = valid_indices[sample_indices]
        else:
            # Uniform random sampling
            if num_valid >= n:
                perm = torch.randperm(num_valid, device=device)[:n]
            else:
                # Not enough samples: sample with replacement
                perm = torch.randint(0, num_valid, (n,), device=device)
            selected = valid_indices[perm]
        
        return self.queue[selected].clone(), self.tiers[selected].clone()
    
    @torch.no_grad()
    def sample_cross_ontology(
        self,
        n: int,
        anchor_ontology: int,
        target_tiers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negatives from OTHER ontologies only.
        
        This ensures true inter-ontology negatives for separation ratio.
        
        Args:
            n: Number of negatives to sample
            anchor_ontology: The anchor's ontology ID (to exclude)
            target_tiers: Tiers to sample from
        
        Returns:
            embeddings: [n, hidden_dim]
            tiers: [n]
        """
        return self.sample(
            n=n,
            target_tiers=target_tiers,
            exclude_ontology=anchor_ontology,
            weighted=True
        )
    
    def get_tier_distribution(self) -> Dict[str, int]:
        """Get current distribution of tiers in the bank (for monitoring)."""
        if not self.is_full:
            valid = self.valid_mask
        else:
            valid = torch.ones(self.config.bank_size, dtype=torch.bool, device=self.queue.device)
        
        distribution = {}
        for tier in OntologicalTier:
            count = ((self.tiers == tier) & valid).sum().item()
            distribution[tier.name] = count
        
        distribution["total"] = int(valid.sum().item())
        return distribution
    
    def __repr__(self) -> str:
        dist = self.get_tier_distribution()
        return (
            f"OntologyMemoryBank("
            f"size={self.current_size}/{self.config.bank_size}, "
            f"L0={dist.get('L0_SAME', 0)}, "
            f"L1={dist.get('L1_SIBLING', 0)}, "
            f"L2={dist.get('L2_COUSIN', 0)}, "
            f"L3={dist.get('L3_ADVERSARIAL', 0)})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM ENCODER (Optional, for MoCo v2 style)
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumEncoder(nn.Module):
    """
    Momentum-updated encoder for stable negative representations.
    
    In MoCo, the key encoder (producing negatives) is updated via EMA
    of the query encoder (producing anchors/positives). This provides:
    - Stable representations for cached negatives
    - Smooth evolution of the embedding space
    
    Update rule: θ_k = m * θ_k + (1 - m) * θ_q
    where m = 0.999 (very slow update)
    """
    
    def __init__(self, encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Create momentum encoder as a copy
        self.momentum_encoder = self._copy_encoder(encoder)
        
        # Freeze momentum encoder (no gradients)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        """Create a deep copy of the encoder."""
        import copy
        return copy.deepcopy(encoder)
    
    @torch.no_grad()
    def update(self):
        """Update momentum encoder via EMA."""
        for param_q, param_k in zip(
            self.encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_k.data = (
                self.momentum * param_k.data +
                (1 - self.momentum) * param_q.data
            )
    
    def encode_queries(self, x: torch.Tensor) -> torch.Tensor:
        """Encode queries/anchors with main encoder (has gradients)."""
        return self.encoder(x)
    
    @torch.no_grad()
    def encode_keys(self, x: torch.Tensor) -> torch.Tensor:
        """Encode keys/negatives with momentum encoder (no gradients)."""
        return self.momentum_encoder(x)


# ═══════════════════════════════════════════════════════════════════════════════
# HARD NEGATIVE MINING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

class HardNegativeMiner:
    """
    Strategies for mining hard negatives from the memory bank.
    
    Hard negatives are critical for learning fine-grained ontological boundaries.
    Three strategies are provided:
    
    1. Tier-based: Sample from L2/L3 tiers (ontologically hard)
    2. Similarity-based: Find negatives close to anchor in embedding space
    3. Semi-hard: Find negatives between positive and margin boundary
    """
    
    @staticmethod
    @torch.no_grad()
    def mine_by_tier(
        bank: OntologyMemoryBank,
        n: int,
        tiers: List[int] = [OntologicalTier.L2_COUSIN, OntologicalTier.L3_ADVERSARIAL]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mine negatives by ontological tier.
        
        Simple but effective: L2 and L3 negatives are hard by definition.
        """
        return bank.sample(n=n, target_tiers=tiers, weighted=True)
    
    @staticmethod
    @torch.no_grad()
    def mine_by_similarity(
        bank: OntologyMemoryBank,
        anchors: torch.Tensor,
        n_per_anchor: int,
        exclude_tiers: List[int] = [OntologicalTier.L0_SAME]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mine negatives closest to each anchor in embedding space.
        
        These are the hardest negatives: ontologically invalid but
        embedding-similar to the anchor.
        
        Args:
            anchors: [batch_size, hidden_dim]
            n_per_anchor: Number of hard negatives per anchor
        
        Returns:
            hard_negatives: [batch_size * n_per_anchor, hidden_dim]
            tiers: [batch_size * n_per_anchor]
        """
        batch_size = anchors.size(0)
        device = anchors.device
        
        # Get valid entries from bank
        valid_mask = bank.valid_mask.clone()
        for tier in exclude_tiers:
            valid_mask &= (bank.tiers != tier)
        
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return bank.sample(n=batch_size * n_per_anchor)
        
        # Compute similarity between anchors and all valid bank entries
        anchors_norm = F.normalize(anchors, p=2, dim=1)  # [B, D]
        bank_norm = F.normalize(bank.queue[valid_indices], p=2, dim=1)  # [V, D]
        
        similarities = torch.mm(anchors_norm, bank_norm.t())  # [B, V]
        
        # For each anchor, get top-k most similar (hardest) negatives
        k = min(n_per_anchor, len(valid_indices))
        _, topk_indices = similarities.topk(k, dim=1)  # [B, k]
        
        # Gather the actual embeddings
        selected_bank_indices = valid_indices[topk_indices.flatten()]  # [B * k]
        hard_negatives = bank.queue[selected_bank_indices]
        hard_tiers = bank.tiers[selected_bank_indices]
        
        return hard_negatives, hard_tiers
    
    @staticmethod
    @torch.no_grad()
    def mine_semi_hard(
        bank: OntologyMemoryBank,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        margin: float = 0.5,
        n_per_anchor: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mine semi-hard negatives (between positive and margin).
        
        Semi-hard negatives satisfy:
            d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin
        
        These provide stable gradients without being too easy or too hard.
        """
        batch_size = anchors.size(0)
        device = anchors.device
        
        # Compute anchor-positive distances
        anchors_norm = F.normalize(anchors, p=2, dim=1)
        positives_norm = F.normalize(positives, p=2, dim=1)
        d_pos = 1 - (anchors_norm * positives_norm).sum(dim=1)  # [B], cosine distance
        
        # Get valid bank entries
        valid_indices = bank.valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return bank.sample(n=batch_size * n_per_anchor)
        
        bank_norm = F.normalize(bank.queue[valid_indices], p=2, dim=1)
        
        # Compute distances to all bank entries
        similarities = torch.mm(anchors_norm, bank_norm.t())  # [B, V]
        d_neg = 1 - similarities  # [B, V], cosine distances
        
        # Find semi-hard: d_pos < d_neg < d_pos + margin
        lower_bound = d_pos.unsqueeze(1)  # [B, 1]
        upper_bound = d_pos.unsqueeze(1) + margin  # [B, 1]
        
        semi_hard_mask = (d_neg > lower_bound) & (d_neg < upper_bound)  # [B, V]
        
        # Sample from semi-hard negatives
        all_negatives = []
        all_tiers = []
        
        for i in range(batch_size):
            candidates = semi_hard_mask[i].nonzero(as_tuple=True)[0]
            if len(candidates) >= n_per_anchor:
                selected = candidates[torch.randperm(len(candidates))[:n_per_anchor]]
            elif len(candidates) > 0:
                selected = candidates[torch.randint(0, len(candidates), (n_per_anchor,))]
            else:
                # Fallback: random from bank
                selected = torch.randint(0, len(valid_indices), (n_per_anchor,), device=device)
            
            all_negatives.append(bank.queue[valid_indices[selected]])
            all_tiers.append(bank.tiers[valid_indices[selected]])
        
        return torch.cat(all_negatives, dim=0), torch.cat(all_tiers, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO: Ontology Memory Bank for Contrastive Learning")
    print("=" * 70)
    
    # Configuration
    config = MemoryBankConfig(
        bank_size=4096,
        hidden_dim=768,
        tier_sampling_weights={
            OntologicalTier.L0_SAME: 0.0,
            OntologicalTier.L1_SIBLING: 0.1,
            OntologicalTier.L2_COUSIN: 0.3,
            OntologicalTier.L3_ADVERSARIAL: 0.6,
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Bank size: {config.bank_size}")
    print(f"Hidden dim: {config.hidden_dim}")
    
    # Initialize memory bank
    bank = OntologyMemoryBank(config).to(device)
    print(f"\nInitial state: {bank}")
    
    # Simulate training: enqueue batches of negatives
    print("\n" + "-" * 70)
    print("Simulating training: enqueueing embeddings...")
    print("-" * 70)
    
    torch.manual_seed(42)
    batch_size = 32
    n_batches = 50
    
    for batch_idx in range(n_batches):
        # Simulate encoded negatives from LLM
        embeddings = torch.randn(batch_size, config.hidden_dim, device=device)
        
        # Assign tiers (in practice, from ontological distance computation)
        # Skew toward L3 for realistic distribution
        tier_probs = torch.tensor([0.05, 0.15, 0.30, 0.50])
        tiers = torch.multinomial(tier_probs, batch_size, replacement=True).to(device)
        
        # Assign ontology IDs (4 ontologies)
        ontology_ids = torch.randint(0, 4, (batch_size,), device=device)
        
        bank.enqueue(embeddings, tiers, ontology_ids)
    
    print(f"After {n_batches} batches: {bank}")
    print(f"Tier distribution: {bank.get_tier_distribution()}")
    
    # Sample hard negatives
    print("\n" + "-" * 70)
    print("Sampling hard negatives...")
    print("-" * 70)
    
    # Standard tier-based sampling
    hard_negs, hard_tiers = bank.sample(
        n=64,
        target_tiers=[OntologicalTier.L2_COUSIN, OntologicalTier.L3_ADVERSARIAL]
    )
    print(f"\nTier-based sampling (64 samples):")
    print(f"  Shape: {hard_negs.shape}")
    print(f"  L2 count: {(hard_tiers == OntologicalTier.L2_COUSIN).sum().item()}")
    print(f"  L3 count: {(hard_tiers == OntologicalTier.L3_ADVERSARIAL).sum().item()}")
    
    # Cross-ontology sampling
    cross_negs, cross_tiers = bank.sample_cross_ontology(
        n=32,
        anchor_ontology=0,
        target_tiers=[OntologicalTier.L3_ADVERSARIAL]
    )
    print(f"\nCross-ontology sampling (excluding ontology 0):")
    print(f"  Shape: {cross_negs.shape}")
    
    # Similarity-based mining
    print("\n" + "-" * 70)
    print("Hard negative mining strategies...")
    print("-" * 70)
    
    anchors = torch.randn(16, config.hidden_dim, device=device)
    positives = anchors + torch.randn(16, config.hidden_dim, device=device) * 0.1
    
    # Mine by similarity
    sim_negs, sim_tiers = HardNegativeMiner.mine_by_similarity(
        bank=bank,
        anchors=anchors,
        n_per_anchor=4
    )
    print(f"\nSimilarity-based mining (top-4 per anchor):")
    print(f"  Shape: {sim_negs.shape}")
    
    # Mine semi-hard
    semi_negs, semi_tiers = HardNegativeMiner.mine_semi_hard(
        bank=bank,
        anchors=anchors,
        positives=positives,
        margin=0.5,
        n_per_anchor=2
    )
    print(f"\nSemi-hard mining (2 per anchor):")
    print(f"  Shape: {semi_negs.shape}")
    
    # Integration example with InfoNCE
    print("\n" + "-" * 70)
    print("Integration with InfoNCE Loss...")
    print("-" * 70)
    
    print("""
    # Training loop integration:
    
    from contrastive_losses import InfoNCELoss
    from memory_bank import OntologyMemoryBank, HardNegativeMiner
    
    bank = OntologyMemoryBank(config)
    infonce = InfoNCELoss(temperature=0.05)
    
    for batch in dataloader:
        anchors, positives, negatives, tiers = encode(batch)
        
        # Option 1: Tier-based hard negatives
        hard_negs, _ = bank.sample(n=64, target_tiers=[2, 3])
        
        # Option 2: Similarity-based (hardest)
        hard_negs, _ = HardNegativeMiner.mine_by_similarity(bank, anchors, n_per_anchor=4)
        
        # Compute loss
        loss = infonce(anchors, positives, hard_negatives=hard_negs)
        loss.backward()
        optimizer.step()
        
        # Update bank with current negatives
        bank.enqueue(negatives.detach(), tiers)
    """)
    
    print("\n✓ Memory Bank demo completed!")
    print("\nKey benefits:")
    print("  - Batch size 32 → access to 4096+ cached negatives")
    print("  - Tier-aware sampling → focus on hard (L2/L3) negatives")
    print("  - Cross-ontology support → proper inter-ontology separation")
    print("  - Multiple mining strategies → similarity, semi-hard, tier-based")
