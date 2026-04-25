# -*- coding: utf-8 -*-
"""
Inductive Embeddings for Out-of-Vocabulary (OOV) Types

This module enables embedding inference for new types not seen during training,
without requiring full model retraining.

Problem:
    Current approach uses static ID-based embeddings. When a new type is introduced,
    it falls back to random initialization, losing all learned structure.

Solution:
    Three inductive strategies:
    1. Neighborhood Aggregation (GraphSAGE-style): Infer from connected types
    2. Text-Based Initialization: Use LLM encoder on type name/description
    3. Ontological Prior: Initialize based on parent/sibling types in hierarchy

Use Cases:
    - Dynamic ontology expansion (new product types, diseases, etc.)
    - Transfer across ontologies (university types -> company types)
    - Few-shot type learning
"""

from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# BASE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class InductiveEmbedder(ABC):
    """
    Abstract base class for inductive embedding strategies.
    
    All strategies must implement:
    - can_embed(): Check if a new type can be embedded
    - embed(): Generate embedding for a new type
    """
    
    @abstractmethod
    def can_embed(self, type_qid: str, context: Dict[str, Any]) -> bool:
        """Check if this strategy can embed the given type."""
        pass
    
    @abstractmethod
    def embed(
        self,
        type_qid: str,
        context: Dict[str, Any],
        existing_embeddings: Dict[str, Any]
    ) -> Any:
        """Generate embedding for a new type."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: NEIGHBORHOOD AGGREGATION (GraphSAGE-style)
# ═══════════════════════════════════════════════════════════════════════════════

class NeighborhoodAggregator(InductiveEmbedder):
    """
    Infer embeddings from local graph neighborhood.
    
    Method:
        new_embedding = aggregate(neighbor_embeddings)
        
    where aggregate can be:
        - mean: Average of all neighbors
        - weighted_mean: Weighted by relation type
        - attention: Learned attention over neighbors
    
    This is similar to GraphSAGE (Hamilton et al., 2017) but applied to
    ontological type graphs rather than entity graphs.
    
    Example:
        New type "ElectricCar" connects to existing types:
        - isA -> "Car" (parent)
        - relatedTo -> "Battery", "Motor"
        - partOf -> "Vehicle"
        
        embedding(ElectricCar) = mean([emb(Car), emb(Battery), emb(Motor), emb(Vehicle)])
    """
    
    AGGREGATION_METHODS = ["mean", "weighted_mean", "max", "attention"]
    
    def __init__(
        self,
        aggregation: str = "weighted_mean",
        relation_weights: Optional[Dict[str, float]] = None,
        embed_dim: int = 64
    ):
        self.aggregation = aggregation
        self.embed_dim = embed_dim
        
        # Default relation weights (higher = more important)
        if relation_weights is None:
            relation_weights = {
                "isA": 2.0,        # Parent is most important
                "subTypeOf": 2.0,
                "partOf": 1.5,
                "relatedTo": 1.0,
                "hasProperty": 0.8,
                "default": 0.5,
            }
        self.relation_weights = relation_weights
    
    def can_embed(self, type_qid: str, context: Dict[str, Any]) -> bool:
        """
        Can embed if the type has at least one known neighbor.
        
        Context should contain:
            - neighbors: List of (neighbor_qid, relation) tuples
        """
        neighbors = context.get("neighbors", [])
        return len(neighbors) > 0
    
    def embed(
        self,
        type_qid: str,
        context: Dict[str, Any],
        existing_embeddings: Dict[str, Any]
    ) -> Any:
        """
        Aggregate neighbor embeddings to create new type embedding.
        
        Args:
            type_qid: The new type to embed
            context: Must contain "neighbors" list
            existing_embeddings: Known type embeddings
        
        Returns:
            Embedding vector for the new type
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required")
        
        neighbors = context.get("neighbors", [])
        
        # Collect neighbor embeddings with weights
        neighbor_embs = []
        weights = []
        
        for neighbor_qid, relation in neighbors:
            if neighbor_qid in existing_embeddings:
                emb = existing_embeddings[neighbor_qid]
                if isinstance(emb, np.ndarray):
                    neighbor_embs.append(emb)
                else:
                    neighbor_embs.append(np.array(emb))
                
                # Get weight for this relation
                weight = self.relation_weights.get(
                    relation, 
                    self.relation_weights.get("default", 1.0)
                )
                weights.append(weight)
        
        if not neighbor_embs:
            # Fallback to random
            return np.random.randn(self.embed_dim) * 0.1
        
        neighbor_embs = np.array(neighbor_embs)
        weights = np.array(weights)
        
        # Aggregate
        if self.aggregation == "mean":
            return np.mean(neighbor_embs, axis=0)
        
        elif self.aggregation == "weighted_mean":
            weights = weights / weights.sum()  # Normalize
            return np.sum(neighbor_embs * weights[:, np.newaxis], axis=0)
        
        elif self.aggregation == "max":
            return np.max(neighbor_embs, axis=0)
        
        else:
            # Default to mean
            return np.mean(neighbor_embs, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: TEXT-BASED INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TextBasedInitializer(InductiveEmbedder):
    """
    Initialize embeddings using LLM encoder on type name/description.
    
    Method:
        1. Format type as text: "A {type_name} is a {description}"
        2. Encode with sentence transformer / LLM
        3. Project to embedding space
    
    This leverages pretrained knowledge about type semantics.
    
    Example:
        type_qid = "GoldenRetriever"
        description = "A breed of dog known for its golden coat"
        
        text = "A GoldenRetriever is a breed of dog known for its golden coat"
        embedding = project(sentence_encoder(text))
    """
    
    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        embed_dim: int = 64,
        projection: Optional[Any] = None
    ):
        self.encoder_name = encoder_name
        self.embed_dim = embed_dim
        self.projection = projection
        self._encoder = None
    
    def _load_encoder(self):
        """Lazy load sentence transformer."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.encoder_name)
            except ImportError:
                raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
        return self._encoder
    
    def can_embed(self, type_qid: str, context: Dict[str, Any]) -> bool:
        """Can always embed if we have a type name."""
        return bool(type_qid) or bool(context.get("label")) or bool(context.get("description"))
    
    def embed(
        self,
        type_qid: str,
        context: Dict[str, Any],
        existing_embeddings: Dict[str, Any]
    ) -> Any:
        """
        Encode type text and project to embedding space.
        
        Context can contain:
            - label: Human-readable type name
            - description: Type description
            - parent: Parent type for context
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required")
        
        # Build text representation
        label = context.get("label", type_qid)
        description = context.get("description", "")
        parent = context.get("parent", "")
        
        if description:
            text = f"A {label} is {description}"
        elif parent:
            text = f"A {label} is a type of {parent}"
        else:
            text = f"The concept of {label}"
        
        # Encode
        try:
            encoder = self._load_encoder()
            raw_embedding = encoder.encode(text)
        except Exception:
            # Fallback to random if encoder fails
            return np.random.randn(self.embed_dim) * 0.1
        
        # Project to target dimension if needed
        if len(raw_embedding) != self.embed_dim:
            if self.projection is not None:
                # Use learned projection
                return self.projection(raw_embedding)
            else:
                # Simple truncation/padding
                if len(raw_embedding) > self.embed_dim:
                    return raw_embedding[:self.embed_dim]
                else:
                    padded = np.zeros(self.embed_dim)
                    padded[:len(raw_embedding)] = raw_embedding
                    return padded
        
        return raw_embedding


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: ONTOLOGICAL PRIOR
# ═══════════════════════════════════════════════════════════════════════════════

class OntologicalPrior(InductiveEmbedder):
    """
    Initialize based on ontological position (parent, siblings, depth).
    
    Method:
        new_embedding = parent_embedding + offset
        
    where offset is computed from:
        - Average sibling offset from parent
        - Depth-based scaling
        - Random noise for uniqueness
    
    This preserves hierarchical structure while allowing new types.
    
    Example:
        New type "Labrador" with parent "Dog" and siblings ["Poodle", "Beagle"]
        
        sibling_offset = mean([emb(Poodle) - emb(Dog), emb(Beagle) - emb(Dog)])
        emb(Labrador) = emb(Dog) + sibling_offset + noise
    """
    
    def __init__(
        self,
        noise_scale: float = 0.05,
        depth_scaling: bool = True,
        embed_dim: int = 64
    ):
        self.noise_scale = noise_scale
        self.depth_scaling = depth_scaling
        self.embed_dim = embed_dim
    
    def can_embed(self, type_qid: str, context: Dict[str, Any]) -> bool:
        """Can embed if we have parent OR siblings."""
        return bool(context.get("parent")) or bool(context.get("siblings"))
    
    def embed(
        self,
        type_qid: str,
        context: Dict[str, Any],
        existing_embeddings: Dict[str, Any]
    ) -> Any:
        """
        Compute embedding from ontological position.
        
        Context should contain:
            - parent: Parent type qid
            - siblings: List of sibling type qids
            - depth: Depth in ontology (optional, for scaling)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required")
        
        parent = context.get("parent")
        siblings = context.get("siblings", [])
        depth = context.get("depth", 1)
        
        # Start with parent embedding if available
        if parent and parent in existing_embeddings:
            base = np.array(existing_embeddings[parent])
        else:
            # Fallback to mean of siblings
            sibling_embs = [existing_embeddings[s] for s in siblings if s in existing_embeddings]
            if sibling_embs:
                base = np.mean(sibling_embs, axis=0)
            else:
                # Final fallback
                return np.random.randn(self.embed_dim) * 0.1
        
        # Compute offset from siblings
        offset = np.zeros_like(base)
        if parent and parent in existing_embeddings:
            parent_emb = existing_embeddings[parent]
            sibling_offsets = []
            for sib in siblings:
                if sib in existing_embeddings:
                    sib_offset = np.array(existing_embeddings[sib]) - parent_emb
                    sibling_offsets.append(sib_offset)
            
            if sibling_offsets:
                # Average sibling offset direction
                avg_offset = np.mean(sibling_offsets, axis=0)
                # Add some variation
                offset = avg_offset * (1 + np.random.randn() * 0.1)
        
        # Apply depth scaling (deeper = smaller offsets)
        if self.depth_scaling:
            scale = 1.0 / (depth + 1)
            offset = offset * scale
        
        # Add noise for uniqueness
        noise = np.random.randn(self.embed_dim) * self.noise_scale
        
        return base + offset + noise


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE EMBEDDER
# ═══════════════════════════════════════════════════════════════════════════════

class InductiveOntologyEmbedder:
    """
    Composite embedder that combines multiple inductive strategies.
    
    Strategy selection order:
    1. If type has known neighbors -> NeighborhoodAggregator
    2. If type has parent/siblings -> OntologicalPrior
    3. If type has text description -> TextBasedInitializer
    4. Fallback -> Random initialization
    
    Usage:
        embedder = InductiveOntologyEmbedder(existing_embeddings)
        
        # Add new type
        new_emb = embedder.embed_new_type(
            type_qid="ElectricCar",
            context={
                "parent": "Car",
                "siblings": ["GasCar", "HybridCar"],
                "neighbors": [("Battery", "partOf"), ("Motor", "hasPart")],
                "description": "A car powered by electric motors"
            }
        )
    """
    
    def __init__(
        self,
        existing_embeddings: Dict[str, Any],
        embed_dim: int = 64,
        strategies: Optional[List[InductiveEmbedder]] = None
    ):
        self.existing_embeddings = existing_embeddings
        self.embed_dim = embed_dim
        
        # Default strategy chain
        if strategies is None:
            strategies = [
                NeighborhoodAggregator(embed_dim=embed_dim),
                OntologicalPrior(embed_dim=embed_dim),
                TextBasedInitializer(embed_dim=embed_dim),
            ]
        self.strategies = strategies
        
        # Track newly embedded types
        self.new_embeddings: Dict[str, Any] = {}
    
    def embed_new_type(
        self,
        type_qid: str,
        context: Dict[str, Any]
    ) -> Any:
        """
        Embed a new type using the best available strategy.
        
        Args:
            type_qid: Unique identifier for new type
            context: Information about the type (neighbors, parent, description, etc.)
        
        Returns:
            Embedding vector
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required")
        
        # Check if already embedded
        if type_qid in self.existing_embeddings:
            return self.existing_embeddings[type_qid]
        if type_qid in self.new_embeddings:
            return self.new_embeddings[type_qid]
        
        # Combine existing and new embeddings for context
        all_embeddings = {**self.existing_embeddings, **self.new_embeddings}
        
        # Try each strategy
        for strategy in self.strategies:
            if strategy.can_embed(type_qid, context):
                try:
                    embedding = strategy.embed(type_qid, context, all_embeddings)
                    self.new_embeddings[type_qid] = embedding
                    return embedding
                except Exception as e:
                    # Try next strategy
                    continue
        
        # Fallback to random
        embedding = np.random.randn(self.embed_dim) * 0.1
        self.new_embeddings[type_qid] = embedding
        return embedding
    
    def embed_batch(
        self,
        types_with_context: List[Tuple[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Embed multiple new types."""
        results = {}
        for type_qid, context in types_with_context:
            results[type_qid] = self.embed_new_type(type_qid, context)
        return results
    
    def get_all_embeddings(self) -> Dict[str, Any]:
        """Get all embeddings (existing + new)."""
        return {**self.existing_embeddings, **self.new_embeddings}


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH MODULE (Optional)
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    
    class LearnedNeighborhoodAggregator(nn.Module):
        """
        Learnable neighborhood aggregation for inductive embeddings.
        
        Unlike the static aggregator, this learns:
        - Relation-specific transformations
        - Attention weights over neighbors
        - Non-linear combination
        """
        
        def __init__(
            self,
            embed_dim: int = 64,
            n_relations: int = 10,
            n_heads: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            self.embed_dim = embed_dim
            
            # Relation embeddings
            self.relation_embeddings = nn.Embedding(n_relations, embed_dim)
            
            # Multi-head attention for aggregation
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        
        def forward(
            self,
            neighbor_embeddings: torch.Tensor,  # [batch, n_neighbors, embed_dim]
            relation_ids: torch.Tensor,          # [batch, n_neighbors]
            mask: Optional[torch.Tensor] = None  # [batch, n_neighbors]
        ) -> torch.Tensor:
            """
            Aggregate neighbor embeddings with learned attention.
            
            Returns:
                [batch, embed_dim] aggregated embeddings
            """
            batch_size, n_neighbors, _ = neighbor_embeddings.shape
            
            # Add relation information
            rel_embs = self.relation_embeddings(relation_ids)  # [B, N, D]
            neighbor_with_rel = neighbor_embeddings + rel_embs
            
            # Create query (learnable or mean of neighbors)
            query = neighbor_embeddings.mean(dim=1, keepdim=True)  # [B, 1, D]
            
            # Attention over neighbors
            attn_output, _ = self.attention(
                query=query,
                key=neighbor_with_rel,
                value=neighbor_embeddings,
                key_padding_mask=mask
            )
            
            # Project
            output = self.output_proj(attn_output.squeeze(1))  # [B, D]
            
            return output


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Inductive Embeddings for OOV Types")
    print("=" * 70)
    
    if not NUMPY_AVAILABLE:
        print("NumPy required for demo")
        return
    
    np.random.seed(42)
    
    # Simulate existing embeddings
    existing_types = ["Car", "Vehicle", "GasCar", "HybridCar", "Battery", "Motor", "Engine"]
    embed_dim = 64
    
    existing_embeddings = {
        t: np.random.randn(embed_dim) for t in existing_types
    }
    
    # Make Car closer to Vehicle (parent relationship)
    existing_embeddings["Car"] = existing_embeddings["Vehicle"] + np.random.randn(embed_dim) * 0.1
    existing_embeddings["GasCar"] = existing_embeddings["Car"] + np.random.randn(embed_dim) * 0.1
    existing_embeddings["HybridCar"] = existing_embeddings["Car"] + np.random.randn(embed_dim) * 0.1
    
    print(f"\nExisting types: {existing_types}")
    print(f"Embedding dimension: {embed_dim}")
    
    # Create composite embedder
    embedder = InductiveOntologyEmbedder(
        existing_embeddings=existing_embeddings,
        embed_dim=embed_dim
    )
    
    # Test 1: New type with neighborhood
    print("\n" + "-" * 70)
    print("Test 1: Neighborhood Aggregation")
    print("-" * 70)
    
    electric_car_emb = embedder.embed_new_type(
        type_qid="ElectricCar",
        context={
            "neighbors": [
                ("Car", "isA"),
                ("Battery", "hasPart"),
                ("Motor", "hasPart"),
            ]
        }
    )
    print(f"Embedded 'ElectricCar' using neighborhood aggregation")
    print(f"  Embedding norm: {np.linalg.norm(electric_car_emb):.3f}")
    
    # Check similarity to Car
    car_sim = np.dot(electric_car_emb, existing_embeddings["Car"]) / (
        np.linalg.norm(electric_car_emb) * np.linalg.norm(existing_embeddings["Car"])
    )
    print(f"  Similarity to 'Car': {car_sim:.3f}")
    
    # Test 2: New type with ontological prior
    print("\n" + "-" * 70)
    print("Test 2: Ontological Prior")
    print("-" * 70)
    
    diesel_car_emb = embedder.embed_new_type(
        type_qid="DieselCar",
        context={
            "parent": "Car",
            "siblings": ["GasCar", "HybridCar"],
            "depth": 2
        }
    )
    print(f"Embedded 'DieselCar' using ontological prior")
    print(f"  Embedding norm: {np.linalg.norm(diesel_car_emb):.3f}")
    
    # Check similarity to siblings
    gas_sim = np.dot(diesel_car_emb, existing_embeddings["GasCar"]) / (
        np.linalg.norm(diesel_car_emb) * np.linalg.norm(existing_embeddings["GasCar"])
    )
    hybrid_sim = np.dot(diesel_car_emb, existing_embeddings["HybridCar"]) / (
        np.linalg.norm(diesel_car_emb) * np.linalg.norm(existing_embeddings["HybridCar"])
    )
    print(f"  Similarity to 'GasCar': {gas_sim:.3f}")
    print(f"  Similarity to 'HybridCar': {hybrid_sim:.3f}")
    
    # Test 3: New type with only text (would need sentence-transformers)
    print("\n" + "-" * 70)
    print("Test 3: Text-Based Initialization")
    print("-" * 70)
    
    try:
        fuel_cell_emb = embedder.embed_new_type(
            type_qid="FuelCellCar",
            context={
                "label": "Fuel Cell Car",
                "description": "A car powered by hydrogen fuel cells"
            }
        )
        print(f"Embedded 'FuelCellCar' using text encoder")
        print(f"  Embedding norm: {np.linalg.norm(fuel_cell_emb):.3f}")
    except ImportError:
        print("  (Skipped - sentence-transformers not installed)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_embs = embedder.get_all_embeddings()
    print(f"Total types: {len(all_embs)}")
    print(f"  Existing: {len(existing_embeddings)}")
    print(f"  New (inductive): {len(embedder.new_embeddings)}")
    
    print("\nNew embeddings created:")
    for qid in embedder.new_embeddings:
        print(f"  - {qid}")
    
    print("\n✓ Inductive embedding demo complete!")
    print("\nKey benefits:")
    print("  - No retraining needed for new types")
    print("  - Preserves ontological structure")
    print("  - Multiple fallback strategies")


if __name__ == "__main__":
    main()
