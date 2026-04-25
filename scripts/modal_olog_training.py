#!/usr/bin/env python3
"""
Modal.com GPU Deployment for Ontological Induction Experiments

This script runs:
1. Ontological embedding training (contrastive learning on type/relation structure)
2. Ontological attention training (type-constrained transformer)
3. Hallucination detection benchmarks

Setup:
    pip install modal
    modal setup  # One-time authentication

Usage:
    modal run scripts/modal_olog_training.py
    modal run scripts/modal_olog_training.py --experiment embeddings
    modal run scripts/modal_olog_training.py --experiment attention
    modal run scripts/modal_olog_training.py --experiment benchmark
"""

import modal

# Define the Modal app
app = modal.App("olog-training")

# Container image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy",
    "networkx",
    "pydantic",
    "matplotlib",
    "scikit-learn",
    "tqdm",
)

# Volume for persisting results
volume = modal.Volume.from_name("olog-results", create_if_missing=True)


@app.function(
    gpu="T4",
    image=image,
    timeout=7200,
    volumes={"/results": volume},
)
def train_ontological_embeddings(
    n_epochs: int = 100,
    embed_dim: int = 64,
    learning_rate: float = 0.001,
    margin: float = 0.5,
    seed: int = 42,
):
    """
    Train ontological embeddings using contrastive learning.
    
    Objective: Similar types/relations should have similar embeddings,
    dissimilar ones should be pushed apart.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from dataclasses import dataclass
    from typing import Dict, List, Tuple, Set
    import json
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define sample ontologies for training
    ONTOLOGIES = {
        "business": {
            "types": ["Customer", "Order", "Product", "Invoice", "Payment", "Shipment"],
            "aspects": [
                ("Customer", "Order", "places"),
                ("Order", "Product", "contains"),
                ("Order", "Invoice", "generates"),
                ("Invoice", "Payment", "requires"),
                ("Payment", "Shipment", "triggers"),
                ("Shipment", "Customer", "delivers_to"),
            ]
        },
        "academic": {
            "types": ["Student", "Course", "Professor", "Department", "Grade", "Transcript"],
            "aspects": [
                ("Student", "Course", "enrolls_in"),
                ("Course", "Professor", "taught_by"),
                ("Professor", "Department", "belongs_to"),
                ("Student", "Grade", "receives"),
                ("Grade", "Course", "for_course"),
                ("Student", "Transcript", "has"),
            ]
        },
        "healthcare": {
            "types": ["Patient", "Doctor", "Diagnosis", "Treatment", "Prescription", "Insurance"],
            "aspects": [
                ("Patient", "Doctor", "sees"),
                ("Doctor", "Diagnosis", "makes"),
                ("Diagnosis", "Treatment", "requires"),
                ("Treatment", "Prescription", "involves"),
                ("Patient", "Insurance", "has"),
                ("Insurance", "Treatment", "covers"),
            ]
        },
        "ecommerce": {
            "types": ["User", "Cart", "Item", "Checkout", "Payment", "Delivery"],
            "aspects": [
                ("User", "Cart", "has"),
                ("Cart", "Item", "contains"),
                ("Cart", "Checkout", "proceeds_to"),
                ("Checkout", "Payment", "requires"),
                ("Payment", "Delivery", "triggers"),
                ("Delivery", "User", "to"),
            ]
        },
    }
    
    # Collect all types and relations
    all_types = set()
    all_relations = set()
    type_to_ontology = {}
    relation_to_ontology = {}
    
    for ont_name, ont in ONTOLOGIES.items():
        for t in ont["types"]:
            all_types.add(t)
            type_to_ontology[t] = ont_name
        for src, tgt, rel in ont["aspects"]:
            all_relations.add(rel)
            relation_to_ontology[rel] = ont_name
    
    all_types = sorted(list(all_types))
    all_relations = sorted(list(all_relations))
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    rel_to_idx = {r: i for i, r in enumerate(all_relations)}
    
    print(f"Total types: {len(all_types)}")
    print(f"Total relations: {len(all_relations)}")
    
    # Embedding model
    class OntologicalEmbeddingModel(nn.Module):
        def __init__(self, n_types, n_relations, embed_dim):
            super().__init__()
            self.type_embeddings = nn.Embedding(n_types, embed_dim)
            self.relation_embeddings = nn.Embedding(n_relations, embed_dim)
            
            # Composition layer for (source, relation, target) triples
            self.compose = nn.Sequential(
                nn.Linear(3 * embed_dim, 2 * embed_dim),
                nn.ReLU(),
                nn.Linear(2 * embed_dim, embed_dim),
            )
            
            # Initialize with small random values
            nn.init.normal_(self.type_embeddings.weight, std=0.1)
            nn.init.normal_(self.relation_embeddings.weight, std=0.1)
        
        def get_type_embedding(self, type_idx):
            return self.type_embeddings(torch.tensor(type_idx, device=self.type_embeddings.weight.device))
        
        def get_relation_embedding(self, rel_idx):
            return self.relation_embeddings(torch.tensor(rel_idx, device=self.relation_embeddings.weight.device))
        
        def get_triple_embedding(self, src_idx, rel_idx, tgt_idx):
            src = self.type_embeddings(torch.tensor(src_idx, device=self.type_embeddings.weight.device))
            rel = self.relation_embeddings(torch.tensor(rel_idx, device=self.relation_embeddings.weight.device))
            tgt = self.type_embeddings(torch.tensor(tgt_idx, device=self.type_embeddings.weight.device))
            combined = torch.cat([src, rel, tgt], dim=-1)
            return self.compose(combined)
    
    # Create positive and negative pairs
    def create_training_pairs():
        """Generate contrastive pairs for training."""
        positive_type_pairs = []  # Types in same ontology
        negative_type_pairs = []  # Types in different ontologies
        positive_rel_pairs = []   # Relations that are composable
        negative_rel_pairs = []   # Relations that cannot compose
        
        # Type pairs
        for t1 in all_types:
            for t2 in all_types:
                if t1 != t2:
                    if type_to_ontology.get(t1) == type_to_ontology.get(t2):
                        positive_type_pairs.append((type_to_idx[t1], type_to_idx[t2]))
                    else:
                        negative_type_pairs.append((type_to_idx[t1], type_to_idx[t2]))
        
        # Relation composability
        composable = set()
        for ont in ONTOLOGIES.values():
            aspects = ont["aspects"]
            for i, (s1, t1, r1) in enumerate(aspects):
                for j, (s2, t2, r2) in enumerate(aspects):
                    if t1 == s2:  # r1 ; r2 is composable
                        composable.add((rel_to_idx[r1], rel_to_idx[r2]))
        
        for r1 in all_relations:
            for r2 in all_relations:
                if r1 != r2:
                    pair = (rel_to_idx[r1], rel_to_idx[r2])
                    if pair in composable:
                        positive_rel_pairs.append(pair)
                    else:
                        negative_rel_pairs.append(pair)
        
        return positive_type_pairs, negative_type_pairs, positive_rel_pairs, negative_rel_pairs
    
    pos_type, neg_type, pos_rel, neg_rel = create_training_pairs()
    print(f"Type pairs - Positive: {len(pos_type)}, Negative: {len(neg_type)}")
    print(f"Relation pairs - Positive: {len(pos_rel)}, Negative: {len(neg_rel)}")
    
    # Initialize model
    model = OntologicalEmbeddingModel(len(all_types), len(all_relations), embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Sample batches
        batch_size = 32
        
        # Type contrastive loss
        if pos_type and neg_type:
            pos_samples = [pos_type[i] for i in np.random.choice(len(pos_type), min(batch_size, len(pos_type)), replace=False)]
            neg_samples = [neg_type[i] for i in np.random.choice(len(neg_type), min(batch_size, len(neg_type)), replace=False)]
            
            for (p1, p2) in pos_samples:
                e1 = model.get_type_embedding(p1)
                e2 = model.get_type_embedding(p2)
                pos_dist = F.pairwise_distance(e1.unsqueeze(0), e2.unsqueeze(0))
                
                # Sample a negative
                n1, n2 = neg_samples[np.random.randint(len(neg_samples))]
                e_neg = model.get_type_embedding(n2)
                neg_dist = F.pairwise_distance(e1.unsqueeze(0), e_neg.unsqueeze(0))
                
                # Triplet loss
                loss = F.relu(pos_dist - neg_dist + margin)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
        
        # Relation contrastive loss
        if pos_rel and neg_rel:
            pos_samples = [pos_rel[i] for i in np.random.choice(len(pos_rel), min(batch_size, len(pos_rel)), replace=False)]
            neg_samples = [neg_rel[i] for i in np.random.choice(len(neg_rel), min(batch_size, len(neg_rel)), replace=False)]
            
            for (p1, p2) in pos_samples:
                e1 = model.get_relation_embedding(p1)
                e2 = model.get_relation_embedding(p2)
                pos_dist = F.pairwise_distance(e1.unsqueeze(0), e2.unsqueeze(0))
                
                n1, n2 = neg_samples[np.random.randint(len(neg_samples))]
                e_neg = model.get_relation_embedding(n2)
                neg_dist = F.pairwise_distance(e1.unsqueeze(0), e_neg.unsqueeze(0))
                
                loss = F.relu(pos_dist - neg_dist + margin)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time:.1f}s")
    
    # Evaluate: Check type clustering
    model.eval()
    with torch.no_grad():
        type_embs = model.type_embeddings.weight.cpu().numpy()
        
        # Compute intra-ontology vs inter-ontology distances
        intra_dists = []
        inter_dists = []
        
        for i, t1 in enumerate(all_types):
            for j, t2 in enumerate(all_types):
                if i < j:
                    dist = np.linalg.norm(type_embs[i] - type_embs[j])
                    if type_to_ontology.get(t1) == type_to_ontology.get(t2):
                        intra_dists.append(dist)
                    else:
                        inter_dists.append(dist)
        
        intra_mean = np.mean(intra_dists) if intra_dists else 0
        inter_mean = np.mean(inter_dists) if inter_dists else 0
        separation = inter_mean / (intra_mean + 1e-8)
    
    print(f"\nEvaluation:")
    print(f"  Intra-ontology distance: {intra_mean:.4f}")
    print(f"  Inter-ontology distance: {inter_mean:.4f}")
    print(f"  Separation ratio: {separation:.4f} (higher = better clustering)")
    
    # Save results
    results = {
        "n_epochs": n_epochs,
        "embed_dim": embed_dim,
        "n_types": len(all_types),
        "n_relations": len(all_relations),
        "final_loss": float(losses[-1]) if losses else 0,
        "training_time": training_time,
        "intra_ontology_dist": float(intra_mean),
        "inter_ontology_dist": float(inter_mean),
        "separation_ratio": float(separation),
        "loss_history": [float(l) for l in losses],
    }
    
    with open("/results/embedding_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model weights
    torch.save(model.state_dict(), "/results/olog_embeddings.pt")
    
    volume.commit()
    
    return results


@app.function(
    gpu="T4",
    image=image,
    timeout=7200,
    volumes={"/results": volume},
)
def train_ontological_attention(
    n_epochs: int = 50,
    embed_dim: int = 64,
    num_heads: int = 4,
    learning_rate: float = 0.0001,
    seed: int = 42,
):
    """
    Train ontological attention with type-constrained masking.
    
    Task: Given a sequence of typed tokens, predict the next valid type.
    The attention mask enforces that only reachable types can be attended to.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Dict, List, Tuple
    import json
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define ontology
    TYPES = ["Customer", "Order", "Product", "Invoice", "Payment", "Shipment"]
    ASPECTS = [
        ("Customer", "Order", "places"),
        ("Order", "Product", "contains"),
        ("Order", "Invoice", "generates"),
        ("Invoice", "Payment", "requires"),
        ("Payment", "Shipment", "triggers"),
        ("Shipment", "Customer", "delivers_to"),
    ]
    
    type_to_idx = {t: i for i, t in enumerate(TYPES)}
    n_types = len(TYPES)
    
    # Build reachability matrix
    reachability = np.eye(n_types)  # Identity for self-loops
    for src, tgt, _ in ASPECTS:
        reachability[type_to_idx[src], type_to_idx[tgt]] = 1
    
    # Transitive closure
    for _ in range(n_types):
        reachability = np.clip(reachability @ reachability + reachability, 0, 1)
    
    print(f"Reachability matrix:\n{reachability}")
    
    # Model
    class OntologicalTransformer(nn.Module):
        def __init__(self, n_types, embed_dim, num_heads):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            
            self.type_embedding = nn.Embedding(n_types, embed_dim)
            self.position_embedding = nn.Embedding(32, embed_dim)
            
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            
            self.output = nn.Linear(embed_dim, n_types)
        
        def forward(self, type_indices, ontological_mask=None):
            """
            Args:
                type_indices: (batch, seq_len) tensor of type indices
                ontological_mask: (seq_len, seq_len) mask where True = block attention
            """
            batch_size, seq_len = type_indices.shape
            
            # Embeddings
            positions = torch.arange(seq_len, device=type_indices.device)
            x = self.type_embedding(type_indices) + self.position_embedding(positions)
            
            # Self-attention with ontological mask
            attn_out, _ = self.attention(x, x, x, attn_mask=ontological_mask)
            x = self.norm1(x + attn_out)
            
            # Feed-forward
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            
            # Predict next type
            return self.output(x)
    
    # Generate training data: valid type sequences
    def generate_valid_sequences(n_samples, max_len=6):
        """Generate sequences that follow valid morphism paths."""
        sequences = []
        for _ in range(n_samples):
            # Start from random type
            seq = [np.random.randint(n_types)]
            for _ in range(max_len - 1):
                current = seq[-1]
                # Find valid next types
                valid_next = np.where(reachability[current] > 0)[0]
                if len(valid_next) > 0:
                    seq.append(np.random.choice(valid_next))
                else:
                    break
            sequences.append(seq)
        return sequences
    
    def generate_invalid_sequences(n_samples, max_len=6):
        """Generate sequences with invalid transitions (for negative examples)."""
        sequences = []
        labels = []  # 1 = valid position, 0 = invalid
        for _ in range(n_samples):
            seq = [np.random.randint(n_types)]
            seq_labels = [1]
            for _ in range(max_len - 1):
                current = seq[-1]
                if np.random.random() < 0.3:  # 30% chance of invalid transition
                    invalid_next = np.where(reachability[current] == 0)[0]
                    if len(invalid_next) > 0:
                        seq.append(np.random.choice(invalid_next))
                        seq_labels.append(0)
                        continue
                # Valid transition
                valid_next = np.where(reachability[current] > 0)[0]
                if len(valid_next) > 0:
                    seq.append(np.random.choice(valid_next))
                    seq_labels.append(1)
            sequences.append(seq)
            labels.append(seq_labels)
        return sequences, labels
    
    # Training
    model = OntologicalTransformer(n_types, embed_dim, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create ontological mask (True = block)
    olog_mask = torch.tensor(1 - reachability, dtype=torch.bool, device=device)
    
    losses = []
    accuracies = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        
        # Generate batch
        valid_seqs = generate_valid_sequences(64, max_len=6)
        
        # Pad sequences
        max_len = max(len(s) for s in valid_seqs)
        padded = np.zeros((len(valid_seqs), max_len), dtype=np.int64)
        for i, seq in enumerate(valid_seqs):
            padded[i, :len(seq)] = seq
        
        type_indices = torch.tensor(padded, device=device)
        
        # Create sequence-level mask from ontological mask
        seq_mask = olog_mask[:max_len, :max_len]
        
        # Forward
        logits = model(type_indices, ontological_mask=seq_mask)
        
        # Loss: predict next token
        targets = torch.roll(type_indices, -1, dims=1)
        targets[:, -1] = type_indices[:, -1]  # Last token predicts itself
        
        loss = F.cross_entropy(logits.view(-1, n_types), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
    
    training_time = time.time() - start_time
    
    # Evaluate: Check if invalid transitions are rejected
    model.eval()
    with torch.no_grad():
        invalid_seqs, invalid_labels = generate_invalid_sequences(100, max_len=6)
        
        n_detected = 0
        n_invalid = 0
        
        for seq, labels in zip(invalid_seqs, invalid_labels):
            if len(seq) < 2:
                continue
            
            type_indices = torch.tensor([seq], device=device)
            logits = model(type_indices)
            
            for i in range(len(seq) - 1):
                if labels[i + 1] == 0:  # Invalid transition
                    n_invalid += 1
                    # Check if the model assigns low probability to invalid next
                    probs = F.softmax(logits[0, i], dim=-1)
                    pred = probs.argmax().item()
                    actual = seq[i + 1]
                    if pred != actual or probs[actual].item() < 0.1:
                        n_detected += 1
        
        detection_rate = n_detected / max(n_invalid, 1)
    
    print(f"\nEvaluation:")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Final accuracy: {accuracies[-1]:.4f}")
    print(f"  Invalid transition detection: {detection_rate:.2%} ({n_detected}/{n_invalid})")
    
    results = {
        "n_epochs": n_epochs,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "final_loss": float(losses[-1]),
        "final_accuracy": float(accuracies[-1]),
        "training_time": training_time,
        "invalid_detection_rate": float(detection_rate),
        "loss_history": [float(l) for l in losses],
        "accuracy_history": [float(a) for a in accuracies],
    }
    
    with open("/results/attention_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), "/results/olog_attention.pt")
    
    volume.commit()
    
    return results


@app.function(
    gpu="T4",
    image=image,
    timeout=3600,
    volumes={"/results": volume},
)
def benchmark_hallucination_detection(seed: int = 42):
    """
    Benchmark hallucination detection using the proof engine.
    
    Tests:
    1. Valid claims (should pass)
    2. Invalid claims - wrong relation (should fail STRICT mode)
    3. Invalid claims - no path (should fail all modes)
    """
    import numpy as np
    import json
    import time
    from typing import Dict, List
    from enum import Enum
    from dataclasses import dataclass
    
    np.random.seed(seed)
    
    # Inline minimal Olog implementation for Modal
    class OlogGraph:
        def __init__(self, name):
            self.name = name
            self.types = {}  # name -> description
            self.aspects = []  # (source, target, label)
            self._reachability = None
        
        def add_type(self, name, desc=""):
            self.types[name] = desc
        
        def add_aspect(self, source, target, label):
            self.aspects.append((source, target, label))
            self._reachability = None
        
        def _compute_reachability(self):
            """Compute transitive closure."""
            types = list(self.types.keys())
            n = len(types)
            type_to_idx = {t: i for i, t in enumerate(types)}
            
            reach = {t: {t} for t in types}  # Identity
            for src, tgt, _ in self.aspects:
                if src in reach:
                    reach[src].add(tgt)
            
            # Transitive closure
            for _ in range(n):
                for t1 in types:
                    for t2 in list(reach.get(t1, set())):
                        if t2 in reach:
                            reach[t1].update(reach[t2])
            
            self._reachability = reach
            return reach
        
        def is_reachable(self, source, target):
            if self._reachability is None:
                self._compute_reachability()
            return target in self._reachability.get(source, set())
        
        def has_direct_edge(self, source, target, label):
            return (source, target, label) in self.aspects
    
    class ProofMode(Enum):
        STRICT = "strict"
        COMPOSITIONAL = "compositional"
        REACHABILITY = "reachability"
    
    def prove_claim(olog, claim, mode):
        """Simplified proof engine."""
        parts = claim.split()
        if len(parts) < 3:
            return {"valid": False, "reason": "Parse error"}
        
        subject = parts[0]
        relation = parts[1]
        obj = " ".join(parts[2:])
        
        if subject not in olog.types:
            return {"valid": False, "reason": f"Unknown type: {subject}"}
        if obj not in olog.types:
            return {"valid": False, "reason": f"Unknown type: {obj}"}
        
        if mode == ProofMode.STRICT:
            if olog.has_direct_edge(subject, obj, relation):
                return {"valid": True, "path": [relation]}
            return {"valid": False, "reason": f"No edge '{relation}' from {subject} to {obj}"}
        
        elif mode == ProofMode.COMPOSITIONAL:
            # Check if relation appears anywhere in a valid path
            if olog.has_direct_edge(subject, obj, relation):
                return {"valid": True, "path": [relation]}
            # For simplicity, just check reachability + relation exists somewhere
            if olog.is_reachable(subject, obj):
                for s, t, r in olog.aspects:
                    if r == relation:
                        return {"valid": True, "path": ["composition"]}
            return {"valid": False, "reason": f"No path with '{relation}'"}
        
        else:  # REACHABILITY
            if olog.is_reachable(subject, obj):
                return {"valid": True, "path": ["reachable"]}
            return {"valid": False, "reason": f"No path from {subject} to {obj}"}
    
    # Create test ontology
    olog = OlogGraph("TestOntology")
    for t in ["Customer", "Order", "Product", "Invoice", "Payment", "Shipment"]:
        olog.add_type(t)
    
    olog.add_aspect("Customer", "Order", "places")
    olog.add_aspect("Order", "Product", "contains")
    olog.add_aspect("Order", "Invoice", "generates")
    olog.add_aspect("Invoice", "Payment", "requires")
    olog.add_aspect("Payment", "Shipment", "triggers")
    olog.add_aspect("Shipment", "Customer", "delivers_to")
    
    # Test cases
    test_cases = [
        # (claim, expected_strict, expected_compositional, expected_reachability)
        ("Customer places Order", True, True, True),
        ("Order generates Invoice", True, True, True),
        ("Invoice requires Payment", True, True, True),
        ("Payment places Customer", False, False, True),  # Wrong relation, but reachable
        ("Customer triggers Shipment", False, False, True),  # Wrong relation
        ("Product generates Invoice", False, False, False),  # No path at all
        ("Invoice places Customer", False, False, True),  # Wrong relation
        ("Shipment delivers_to Customer", True, True, True),  # Valid direct
        ("Customer contains Product", False, False, True),  # Wrong relation
        ("Order requires Payment", False, False, True),  # Reachable but wrong relation
    ]
    
    print("=" * 60)
    print("  HALLUCINATION DETECTION BENCHMARK")
    print("=" * 60)
    
    results = {mode.value: {"correct": 0, "total": 0, "details": []} for mode in ProofMode}
    
    for claim, exp_strict, exp_comp, exp_reach in test_cases:
        for mode, expected in [
            (ProofMode.STRICT, exp_strict),
            (ProofMode.COMPOSITIONAL, exp_comp),
            (ProofMode.REACHABILITY, exp_reach),
        ]:
            result = prove_claim(olog, claim, mode)
            actual = result["valid"]
            correct = (actual == expected)
            
            results[mode.value]["total"] += 1
            if correct:
                results[mode.value]["correct"] += 1
            
            results[mode.value]["details"].append({
                "claim": claim,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "reason": result.get("reason"),
            })
    
    # Print results
    for mode in ProofMode:
        r = results[mode.value]
        acc = r["correct"] / r["total"] * 100
        print(f"\n{mode.value.upper()} MODE: {r['correct']}/{r['total']} ({acc:.0f}%)")
        for d in r["details"]:
            icon = "✓" if d["correct"] else "✗"
            print(f"  {icon} \"{d['claim']}\" - expected={d['expected']}, got={d['actual']}")
    
    # Summary metrics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Calculate hallucination detection rate (false claims correctly rejected)
    hallucination_cases = [c for c in test_cases if not c[1]]  # Cases where STRICT should fail
    
    strict_detection = sum(
        1 for d in results["strict"]["details"]
        if not d["expected"] and not d["actual"]
    ) / max(len(hallucination_cases), 1)
    
    comp_detection = sum(
        1 for d in results["compositional"]["details"]
        if not d["expected"] and not d["actual"]
    ) / max(len(hallucination_cases), 1)
    
    reach_detection = sum(
        1 for d in results["reachability"]["details"]
        if not d["expected"] and not d["actual"]
    ) / max(len(hallucination_cases), 1)
    
    print(f"Hallucination Detection Rate:")
    print(f"  STRICT:        {strict_detection:.1%}")
    print(f"  COMPOSITIONAL: {comp_detection:.1%}")
    print(f"  REACHABILITY:  {reach_detection:.1%}")
    
    final_results = {
        "strict_accuracy": results["strict"]["correct"] / results["strict"]["total"],
        "compositional_accuracy": results["compositional"]["correct"] / results["compositional"]["total"],
        "reachability_accuracy": results["reachability"]["correct"] / results["reachability"]["total"],
        "strict_hallucination_detection": strict_detection,
        "compositional_hallucination_detection": comp_detection,
        "reachability_hallucination_detection": reach_detection,
        "details": results,
    }
    
    with open("/results/benchmark_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    volume.commit()
    
    return final_results


@app.local_entrypoint()
def main(experiment: str = "all", n_epochs: int = 100):
    """
    Entry point for `modal run`.
    
    Args:
        experiment: "embeddings", "attention", "benchmark", or "all"
        n_epochs: Number of training epochs
    """
    print(f"Starting Ontological Induction Experiment: {experiment}")
    print("=" * 60)
    
    if experiment in ["embeddings", "all"]:
        print("\n[1/3] Training Ontological Embeddings...")
        emb_results = train_ontological_embeddings.remote(n_epochs=n_epochs)
        print(f"  Separation ratio: {emb_results['separation_ratio']:.4f}")
    
    if experiment in ["attention", "all"]:
        print("\n[2/3] Training Ontological Attention...")
        attn_results = train_ontological_attention.remote(n_epochs=n_epochs // 2)
        print(f"  Invalid detection: {attn_results['invalid_detection_rate']:.1%}")
    
    if experiment in ["benchmark", "all"]:
        print("\n[3/3] Running Hallucination Detection Benchmark...")
        bench_results = benchmark_hallucination_detection.remote()
        print(f"  STRICT detection: {bench_results['strict_hallucination_detection']:.1%}")
    
    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("Results saved to Modal volume 'olog-results'")
    print("=" * 60)
