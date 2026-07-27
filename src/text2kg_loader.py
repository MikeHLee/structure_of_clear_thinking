# -*- coding: utf-8 -*-
"""
Text2KGBench Data Loader for Scaled Ontological Embedding Training

This module loads and processes the Text2KGBench dataset for training
ontological embeddings at scale (331+ types, 430 relations, 7,943+ triples).

Dataset Structure:
    Text2KGBench/data/dbpedia_webnlg/
    ├── ontologies/           # 19 domain ontologies (JSON)
    │   ├── 1_university_ontology.json
    │   ├── 2_musicalwork_ontology.json
    │   └── ...
    ├── train/                # Training examples
    └── test/                 # Test examples

Ontology JSON Format:
    {
        "title": "University Ontology",
        "id": "ont_1_university",
        "concepts": [{"qid": "University", "label": "University"}, ...],
        "relations": [{"pid": "director", "domain": "University", "range": "Person"}, ...]
    }

Usage:
    loader = Text2KGBenchLoader(data_dir="training_data/Text2KGBench")
    ontologies = loader.load_all_ontologies()
    triples = loader.to_triples(ontologies)
    
    # For PyTorch DataLoader
    dataset = OntologyTripleDataset(triples, ontologies)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import random

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Concept:
    """A type/concept in an ontology."""
    qid: str              # Unique identifier
    label: str            # Human-readable label
    ontology_id: str = "" # Source ontology


@dataclass
class Relation:
    """A relation/property connecting types."""
    pid: str              # Unique identifier  
    label: str            # Human-readable label
    domain: str           # Source type qid
    range: str            # Target type qid (or literal type like "string", "number")
    ontology_id: str = "" # Source ontology


@dataclass 
class Ontology:
    """A complete ontology with concepts and relations."""
    id: str
    title: str
    concepts: List[Concept] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    
    @property
    def types(self) -> Set[str]:
        """All concept qids."""
        return {c.qid for c in self.concepts}
    
    @property
    def morphisms(self) -> List[Tuple[str, str, str]]:
        """Relations as (domain, relation, range) triples."""
        return [(r.domain, r.pid, r.range) for r in self.relations 
                if r.range not in ("string", "number", "date", "")]
    
    def get_successors(self, type_qid: str) -> Dict[str, Set[str]]:
        """Get reachable types from a given type via relations."""
        successors = defaultdict(set)
        for r in self.relations:
            if r.domain == type_qid and r.range not in ("string", "number", "date", ""):
                successors[r.pid].add(r.range)
        return dict(successors)


@dataclass
class ScaleStats:
    """Statistics for scaled dataset."""
    n_ontologies: int = 0
    n_types: int = 0
    n_relations: int = 0
    n_triples: int = 0
    types_per_ontology: Dict[str, int] = field(default_factory=dict)
    relations_per_ontology: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class Text2KGBenchLoader:
    """
    Loader for Text2KGBench ontology data.
    
    Supports:
    - Loading individual or all ontologies
    - Converting to various formats (triples, PyKEEN, our format)
    - Computing reachability graphs for attention masking
    - Generating training examples with hard negatives
    """
    
    # Literal types to exclude from relation targets
    LITERAL_TYPES = {"string", "number", "date", ""}
    
    def __init__(
        self,
        data_dir: str = "training_data/Text2KGBench",
        dataset: str = "dbpedia_webnlg"
    ):
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.ontologies_dir = self.data_dir / "data" / dataset / "ontologies"
        
        # Cache
        self._ontologies: Dict[str, Ontology] = {}
        self._all_types: Set[str] = set()
        self._all_relations: Set[str] = set()
        self._type_to_ontology: Dict[str, str] = {}
    
    def load_ontology(self, json_path: Path) -> Ontology:
        """Load a single ontology from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        ont_id = data.get("id", json_path.stem)
        title = data.get("title", ont_id)
        
        # Parse concepts
        concepts = []
        for c in data.get("concepts", []):
            concepts.append(Concept(
                qid=c["qid"],
                label=c.get("label", c["qid"]),
                ontology_id=ont_id
            ))
        
        # Parse relations (only those with non-literal ranges)
        relations = []
        for r in data.get("relations", []):
            relations.append(Relation(
                pid=r["pid"],
                label=r.get("label", r["pid"]),
                domain=r["domain"],
                range=r["range"],
                ontology_id=ont_id
            ))
        
        return Ontology(
            id=ont_id,
            title=title,
            concepts=concepts,
            relations=relations
        )
    
    def load_all_ontologies(self) -> Dict[str, Ontology]:
        """Load all ontologies from the dataset."""
        if self._ontologies:
            return self._ontologies
        
        ontology_files = sorted(self.ontologies_dir.glob("*_ontology.json"))
        
        for ont_file in ontology_files:
            ont = self.load_ontology(ont_file)
            self._ontologies[ont.id] = ont
            
            # Update global indices
            for c in ont.concepts:
                self._all_types.add(c.qid)
                self._type_to_ontology[c.qid] = ont.id
            
            for r in ont.relations:
                self._all_relations.add(r.pid)
        
        return self._ontologies
    
    def get_stats(self) -> ScaleStats:
        """Get dataset statistics."""
        if not self._ontologies:
            self.load_all_ontologies()
        
        stats = ScaleStats(
            n_ontologies=len(self._ontologies),
            n_types=len(self._all_types),
            n_relations=len(self._all_relations),
        )
        
        total_triples = 0
        for ont_id, ont in self._ontologies.items():
            stats.types_per_ontology[ont_id] = len(ont.concepts)
            stats.relations_per_ontology[ont_id] = len(ont.relations)
            total_triples += len(ont.morphisms)
        
        stats.n_triples = total_triples
        return stats
    
    def to_triples(
        self,
        include_ontology_membership: bool = True
    ) -> List[Tuple[str, str, str]]:
        """
        Convert all ontologies to flat triple list.
        
        Returns:
            List of (head, relation, tail) triples
        """
        if not self._ontologies:
            self.load_all_ontologies()
        
        triples = []
        
        for ont_id, ont in self._ontologies.items():
            # Add relation triples
            for domain, rel, range_ in ont.morphisms:
                triples.append((domain, rel, range_))
            
            # Add ontology membership
            if include_ontology_membership:
                for c in ont.concepts:
                    triples.append((c.qid, "belongsTo", f"ont_{ont_id}"))
        
        return triples
    
    def build_reachability_graph(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Build type-to-reachable-types graph for attention masking.
        
        Returns:
            {type_qid: {relation_pid: {reachable_type_qids}}}
        """
        if not self._ontologies:
            self.load_all_ontologies()
        
        reach = defaultdict(lambda: defaultdict(set))
        
        for ont in self._ontologies.values():
            for r in ont.relations:
                if r.range not in self.LITERAL_TYPES:
                    reach[r.domain][r.pid].add(r.range)
        
        return dict(reach)
    
    def build_successor_map(self) -> Dict[str, Set[str]]:
        """
        Build simple type-to-successor-types map (relation-agnostic).
        
        Returns:
            {type_qid: {all_reachable_type_qids}}
        """
        reach = self.build_reachability_graph()
        
        successor_map = {}
        for type_qid, rel_dict in reach.items():
            all_successors = set()
            for successors in rel_dict.values():
                all_successors.update(successors)
            successor_map[type_qid] = all_successors
        
        return successor_map
    
    def to_our_format(self) -> Dict[str, Dict]:
        """
        Convert to our toy ontology format for compatibility.
        
        Returns:
            {ont_name: {"types": [...], "morphisms": [...]}}
        """
        if not self._ontologies:
            self.load_all_ontologies()
        
        result = {}
        for ont_id, ont in self._ontologies.items():
            result[ont_id] = {
                "types": [c.qid for c in ont.concepts],
                "morphisms": ont.morphisms
            }
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    
    class OntologyTripleDataset(Dataset):
        """
        PyTorch Dataset for ontological triple training.
        
        Generates:
        - Positive samples: valid (domain, relation, range) triples
        - Negative samples: corrupted triples (wrong domain or range)
        
        Supports tier-based negative sampling:
        - L0: Same type (trivial)
        - L1: Sibling in same ontology
        - L2: Different ontology, same relation domain/range
        - L3: Random type (adversarial)
        """
        
        def __init__(
            self,
            loader: Text2KGBenchLoader,
            n_negatives_per_positive: int = 4,
            tier_distribution: Dict[int, float] = None,
            seed: int = 42
        ):
            self.loader = loader
            self.n_negatives = n_negatives_per_positive
            self.rng = random.Random(seed)
            
            # Default tier distribution
            if tier_distribution is None:
                tier_distribution = {
                    0: 0.0,   # L0: never (same type = trivial)
                    1: 0.2,   # L1: sibling
                    2: 0.3,   # L2: cousin  
                    3: 0.5,   # L3: random (adversarial)
                }
            self.tier_distribution = tier_distribution
            
            # Load data
            self.ontologies = loader.load_all_ontologies()
            self.triples = loader.to_triples(include_ontology_membership=False)
            self.successor_map = loader.build_successor_map()
            self.all_types = list(loader._all_types)
            self.type_to_ontology = loader._type_to_ontology
            
            # Group types by ontology
            self.ontology_types = defaultdict(list)
            for t, ont in self.type_to_ontology.items():
                self.ontology_types[ont].append(t)
        
        def __len__(self) -> int:
            return len(self.triples)
        
        def __getitem__(self, idx: int) -> Dict[str, Any]:
            """
            Get a training example with positive and negative samples.
            
            Returns:
                {
                    "anchor_type": str,
                    "positive_type": str,
                    "relation": str,
                    "negative_types": List[str],
                    "negative_tiers": List[int],
                }
            """
            head, rel, tail = self.triples[idx]
            
            # Generate negatives
            negatives = []
            tiers = []
            
            for _ in range(self.n_negatives):
                tier = self._sample_tier()
                neg_type = self._sample_negative(head, tail, tier)
                negatives.append(neg_type)
                tiers.append(tier)
            
            return {
                "anchor_type": head,
                "positive_type": tail,
                "relation": rel,
                "negative_types": negatives,
                "negative_tiers": tiers,
            }
        
        def _sample_tier(self) -> int:
            """Sample a negative tier based on distribution."""
            r = self.rng.random()
            cumulative = 0.0
            for tier, prob in self.tier_distribution.items():
                cumulative += prob
                if r < cumulative:
                    return tier
            return 3  # Default to L3
        
        def _sample_negative(self, anchor: str, positive: str, tier: int) -> str:
            """
            Sample a negative type based on tier.
            
            Tiers:
            - L0: Same as positive (trivial - avoided)
            - L1: Same ontology, different type
            - L2: Different ontology, semantically similar
            - L3: Random type (adversarial)
            """
            anchor_ont = self.type_to_ontology.get(anchor, "")
            valid_successors = self.successor_map.get(anchor, set())
            
            if tier == 0:
                # L0: Same type (shouldn't happen with default distribution)
                return positive
            
            elif tier == 1:
                # L1: Sibling - same ontology, different type, not valid successor
                siblings = [t for t in self.ontology_types[anchor_ont]
                           if t != positive and t not in valid_successors]
                if siblings:
                    return self.rng.choice(siblings)
                # Fallback to L3
                return self._sample_negative(anchor, positive, 3)
            
            elif tier == 2:
                # L2: Cousin - different ontology
                other_onts = [ont for ont in self.ontology_types.keys() if ont != anchor_ont]
                if other_onts:
                    other_ont = self.rng.choice(other_onts)
                    cousins = [t for t in self.ontology_types[other_ont]
                              if t not in valid_successors]
                    if cousins:
                        return self.rng.choice(cousins)
                # Fallback to L3
                return self._sample_negative(anchor, positive, 3)
            
            else:
                # L3: Random type (adversarial)
                invalid_types = [t for t in self.all_types 
                                if t not in valid_successors and t != positive]
                if invalid_types:
                    return self.rng.choice(invalid_types)
                return self.rng.choice(self.all_types)
    
    
    class TextualTripleDataset(Dataset):
        """
        Dataset that converts triples to textual statements for LLM encoding.
        
        Formats triples as:
        - "[CLS] {head} has relation {rel} to {tail} [SEP]"
        - "[CLS] A {head} can {rel} a {tail} [SEP]"
        
        For use with sentence transformers or LLM encoders.
        """
        
        TEMPLATES = [
            "{head} has relation {rel} to {tail}",
            "A {head} can {rel} a {tail}",
            "{head} {rel} {tail}",
            "The {head} is connected to {tail} via {rel}",
        ]
        
        def __init__(
            self,
            loader: Text2KGBenchLoader,
            template_idx: int = 0,
            n_negatives: int = 4,
            seed: int = 42
        ):
            self.loader = loader
            self.template = self.TEMPLATES[template_idx]
            self.n_negatives = n_negatives
            self.rng = random.Random(seed)
            
            self.ontologies = loader.load_all_ontologies()
            self.triples = loader.to_triples(include_ontology_membership=False)
            self.successor_map = loader.build_successor_map()
            self.all_types = list(loader._all_types)
        
        def __len__(self) -> int:
            return len(self.triples)
        
        def __getitem__(self, idx: int) -> Dict[str, str]:
            """
            Get textual representations of anchor, positive, and negatives.
            
            Returns:
                {
                    "anchor_text": str,
                    "positive_text": str,
                    "negative_texts": List[str],
                }
            """
            head, rel, tail = self.triples[idx]
            
            # Format as text
            anchor_text = self.template.format(head=head, rel=rel, tail=tail)
            positive_text = anchor_text  # Same for contrastive anchor
            
            # Generate negative texts
            valid_successors = self.successor_map.get(head, set())
            invalid_types = [t for t in self.all_types 
                           if t not in valid_successors and t != tail]
            
            negative_texts = []
            for _ in range(self.n_negatives):
                if invalid_types:
                    neg_tail = self.rng.choice(invalid_types)
                else:
                    neg_tail = self.rng.choice(self.all_types)
                
                neg_text = self.template.format(head=head, rel=rel, tail=neg_tail)
                negative_texts.append(neg_text)
            
            return {
                "anchor_text": anchor_text,
                "positive_text": positive_text,
                "negative_texts": negative_texts,
                "head": head,
                "rel": rel,
                "tail": tail,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL GPU TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

MODAL_TRAINING_TEMPLATE = '''
"""
Modal GPU training script for scaled ontological embeddings.

Run with: modal run text2kg_loader.py::train_scaled
"""

import modal

app = modal.App("olog-embeddings-scaled")

image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy", 
    "tqdm",
    "sentence-transformers",
)

@app.function(
    image=image,
    gpu="T4",
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_scaled(
    embed_dim: int = 128,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 0.001,
):
    """Train scaled ontological embeddings on Modal GPU."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Import our modules (would need to be uploaded)
    # from text2kg_loader import Text2KGBenchLoader, OntologyTripleDataset
    # from contrastive_losses import InfoNCELoss
    # from memory_bank import OntologyMemoryBank
    
    print(f"Training with embed_dim={embed_dim}, batch_size={batch_size}")
    print(f"Device: {{torch.cuda.get_device_name(0)}}")
    
    # ... training loop ...
    
    return {{"status": "completed", "epochs": epochs}}


if __name__ == "__main__":
    # Local test
    loader = Text2KGBenchLoader()
    stats = loader.get_stats()
    print(f"Loaded {{stats.n_ontologies}} ontologies")
    print(f"Types: {{stats.n_types}}, Relations: {{stats.n_relations}}, Triples: {{stats.n_triples}}")
'''


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Demo the Text2KGBench loader."""
    print("=" * 70)
    print("Text2KGBench Data Loader Demo")
    print("=" * 70)
    
    # Initialize loader
    loader = Text2KGBenchLoader()
    
    # Load ontologies
    print("\nLoading ontologies...")
    ontologies = loader.load_all_ontologies()
    
    # Get stats
    stats = loader.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  Ontologies: {stats.n_ontologies}")
    print(f"  Types: {stats.n_types}")
    print(f"  Relations: {stats.n_relations}")
    print(f"  Triples: {stats.n_triples}")
    
    # Show per-ontology breakdown
    print(f"\nPer-Ontology Breakdown:")
    for ont_id, n_types in sorted(stats.types_per_ontology.items()):
        n_rels = stats.relations_per_ontology[ont_id]
        print(f"  {ont_id}: {n_types} types, {n_rels} relations")
    
    # Build reachability graph
    print(f"\nBuilding reachability graph...")
    reach = loader.build_successor_map()
    print(f"  Types with successors: {len(reach)}")
    
    # Sample some reachability
    sample_type = list(reach.keys())[0] if reach else None
    if sample_type:
        print(f"  Example: {sample_type} -> {reach[sample_type]}")
    
    # Convert to triples
    print(f"\nConverting to triples...")
    triples = loader.to_triples()
    print(f"  Total triples (with membership): {len(triples)}")
    
    # PyTorch dataset demo
    if TORCH_AVAILABLE:
        print(f"\nCreating PyTorch Dataset...")
        dataset = OntologyTripleDataset(loader, n_negatives_per_positive=4)
        print(f"  Dataset size: {len(dataset)}")
        
        # Sample item
        sample = dataset[0]
        print(f"  Sample item:")
        print(f"    Anchor: {sample['anchor_type']}")
        print(f"    Positive: {sample['positive_type']}")
        print(f"    Relation: {sample['relation']}")
        print(f"    Negatives: {sample['negative_types']}")
        print(f"    Tiers: {sample['negative_tiers']}")
        
        # Textual dataset demo
        print(f"\nCreating Textual Dataset (for LLM encoding)...")
        text_dataset = TextualTripleDataset(loader, template_idx=0)
        text_sample = text_dataset[0]
        print(f"  Anchor text: \"{text_sample['anchor_text']}\"")
        print(f"  Negative texts: {text_sample['negative_texts'][:2]}")
    
    # Comparison to toy scale
    print(f"\n" + "=" * 70)
    print("Scale Comparison: Toy vs. Text2KGBench")
    print("=" * 70)
    print(f"| Metric | Toy Scale | Text2KGBench |")
    print(f"|--------|-----------|--------------|")
    print(f"| Ontologies | 4 | {stats.n_ontologies} |")
    print(f"| Types | 24 | {stats.n_types} |")
    print(f"| Relations | 24 | {stats.n_relations} |")
    print(f"| Triples | ~24 | {stats.n_triples} |")
    print(f"| Expected separation | 2.71× | 1.5-2.5× |")
    
    print(f"\n✓ Text2KGBench loader ready for scaled training!")
    print(f"\nNext steps:")
    print(f"  1. Run: modal run text2kg_loader.py::train_scaled")
    print(f"  2. Use with InfoNCE + Memory Bank for training")


if __name__ == "__main__":
    main()
