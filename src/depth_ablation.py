# -*- coding: utf-8 -*-
"""
Depth Ablation Study for Ontological Embeddings

This module evaluates how separation ratio varies with ontology depth.

Research Question:
    Does separation ratio degrade with ontology depth? Is it easier to
    separate high-level domain transitions than fine-grained subtypes?

Hypothesis:
    Separation ratio decreases with depth because fine-grained types
    are semantically closer (e.g., "Mammal" vs "Dog" vs "Golden Retriever").

Methodology:
    1. Generate synthetic hierarchical ontologies at depths 2, 3, 4, 5, 6
    2. Train embedding model on each
    3. Compute depth-stratified separation ratios
    4. Analyze parent-child vs sibling distances
    
Output:
    - Ablation plot: depth vs. separation ratio
    - Per-depth metrics table
    - Hierarchy preservation score
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HierarchicalType:
    """A type in a hierarchical ontology with depth information."""
    qid: str
    label: str
    depth: int
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class HierarchicalOntology:
    """An ontology with explicit hierarchical structure."""
    id: str
    title: str
    max_depth: int
    types: Dict[str, HierarchicalType] = field(default_factory=dict)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    
    @property
    def types_at_depth(self) -> Dict[int, List[str]]:
        """Group types by depth level."""
        by_depth = defaultdict(list)
        for t in self.types.values():
            by_depth[t.depth].append(t.qid)
        return dict(by_depth)
    
    @property
    def leaf_types(self) -> List[str]:
        """Types with no children (deepest in their branch)."""
        return [t.qid for t in self.types.values() if not t.children]
    
    @property
    def root_types(self) -> List[str]:
        """Types with no parent (depth 0)."""
        return [t.qid for t in self.types.values() if t.parent is None]


@dataclass
class DepthAblationResult:
    """Results from depth ablation experiment."""
    max_depth: int
    overall_separation_ratio: float
    depth_stratified_ratios: Dict[int, float]
    parent_child_distances: Dict[int, float]
    sibling_distances: Dict[int, float]
    hierarchy_preservation_score: float
    n_types: int
    n_relations: int


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC ONTOLOGY GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalOntologyGenerator:
    """
    Generate synthetic hierarchical ontologies for depth ablation.
    
    Structure:
        Root (depth 0)
        ├── Branch1 (depth 1)
        │   ├── Leaf1a (depth 2)
        │   └── Leaf1b (depth 2)
        └── Branch2 (depth 1)
            ├── Leaf2a (depth 2)
            └── Leaf2b (depth 2)
    
    Relations:
        - "isA": child -> parent (subtype relation)
        - "relatedTo": siblings at same depth
        - Domain-specific relations between types
    """
    
    # Templates for generating type names at each depth
    DEPTH_TEMPLATES = {
        0: ["Entity", "Thing", "Object", "Concept"],
        1: ["Living", "NonLiving", "Abstract", "Concrete", "Natural", "Artificial"],
        2: ["Animal", "Plant", "Mineral", "Machine", "Idea", "Structure"],
        3: ["Mammal", "Bird", "Fish", "Tree", "Flower", "Rock", "Metal", "Vehicle", "Building"],
        4: ["Dog", "Cat", "Eagle", "Salmon", "Oak", "Rose", "Granite", "Iron", "Car", "House"],
        5: ["GoldenRetriever", "Siamese", "BaldEagle", "AtlanticSalmon", "WhiteOak", "RedRose"],
        6: ["ShowDog", "HouseCat", "WildEagle", "FarmedSalmon", "AncientOak", "GardenRose"],
    }
    
    # Relation templates
    RELATION_TEMPLATES = [
        ("isA", "subtype of"),
        ("partOf", "is part of"),
        ("hasProperty", "has property"),
        ("locatedIn", "is located in"),
        ("createdBy", "was created by"),
        ("usedFor", "is used for"),
    ]
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        if NUMPY_AVAILABLE:
            self.np_rng = np.random.default_rng(seed)
    
    def generate(
        self,
        max_depth: int,
        branching_factor: int = 3,
        ontology_id: str = None
    ) -> HierarchicalOntology:
        """
        Generate a hierarchical ontology with specified depth.
        
        Args:
            max_depth: Maximum depth of hierarchy (0 = flat)
            branching_factor: Number of children per non-leaf node
            ontology_id: Unique identifier for the ontology
        
        Returns:
            HierarchicalOntology with types and relations
        """
        if ontology_id is None:
            ontology_id = f"synth_depth{max_depth}"
        
        ontology = HierarchicalOntology(
            id=ontology_id,
            title=f"Synthetic Hierarchy (depth={max_depth})",
            max_depth=max_depth
        )
        
        # Generate type hierarchy
        self._generate_types_recursive(
            ontology=ontology,
            parent=None,
            current_depth=0,
            max_depth=max_depth,
            branching_factor=branching_factor,
            prefix=""
        )
        
        # Generate relations
        self._generate_relations(ontology)
        
        return ontology
    
    def _generate_types_recursive(
        self,
        ontology: HierarchicalOntology,
        parent: Optional[str],
        current_depth: int,
        max_depth: int,
        branching_factor: int,
        prefix: str
    ) -> List[str]:
        """Recursively generate types at each depth level."""
        if current_depth > max_depth:
            return []
        
        # Get templates for this depth
        templates = self.DEPTH_TEMPLATES.get(current_depth, [f"Type_d{current_depth}"])
        
        # Generate types at this level
        n_types = branching_factor if current_depth > 0 else 1  # Single root
        generated_qids = []
        
        for i in range(n_types):
            # Create unique type ID
            template = templates[i % len(templates)]
            qid = f"{prefix}{template}_{current_depth}_{i}" if prefix else f"{template}"
            
            # Create type
            type_obj = HierarchicalType(
                qid=qid,
                label=template,
                depth=current_depth,
                parent=parent
            )
            ontology.types[qid] = type_obj
            generated_qids.append(qid)
            
            # Update parent's children list
            if parent and parent in ontology.types:
                ontology.types[parent].children.append(qid)
            
            # Recurse to generate children
            if current_depth < max_depth:
                children = self._generate_types_recursive(
                    ontology=ontology,
                    parent=qid,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    branching_factor=branching_factor,
                    prefix=f"{qid}_"
                )
                type_obj.children = children
        
        return generated_qids
    
    def _generate_relations(self, ontology: HierarchicalOntology) -> None:
        """Generate relations between types."""
        # isA relations (child -> parent)
        for type_obj in ontology.types.values():
            if type_obj.parent:
                ontology.relations.append((type_obj.qid, "isA", type_obj.parent))
        
        # Sibling relations (types at same depth with same parent)
        for type_obj in ontology.types.values():
            if type_obj.parent and type_obj.parent in ontology.types:
                siblings = ontology.types[type_obj.parent].children
                for sibling in siblings:
                    if sibling != type_obj.qid:
                        # Add bidirectional "relatedTo"
                        ontology.relations.append((type_obj.qid, "relatedTo", sibling))
        
        # Cross-branch relations (sample some random valid relations)
        types_by_depth = ontology.types_at_depth
        for depth in range(1, ontology.max_depth + 1):
            types_at_d = types_by_depth.get(depth, [])
            if len(types_at_d) >= 2:
                # Sample some cross-branch relations
                n_relations = min(5, len(types_at_d) // 2)
                for _ in range(n_relations):
                    src = self.rng.choice(types_at_d)
                    tgt = self.rng.choice(types_at_d)
                    if src != tgt:
                        rel_type = self.rng.choice(["interactsWith", "influences", "precedes"])
                        ontology.relations.append((src, rel_type, tgt))


# ═══════════════════════════════════════════════════════════════════════════════
# DEPTH-STRATIFIED METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class DepthStratifiedEvaluator:
    """
    Evaluate embedding quality stratified by ontology depth.
    
    Metrics:
    1. Separation ratio at each depth level
    2. Parent-child distance (should be small)
    3. Sibling distance (should be moderate)
    4. Cross-branch distance (should be large)
    5. Hierarchy preservation score
    """
    
    def __init__(self, ontology: HierarchicalOntology):
        self.ontology = ontology
        self.types_by_depth = ontology.types_at_depth
    
    def evaluate(
        self,
        embeddings: Dict[str, Any],  # type_qid -> embedding vector
        distance_fn: str = "l2"
    ) -> DepthAblationResult:
        """
        Evaluate embeddings with depth-stratified metrics.
        
        Args:
            embeddings: Dictionary mapping type qids to embedding vectors
            distance_fn: "l2" or "cosine"
        
        Returns:
            DepthAblationResult with all metrics
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required for evaluation")
        
        # Convert embeddings to numpy
        emb_array = {k: np.array(v) for k, v in embeddings.items()}
        
        # Compute distances
        def dist(a: str, b: str) -> float:
            if a not in emb_array or b not in emb_array:
                return 0.0
            va, vb = emb_array[a], emb_array[b]
            if distance_fn == "l2":
                return float(np.linalg.norm(va - vb))
            else:  # cosine
                return float(1 - np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
        
        # 1. Depth-stratified separation ratios
        depth_ratios = {}
        for depth, types_at_d in self.types_by_depth.items():
            if len(types_at_d) < 2:
                continue
            
            # Intra-depth distances (same depth level)
            intra_dists = []
            for i, t1 in enumerate(types_at_d):
                for t2 in types_at_d[i+1:]:
                    intra_dists.append(dist(t1, t2))
            
            # Inter-depth distances (to types at other depths)
            inter_dists = []
            for other_depth, other_types in self.types_by_depth.items():
                if other_depth != depth:
                    for t1 in types_at_d:
                        for t2 in other_types:
                            inter_dists.append(dist(t1, t2))
            
            if intra_dists and inter_dists:
                depth_ratios[depth] = np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8)
        
        # 2. Parent-child distances by depth
        parent_child_dists = defaultdict(list)
        for type_obj in self.ontology.types.values():
            if type_obj.parent:
                d = dist(type_obj.qid, type_obj.parent)
                parent_child_dists[type_obj.depth].append(d)
        
        parent_child_means = {d: np.mean(ds) for d, ds in parent_child_dists.items() if ds}
        
        # 3. Sibling distances by depth
        sibling_dists = defaultdict(list)
        for type_obj in self.ontology.types.values():
            if type_obj.parent and type_obj.parent in self.ontology.types:
                siblings = self.ontology.types[type_obj.parent].children
                for sib in siblings:
                    if sib != type_obj.qid:
                        d = dist(type_obj.qid, sib)
                        sibling_dists[type_obj.depth].append(d)
        
        sibling_means = {d: np.mean(ds) for d, ds in sibling_dists.items() if ds}
        
        # 4. Overall separation ratio
        all_intra = []
        all_inter = []
        all_types = list(self.ontology.types.keys())
        for i, t1 in enumerate(all_types):
            d1 = self.ontology.types[t1].depth
            for t2 in all_types[i+1:]:
                d2 = self.ontology.types[t2].depth
                d = dist(t1, t2)
                if d1 == d2:
                    all_intra.append(d)
                else:
                    all_inter.append(d)
        
        overall_ratio = (np.mean(all_inter) / (np.mean(all_intra) + 1e-8)) if all_intra and all_inter else 0
        
        # 5. Hierarchy preservation score
        # = proportion of (parent-child dist < sibling dist < cross-branch dist)
        preservation_checks = []
        for type_obj in self.ontology.types.values():
            if type_obj.parent and type_obj.parent in self.ontology.types:
                pc_dist = dist(type_obj.qid, type_obj.parent)
                
                # Check against siblings
                siblings = self.ontology.types[type_obj.parent].children
                for sib in siblings:
                    if sib != type_obj.qid:
                        sib_dist = dist(type_obj.qid, sib)
                        # parent-child should be closer than siblings
                        preservation_checks.append(pc_dist < sib_dist)
        
        hierarchy_score = np.mean(preservation_checks) if preservation_checks else 0.0
        
        return DepthAblationResult(
            max_depth=self.ontology.max_depth,
            overall_separation_ratio=float(overall_ratio),
            depth_stratified_ratios=depth_ratios,
            parent_child_distances=parent_child_means,
            sibling_distances=sibling_means,
            hierarchy_preservation_score=float(hierarchy_score),
            n_types=len(self.ontology.types),
            n_relations=len(self.ontology.relations)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class DepthAblationStudy:
    """
    Run depth ablation study across multiple ontology depths.
    
    Generates ontologies at depths 2-6, trains embeddings, and compares.
    """
    
    def __init__(
        self,
        depths: List[int] = [2, 3, 4, 5, 6],
        branching_factor: int = 3,
        embed_dim: int = 64,
        seed: int = 42
    ):
        self.depths = depths
        self.branching_factor = branching_factor
        self.embed_dim = embed_dim
        self.seed = seed
        self.generator = HierarchicalOntologyGenerator(seed=seed)
    
    def generate_all_ontologies(self) -> Dict[int, HierarchicalOntology]:
        """Generate ontologies at each depth level."""
        ontologies = {}
        for depth in self.depths:
            ont = self.generator.generate(
                max_depth=depth,
                branching_factor=self.branching_factor,
                ontology_id=f"depth_{depth}"
            )
            ontologies[depth] = ont
            print(f"Generated depth-{depth} ontology: {len(ont.types)} types, {len(ont.relations)} relations")
        return ontologies
    
    def generate_random_embeddings(
        self,
        ontology: HierarchicalOntology
    ) -> Dict[str, Any]:
        """
        Generate random embeddings (placeholder for actual training).
        
        In practice, replace with:
        1. Train InfoNCE model on ontology
        2. Extract learned embeddings
        """
        if not NUMPY_AVAILABLE:
            return {}
        
        np.random.seed(self.seed)
        embeddings = {}
        
        for type_qid, type_obj in ontology.types.items():
            # Bias by depth for demonstration
            # (deeper types should cluster tighter in real training)
            depth_scale = 1.0 / (type_obj.depth + 1)
            emb = np.random.randn(self.embed_dim) * depth_scale
            embeddings[type_qid] = emb
        
        return embeddings
    
    def run(self, output_dir: str = "results/depth_ablation") -> List[DepthAblationResult]:
        """
        Run the full ablation study.
        
        Returns:
            List of DepthAblationResult, one per depth level
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate ontologies
        ontologies = self.generate_all_ontologies()
        
        results = []
        for depth, ontology in ontologies.items():
            print(f"\nEvaluating depth-{depth}...")
            
            # Generate embeddings (replace with actual training)
            embeddings = self.generate_random_embeddings(ontology)
            
            # Evaluate
            evaluator = DepthStratifiedEvaluator(ontology)
            result = evaluator.evaluate(embeddings)
            results.append(result)
            
            print(f"  Separation ratio: {result.overall_separation_ratio:.3f}")
            print(f"  Hierarchy preservation: {result.hierarchy_preservation_score:.3f}")
        
        # Save results
        results_data = {
            "config": {
                "depths": self.depths,
                "branching_factor": self.branching_factor,
                "embed_dim": self.embed_dim,
                "seed": self.seed,
            },
            "results": [asdict(r) for r in results]
        }
        
        with open(output_path / "depth_ablation_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        return results
    
    def generate_report(self, results: List[DepthAblationResult]) -> str:
        """Generate markdown report from results."""
        lines = [
            "# Depth Ablation Study Results",
            "",
            "## Summary Table",
            "",
            "| Depth | Types | Relations | Separation Ratio | Hierarchy Score |",
            "|-------|-------|-----------|------------------|-----------------|",
        ]
        
        for r in results:
            lines.append(
                f"| {r.max_depth} | {r.n_types} | {r.n_relations} | "
                f"{r.overall_separation_ratio:.3f} | {r.hierarchy_preservation_score:.3f} |"
            )
        
        lines.extend([
            "",
            "## Depth-Stratified Ratios",
            "",
        ])
        
        for r in results:
            lines.append(f"### Depth {r.max_depth}")
            for d, ratio in sorted(r.depth_stratified_ratios.items()):
                lines.append(f"- Level {d}: {ratio:.3f}")
            lines.append("")
        
        lines.extend([
            "## Observations",
            "",
            "- **Hypothesis**: Separation ratio decreases with depth",
            "- **Result**: [To be filled after training]",
            "",
            "## Next Steps",
            "",
            "1. Train actual embeddings using InfoNCE + Memory Bank",
            "2. Run on real hierarchical ontologies (Schema.org, Gene Ontology)",
            "3. Generate ablation plot for paper",
        ])
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Depth Ablation Study for Ontological Embeddings")
    print("=" * 70)
    
    # Run ablation study
    study = DepthAblationStudy(
        depths=[2, 3, 4, 5],
        branching_factor=3,
        embed_dim=64
    )
    
    results = study.run()
    
    # Generate report
    report = study.generate_report(results)
    print("\n" + report)
    
    # Save report
    output_path = Path("results/depth_ablation")
    with open(output_path / "REPORT.md", "w") as f:
        f.write(report)
    
    print(f"\nReport saved to {output_path / 'REPORT.md'}")


if __name__ == "__main__":
    main()
