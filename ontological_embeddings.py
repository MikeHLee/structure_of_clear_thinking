"""
Ontological Embeddings via Hydration Manifests

Implements graph-based embeddings for Ologs through selective edge materialization
during graph walks. Unlike dense vector embeddings, Hydration Manifests preserve
the categorical structure and enable interpretable, composable representations.

Key Concepts:
- Hydration Manifest: A sparse representation specifying which paths to materialize
- Graph Walk: Traverse Olog following morphism composition
- Selective Hydration: Only expand relevant subgraphs based on query/context
- Categorical Embedding: Embeddings that respect functorial composition

Usage:
    from ontological_embeddings import OlogEmbedder, HydrationManifest
    
    embedder = OlogEmbedder(olog_graph)
    manifest = embedder.create_manifest(root="Customer", depth=2)
    embedding = embedder.hydrate(manifest)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np

from olog_core import OlogGraph, OlogNode, OlogMorphism
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_olog_types(olog: OlogGraph) -> Dict[str, OlogNode]:
    """Extract types from OlogGraph."""
    types = {}
    for node in olog.graph.nodes():
        data = olog.graph.nodes[node].get('data')
        if data:
            types[node] = data
        else:
            types[node] = OlogNode(name=node)
    return types


def get_olog_morphisms(olog: OlogGraph) -> Dict[str, OlogMorphism]:
    """Extract morphisms from OlogGraph."""
    morphisms = {}
    for u, v, key, data in olog.graph.edges(keys=True, data=True):
        morph_data = data.get('data')
        if morph_data:
            morphisms[f"{u}->{v}:{key}"] = morph_data
        else:
            morphisms[f"{u}->{v}:{key}"] = OlogMorphism(source=u, target=v, label=key)
    return morphisms


class HydrationStrategy(Enum):
    """Strategy for selecting which edges to hydrate."""
    FULL = "full"           # Hydrate all reachable edges
    BFS = "bfs"             # Breadth-first up to depth limit
    DFS = "dfs"             # Depth-first up to depth limit  
    TYPED = "typed"         # Only edges matching type constraints
    WEIGHTED = "weighted"   # Prioritize by edge weights/scores
    SEMANTIC = "semantic"   # Use semantic similarity for selection


@dataclass
class HydrationManifest:
    """
    Specifies which parts of an Olog to materialize (hydrate).
    
    A manifest is a sparse representation that can be:
    - Serialized efficiently
    - Composed with other manifests
    - Used to reconstruct subgraphs on demand
    """
    root_type: str
    included_types: Set[str] = field(default_factory=set)
    included_aspects: Set[Tuple[str, str, str]] = field(default_factory=set)  # (source, label, target)
    depth: int = 2
    strategy: HydrationStrategy = HydrationStrategy.BFS
    type_constraints: Optional[Set[str]] = None
    edge_constraints: Optional[Set[str]] = None
    
    def __post_init__(self):
        self.included_types.add(self.root_type)
    
    def add_path(self, path: List[Tuple[str, str, str]]):
        """Add a complete path to the manifest."""
        for source, label, target in path:
            self.included_types.add(source)
            self.included_types.add(target)
            self.included_aspects.add((source, label, target))
    
    def merge(self, other: 'HydrationManifest') -> 'HydrationManifest':
        """Merge two manifests (union of hydrated elements)."""
        merged = HydrationManifest(
            root_type=self.root_type,
            included_types=self.included_types | other.included_types,
            included_aspects=self.included_aspects | other.included_aspects,
            depth=max(self.depth, other.depth),
            strategy=self.strategy,
        )
        return merged
    
    def intersect(self, other: 'HydrationManifest') -> 'HydrationManifest':
        """Intersect two manifests (common hydrated elements)."""
        return HydrationManifest(
            root_type=self.root_type,
            included_types=self.included_types & other.included_types,
            included_aspects=self.included_aspects & other.included_aspects,
            depth=min(self.depth, other.depth),
            strategy=self.strategy,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest for storage/transmission."""
        return {
            "root_type": self.root_type,
            "included_types": list(self.included_types),
            "included_aspects": [list(a) for a in self.included_aspects],
            "depth": self.depth,
            "strategy": self.strategy.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HydrationManifest':
        """Deserialize manifest."""
        return cls(
            root_type=data["root_type"],
            included_types=set(data["included_types"]),
            included_aspects={tuple(a) for a in data["included_aspects"]},
            depth=data["depth"],
            strategy=HydrationStrategy(data["strategy"]),
        )


@dataclass
class OlogEmbedding:
    """
    A categorical embedding derived from hydrating a manifest.
    
    Unlike dense vectors, this preserves:
    - Type structure (which nodes are included)
    - Morphism composition (how edges connect)
    - Path equivalences (commutative facts)
    """
    manifest: HydrationManifest
    type_vectors: Dict[str, np.ndarray]  # Type name -> embedding vector
    edge_vectors: Dict[Tuple[str, str, str], np.ndarray]  # Edge -> embedding
    composition_matrix: Optional[np.ndarray] = None  # Path composition structure
    
    @property
    def dim(self) -> int:
        """Embedding dimension."""
        if self.type_vectors:
            return next(iter(self.type_vectors.values())).shape[0]
        return 0
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense vector (loses structure, for compatibility)."""
        vectors = list(self.type_vectors.values()) + list(self.edge_vectors.values())
        if not vectors:
            return np.zeros(64)
        # Aggregate via mean pooling
        return np.mean(vectors, axis=0)
    
    def similarity(self, other: 'OlogEmbedding') -> float:
        """Compute categorical similarity (respects structure)."""
        # Type overlap (Jaccard)
        type_overlap = len(self.manifest.included_types & other.manifest.included_types)
        type_union = len(self.manifest.included_types | other.manifest.included_types)
        type_sim = type_overlap / type_union if type_union > 0 else 0
        
        # Edge overlap
        edge_overlap = len(self.manifest.included_aspects & other.manifest.included_aspects)
        edge_union = len(self.manifest.included_aspects | other.manifest.included_aspects)
        edge_sim = edge_overlap / edge_union if edge_union > 0 else 0
        
        # Vector similarity for shared elements
        shared_types = self.manifest.included_types & other.manifest.included_types
        if shared_types:
            vec_sims = []
            for t in shared_types:
                if t in self.type_vectors and t in other.type_vectors:
                    v1, v2 = self.type_vectors[t], other.type_vectors[t]
                    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    vec_sims.append(cos_sim)
            vec_sim = np.mean(vec_sims) if vec_sims else 0
        else:
            vec_sim = 0
        
        # Weighted combination
        return 0.3 * type_sim + 0.3 * edge_sim + 0.4 * vec_sim


class OlogEmbedder:
    """
    Creates ontological embeddings from OlogGraphs via Hydration Manifests.
    
    The embedder performs graph walks to selectively hydrate subgraphs,
    then produces structured embeddings that preserve categorical semantics.
    """
    
    def __init__(
        self,
        olog: OlogGraph,
        embedding_dim: int = 64,
        use_pretrained: bool = False,
        encoder_model: Optional[str] = None,
    ):
        self.olog = olog
        self.embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained
        self.encoder_model = encoder_model
        
        # Type embedding cache
        self._type_embeddings: Dict[str, np.ndarray] = {}
        self._edge_embeddings: Dict[Tuple[str, str, str], np.ndarray] = {}
        
        # Extract types and morphisms from olog
        self.types = get_olog_types(olog)
        self.morphisms = get_olog_morphisms(olog)
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize base embeddings for types and edges."""
        if self.use_pretrained and self.encoder_model:
            self._initialize_pretrained()
        else:
            self._initialize_random()
    
    def _initialize_random(self):
        """Initialize with random embeddings (trainable)."""
        np.random.seed(42)
        
        # Type embeddings
        for type_name in self.types:
            self._type_embeddings[type_name] = np.random.randn(self.embedding_dim) * 0.1
        
        # Edge embeddings (based on label)
        for morph in self.morphisms.values():
            key = (morph.source, morph.label, morph.target)
            # Hash-based initialization for consistency
            label_hash = hash(morph.label) % 10000
            np.random.seed(label_hash)
            self._edge_embeddings[key] = np.random.randn(self.embedding_dim) * 0.1
    
    def _initialize_pretrained(self):
        """Initialize using pretrained sentence embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.encoder_model or "all-MiniLM-L6-v2")
            
            # Embed type names + descriptions
            for type_name, node in self.types.items():
                text = f"{type_name}: {node.description}" if node.description else type_name
                emb = model.encode(text)
                # Project to target dimension
                if len(emb) != self.embedding_dim:
                    emb = emb[:self.embedding_dim] if len(emb) > self.embedding_dim else \
                          np.pad(emb, (0, self.embedding_dim - len(emb)))
                self._type_embeddings[type_name] = emb
            
            # Embed edge labels
            for morph in self.morphisms.values():
                key = (morph.source, morph.label, morph.target)
                text = f"{morph.source} {morph.label} {morph.target}"
                emb = model.encode(text)
                if len(emb) != self.embedding_dim:
                    emb = emb[:self.embedding_dim] if len(emb) > self.embedding_dim else \
                          np.pad(emb, (0, self.embedding_dim - len(emb)))
                self._edge_embeddings[key] = emb
                
        except ImportError:
            logger.warning("sentence-transformers not installed, using random embeddings")
            self._initialize_random()
    
    def create_manifest(
        self,
        root: str,
        depth: int = 2,
        strategy: HydrationStrategy = HydrationStrategy.BFS,
        type_filter: Optional[Set[str]] = None,
        edge_filter: Optional[Set[str]] = None,
    ) -> HydrationManifest:
        """
        Create a hydration manifest by walking the graph from root.
        
        Args:
            root: Starting type node
            depth: Maximum walk depth
            strategy: How to select edges to include
            type_filter: Only include these types (None = all)
            edge_filter: Only include edges with these labels (None = all)
        
        Returns:
            HydrationManifest specifying which elements to hydrate
        """
        if root not in self.types:
            raise ValueError(f"Unknown type: {root}")
        
        manifest = HydrationManifest(
            root_type=root,
            depth=depth,
            strategy=strategy,
            type_constraints=type_filter,
            edge_constraints=edge_filter,
        )
        
        if strategy == HydrationStrategy.BFS:
            self._walk_bfs(manifest, root, depth, type_filter, edge_filter)
        elif strategy == HydrationStrategy.DFS:
            self._walk_dfs(manifest, root, depth, type_filter, edge_filter, set())
        elif strategy == HydrationStrategy.FULL:
            self._walk_bfs(manifest, root, len(self.olog.types), type_filter, edge_filter)
        else:
            # Default to BFS
            self._walk_bfs(manifest, root, depth, type_filter, edge_filter)
        
        return manifest
    
    def _walk_bfs(
        self,
        manifest: HydrationManifest,
        root: str,
        max_depth: int,
        type_filter: Optional[Set[str]],
        edge_filter: Optional[Set[str]],
    ):
        """Breadth-first walk to build manifest."""
        visited = {root}
        frontier = [(root, 0)]
        
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            
            # Get outgoing edges
            for morph in self.morphisms.values():
                if morph.source != current:
                    continue
                
                target = morph.target
                
                # Apply filters
                if type_filter and target not in type_filter:
                    continue
                if edge_filter and morph.label not in edge_filter:
                    continue
                
                # Add to manifest
                manifest.included_types.add(target)
                manifest.included_aspects.add((morph.source, morph.label, morph.target))
                
                if target not in visited:
                    visited.add(target)
                    frontier.append((target, depth + 1))
    
    def _walk_dfs(
        self,
        manifest: HydrationManifest,
        current: str,
        remaining_depth: int,
        type_filter: Optional[Set[str]],
        edge_filter: Optional[Set[str]],
        visited: Set[str],
    ):
        """Depth-first walk to build manifest."""
        if remaining_depth <= 0 or current in visited:
            return
        
        visited.add(current)
        
        for morph in self.morphisms.values():
            if morph.source != current:
                continue
            
            target = morph.target
            
            if type_filter and target not in type_filter:
                continue
            if edge_filter and morph.label not in edge_filter:
                continue
            
            manifest.included_types.add(target)
            manifest.included_aspects.add((morph.source, morph.label, morph.target))
            
            self._walk_dfs(manifest, target, remaining_depth - 1, 
                          type_filter, edge_filter, visited)
    
    def hydrate(self, manifest: HydrationManifest) -> OlogEmbedding:
        """
        Hydrate a manifest into a full embedding.
        
        This materializes the subgraph specified by the manifest and
        computes embeddings for all included elements.
        """
        type_vectors = {}
        edge_vectors = {}
        
        # Collect type embeddings
        for type_name in manifest.included_types:
            if type_name in self._type_embeddings:
                type_vectors[type_name] = self._type_embeddings[type_name].copy()
            else:
                # Generate embedding for unknown type
                type_vectors[type_name] = np.random.randn(self.embedding_dim) * 0.1
        
        # Collect edge embeddings
        for edge in manifest.included_aspects:
            if edge in self._edge_embeddings:
                edge_vectors[edge] = self._edge_embeddings[edge].copy()
            else:
                # Generate embedding for unknown edge
                edge_vectors[edge] = np.random.randn(self.embedding_dim) * 0.1
        
        # Build composition matrix (captures path structure)
        composition_matrix = self._build_composition_matrix(manifest)
        
        return OlogEmbedding(
            manifest=manifest,
            type_vectors=type_vectors,
            edge_vectors=edge_vectors,
            composition_matrix=composition_matrix,
        )
    
    def _build_composition_matrix(self, manifest: HydrationManifest) -> np.ndarray:
        """
        Build a matrix capturing morphism composition structure.
        
        Entry (i,j) indicates whether type_i connects to type_j via some path.
        """
        types = sorted(manifest.included_types)
        n = len(types)
        type_to_idx = {t: i for i, t in enumerate(types)}
        
        # Adjacency matrix
        adj = np.zeros((n, n))
        for source, label, target in manifest.included_aspects:
            if source in type_to_idx and target in type_to_idx:
                adj[type_to_idx[source], type_to_idx[target]] = 1
        
        # Compute reachability (transitive closure for composition)
        reachability = adj.copy()
        for _ in range(n):
            reachability = np.minimum(reachability + reachability @ adj, 1)
        
        return reachability
    
    def embed_query(self, query: str, context_types: Optional[List[str]] = None) -> OlogEmbedding:
        """
        Create an embedding for a natural language query.
        
        Uses the query to determine which parts of the Olog to hydrate,
        then creates an embedding focused on relevant substructure.
        """
        # Find relevant root types
        if context_types:
            roots = [t for t in context_types if t in self.olog.types]
        else:
            # Use all types as potential roots
            roots = list(self.types.keys())
        
        if not roots:
            # Return empty embedding
            return OlogEmbedding(
                manifest=HydrationManifest(root_type="Unknown"),
                type_vectors={},
                edge_vectors={},
            )
        
        # Create and merge manifests from each root
        manifests = [self.create_manifest(root, depth=2) for root in roots[:3]]
        merged = manifests[0]
        for m in manifests[1:]:
            merged = merged.merge(m)
        
        return self.hydrate(merged)
    
    def compose_embeddings(
        self,
        emb1: OlogEmbedding,
        emb2: OlogEmbedding,
        operation: str = "merge"
    ) -> OlogEmbedding:
        """
        Compose two embeddings categorically.
        
        Operations:
        - merge: Union of hydrated elements
        - intersect: Common elements only
        - chain: Sequential composition (if compatible)
        """
        if operation == "merge":
            merged_manifest = emb1.manifest.merge(emb2.manifest)
        elif operation == "intersect":
            merged_manifest = emb1.manifest.intersect(emb2.manifest)
        else:
            merged_manifest = emb1.manifest.merge(emb2.manifest)
        
        return self.hydrate(merged_manifest)


def demo():
    """Demonstrate ontological embeddings."""
    print("=" * 60)
    print("  ONTOLOGICAL EMBEDDINGS DEMO")
    print("=" * 60)
    
    # Create sample Olog
    olog = OlogGraph(name="ECommerceOntology")
    
    # Add types
    olog.add_type("Customer", "A person who purchases products")
    olog.add_type("Order", "A request to purchase products")
    olog.add_type("Product", "An item available for purchase")
    olog.add_type("Invoice", "A document requesting payment")
    olog.add_type("Payment", "A completed transaction")
    olog.add_type("Inventory", "Stock of products")
    
    # Add aspects (morphisms) - signature: add_aspect(source, target, label)
    olog.add_aspect("Customer", "Order", "places")
    olog.add_aspect("Order", "Product", "contains")
    olog.add_aspect("Order", "Invoice", "generates")
    olog.add_aspect("Invoice", "Payment", "requires")
    olog.add_aspect("Payment", "Order", "confirms")
    olog.add_aspect("Product", "Inventory", "from")
    
    # Create embedder
    embedder = OlogEmbedder(olog, embedding_dim=64)
    
    print("\n[OLOG STRUCTURE]")
    print(f"  Types: {list(olog.graph.nodes())}")
    print(f"  Aspects: {olog.graph.number_of_edges()}")
    
    # Create manifests with different roots
    print("\n[HYDRATION MANIFESTS]")
    
    manifest_customer = embedder.create_manifest("Customer", depth=2)
    print(f"\n  From Customer (depth=2):")
    print(f"    Types: {manifest_customer.included_types}")
    print(f"    Aspects: {len(manifest_customer.included_aspects)}")
    
    manifest_product = embedder.create_manifest("Product", depth=2)
    print(f"\n  From Product (depth=2):")
    print(f"    Types: {manifest_product.included_types}")
    print(f"    Aspects: {len(manifest_product.included_aspects)}")
    
    # Hydrate to embeddings
    print("\n[ONTOLOGICAL EMBEDDINGS]")
    
    emb_customer = embedder.hydrate(manifest_customer)
    emb_product = embedder.hydrate(manifest_product)
    
    print(f"\n  Customer embedding:")
    print(f"    Dimension: {emb_customer.dim}")
    print(f"    Types included: {len(emb_customer.type_vectors)}")
    print(f"    Dense vector norm: {np.linalg.norm(emb_customer.to_dense()):.4f}")
    
    print(f"\n  Product embedding:")
    print(f"    Dimension: {emb_product.dim}")
    print(f"    Types included: {len(emb_product.type_vectors)}")
    print(f"    Dense vector norm: {np.linalg.norm(emb_product.to_dense()):.4f}")
    
    # Compute similarity
    print("\n[CATEGORICAL SIMILARITY]")
    sim = emb_customer.similarity(emb_product)
    print(f"  Customer <-> Product similarity: {sim:.4f}")
    
    # Compose embeddings
    print("\n[EMBEDDING COMPOSITION]")
    merged = embedder.compose_embeddings(emb_customer, emb_product, "merge")
    print(f"  Merged embedding types: {merged.manifest.included_types}")
    
    # Composition matrix
    print("\n[COMPOSITION MATRIX]")
    print(f"  Shape: {merged.composition_matrix.shape}")
    print(f"  Reachability structure:")
    types = sorted(merged.manifest.included_types)
    for i, t in enumerate(types):
        reachable = [types[j] for j in range(len(types)) 
                    if merged.composition_matrix[i, j] > 0]
        if reachable:
            print(f"    {t} -> {reachable}")
    
    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


def load_ontologies_from_text2kg(data_dir: Path) -> List[OlogGraph]:
    """
    Load ontology graphs from Text2KGBench TTL files.
    
    This provides real ontology structures for training embeddings.
    """
    try:
        from rdflib import Graph as RDFGraph, RDF, RDFS, OWL
    except ImportError:
        logger.error("rdflib required: pip install rdflib")
        return []
    
    ologs = []
    ttl_dirs = [
        data_dir / "Text2KGBench" / "data" / "wikidata_tekgen" / "ontologies" / "owl",
        data_dir / "Text2KGBench" / "data" / "dbpedia_webnlg" / "ontologies" / "owl",
    ]
    
    for ont_dir in ttl_dirs:
        if not ont_dir.exists():
            continue
        
        for ttl_file in ont_dir.glob("*.ttl"):
            try:
                g = RDFGraph()
                g.parse(ttl_file, format="turtle")
                
                olog = OlogGraph(name=ttl_file.stem)
                
                # Extract classes as types
                for s in g.subjects(RDF.type, OWL.Class):
                    label = g.value(s, RDFS.label)
                    name = str(label) if label else str(s).split("/")[-1].split("#")[-1]
                    desc = g.value(s, RDFS.comment)
                    olog.add_type(name, str(desc) if desc else "")
                
                # Extract properties as potential morphisms
                for s in g.subjects(RDF.type, OWL.ObjectProperty):
                    label = g.value(s, RDFS.label)
                    prop_name = str(label) if label else str(s).split("/")[-1].split("#")[-1]
                    domain = g.value(s, RDFS.domain)
                    range_ = g.value(s, RDFS.range)
                    
                    if domain and range_:
                        dom_name = str(domain).split("/")[-1].split("#")[-1]
                        rng_name = str(range_).split("/")[-1].split("#")[-1]
                        
                        # Ensure types exist
                        if dom_name not in [n for n in olog.graph.nodes()]:
                            olog.add_type(dom_name)
                        if rng_name not in [n for n in olog.graph.nodes()]:
                            olog.add_type(rng_name)
                        
                        olog.add_aspect(dom_name, rng_name, prop_name)
                
                if olog.graph.number_of_nodes() > 0:
                    ologs.append(olog)
                    logger.info(f"Loaded ontology: {olog.name} ({olog.graph.number_of_nodes()} types, {olog.graph.number_of_edges()} relations)")
                    
            except Exception as e:
                logger.warning(f"Failed to parse {ttl_file}: {e}")
    
    return ologs


def train_embeddings_on_ontologies(ologs: List[OlogGraph], embedding_dim: int = 64) -> Dict[str, np.ndarray]:
    """
    Train unified embeddings across multiple ontology graphs.
    
    Returns a dictionary mapping type/relation names to embedding vectors.
    """
    # Collect all types and relations
    all_types = set()
    all_relations = set()
    
    for olog in ologs:
        types = get_olog_types(olog)
        morphisms = get_olog_morphisms(olog)
        all_types.update(types.keys())
        for m in morphisms.values():
            all_relations.add(m.label)
    
    logger.info(f"Total unique types: {len(all_types)}")
    logger.info(f"Total unique relations: {len(all_relations)}")
    
    # Initialize embeddings
    embeddings = {}
    np.random.seed(42)
    
    for t in all_types:
        embeddings[f"type:{t}"] = np.random.randn(embedding_dim) * 0.1
    for r in all_relations:
        embeddings[f"rel:{r}"] = np.random.randn(embedding_dim) * 0.1
    
    # TODO: Train via contrastive learning on graph structure
    # For now, return initialized embeddings
    
    return embeddings


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        # Load and train on Text2KGBench ontologies
        data_dir = Path(__file__).parent / "training_data"
        print("Loading ontologies from Text2KGBench...")
        ologs = load_ontologies_from_text2kg(data_dir)
        print(f"\nLoaded {len(ologs)} ontology graphs")
        
        if ologs:
            embeddings = train_embeddings_on_ontologies(ologs)
            print(f"Generated {len(embeddings)} embeddings")
    else:
        demo()
