"""
Semantic Analysis Module for Ontological Contradiction Detection

Detects semantic contradictions in Olog edge labels using:
1. Antonym detection via WordNet/hardcoded pairs
2. Embedding similarity via sentence-transformers (optional)
3. ConceptNet relations (optional, requires API)

The core insight: two paths to the same target are semantically contradictory
if their edge labels have opposing meanings (e.g., "increases" vs "reduces").
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContradictionType(Enum):
    """Types of semantic contradictions."""
    ANTONYM = "antonym"              # Direct opposite (increase/decrease)
    NEGATION = "negation"            # Negated form (allow/disallow)
    INCOMPATIBLE = "incompatible"    # Mutually exclusive (buy/sell from same agent)
    TEMPORAL = "temporal"            # Temporal impossibility (before/after)


@dataclass
class SemanticContradiction:
    """Represents a detected semantic contradiction."""
    path_a_labels: List[str]
    path_b_labels: List[str]
    contradiction_type: ContradictionType
    conflicting_pair: Tuple[str, str]
    confidence: float  # 0.0 to 1.0
    explanation: str


@dataclass
class SemanticAnalysisResult:
    """Result of semantic analysis on an Olog."""
    contradictions: List[SemanticContradiction] = field(default_factory=list)
    consistency_score: float = 1.0
    analyzed_paths: int = 0
    

class AntonymDatabase:
    """
    Database of antonym pairs for contradiction detection.
    
    Organized by semantic domain for efficient lookup.
    """
    
    # Core antonym pairs organized by domain
    ANTONYMS = {
        # Quantity/Amount
        "quantity": [
            ("increase", "decrease"),
            ("increases", "decreases"),
            ("increasing", "decreasing"),
            ("increase", "reduce"),
            ("increases", "reduces"),
            ("increasing", "reducing"),
            ("add", "remove"),
            ("adds", "removes"),
            ("add", "subtract"),
            ("adds", "subtracts"),
            ("grow", "shrink"),
            ("grows", "shrinks"),
            ("expand", "contract"),
            ("expands", "contracts"),
            ("raise", "lower"),
            ("raises", "lowers"),
            ("more", "less"),
            ("gain", "lose"),
            ("gains", "loses"),
            ("increment", "decrement"),
            ("increments", "decrements"),
        ],
        # State changes
        "state": [
            ("create", "destroy"),
            ("creates", "destroys"),
            ("open", "close"),
            ("opens", "closes"),
            ("start", "stop"),
            ("starts", "stops"),
            ("begin", "end"),
            ("begins", "ends"),
            ("activate", "deactivate"),
            ("activates", "deactivates"),
            ("enable", "disable"),
            ("enables", "disables"),
            ("allow", "forbid"),
            ("allows", "forbids"),
            ("permit", "deny"),
            ("permits", "denies"),
        ],
        # Movement/Direction
        "direction": [
            ("push", "pull"),
            ("pushes", "pulls"),
            ("send", "receive"),
            ("sends", "receives"),
            ("give", "take"),
            ("gives", "takes"),
            ("export", "import"),
            ("exports", "imports"),
            ("upload", "download"),
            ("uploads", "downloads"),
            ("enter", "exit"),
            ("enters", "exits"),
            ("arrive", "depart"),
            ("arrives", "departs"),
        ],
        # Transactions
        "transaction": [
            ("buy", "sell"),
            ("buys", "sells"),
            ("credit", "debit"),
            ("credits", "debits"),
            ("deposit", "withdraw"),
            ("deposits", "withdraws"),
            ("borrow", "lend"),
            ("borrows", "lends"),
            ("charge", "refund"),
            ("charges", "refunds"),
        ],
        # Relationships
        "relationship": [
            ("connect", "disconnect"),
            ("connects", "disconnects"),
            ("attach", "detach"),
            ("attaches", "detaches"),
            ("link", "unlink"),
            ("links", "unlinks"),
            ("join", "separate"),
            ("joins", "separates"),
            ("include", "exclude"),
            ("includes", "excludes"),
        ],
        # Inventory/Stock specific
        "inventory": [
            ("reduce", "replenish"),
            ("reduces", "replenishes"),
            ("deplete", "restock"),
            ("depletes", "restocks"),
            ("consume", "produce"),
            ("consumes", "produces"),
        ],
    }
    
    def __init__(self):
        # Build bidirectional lookup
        self._antonym_map: Dict[str, Set[str]] = {}
        self._build_lookup()
    
    def _build_lookup(self):
        """Build efficient lookup structure."""
        for domain, pairs in self.ANTONYMS.items():
            for word_a, word_b in pairs:
                # Normalize to lowercase
                a_lower = word_a.lower()
                b_lower = word_b.lower()
                
                if a_lower not in self._antonym_map:
                    self._antonym_map[a_lower] = set()
                if b_lower not in self._antonym_map:
                    self._antonym_map[b_lower] = set()
                
                self._antonym_map[a_lower].add(b_lower)
                self._antonym_map[b_lower].add(a_lower)
    
    def are_antonyms(self, word_a: str, word_b: str) -> bool:
        """Check if two words are antonyms."""
        a_lower = word_a.lower().strip()
        b_lower = word_b.lower().strip()
        
        if a_lower in self._antonym_map:
            return b_lower in self._antonym_map[a_lower]
        return False
    
    def get_antonyms(self, word: str) -> Set[str]:
        """Get all known antonyms for a word."""
        return self._antonym_map.get(word.lower().strip(), set())


class EmbeddingAnalyzer:
    """
    Embedding-based semantic similarity analyzer.
    
    Uses sentence-transformers for dense embeddings, falls back to
    simple heuristics if not available.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = None
        self._model_name = model_name
        self._available = False
        self._load_model()
    
    def _load_model(self):
        """Attempt to load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._available = True
            logger.info(f"Loaded embedding model: {self._model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using heuristic fallback.")
            self._available = False
    
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between two texts."""
        if not self._available:
            return self._heuristic_similarity(text_a, text_b)
        
        embeddings = self._model.encode([text_a, text_b])
        # Cosine similarity
        dot = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
        norm_a = math.sqrt(sum(a * a for a in embeddings[0]))
        norm_b = math.sqrt(sum(b * b for b in embeddings[1]))
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
    
    def _heuristic_similarity(self, text_a: str, text_b: str) -> float:
        """Simple heuristic similarity when embeddings unavailable."""
        # Jaccard similarity on character trigrams
        def trigrams(s):
            s = s.lower()
            return set(s[i:i+3] for i in range(len(s)-2))
        
        tri_a = trigrams(text_a)
        tri_b = trigrams(text_b)
        
        if not tri_a or not tri_b:
            return 0.0
        
        intersection = len(tri_a & tri_b)
        union = len(tri_a | tri_b)
        return intersection / union if union > 0 else 0.0
    
    def are_semantically_opposite(self, text_a: str, text_b: str, threshold: float = 0.3) -> bool:
        """
        Detect if two texts are semantically opposite.
        
        This is tricky because antonyms often have HIGH similarity in embedding space
        (e.g., "increase" and "decrease" are both about quantity change).
        
        Strategy: Check if similarity is moderate (same domain) but words differ in polarity.
        """
        sim = self.compute_similarity(text_a, text_b)
        
        # If very similar (>0.8), probably synonyms or same word
        # If very dissimilar (<0.3), probably unrelated domains
        # Antonyms often fall in the 0.4-0.7 range (same domain, opposite meaning)
        
        # For now, return True if in "suspicious" range and we need further analysis
        return 0.3 < sim < 0.85


class SemanticContradictionDetector:
    """
    Main detector for semantic contradictions in Ologs.
    
    Combines multiple strategies:
    1. Antonym lookup (fast, precise)
    2. Negation patterns (prefix-based)
    3. Embedding analysis (semantic similarity)
    """
    
    # Negation prefixes that flip meaning
    NEGATION_PREFIXES = ["un", "dis", "non", "de", "in", "im", "ir", "il"]
    
    def __init__(self, use_embeddings: bool = False):
        self.antonym_db = AntonymDatabase()
        self.embedding_analyzer = EmbeddingAnalyzer() if use_embeddings else None
    
    def detect_contradictions(
        self,
        path_pairs: List[Tuple[List[str], List[str]]],
    ) -> SemanticAnalysisResult:
        """
        Detect semantic contradictions between pairs of paths.
        
        Args:
            path_pairs: List of (path_a_labels, path_b_labels) tuples
                       representing paths that should be semantically equivalent.
        
        Returns:
            SemanticAnalysisResult with detected contradictions.
        """
        result = SemanticAnalysisResult(analyzed_paths=len(path_pairs))
        
        for path_a, path_b in path_pairs:
            contradictions = self._analyze_path_pair(path_a, path_b)
            result.contradictions.extend(contradictions)
        
        # Calculate consistency score
        if path_pairs:
            # Penalize based on number and confidence of contradictions
            total_penalty = sum(c.confidence for c in result.contradictions)
            result.consistency_score = max(0.0, 1.0 - (total_penalty / len(path_pairs)))
        
        return result
    
    def _analyze_path_pair(
        self,
        path_a: List[str],
        path_b: List[str]
    ) -> List[SemanticContradiction]:
        """Analyze a single pair of paths for contradictions."""
        contradictions = []
        
        # Strategy 1: Check each label in path_a against each in path_b
        for label_a in path_a:
            for label_b in path_b:
                contradiction = self._check_label_pair(label_a, label_b, path_a, path_b)
                if contradiction:
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _check_label_pair(
        self,
        label_a: str,
        label_b: str,
        path_a: List[str],
        path_b: List[str]
    ) -> Optional[SemanticContradiction]:
        """Check if two labels are semantically contradictory."""
        
        # Normalize labels
        a_norm = label_a.lower().strip()
        b_norm = label_b.lower().strip()
        
        # Same label = no contradiction
        if a_norm == b_norm:
            return None
        
        # Check 1: Direct antonyms
        if self.antonym_db.are_antonyms(a_norm, b_norm):
            return SemanticContradiction(
                path_a_labels=path_a,
                path_b_labels=path_b,
                contradiction_type=ContradictionType.ANTONYM,
                conflicting_pair=(label_a, label_b),
                confidence=0.95,
                explanation=f"'{label_a}' and '{label_b}' are antonyms (opposite meanings)"
            )
        
        # Check 2: Negation patterns
        negation = self._check_negation(a_norm, b_norm)
        if negation:
            return SemanticContradiction(
                path_a_labels=path_a,
                path_b_labels=path_b,
                contradiction_type=ContradictionType.NEGATION,
                conflicting_pair=(label_a, label_b),
                confidence=0.85,
                explanation=f"'{label_a}' is the negated form of '{label_b}'"
            )
        
        # Check 3: Embedding-based detection (if available)
        if self.embedding_analyzer and self.embedding_analyzer._available:
            if self.embedding_analyzer.are_semantically_opposite(a_norm, b_norm):
                # Additional check: low similarity might indicate contradiction
                sim = self.embedding_analyzer.compute_similarity(a_norm, b_norm)
                if sim < 0.5:  # Different semantic domain
                    return SemanticContradiction(
                        path_a_labels=path_a,
                        path_b_labels=path_b,
                        contradiction_type=ContradictionType.INCOMPATIBLE,
                        conflicting_pair=(label_a, label_b),
                        confidence=0.6,
                        explanation=f"'{label_a}' and '{label_b}' appear semantically incompatible (sim={sim:.2f})"
                    )
        
        return None
    
    def _check_negation(self, word_a: str, word_b: str) -> bool:
        """Check if one word is a negated form of the other."""
        for prefix in self.NEGATION_PREFIXES:
            # Check if word_a = prefix + word_b
            if word_a.startswith(prefix) and word_a[len(prefix):] == word_b:
                return True
            # Check if word_b = prefix + word_a
            if word_b.startswith(prefix) and word_b[len(prefix):] == word_a:
                return True
        return False


def integrate_with_olog(olog_graph) -> SemanticAnalysisResult:
    """
    Integrate semantic analysis with an OlogGraph.
    
    Extracts commutative facts and analyzes their paths for semantic contradictions.
    """
    from olog_core import OlogGraph
    
    detector = SemanticContradictionDetector(use_embeddings=False)
    
    # Extract path pairs from facts
    path_pairs = []
    for fact in olog_graph.facts:
        path_pairs.append((fact.path_a_labels, fact.path_b_labels))
    
    return detector.detect_contradictions(path_pairs)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate semantic contradiction detection."""
    print("=" * 60)
    print("  SEMANTIC CONTRADICTION DETECTION DEMO")
    print("=" * 60)
    
    detector = SemanticContradictionDetector(use_embeddings=False)
    
    # Test cases
    test_cases = [
        {
            "name": "Antonym: increase vs decrease",
            "path_a": ["places", "increases"],
            "path_b": ["places", "generates", "decreases"],
        },
        {
            "name": "Antonym: reduce vs replenish", 
            "path_a": ["order", "reduces"],
            "path_b": ["invoice", "replenishes"],
        },
        {
            "name": "Negation: connect vs disconnect",
            "path_a": ["connects"],
            "path_b": ["disconnects"],
        },
        {
            "name": "No contradiction: creates vs generates",
            "path_a": ["creates"],
            "path_b": ["generates"],
        },
        {
            "name": "Transaction: buy vs sell",
            "path_a": ["customer", "buys"],
            "path_b": ["vendor", "sells"],
        },
    ]
    
    for case in test_cases:
        print(f"\n[TEST: {case['name']}]")
        print(f"  Path A: {case['path_a']}")
        print(f"  Path B: {case['path_b']}")
        
        result = detector.detect_contradictions([(case["path_a"], case["path_b"])])
        
        if result.contradictions:
            for c in result.contradictions:
                print(f"  ⚠ CONTRADICTION DETECTED:")
                print(f"    Type: {c.contradiction_type.value}")
                print(f"    Pair: {c.conflicting_pair}")
                print(f"    Confidence: {c.confidence:.2f}")
                print(f"    Explanation: {c.explanation}")
        else:
            print(f"  ✓ No contradiction detected")
    
    # Integration demo with OlogGraph
    print("\n" + "=" * 60)
    print("  INTEGRATION WITH OLOG")
    print("=" * 60)
    
    from olog_core import OlogGraph, CommutativeFact
    
    olog = OlogGraph("SemanticTest")
    for t in ["Customer", "Order", "Invoice", "Inventory"]:
        olog.add_type(t)
    
    olog.add_aspect("Customer", "Order", "places")
    olog.add_aspect("Order", "Inventory", "reduces")
    olog.add_aspect("Order", "Invoice", "generates")
    olog.add_aspect("Invoice", "Inventory", "increases")  # CONTRADICTION!
    
    # Add the contradictory fact
    olog.add_fact(CommutativeFact(
        source_node="Customer",
        target_node="Inventory",
        path_a_labels=["places", "reduces"],
        path_b_labels=["places", "generates", "increases"]
    ))
    
    # Run semantic analysis
    result = integrate_with_olog(olog)
    
    print(f"\n[SEMANTIC ANALYSIS RESULT]")
    print(f"  Paths analyzed: {result.analyzed_paths}")
    print(f"  Contradictions found: {len(result.contradictions)}")
    print(f"  Semantic consistency score: {result.consistency_score:.2f}")
    
    for c in result.contradictions:
        print(f"\n  ⚠ CONTRADICTION:")
        print(f"    {c.explanation}")


if __name__ == "__main__":
    demo()
