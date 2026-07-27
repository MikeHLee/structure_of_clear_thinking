"""
Intent Parser Module for Ontological Query Processing

This module implements hierarchical semantic tokenization for natural language
query intent parsing, mapping user queries to Olog traversal operations.

Architecture (hierarchical semantic layers):
    Level 0: Surface tokenization (raw text → tokens)
    Level 1a: Slot assignment (tokens → semantic roles)
    Level 1b: Type grounding (roles → Olog types)
    Level 1c: Constraint extraction (types → adicity/relations)
    Level 2: Intent classification + execution plan

The Intent Parser bridges natural language queries to:
    - ProofSearcher: For relational queries ("How does X relate to Y?")
    - HydrationManifest: For retrieval queries ("Show me all X")
    - OlogGraph mutations: For update queries ("Tag X as Y")

Key Insight (from GEMINI_HANDOFF.md):
    Q = Rules (what information is needed)
    K = Graphs (valid connection structure)
    V = Objects (hydrated instances)
    
    Intent parsing extracts Q, uses K to validate, retrieves V.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set, Any
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Core Data Structures
# =============================================================================

class QueryIntent(Enum):
    """
    Types of ontological query operations.
    
    These map directly to Olog traversal strategies:
    - RETRIEVE: Type enumeration (list all instances of a type)
    - RELATE: Proof search (find path between types)
    - AGGREGATE: Type hierarchy traversal (meta-level queries)
    - UPDATE: Morphism creation/modification
    - NAVIGATE: Context hydration (expand neighborhood)
    """
    RETRIEVE = "retrieve"      # "Find files about X" → enumerate type
    RELATE = "relate"          # "How does X connect to Y?" → proof search
    AGGREGATE = "aggregate"    # "What categories exist?" → type hierarchy
    UPDATE = "update"          # "Tag X as Y" → add morphism
    NAVIGATE = "navigate"      # "Show X's context" → hydrate neighborhood
    UNKNOWN = "unknown"        # Fallback for unrecognized queries


class SemanticRole(Enum):
    """
    Semantic roles in query structure.
    
    Based on AMR role mapping from hybrid_encoder.py.
    """
    SOURCE = "source"          # Starting point of relation (ARG0)
    TARGET = "target"          # Endpoint of relation (ARG1)
    RELATION = "relation"      # The connecting morphism
    MODIFIER = "modifier"      # Constraints/filters (ARG2+)
    ENTITY = "entity"          # Standalone entity mention
    ATTRIBUTE = "attribute"    # Property/slot reference
    QUANTIFIER = "quantifier"  # Aggregation operator (all, some, count)


@dataclass
class SemanticToken:
    """
    Multi-level semantic token following hierarchical semantic layers.
    
    This extends OntologicalToken from hybrid_encoder.py with intent-specific
    fields for query processing.
    
    Attributes:
        surface_form: Original text span (Level 0)
        lemma: Normalized form for matching
        slot_type: Semantic role in query (Level 1a)
        olog_type: Matched Olog node type (Level 1b)
        morphism_hint: Potential relation label (Level 1c)
        adicity: Expected connection count
        intent_role: Role in query execution plan (Level 2)
        confidence: Match confidence score
        context_path: Path in Olog from root (for disambiguation)
    """
    surface_form: str
    lemma: str = ""
    slot_type: SemanticRole = SemanticRole.ENTITY
    olog_type: Optional[str] = None
    morphism_hint: Optional[str] = None
    adicity: int = 1
    intent_role: str = "entity"
    confidence: float = 1.0
    context_path: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.lemma:
            self.lemma = self.surface_form.lower().strip()
    
    def __repr__(self):
        type_str = f"→{self.olog_type}" if self.olog_type else ""
        return f"SToken({self.surface_form!r}{type_str}:{self.slot_type.value})"


@dataclass
class QueryPlan:
    """
    Execution plan for an ontological query.
    
    Contains all information needed to execute the query against an OlogGraph.
    """
    intent: QueryIntent
    tokens: List[SemanticToken]
    source_type: Optional[str] = None
    target_type: Optional[str] = None
    relation_label: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    depth: int = 2  # For hydration/navigation
    confidence: float = 1.0
    raw_query: str = ""
    
    def __repr__(self):
        return f"QueryPlan({self.intent.value}: {self.source_type}→{self.target_type})"


# =============================================================================
# Intent Classification
# =============================================================================

class IntentClassifier:
    """
    Rule-based intent classifier with pattern matching.
    
    Uses linguistic patterns to classify query intent before type grounding.
    Can be replaced with a fine-tuned model for production use.
    """
    
    # Pattern definitions: (regex, intent, confidence)
    PATTERNS = [
        # Relational patterns (check first - higher priority)
        (r'\bhow\b.*\b(relate|connect|link|associated)\b', 
         QueryIntent.RELATE, 0.95),
        (r'\bshow\b.*\bhow\b', 
         QueryIntent.RELATE, 0.95),
        (r'\b(relationship|connection|path)\b.*\b(between|from|to)\b', 
         QueryIntent.RELATE, 0.9),
        (r'\bfrom\b.+\bto\b', 
         QueryIntent.RELATE, 0.6),
        
        # Retrieval patterns
        (r'\b(find|get|show|list|what are|give me)\b.*\b(all|the|my)\b', 
         QueryIntent.RETRIEVE, 0.9),
        (r'\b(find|search|look for)\b', 
         QueryIntent.RETRIEVE, 0.7),
        
        # Aggregation patterns
        (r'\b(how many|count|total|sum|average)\b', 
         QueryIntent.AGGREGATE, 0.9),
        (r'\b(what|which)\b.*\b(types|categories|kinds)\b', 
         QueryIntent.AGGREGATE, 0.85),
        (r'\b(group|cluster|organize)\b.*\bby\b', 
         QueryIntent.AGGREGATE, 0.8),
        
        # Update patterns
        (r'\b(tag|label|mark|categorize|classify)\b.*\bas\b', 
         QueryIntent.UPDATE, 0.95),
        (r'\b(add|create|insert|set)\b.*\b(to|for|on)\b', 
         QueryIntent.UPDATE, 0.8),
        (r'\b(update|modify|change)\b', 
         QueryIntent.UPDATE, 0.7),
        
        # Navigation patterns
        (r'\b(show|display|expand)\b.*\b(context|neighborhood|related)\b', 
         QueryIntent.NAVIGATE, 0.9),
        (r'\b(what is|tell me about|describe)\b', 
         QueryIntent.NAVIGATE, 0.6),
    ]
    
    def classify(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify query intent using pattern matching.
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple of (QueryIntent, confidence)
        """
        query_lower = query.lower()
        
        best_intent = QueryIntent.UNKNOWN
        best_confidence = 0.0
        
        for pattern, intent, confidence in self.PATTERNS:
            if re.search(pattern, query_lower):
                if confidence > best_confidence:
                    best_intent = intent
                    best_confidence = confidence
        
        # Default to RETRIEVE if no pattern matches
        if best_intent == QueryIntent.UNKNOWN:
            best_intent = QueryIntent.RETRIEVE
            best_confidence = 0.3
        
        return best_intent, best_confidence


# =============================================================================
# Type Grounding
# =============================================================================

class TypeGrounder:
    """
    Grounds surface tokens to Olog types using fuzzy matching.
    
    Uses multiple strategies:
    1. Exact match (highest confidence)
    2. Lemma match (normalized forms)
    3. Substring match (partial matches)
    4. Semantic similarity (if embeddings available)
    """
    
    def __init__(self, olog_types: Set[str], morphism_labels: Set[str] = None):
        """
        Initialize with available Olog vocabulary.
        
        Args:
            olog_types: Set of type names from OlogGraph
            morphism_labels: Set of morphism/aspect labels
        """
        self.olog_types = olog_types
        self.morphism_labels = morphism_labels or set()
        
        # Build normalized lookup tables
        self._type_lookup = {t.lower(): t for t in olog_types}
        self._morphism_lookup = {m.lower(): m for m in morphism_labels}
    
    def ground_token(self, token: SemanticToken) -> SemanticToken:
        """
        Ground a semantic token to Olog types/morphisms.
        
        Args:
            token: SemanticToken with surface form
            
        Returns:
            SemanticToken with olog_type and/or morphism_hint filled
        """
        lemma = token.lemma
        
        # Strategy 1: Exact match on types
        if lemma in self._type_lookup:
            token.olog_type = self._type_lookup[lemma]
            token.confidence = 1.0
            return token
        
        # Strategy 2: Exact match on morphisms
        if lemma in self._morphism_lookup:
            token.morphism_hint = self._morphism_lookup[lemma]
            token.slot_type = SemanticRole.RELATION
            token.confidence = 1.0
            return token
        
        # Strategy 3: Substring match on types
        for type_lower, type_orig in self._type_lookup.items():
            if lemma in type_lower or type_lower in lemma:
                token.olog_type = type_orig
                token.confidence = 0.7
                return token
        
        # Strategy 4: Substring match on morphisms
        for morph_lower, morph_orig in self._morphism_lookup.items():
            if lemma in morph_lower or morph_lower in lemma:
                token.morphism_hint = morph_orig
                token.slot_type = SemanticRole.RELATION
                token.confidence = 0.7
                return token
        
        # No match - keep as ungrounded entity
        token.confidence = 0.3
        return token
    
    @classmethod
    def from_olog(cls, olog) -> 'TypeGrounder':
        """
        Create TypeGrounder from an OlogGraph instance.
        
        Args:
            olog: OlogGraph instance
            
        Returns:
            TypeGrounder initialized with Olog vocabulary
        """
        types = set(olog.graph.nodes())
        morphisms = set()
        
        # OlogGraph uses MultiDiGraph with key=label
        # edges() returns (u, v, key) tuples for MultiDiGraph
        for u, v, key in olog.graph.edges(keys=True):
            if key:  # key is the label
                morphisms.add(key)
        
        return cls(types, morphisms)


# =============================================================================
# Semantic Role Labeler
# =============================================================================

class SemanticRoleLabeler:
    """
    Assigns semantic roles to tokens based on query structure.
    
    Uses syntactic patterns and position heuristics.
    For production, integrate with actual AMR parser from hybrid_encoder.py.
    """
    
    # Prepositions that indicate roles
    SOURCE_PREPS = {'from', 'by', 'through'}
    TARGET_PREPS = {'to', 'into', 'for', 'about'}
    MODIFIER_PREPS = {'with', 'without', 'like', 'containing', 'where', 'when'}
    
    # Question words that indicate aggregation
    QUANTIFIERS = {'how many', 'count', 'total', 'all', 'some', 'any', 'every'}
    
    def label_tokens(
        self, 
        tokens: List[SemanticToken], 
        intent: QueryIntent
    ) -> List[SemanticToken]:
        """
        Assign semantic roles to tokens based on intent and structure.
        
        Args:
            tokens: List of semantic tokens
            intent: Classified query intent
            
        Returns:
            Tokens with slot_type assigned
        """
        if not tokens:
            return tokens
        
        if intent == QueryIntent.RELATE:
            return self._label_relational(tokens)
        elif intent == QueryIntent.RETRIEVE:
            return self._label_retrieval(tokens)
        elif intent == QueryIntent.UPDATE:
            return self._label_update(tokens)
        elif intent == QueryIntent.AGGREGATE:
            return self._label_aggregate(tokens)
        else:
            return self._label_navigation(tokens)
    
    def _label_relational(self, tokens: List[SemanticToken]) -> List[SemanticToken]:
        """Label tokens for relational query (finding paths)."""
        found_source = False
        found_target = False
        seen_to_prep = False
        
        for i, token in enumerate(tokens):
            lemma = token.lemma
            
            # Track "to" preposition for source/target boundary
            if lemma == 'to':
                seen_to_prep = True
                continue
            
            # Skip prepositions
            if lemma in self.SOURCE_PREPS | self.TARGET_PREPS:
                if lemma in self.TARGET_PREPS:
                    seen_to_prep = True
                continue
            
            # If token has morphism hint, it's a relation
            if token.morphism_hint:
                token.slot_type = SemanticRole.RELATION
                continue
            
            # Assign based on position relative to "to" preposition
            if token.olog_type:
                if not seen_to_prep and not found_source:
                    token.slot_type = SemanticRole.SOURCE
                    found_source = True
                elif seen_to_prep and not found_target:
                    token.slot_type = SemanticRole.TARGET
                    found_target = True
                elif not found_source:
                    # Fallback: first grounded type is source
                    token.slot_type = SemanticRole.SOURCE
                    found_source = True
                elif not found_target:
                    # Fallback: second grounded type is target
                    token.slot_type = SemanticRole.TARGET
                    found_target = True
        
        return tokens
    
    def _label_retrieval(self, tokens: List[SemanticToken]) -> List[SemanticToken]:
        """Label tokens for retrieval query (type enumeration)."""
        found_entity = False
        
        for token in tokens:
            if token.olog_type and not found_entity:
                token.slot_type = SemanticRole.ENTITY
                found_entity = True
            elif token.morphism_hint:
                token.slot_type = SemanticRole.RELATION
            elif token.lemma in self.MODIFIER_PREPS:
                continue  # Skip prepositions
            elif found_entity:
                token.slot_type = SemanticRole.MODIFIER
        
        return tokens
    
    def _label_update(self, tokens: List[SemanticToken]) -> List[SemanticToken]:
        """Label tokens for update query (morphism creation)."""
        found_source = False
        found_as = False
        
        for token in tokens:
            if token.lemma == 'as':
                found_as = True
                continue
            
            if token.olog_type:
                if not found_source:
                    token.slot_type = SemanticRole.SOURCE
                    found_source = True
                elif found_as:
                    token.slot_type = SemanticRole.TARGET
            elif token.morphism_hint:
                token.slot_type = SemanticRole.RELATION
        
        return tokens
    
    def _label_aggregate(self, tokens: List[SemanticToken]) -> List[SemanticToken]:
        """Label tokens for aggregation query."""
        for token in tokens:
            if token.lemma in self.QUANTIFIERS:
                token.slot_type = SemanticRole.QUANTIFIER
            elif token.olog_type:
                token.slot_type = SemanticRole.ENTITY
        
        return tokens
    
    def _label_navigation(self, tokens: List[SemanticToken]) -> List[SemanticToken]:
        """Label tokens for navigation query (context expansion)."""
        for token in tokens:
            if token.olog_type:
                token.slot_type = SemanticRole.ENTITY
        
        return tokens


# =============================================================================
# Main Intent Parser
# =============================================================================

class IntentParser:
    """
    Main entry point for natural language query parsing.
    
    Implements the hierarchical semantic tokenization pipeline:
    1. Surface tokenization
    2. Intent classification
    3. Type grounding
    4. Semantic role labeling
    5. Query plan construction
    
    Usage:
        parser = IntentParser.from_olog(olog_graph)
        plan = parser.parse("How does Customer relate to Order?")
        # plan.intent == QueryIntent.RELATE
        # plan.source_type == "Customer"
        # plan.target_type == "Order"
    """
    
    def __init__(
        self,
        type_grounder: TypeGrounder,
        intent_classifier: IntentClassifier = None,
        role_labeler: SemanticRoleLabeler = None,
    ):
        """
        Initialize the intent parser with components.
        
        Args:
            type_grounder: TypeGrounder for Olog vocabulary
            intent_classifier: Optional custom intent classifier
            role_labeler: Optional custom role labeler
        """
        self.type_grounder = type_grounder
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.role_labeler = role_labeler or SemanticRoleLabeler()
        
        # Stop words to filter during tokenization
        self._stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'that', 'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me',
            'him', 'her', 'them', 'us', 'this', 'these', 'those', 'what',
            'which', 'who', 'whom', 'whose', 'how', 'when', 'where', 'why',
        }
    
    @classmethod
    def from_olog(cls, olog) -> 'IntentParser':
        """
        Create IntentParser from an OlogGraph instance.
        
        Args:
            olog: OlogGraph instance
            
        Returns:
            Configured IntentParser
        """
        type_grounder = TypeGrounder.from_olog(olog)
        return cls(type_grounder)
    
    def parse(self, query: str) -> QueryPlan:
        """
        Parse a natural language query into an executable QueryPlan.
        
        This is the main entry point implementing the full pipeline.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryPlan ready for execution
        """
        logger.debug(f"Parsing query: {query}")
        
        # Step 1: Surface tokenization
        tokens = self._tokenize(query)
        logger.debug(f"Surface tokens: {[t.surface_form for t in tokens]}")
        
        # Step 2: Intent classification
        intent, intent_confidence = self.intent_classifier.classify(query)
        logger.debug(f"Intent: {intent.value} (confidence: {intent_confidence:.2f})")
        
        # Step 3: Type grounding
        grounded_tokens = [self.type_grounder.ground_token(t) for t in tokens]
        logger.debug(f"Grounded tokens: {grounded_tokens}")
        
        # Step 4: Semantic role labeling
        labeled_tokens = self.role_labeler.label_tokens(grounded_tokens, intent)
        logger.debug(f"Labeled tokens: {labeled_tokens}")
        
        # Step 5: Build query plan
        plan = self._build_plan(query, intent, labeled_tokens, intent_confidence)
        logger.debug(f"Query plan: {plan}")
        
        return plan
    
    def _tokenize(self, query: str) -> List[SemanticToken]:
        """
        Basic tokenization with stop word filtering.
        
        For production, replace with proper NLP tokenizer + lemmatizer.
        """
        # Simple whitespace + punctuation tokenization
        raw_tokens = re.findall(r'\b\w+\b', query.lower())
        
        tokens = []
        for raw in raw_tokens:
            # Keep content words and known vocabulary
            if raw not in self._stop_words or raw in {'all', 'from', 'to', 'as', 'by'}:
                tokens.append(SemanticToken(
                    surface_form=raw,
                    lemma=raw,
                ))
        
        return tokens
    
    def _build_plan(
        self,
        query: str,
        intent: QueryIntent,
        tokens: List[SemanticToken],
        confidence: float,
    ) -> QueryPlan:
        """
        Build execution plan from classified and labeled tokens.
        """
        plan = QueryPlan(
            intent=intent,
            tokens=tokens,
            confidence=confidence,
            raw_query=query,
        )
        
        # Extract source, target, relation based on roles
        for token in tokens:
            if token.slot_type == SemanticRole.SOURCE and token.olog_type:
                plan.source_type = token.olog_type
            elif token.slot_type == SemanticRole.TARGET and token.olog_type:
                plan.target_type = token.olog_type
            elif token.slot_type == SemanticRole.ENTITY and token.olog_type:
                # For non-relational queries, use as source
                if not plan.source_type:
                    plan.source_type = token.olog_type
            elif token.slot_type == SemanticRole.RELATION and token.morphism_hint:
                plan.relation_label = token.morphism_hint
            elif token.slot_type == SemanticRole.MODIFIER:
                # Collect filters
                if token.olog_type:
                    plan.filters[token.olog_type] = True
        
        return plan


# =============================================================================
# Query Executor (Integration with OlogGraph)
# =============================================================================

class QueryExecutor:
    """
    Executes QueryPlans against an OlogGraph.
    
    Integrates with:
    - ProofSearcher: For RELATE queries
    - HydrationManifest: For NAVIGATE queries
    - Direct OlogGraph methods: For RETRIEVE/UPDATE
    """
    
    def __init__(self, olog):
        """
        Initialize with an OlogGraph instance.
        
        Args:
            olog: OlogGraph to query
        """
        self.olog = olog
    
    def execute(self, plan: QueryPlan) -> Dict[str, Any]:
        """
        Execute a query plan and return results.
        
        Args:
            plan: QueryPlan from IntentParser
            
        Returns:
            Dict with results, metadata, and any errors
        """
        result = {
            "intent": plan.intent.value,
            "query": plan.raw_query,
            "source": plan.source_type,
            "target": plan.target_type,
            "data": None,
            "error": None,
        }
        
        try:
            if plan.intent == QueryIntent.RETRIEVE:
                result["data"] = self._execute_retrieve(plan)
            elif plan.intent == QueryIntent.RELATE:
                result["data"] = self._execute_relate(plan)
            elif plan.intent == QueryIntent.AGGREGATE:
                result["data"] = self._execute_aggregate(plan)
            elif plan.intent == QueryIntent.UPDATE:
                result["data"] = self._execute_update(plan)
            elif plan.intent == QueryIntent.NAVIGATE:
                result["data"] = self._execute_navigate(plan)
            else:
                result["error"] = f"Unknown intent: {plan.intent}"
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Query execution failed: {e}")
        
        return result
    
    def _execute_retrieve(self, plan: QueryPlan) -> Dict:
        """Retrieve instances of a type."""
        if not plan.source_type:
            return {"error": "No type specified for retrieval"}
        
        # Check if type exists
        if plan.source_type not in self.olog.graph.nodes():
            return {"error": f"Type '{plan.source_type}' not found in Olog"}
        
        # Get outgoing edges (instances/aspects)
        edges = list(self.olog.graph.out_edges(plan.source_type, data=True))
        
        return {
            "type": plan.source_type,
            "aspects": [
                {
                    "target": e[1],
                    "label": e[2].get("label", ""),
                }
                for e in edges
            ],
            "count": len(edges),
        }
    
    def _execute_relate(self, plan: QueryPlan) -> Dict:
        """Find relationship path between types."""
        if not plan.source_type or not plan.target_type:
            return {"error": "Both source and target types required for relational query"}
        
        # Check if types exist
        for t in [plan.source_type, plan.target_type]:
            if t not in self.olog.graph.nodes():
                return {"error": f"Type '{t}' not found in Olog"}
        
        # Find paths using BFS
        import networkx as nx
        try:
            paths = list(nx.all_simple_paths(
                self.olog.graph,
                plan.source_type,
                plan.target_type,
                cutoff=5,  # Limit path length
            ))
        except nx.NetworkXNoPath:
            paths = []
        
        # Format paths with edge labels
        formatted_paths = []
        for path in paths:
            path_with_labels = []
            for i in range(len(path) - 1):
                edge_data = self.olog.graph.get_edge_data(path[i], path[i+1])
                label = edge_data.get("label", "") if edge_data else ""
                path_with_labels.append({
                    "from": path[i],
                    "to": path[i+1],
                    "via": label,
                })
            formatted_paths.append(path_with_labels)
        
        return {
            "source": plan.source_type,
            "target": plan.target_type,
            "paths": formatted_paths,
            "connected": len(paths) > 0,
        }
    
    def _execute_aggregate(self, plan: QueryPlan) -> Dict:
        """Aggregate type information."""
        all_types = list(self.olog.graph.nodes())
        
        # Count edges per type
        type_stats = {}
        for t in all_types:
            out_degree = self.olog.graph.out_degree(t)
            in_degree = self.olog.graph.in_degree(t)
            type_stats[t] = {
                "outgoing": out_degree,
                "incoming": in_degree,
                "total_connections": out_degree + in_degree,
            }
        
        return {
            "total_types": len(all_types),
            "types": all_types,
            "statistics": type_stats,
        }
    
    def _execute_update(self, plan: QueryPlan) -> Dict:
        """Create new morphism between types."""
        if not plan.source_type or not plan.target_type:
            return {"error": "Both source and target required for update"}
        
        label = plan.relation_label or "relates_to"
        
        # Add the aspect
        try:
            self.olog.add_aspect(
                plan.source_type,
                plan.target_type,
                label,
            )
            return {
                "success": True,
                "added": {
                    "source": plan.source_type,
                    "target": plan.target_type,
                    "label": label,
                },
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_navigate(self, plan: QueryPlan) -> Dict:
        """Navigate/expand type neighborhood."""
        if not plan.source_type:
            return {"error": "No type specified for navigation"}
        
        if plan.source_type not in self.olog.graph.nodes():
            return {"error": f"Type '{plan.source_type}' not found"}
        
        # Get 2-hop neighborhood
        import networkx as nx
        neighbors_1 = set(self.olog.graph.successors(plan.source_type))
        neighbors_1.update(self.olog.graph.predecessors(plan.source_type))
        
        neighbors_2 = set()
        for n in neighbors_1:
            neighbors_2.update(self.olog.graph.successors(n))
            neighbors_2.update(self.olog.graph.predecessors(n))
        neighbors_2.discard(plan.source_type)
        
        return {
            "center": plan.source_type,
            "direct_neighbors": list(neighbors_1),
            "extended_neighbors": list(neighbors_2 - neighbors_1),
            "depth": plan.depth,
        }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the Intent Parser with a sample Olog."""
    print("=" * 70)
    print("  INTENT PARSER DEMO")
    print("  Hierarchical Semantic Tokenization for Ontological Queries")
    print("=" * 70)
    
    # Import OlogGraph
    from olog_core import OlogGraph
    
    # Create sample e-commerce Olog
    olog = OlogGraph("E-Commerce")
    
    # Add types
    for t in ["Customer", "Order", "Product", "Payment", "Delivery", "Cart", "Invoice"]:
        olog.add_type(t)
    
    # Add aspects (morphisms)
    olog.add_aspect("Customer", "Cart", "has")
    olog.add_aspect("Customer", "Order", "places")
    olog.add_aspect("Cart", "Product", "contains")
    olog.add_aspect("Order", "Product", "includes")
    olog.add_aspect("Order", "Payment", "requires")
    olog.add_aspect("Order", "Delivery", "triggers")
    olog.add_aspect("Payment", "Invoice", "generates")
    olog.add_aspect("Delivery", "Customer", "shipped_to")
    
    # Create parser
    parser = IntentParser.from_olog(olog)
    executor = QueryExecutor(olog)
    
    # Test queries
    test_queries = [
        "How does Customer relate to Invoice?",
        "Find all Products",
        "What categories exist?",
        "Show me Order's context",
        "Tag Payment as completed",
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"QUERY: \"{query}\"")
        print('─' * 70)
        
        # Parse
        plan = parser.parse(query)
        print(f"\n[PARSE RESULT]")
        print(f"  Intent: {plan.intent.value}")
        print(f"  Source: {plan.source_type}")
        print(f"  Target: {plan.target_type}")
        print(f"  Relation: {plan.relation_label}")
        print(f"  Confidence: {plan.confidence:.2f}")
        print(f"  Tokens: {[str(t) for t in plan.tokens]}")
        
        # Execute
        result = executor.execute(plan)
        print(f"\n[EXECUTION RESULT]")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        else:
            print(f"  Data: {result['data']}")
    
    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
