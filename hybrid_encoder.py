"""
Hybrid Encoder Architecture: AMR + LLM Pipeline for Ontological Induction

Architecture:
    Raw Text
        ↓
    [Stage 1: AMR Parser] — Syntactic-semantic grounding
        ↓
    AMRGraph (penman format)
        ↓
    [Stage 2: Concept Extractor] — Extract nodes, roles, coreferences
        ↓
    IntermediateSemanticGraph
        ↓
    [Stage 3: LLM Refiner] — Domain mapping, implicit relations, disambiguation
        ↓
    OlogGraph (categorical structure)

Key Design Principles:
1. AMR provides *grounded* structure — prevents hallucination at syntactic level
2. LLM provides *semantic completion* — fills implicit relations, maps to domain ontology
3. Consistency checking via OlogGraph.generate_health_report() validates final output
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

from olog_core import OlogGraph, OlogNode, OlogMorphism, CommutativeFact

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Stage 1: AMR Parser Interface
# =============================================================================

@dataclass
class AMRConcept:
    """A concept node extracted from AMR."""
    variable: str          # e.g., "c" in (c / customer)
    concept: str           # e.g., "customer"
    instance_of: str = ""  # PropBank frame if applicable
    
@dataclass 
class AMRRelation:
    """A relation (role) extracted from AMR."""
    source_var: str
    role: str              # e.g., ":ARG0", ":location", ":time"
    target_var: str
    
@dataclass
class AMRParseResult:
    """Structured output from AMR parsing."""
    raw_amr: str
    concepts: List[AMRConcept] = field(default_factory=list)
    relations: List[AMRRelation] = field(default_factory=list)
    root_var: str = ""


class AMRParserBackend(ABC):
    """Abstract interface for AMR parsing backends."""
    
    @abstractmethod
    def parse(self, text: str) -> AMRParseResult:
        """Parse text into AMR structure."""
        pass


class MockAMRParser(AMRParserBackend):
    """Mock parser for testing without amrlib dependency."""
    
    def parse(self, text: str) -> AMRParseResult:
        # Simple heuristic extraction for demo
        # In production, use amrlib or transition-amr-parser
        words = text.lower().split()
        concepts = []
        relations = []
        
        # Extract nouns as concepts (simplified)
        noun_indicators = ['customer', 'order', 'invoice', 'product', 'inventory', 
                          'payment', 'shipment', 'account', 'user', 'item']
        verb_indicators = ['place', 'places', 'generate', 'generates', 'create', 
                          'creates', 'reduce', 'reduces', 'increase', 'increases',
                          'buy', 'buys', 'sell', 'sells', 'ship', 'ships']
        
        var_counter = 0
        concept_vars = {}
        
        for word in words:
            word_clean = word.strip('.,;:!?')
            if word_clean in noun_indicators and word_clean not in concept_vars:
                var = f"x{var_counter}"
                var_counter += 1
                concepts.append(AMRConcept(variable=var, concept=word_clean.capitalize()))
                concept_vars[word_clean] = var
        
        # Extract verb relations (simplified: connect adjacent nouns via verbs)
        for i, word in enumerate(words):
            word_clean = word.strip('.,;:!?')
            if word_clean in verb_indicators:
                # Find preceding and following nouns
                prev_noun = None
                next_noun = None
                for j in range(i-1, -1, -1):
                    w = words[j].strip('.,;:!?')
                    if w in noun_indicators:
                        prev_noun = w
                        break
                for j in range(i+1, len(words)):
                    w = words[j].strip('.,;:!?')
                    if w in noun_indicators:
                        next_noun = w
                        break
                
                if prev_noun and next_noun and prev_noun in concept_vars and next_noun in concept_vars:
                    relations.append(AMRRelation(
                        source_var=concept_vars[prev_noun],
                        role=f":ARG1-of({word_clean})",
                        target_var=concept_vars[next_noun]
                    ))
        
        return AMRParseResult(
            raw_amr=f"# Mock AMR for: {text[:50]}...",
            concepts=concepts,
            relations=relations,
            root_var=concepts[0].variable if concepts else ""
        )


class AmrlibParser(AMRParserBackend):
    """Production AMR parser using amrlib."""
    
    def __init__(self):
        self._available = False
        self._fallback = MockAMRParser()
        self.stog = None
        
        try:
            import amrlib
            
            # Try to load the model
            try:
                self.stog = amrlib.load_stog_model()
                self._available = True
                logger.info("amrlib model loaded successfully")
            except Exception as model_err:
                logger.warning(
                    f"amrlib installed but model not found: {model_err}\n"
                    "Run 'python setup_amr_model.py' to download the model."
                )
                
        except ImportError:
            logger.warning(
                "amrlib not installed. Using mock parser.\n"
                "Install with: pip install amrlib"
            )
    
    def parse(self, text: str) -> AMRParseResult:
        if not self._available:
            return self._fallback.parse(text)
        
        import penman
        
        graphs = self.stog.parse_sents([text])
        if not graphs:
            return AMRParseResult(raw_amr="", concepts=[], relations=[])
        
        amr_str = graphs[0]
        graph = penman.decode(amr_str)
        
        concepts = []
        relations = []
        
        # Extract instances (concepts)
        for instance in graph.instances():
            concepts.append(AMRConcept(
                variable=instance.source,
                concept=instance.target,
                instance_of=instance.target
            ))
        
        # Extract edges (relations)
        for edge in graph.edges():
            relations.append(AMRRelation(
                source_var=edge.source,
                role=edge.role,
                target_var=edge.target
            ))
        
        return AMRParseResult(
            raw_amr=amr_str,
            concepts=concepts,
            relations=relations,
            root_var=graph.top if graph.top else ""
        )


# =============================================================================
# Stage 2: Intermediate Semantic Graph
# =============================================================================

@dataclass
class SemanticNode:
    """A semantically-typed node ready for Olog conversion."""
    id: str
    label: str
    semantic_type: str  # "Entity", "Event", "Property", "Relation"
    source_amr_var: str
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticEdge:
    """A typed edge in the semantic graph."""
    source_id: str
    target_id: str
    label: str
    role_type: str  # "Agent", "Patient", "Theme", "Instrument", "Location", etc.
    source_amr_role: str

@dataclass
class IntermediateSemanticGraph:
    """Bridge structure between AMR and Olog."""
    nodes: List[SemanticNode] = field(default_factory=list)
    edges: List[SemanticEdge] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "nodes": [
                {"id": n.id, "label": n.label, "type": n.semantic_type, "attrs": n.attributes}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source_id, "target": e.target_id, "label": e.label, "role": e.role_type}
                for e in self.edges
            ]
        }


class ConceptExtractor:
    """Extracts semantic graph from AMR parse result."""
    
    # Mapping from AMR roles to semantic role types
    ROLE_MAPPING = {
        ":ARG0": "Agent",
        ":ARG1": "Patient", 
        ":ARG2": "Recipient",
        ":ARG3": "Instrument",
        ":location": "Location",
        ":time": "Temporal",
        ":manner": "Manner",
        ":purpose": "Purpose",
        ":cause": "Cause",
        ":mod": "Modifier",
        ":poss": "Possessor",
        ":part": "PartOf",
        ":domain": "Domain",
    }
    
    def extract(self, amr_result: AMRParseResult) -> IntermediateSemanticGraph:
        """Convert AMR to intermediate semantic graph."""
        graph = IntermediateSemanticGraph()
        var_to_node_id = {}
        
        # Create semantic nodes from AMR concepts
        for concept in amr_result.concepts:
            node_id = f"node_{concept.variable}"
            semantic_type = self._infer_semantic_type(concept.concept)
            
            node = SemanticNode(
                id=node_id,
                label=concept.concept,
                semantic_type=semantic_type,
                source_amr_var=concept.variable,
                attributes={"instance_of": concept.instance_of}
            )
            graph.nodes.append(node)
            var_to_node_id[concept.variable] = node_id
        
        # Create semantic edges from AMR relations
        for relation in amr_result.relations:
            if relation.source_var not in var_to_node_id:
                continue
            if relation.target_var not in var_to_node_id:
                continue
                
            role_type = self._map_role(relation.role)
            edge_label = self._derive_edge_label(relation.role)
            
            edge = SemanticEdge(
                source_id=var_to_node_id[relation.source_var],
                target_id=var_to_node_id[relation.target_var],
                label=edge_label,
                role_type=role_type,
                source_amr_role=relation.role
            )
            graph.edges.append(edge)
        
        return graph
    
    def _infer_semantic_type(self, concept: str) -> str:
        """Infer semantic type from concept name."""
        # Events typically end in certain patterns or are verbs
        event_patterns = ['-01', '-02', '-03', '-04', '-05']  # PropBank frames
        if any(concept.endswith(p) for p in event_patterns):
            return "Event"
        
        # Properties/attributes
        property_indicators = ['large', 'small', 'new', 'old', 'good', 'bad']
        if concept.lower() in property_indicators:
            return "Property"
        
        # Default to Entity
        return "Entity"
    
    def _map_role(self, amr_role: str) -> str:
        """Map AMR role to semantic role type."""
        # Handle compound roles like ":ARG1-of(places)"
        base_role = amr_role.split('(')[0].split('-of')[0]
        return self.ROLE_MAPPING.get(base_role, "Related")
    
    def _derive_edge_label(self, amr_role: str) -> str:
        """Derive human-readable edge label from AMR role."""
        # Extract verb from compound roles like ":ARG1-of(places)"
        if '(' in amr_role and ')' in amr_role:
            verb = amr_role.split('(')[1].rstrip(')')
            return verb
        return amr_role.lstrip(':')


# =============================================================================
# Stage 3: LLM Refiner
# =============================================================================

class LLMBackend(ABC):
    """Abstract interface for LLM backends."""
    
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Generate completion for prompt."""
        pass


class MockLLMBackend(LLMBackend):
    """Mock LLM for testing - extracts structure from the prompt."""
    
    def complete(self, prompt: str, system: str = "") -> str:
        # Parse the semantic graph from the prompt and convert to Olog format
        import re
        
        types = []
        aspects = []
        
        # Extract nodes from the JSON in the prompt
        try:
            # Find the semantic graph JSON block - it starts after "semantic graph extracted from text:"
            start_marker = "semantic graph extracted from text:"
            if start_marker in prompt:
                json_start = prompt.find("{", prompt.find(start_marker))
                if json_start != -1:
                    # Find matching closing brace by counting
                    depth = 0
                    json_end = json_start
                    for i, c in enumerate(prompt[json_start:]):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                json_end = json_start + i + 1
                                break
                    sg = json.loads(prompt[json_start:json_end])
                seen_labels = set()
                for node in sg.get("nodes", []):
                    label = node.get("label", "")
                    if label and label not in seen_labels:
                        types.append({"name": label, "description": f"A {label.lower()} in the domain"})
                        seen_labels.add(label)
                
                for edge in sg.get("edges", []):
                    # Map node IDs to labels
                    source_label = None
                    target_label = None
                    for node in sg.get("nodes", []):
                        if node.get("id") == edge.get("source"):
                            source_label = node.get("label")
                        if node.get("id") == edge.get("target"):
                            target_label = node.get("label")
                    
                    if source_label and target_label:
                        aspects.append({
                            "source": source_label,
                            "target": target_label,
                            "label": edge.get("label", "related_to"),
                            "description": f"{source_label} {edge.get('label', 'relates to')} {target_label}"
                        })
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return json.dumps({
            "types": types,
            "aspects": aspects,
            "facts": [],
            "reasoning": "Mock LLM: Direct conversion from semantic graph"
        })


class AnthropicBackend(LLMBackend):
    """Claude API backend."""
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key. Using mock backend.")
            self._fallback = MockLLMBackend()
            self._available = False
        else:
            self._available = True
    
    def complete(self, prompt: str, system: str = "") -> str:
        if not self._available:
            return self._fallback.complete(prompt, system)
        
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text


class OllamaBackend(LLMBackend):
    """Local Ollama backend."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "zac/phi4-tools:latest"
        self.api_url = "http://localhost:11434/api/generate"
    
    def complete(self, prompt: str, system: str = "") -> str:
        import requests
        import json
        
        full_prompt = f"System: {system}\n\nUser: {prompt}\n\nAssistant: "
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            logger.info(f"Calling Ollama ({self.model_name})...")
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return "{}"


class TriplexBackend(LLMBackend):
    """Specialized backend for sciphi/triplex triplet extraction."""
    
    def __init__(self, model_name: str = "sciphi/triplex:latest"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
    def complete(self, prompt: str, system: str = "") -> str:
        import requests
        import json
        
        # Triplex expects a specific prompt format for extraction
        # It doesn't use the standard system/user split the same way for JSON.
        # But we can try to force it to return JSON-like structure.
        # However, it's better at just returning triplets.
        
        # We will wrap the prompt to ensure it fits the 'extract' task
        extraction_prompt = f"Extract all entities and relationships from the following text as a JSON object with 'types' and 'aspects'.\n\nText: {prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": extraction_prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            logger.info(f"Calling Triplex ({self.model_name})...")
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Triplex call failed: {e}")
            return "{}"


class OlogRefiner:
    """Uses LLM to refine semantic graph into proper Olog."""
    
    SYSTEM_PROMPT = """You are an expert in category theory and ontology engineering.
Your task is to extract a COMPREHENSIVE Olog (Ontology Log) from technical text.

An Olog consists of:
1. Types (Objects): Represent sets of things (e.g., 'LLM', 'Context Window', 'Virtual Memory').
2. Aspects (Morphisms): Functional relationships between types (e.g., 'Virtual Memory' --(manages)--> 'LLM Context').
3. Facts (Commutative Diagrams): Paths that are equivalent.

Rules for a COMPLETE Olog:
- Find EVERY entity mentioned that can be a 'Type'.
- Find EVERY relationship between these entities.
- Morphisms MUST be functional (A has at most one B).
- Output as MANY valid aspects as possible to create a dense graph.
- All aspects must be readable as "a [Source] has [Aspect] which is a [Target]".

You must output valid JSON only."""

    REFINEMENT_PROMPT_TEMPLATE = """Extract a detailed Olog from this text:

"{original_text}"

{semantic_graph_context}

Be exhaustive. Find all implicit relationships.

Output JSON format:
{{
    "types": [
        {{"name": "TypeName", "description": "..."}}
    ],
    "aspects": [
        {{"source": "SourceType", "target": "TargetType", "label": "aspect_label", "description": "..."}}
    ],
    "facts": [
        {{"source": "StartType", "target": "EndType", "path_a": ["label1"], "path_b": ["label2"], "justification": "..."}}
    ],
    "reasoning": "Brief explanation of the core categorical structure"
}}"""

    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        self.llm = llm_backend or MockLLMBackend()
    
    def refine(self, semantic_graph: IntermediateSemanticGraph, original_text: str, extra_context: str = "") -> Dict:
        """Refine semantic graph using LLM."""
        sg_context = ""
        if semantic_graph.nodes:
            sg_context = f"Initial semantic graph nodes: {[n.label for n in semantic_graph.nodes]}\n"
            
        prompt = self.REFINEMENT_PROMPT_TEMPLATE.format(
            semantic_graph_context=sg_context + extra_context,
            original_text=original_text
        )
        
        try:
            response = self.llm.complete(prompt, self.SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"LLM backend failed ({e}), falling back to mock refinement.")
            fallback_llm = MockLLMBackend()
            response = fallback_llm.complete(prompt, self.SYSTEM_PROMPT)
        
        try:
            # Try to parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response}")
            # Return minimal valid structure
            return {
                "types": [{"name": n.label, "description": ""} for n in semantic_graph.nodes],
                "aspects": [{"source": e.source_id.replace("node_", ""), 
                            "target": e.target_id.replace("node_", ""),
                            "label": e.label, "description": ""} for e in semantic_graph.edges],
                "facts": [],
                "reasoning": "Fallback: Direct conversion from semantic graph"
            }


# =============================================================================
# Hybrid Encoder: Full Pipeline
# =============================================================================

class HybridOlogEncoder:
    """
    The main encoder that combines AMR parsing with LLM refinement.
    
    Pipeline:
        Text → AMR → SemanticGraph → LLM Refinement → OlogGraph
    """
    
    def __init__(
        self,
        amr_parser: Optional[AMRParserBackend] = None,
        llm_backend: Optional[LLMBackend] = None,
        use_mock: bool = True
    ):
        if use_mock:
            self.amr_parser = MockAMRParser()
            self.llm_backend = MockLLMBackend()
        else:
            self.amr_parser = amr_parser or AmrlibParser()
            self.llm_backend = llm_backend or AnthropicBackend()
        
        self.concept_extractor = ConceptExtractor()
        self.refiner = OlogRefiner(self.llm_backend)
    
    def encode(self, text: str, olog_name: str = "InducedOlog", iterations: int = 2) -> Tuple[OlogGraph, Dict]:
        """
        Full encoding pipeline with iterative refinement using shadow morphisms.
        """
        metadata = {"input_text": text, "stages": {}}
        
        # Stage 1: AMR Parsing
        logger.info("Stage 1: AMR Parsing")
        try:
            amr_result = self.amr_parser.parse(text)
        except Exception as e:
            logger.warning(f"AMR Parsing failed: {e}")
            amr_result = AMRParseResult(raw_amr="", concepts=[], relations=[])

        metadata["stages"]["amr"] = {"concept_count": len(amr_result.concepts)}
        
        # Stage 2: Concept Extraction
        semantic_graph = self.concept_extractor.extract(amr_result)
        
        # Stage 3: Iterative Refinement
        current_olog = OlogGraph(olog_name)
        all_refined_outputs = []
        
        for i in range(iterations):
            logger.info(f"Refinement Iteration {i+1}/{iterations}")
            
            # Detect shadows from previous pass (empty on first pass)
            shadows = self.detect_shadow_morphisms(semantic_graph, current_olog)
            
            # Enhance prompt with shadows if they exist
            shadow_context = ""
            if shadows:
                shadow_context = "\nPotential relationships to verify (from syntactic analysis):\n"
                for s in shadows:
                    shadow_context += f"- Could '{s['source']}' have an aspect '{s['amr_label']}' targeting '{s['target']}'?\n"
            
            # Get refined structure from LLM
            # We modify the refiner slightly to take this extra context
            refined = self.refiner.refine(semantic_graph, text, extra_context=shadow_context)
            all_refined_outputs.append(refined)
            
            # Update Olog
            current_olog = self._build_olog(refined, olog_name)
            
        metadata["stages"]["llm_refinement"] = all_refined_outputs[-1]
        metadata["stages"]["shadow_morphisms"] = self.detect_shadow_morphisms(semantic_graph, current_olog)
        
        # Final Health Report
        metadata["health_report"] = current_olog.generate_health_report()
        
        return current_olog, metadata
    
    def detect_shadow_morphisms(self, semantic_graph: IntermediateSemanticGraph, olog: OlogGraph) -> List[Dict]:
        """
        Identify 'Shadow Morphisms': Relations present in the syntactic/semantic graph
        but missing from the final Olog. These represent the 'search space' for
        future LLM passes or 'invalid' morphisms if they were explicitly rejected.
        """
        shadows = []
        
        # Map SemanticNode IDs to Olog Type Names (heuristic matching)
        # We need to know which Olog Type corresponds to which Semantic Node
        # This is fuzzy because the LLM might have renamed things.
        
        # 1. Build a fuzzy map
        node_id_to_type = {}
        olog_types = set(olog.graph.nodes())
        
        for node in semantic_graph.nodes:
            # Exact match?
            if node.label in olog_types:
                node_id_to_type[node.id] = node.label
                continue
            
            # Case-insensitive match?
            for ot in olog_types:
                if ot.lower() == node.label.lower():
                    node_id_to_type[node.id] = ot
                    break
            
            # If still not found, check partial containment (risky but useful for shadows)
            if node.id not in node_id_to_type:
                for ot in olog_types:
                    if node.label.lower() in ot.lower() or ot.lower() in node.label.lower():
                        node_id_to_type[node.id] = ot
                        break
                        
        # 2. Check every Semantic Edge
        for edge in semantic_graph.edges:
            src_type = node_id_to_type.get(edge.source_id)
            tgt_type = node_id_to_type.get(edge.target_id)
            
            if src_type and tgt_type:
                # Does this edge exist in the Olog?
                # Check for ANY edge between these two types in the correct direction
                if not olog.graph.has_edge(src_type, tgt_type):
                    shadows.append({
                        "source": src_type,
                        "target": tgt_type,
                        "amr_label": edge.label,
                        "amr_role": edge.role_type,
                        "reason": "Syntactic dependency not promoted to Olog aspect"
                    })
                    
        return shadows

    def _build_olog(self, refined: Dict, name: str) -> OlogGraph:
        """Convert refined LLM output to OlogGraph."""
        olog = OlogGraph(name)
        
        # Add types
        for type_def in refined.get("types", []):
            olog.add_type(
                name=type_def["name"],
                description=type_def.get("description", "")
            )
        
        # Add aspects
        for aspect in refined.get("aspects", []):
            try:
                olog.add_aspect(
                    source=aspect["source"],
                    target=aspect["target"],
                    label=aspect["label"],
                    description=aspect.get("description", "")
                )
            except ValueError as e:
                logger.warning(f"Skipping invalid aspect: {e}")
        
        # Add facts
        for fact in refined.get("facts", []):
            try:
                olog.add_fact(CommutativeFact(
                    source_node=fact["source"],
                    target_node=fact["target"],
                    path_a_labels=fact["path_a"],
                    path_b_labels=fact["path_b"]
                ))
            except ValueError as e:
                logger.warning(f"Skipping invalid fact: {e}")
        
        return olog


# =============================================================================
# Ontological Tokenizer
# =============================================================================

class OntologicalToken:
    """
    An ontological token - a semantically grounded unit.
    
    Unlike statistical tokens (BPE, WordPiece), ontological tokens
    are grounded in categorical structure.
    """
    
    def __init__(
        self,
        surface_form: str,
        olog_type: str,
        semantic_role: str,
        context_path: List[str] = None
    ):
        self.surface_form = surface_form  # Original text span
        self.olog_type = olog_type        # Corresponding Olog type
        self.semantic_role = semantic_role # Role in sentence (Agent, Patient, etc.)
        self.context_path = context_path or []  # Path in Olog from root
    
    def __repr__(self):
        return f"OToken({self.surface_form!r} → {self.olog_type}:{self.semantic_role})"


class OntologicalTokenizer:
    """
    Tokenizer that produces ontologically-grounded tokens.
    
    Unlike BPE/WordPiece which are purely statistical, this tokenizer
    grounds each token in the Olog structure.
    """
    
    def __init__(self, encoder: HybridOlogEncoder):
        self.encoder = encoder
        self._olog_cache: Dict[str, OlogGraph] = {}
    
    def tokenize(self, text: str) -> Tuple[List[OntologicalToken], OlogGraph]:
        """
        Tokenize text into ontological tokens.
        
        Returns:
            Tuple of (list of OntologicalTokens, the induced OlogGraph)
        """
        # Encode text to Olog
        olog, metadata = self.encoder.encode(text)
        
        # Extract tokens from semantic graph
        tokens = []
        semantic_graph = metadata["stages"].get("semantic_graph", {})
        
        for node in semantic_graph.get("nodes", []):
            # Find corresponding edges for semantic role
            role = "Entity"  # Default
            for edge in semantic_graph.get("edges", []):
                if edge["target"] == node["id"]:
                    role = edge.get("role", "Related")
                    break
            
            token = OntologicalToken(
                surface_form=node["label"],
                olog_type=node["label"],
                semantic_role=role
            )
            tokens.append(token)
        
        return tokens, olog
    
    def batch_tokenize(self, texts: List[str]) -> List[Tuple[List[OntologicalToken], OlogGraph]]:
        """Tokenize multiple texts."""
        return [self.tokenize(text) for text in texts]


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """Demonstrate the hybrid encoder pipeline."""
    print("=" * 60)
    print("  HYBRID OLOG ENCODER DEMO")
    print("=" * 60)
    
    # Create encoder with mock backends for testing
    encoder = HybridOlogEncoder(use_mock=True)
    
    # Test text
    text = "A customer places an order. The order generates an invoice. The invoice is sent to the customer."
    
    print(f"\nInput text: {text}\n")
    
    # Run encoding
    olog, metadata = encoder.encode(text, "CustomerOrderOlog")
    
    # Display results
    print("\n[STAGE 1: AMR PARSING]")
    print(f"  Concepts found: {metadata['stages']['amr']['concept_count']}")
    print(f"  Relations found: {metadata['stages']['amr']['relation_count']}")
    
    print("\n[STAGE 2: SEMANTIC GRAPH]")
    sg = metadata['stages']['semantic_graph']
    for node in sg['nodes']:
        print(f"  Node: {node['label']} ({node['type']})")
    for edge in sg['edges']:
        print(f"  Edge: {edge['source']} --{edge['label']}--> {edge['target']}")
    
    print("\n[STAGE 3: OLOG STRUCTURE]")
    print(f"  Types: {olog.graph.number_of_nodes()}")
    print(f"  Aspects: {olog.graph.number_of_edges()}")
    print(f"  Facts: {len(olog.facts)}")
    
    print("\n[HEALTH REPORT]")
    health = metadata['health_report']
    print(f"  Status: {health['status']}")
    print(f"  Consistency Score: {health['semantic_consistency_score']:.2f}")
    print(f"  Obstructions: {health['obstruction_count']}")
    
    # Demo tokenizer
    print("\n[ONTOLOGICAL TOKENIZATION]")
    tokenizer = OntologicalTokenizer(encoder)
    tokens, _ = tokenizer.tokenize(text)
    for token in tokens:
        print(f"  {token}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
