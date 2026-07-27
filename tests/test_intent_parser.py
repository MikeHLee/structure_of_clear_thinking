"""
Tests for Intent Parser Module

Tests the hierarchical semantic tokenization pipeline:
1. Intent classification
2. Type grounding
3. Semantic role labeling
4. Query plan construction
5. Query execution
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from intent_parser import (
    IntentParser,
    IntentClassifier,
    TypeGrounder,
    SemanticRoleLabeler,
    QueryExecutor,
    QueryIntent,
    SemanticRole,
    SemanticToken,
    QueryPlan,
)
from olog_core import OlogGraph


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ecommerce_olog():
    """Create a sample e-commerce OlogGraph for testing."""
    olog = OlogGraph("E-Commerce")
    
    # Add types
    for t in ["Customer", "Order", "Product", "Payment", "Delivery", "Cart", "Invoice"]:
        olog.add_type(t)
    
    # Add aspects
    olog.add_aspect("Customer", "Cart", "has")
    olog.add_aspect("Customer", "Order", "places")
    olog.add_aspect("Cart", "Product", "contains")
    olog.add_aspect("Order", "Product", "includes")
    olog.add_aspect("Order", "Payment", "requires")
    olog.add_aspect("Order", "Delivery", "triggers")
    olog.add_aspect("Payment", "Invoice", "generates")
    olog.add_aspect("Delivery", "Customer", "shipped_to")
    
    return olog


@pytest.fixture
def parser(ecommerce_olog):
    """Create IntentParser from e-commerce Olog."""
    return IntentParser.from_olog(ecommerce_olog)


@pytest.fixture
def executor(ecommerce_olog):
    """Create QueryExecutor from e-commerce Olog."""
    return QueryExecutor(ecommerce_olog)


# =============================================================================
# Intent Classification Tests
# =============================================================================

class TestIntentClassifier:
    
    def test_retrieve_intent(self):
        classifier = IntentClassifier()
        
        queries = [
            "Find all products",
            "Show me the customers",
            "List all orders",
            "Get the invoices",
        ]
        
        for query in queries:
            intent, confidence = classifier.classify(query)
            assert intent == QueryIntent.RETRIEVE, f"Failed for: {query}"
            assert confidence > 0.5
    
    def test_relate_intent(self):
        classifier = IntentClassifier()
        
        queries = [
            "How does Customer relate to Invoice?",
            "What is the relationship between Order and Product?",
            "Show connection from Payment to Delivery",
            "Path between Customer and Product",
        ]
        
        for query in queries:
            intent, confidence = classifier.classify(query)
            assert intent == QueryIntent.RELATE, f"Failed for: {query}"
            assert confidence > 0.5
    
    def test_aggregate_intent(self):
        classifier = IntentClassifier()
        
        queries = [
            "How many products exist?",
            "Count all orders",
            "What types are there?",
            "What categories exist?",
        ]
        
        for query in queries:
            intent, confidence = classifier.classify(query)
            assert intent == QueryIntent.AGGREGATE, f"Failed for: {query}"
    
    def test_update_intent(self):
        classifier = IntentClassifier()
        
        queries = [
            "Tag Payment as completed",
            "Label Order as shipped",
            "Mark Customer as premium",
        ]
        
        for query in queries:
            intent, confidence = classifier.classify(query)
            assert intent == QueryIntent.UPDATE, f"Failed for: {query}"
    
    def test_navigate_intent(self):
        classifier = IntentClassifier()
        
        queries = [
            "Show Order's context",
            "Expand Customer neighborhood",
            "Display related entities for Product",
        ]
        
        for query in queries:
            intent, confidence = classifier.classify(query)
            assert intent == QueryIntent.NAVIGATE, f"Failed for: {query}"


# =============================================================================
# Type Grounding Tests
# =============================================================================

class TestTypeGrounder:
    
    def test_exact_match(self, ecommerce_olog):
        grounder = TypeGrounder.from_olog(ecommerce_olog)
        
        token = SemanticToken(surface_form="customer")
        grounded = grounder.ground_token(token)
        
        assert grounded.olog_type == "Customer"
        assert grounded.confidence == 1.0
    
    def test_substring_match(self, ecommerce_olog):
        grounder = TypeGrounder.from_olog(ecommerce_olog)
        
        token = SemanticToken(surface_form="products")
        grounded = grounder.ground_token(token)
        
        assert grounded.olog_type == "Product"
        assert grounded.confidence == 0.7
    
    def test_morphism_match(self, ecommerce_olog):
        grounder = TypeGrounder.from_olog(ecommerce_olog)
        
        # Verify morphisms are extracted
        assert "places" in grounder.morphism_labels
        assert "has" in grounder.morphism_labels
        
        token = SemanticToken(surface_form="places")
        grounded = grounder.ground_token(token)
        
        assert grounded.morphism_hint == "places"
        assert grounded.slot_type == SemanticRole.RELATION
    
    def test_no_match(self, ecommerce_olog):
        grounder = TypeGrounder.from_olog(ecommerce_olog)
        
        token = SemanticToken(surface_form="foobar")
        grounded = grounder.ground_token(token)
        
        assert grounded.olog_type is None
        assert grounded.confidence == 0.3


# =============================================================================
# Semantic Role Labeling Tests
# =============================================================================

class TestSemanticRoleLabeler:
    
    def test_relational_roles(self):
        labeler = SemanticRoleLabeler()
        
        tokens = [
            SemanticToken("customer", olog_type="Customer"),
            SemanticToken("to"),
            SemanticToken("invoice", olog_type="Invoice"),
        ]
        
        labeled = labeler.label_tokens(tokens, QueryIntent.RELATE)
        
        assert labeled[0].slot_type == SemanticRole.SOURCE
        assert labeled[2].slot_type == SemanticRole.TARGET
    
    def test_retrieval_roles(self):
        labeler = SemanticRoleLabeler()
        
        tokens = [
            SemanticToken("find"),
            SemanticToken("products", olog_type="Product"),
        ]
        
        labeled = labeler.label_tokens(tokens, QueryIntent.RETRIEVE)
        
        assert labeled[1].slot_type == SemanticRole.ENTITY


# =============================================================================
# Full Parser Integration Tests
# =============================================================================

class TestIntentParser:
    
    def test_relational_query(self, parser):
        plan = parser.parse("How does Customer relate to Invoice?")
        
        assert plan.intent == QueryIntent.RELATE
        assert plan.source_type == "Customer"
        assert plan.target_type == "Invoice"
        assert plan.confidence > 0.8
    
    def test_retrieve_query(self, parser):
        plan = parser.parse("Find all Products")
        
        assert plan.intent == QueryIntent.RETRIEVE
        assert plan.source_type == "Product"
    
    def test_aggregate_query(self, parser):
        plan = parser.parse("What categories exist?")
        
        assert plan.intent == QueryIntent.AGGREGATE
    
    def test_navigate_query(self, parser):
        plan = parser.parse("Show me Order's context")
        
        assert plan.intent == QueryIntent.NAVIGATE
        assert plan.source_type == "Order"


# =============================================================================
# Query Execution Tests
# =============================================================================

class TestQueryExecutor:
    
    def test_retrieve_execution(self, parser, executor):
        plan = parser.parse("Find all Orders")
        result = executor.execute(plan)
        
        assert result["error"] is None
        assert result["data"]["type"] == "Order"
        assert "aspects" in result["data"]
    
    def test_relate_execution(self, parser, executor):
        plan = parser.parse("How does Customer relate to Invoice?")
        result = executor.execute(plan)
        
        assert result["error"] is None
        assert result["data"]["connected"] is True
        assert len(result["data"]["paths"]) > 0
        
        # Verify path structure
        path = result["data"]["paths"][0]
        assert path[0]["from"] == "Customer"
        assert path[-1]["to"] == "Invoice"
    
    def test_aggregate_execution(self, parser, executor):
        plan = parser.parse("What categories exist?")
        result = executor.execute(plan)
        
        assert result["error"] is None
        assert result["data"]["total_types"] == 7
        assert "Customer" in result["data"]["types"]
    
    def test_navigate_execution(self, parser, executor):
        plan = parser.parse("Show Order's context")
        result = executor.execute(plan)
        
        assert result["error"] is None
        assert result["data"]["center"] == "Order"
        assert "Customer" in result["data"]["direct_neighbors"]
    
    def test_invalid_type_error(self, parser, executor):
        plan = QueryPlan(
            intent=QueryIntent.RETRIEVE,
            tokens=[],
            source_type="NonExistent",
        )
        result = executor.execute(plan)
        
        # Error is returned inside data dict for retrieve
        assert result["data"] is not None
        assert "error" in result["data"]
        assert "not found" in result["data"]["error"]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    
    def test_empty_query(self, parser):
        plan = parser.parse("")
        assert plan.intent is not None  # Should default to something
    
    def test_unknown_types_query(self, parser):
        plan = parser.parse("Find all foobar items")
        # Should still parse, just without grounded types
        assert plan.intent == QueryIntent.RETRIEVE
    
    def test_complex_query(self, parser):
        plan = parser.parse(
            "Show me how Customer relates to Product through Order"
        )
        assert plan.intent == QueryIntent.RELATE
        assert plan.source_type == "Customer"
        assert plan.target_type == "Product"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
