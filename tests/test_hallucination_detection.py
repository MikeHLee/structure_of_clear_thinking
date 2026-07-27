#!/usr/bin/env python3
"""
Integration tests for hallucination detection.

Tests the full pipeline:
1. Olog construction
2. Proof engine (all 3 modes)
3. Ontological attention masking
4. End-to-end claim verification

Run:
    python -m pytest tests/test_hallucination_detection.py -v
    python tests/test_hallucination_detection.py  # Standalone
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from typing import List, Tuple

from olog_core import OlogGraph
from proof_objects import ProofEngine, ProofMode, ProofStatus, Claim
from ontological_attention import OntologicalAttention, TypedToken, RelationAwareEmbedding


class TestProofModes(unittest.TestCase):
    """Test different proof modes for hallucination detection."""
    
    def setUp(self):
        """Create test ontology."""
        self.olog = OlogGraph(name="TestOntology")
        
        # Types
        for t in ["Customer", "Order", "Product", "Invoice", "Payment", "Shipment"]:
            self.olog.add_type(t, f"A {t.lower()}")
        
        # Aspects (edges)
        self.olog.add_aspect("Customer", "Order", "places")
        self.olog.add_aspect("Order", "Product", "contains")
        self.olog.add_aspect("Order", "Invoice", "generates")
        self.olog.add_aspect("Invoice", "Payment", "requires")
        self.olog.add_aspect("Payment", "Shipment", "triggers")
        self.olog.add_aspect("Shipment", "Customer", "delivers_to")
    
    def test_strict_mode_valid_claims(self):
        """STRICT mode should accept direct edges with exact labels."""
        engine = ProofEngine(self.olog, mode=ProofMode.STRICT)
        
        valid_claims = [
            "Customer places Order",
            "Order generates Invoice",
            "Invoice requires Payment",
            "Payment triggers Shipment",
            "Shipment delivers_to Customer",
        ]
        
        for claim in valid_claims:
            proof = engine.prove(claim)
            self.assertEqual(proof.status, ProofStatus.VALID, f"Should be valid: {claim}")
    
    def test_strict_mode_rejects_wrong_relation(self):
        """STRICT mode should reject claims with wrong relation labels."""
        engine = ProofEngine(self.olog, mode=ProofMode.STRICT)
        
        invalid_claims = [
            ("Payment places Customer", "Wrong relation - should be via triggers/delivers_to"),
            ("Customer triggers Shipment", "triggers is Payment->Shipment, not Customer->"),
            ("Order requires Payment", "requires is Invoice->Payment, not Order->"),
            ("Invoice places Customer", "places is Customer->Order, not Invoice->"),
        ]
        
        for claim, reason in invalid_claims:
            proof = engine.prove(claim)
            self.assertEqual(proof.status, ProofStatus.INVALID, f"Should be invalid ({reason}): {claim}")
    
    def test_strict_mode_rejects_no_path(self):
        """STRICT mode should reject claims with no path at all."""
        engine = ProofEngine(self.olog, mode=ProofMode.STRICT)
        
        proof = engine.prove("Product generates Invoice")
        self.assertEqual(proof.status, ProofStatus.INVALID)
        self.assertIn("No edge", proof.failure_reason)
    
    def test_compositional_mode_allows_valid_composition(self):
        """COMPOSITIONAL mode should allow claims where relation appears in path."""
        engine = ProofEngine(self.olog, mode=ProofMode.COMPOSITIONAL)
        
        # "Customer triggers Shipment" - triggers appears in the path
        # Customer -> Order -> Invoice -> Payment --(triggers)--> Shipment
        proof = engine.prove("Customer triggers Shipment")
        self.assertEqual(proof.status, ProofStatus.VALID)
    
    def test_reachability_mode_allows_any_path(self):
        """REACHABILITY mode should allow any reachable pair."""
        engine = ProofEngine(self.olog, mode=ProofMode.REACHABILITY)
        
        # Any claim between reachable types should pass
        reachable_claims = [
            "Customer places Order",
            "Payment places Customer",  # Wrong relation but reachable via cycle
            "Customer triggers Shipment",  # Wrong relation but reachable
        ]
        
        for claim in reachable_claims:
            proof = engine.prove(claim)
            self.assertEqual(proof.status, ProofStatus.VALID, f"Should be reachable: {claim}")
    
    def test_all_modes_reject_unreachable(self):
        """All modes should reject claims between unreachable types."""
        for mode in [ProofMode.STRICT, ProofMode.COMPOSITIONAL, ProofMode.REACHABILITY]:
            engine = ProofEngine(self.olog, mode=mode)
            
            # Product is a leaf - nothing is reachable from it except itself
            proof = engine.prove("Product generates Invoice")
            self.assertEqual(proof.status, ProofStatus.INVALID, 
                           f"{mode.value} should reject Product->Invoice")
    
    def test_identity_proof(self):
        """Identity claims (A = A) should always be valid."""
        engine = ProofEngine(self.olog, mode=ProofMode.STRICT)
        
        # Self-referential claim
        proof = engine.prove("Customer is Customer")
        self.assertEqual(proof.status, ProofStatus.VALID)


class TestOntologicalAttention(unittest.TestCase):
    """Test type-constrained attention masking."""
    
    def setUp(self):
        """Create test ontology and attention module."""
        self.olog = OlogGraph(name="TestOntology")
        
        for t in ["Customer", "Order", "Product", "Invoice", "Payment"]:
            self.olog.add_type(t)
        
        self.olog.add_aspect("Customer", "Order", "places")
        self.olog.add_aspect("Order", "Product", "contains")
        self.olog.add_aspect("Order", "Invoice", "generates")
        self.olog.add_aspect("Invoice", "Payment", "requires")
        
        self.attention = OntologicalAttention(self.olog, embed_dim=64)
    
    def test_reachability_computed_correctly(self):
        """Reachability should include transitive closure."""
        reach = self.attention._reachability
        
        # Direct edges
        self.assertIn("Order", reach.get("Customer", set()))
        self.assertIn("Product", reach.get("Order", set()))
        
        # Transitive
        self.assertIn("Product", reach.get("Customer", set()))
        self.assertIn("Invoice", reach.get("Customer", set()))
        self.assertIn("Payment", reach.get("Customer", set()))
        
        # Not reachable (reverse direction)
        self.assertNotIn("Customer", reach.get("Product", set()))
        self.assertNotIn("Customer", reach.get("Invoice", set()))
    
    def test_attention_mask_blocks_invalid(self):
        """Attention mask should block invalid type pairs."""
        tokens = [
            TypedToken("customer", 0, olog_type="Customer"),
            TypedToken("order", 1, olog_type="Order"),
            TypedToken("product", 2, olog_type="Product"),
        ]
        
        mask = self.attention.create_attention_mask(tokens)
        
        # Customer can attend to Order and Product
        self.assertEqual(mask.mask[0, 1], 1)  # Customer -> Order
        self.assertEqual(mask.mask[0, 2], 1)  # Customer -> Product (via Order)
        
        # Product cannot attend to Customer or Order (reverse direction)
        self.assertEqual(mask.mask[2, 0], 0)  # Product -> Customer blocked
        self.assertEqual(mask.mask[2, 1], 0)  # Product -> Order blocked
    
    def test_untyped_tokens_pass_through(self):
        """Untyped tokens should be able to attend to anything."""
        tokens = [
            TypedToken("the", 0),  # Untyped
            TypedToken("customer", 1, olog_type="Customer"),
            TypedToken("product", 2, olog_type="Product"),
        ]
        
        mask = self.attention.create_attention_mask(tokens)
        
        # Untyped token can attend to everything
        self.assertEqual(mask.mask[0, 1], 1)
        self.assertEqual(mask.mask[0, 2], 1)
        
        # Everything can attend to untyped token
        self.assertEqual(mask.mask[1, 0], 1)
        self.assertEqual(mask.mask[2, 0], 1)
    
    def test_forward_pass_shape(self):
        """Forward pass should produce correct output shape."""
        tokens = [
            TypedToken("customer", 0, olog_type="Customer"),
            TypedToken("places", 1, is_relation=True, relation_label="places"),
            TypedToken("order", 2, olog_type="Order"),
        ]
        
        output, weights = self.attention.forward(tokens, return_attention=True)
        
        self.assertEqual(output.shape, (3, 64))  # (seq_len, embed_dim)
        self.assertEqual(weights.shape, (3, 3))  # (seq_len, seq_len)


class TestRelationAwareEmbeddings(unittest.TestCase):
    """Test relation-aware embedding composition."""
    
    def setUp(self):
        """Create test ontology and embedder."""
        self.olog = OlogGraph(name="TestOntology")
        
        for t in ["A", "B", "C", "D"]:
            self.olog.add_type(t)
        
        self.olog.add_aspect("A", "B", "f")
        self.olog.add_aspect("B", "C", "g")
        self.olog.add_aspect("A", "C", "h")  # h = g ∘ f
        
        self.embedder = RelationAwareEmbedding(self.olog, embed_dim=64)
    
    def test_type_embeddings_exist(self):
        """All types should have embeddings."""
        for t in ["A", "B", "C", "D"]:
            emb = self.embedder.embed_type(t)
            self.assertEqual(emb.shape, (64,))
    
    def test_relation_embeddings_exist(self):
        """All relations should have embeddings."""
        for src, tgt, rel in [("A", "B", "f"), ("B", "C", "g"), ("A", "C", "h")]:
            emb = self.embedder.embed_relation(src, rel, tgt)
            self.assertEqual(emb.shape, (64,))
    
    def test_composition_produces_embedding(self):
        """Composing relations should produce valid embedding."""
        composed = self.embedder.compose_relations([
            ("A", "f", "B"),
            ("B", "g", "C"),
        ])
        
        self.assertEqual(composed.shape, (64,))
    
    def test_similarity_function(self):
        """Similarity should be in [-1, 1] range."""
        emb1 = self.embedder.embed_type("A")
        emb2 = self.embedder.embed_type("B")
        
        sim = self.embedder.similarity(emb1, emb2)
        self.assertGreaterEqual(sim, -1)
        self.assertLessEqual(sim, 1)


class TestEndToEndHallucinationDetection(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Create realistic ontology."""
        self.olog = OlogGraph(name="ECommerceOntology")
        
        types = [
            ("User", "A registered user"),
            ("Cart", "Shopping cart"),
            ("Item", "A purchasable item"),
            ("Checkout", "Checkout process"),
            ("Payment", "Payment transaction"),
            ("Order", "Completed order"),
            ("Delivery", "Delivery process"),
        ]
        
        for name, desc in types:
            self.olog.add_type(name, desc)
        
        aspects = [
            ("User", "Cart", "has"),
            ("Cart", "Item", "contains"),
            ("Cart", "Checkout", "proceeds_to"),
            ("Checkout", "Payment", "requires"),
            ("Payment", "Order", "creates"),
            ("Order", "Delivery", "triggers"),
            ("Delivery", "User", "to"),
        ]
        
        for src, tgt, label in aspects:
            self.olog.add_aspect(src, tgt, label)
        
        self.strict_engine = ProofEngine(self.olog, mode=ProofMode.STRICT)
    
    def test_valid_business_flow(self):
        """Valid e-commerce flow claims should pass."""
        valid_claims = [
            "User has Cart",
            "Cart contains Item",
            "Cart proceeds_to Checkout",
            "Checkout requires Payment",
            "Payment creates Order",
            "Order triggers Delivery",
        ]
        
        for claim in valid_claims:
            proof = self.strict_engine.prove(claim)
            self.assertTrue(proof.is_valid, f"Should be valid: {claim}")
    
    def test_hallucinated_shortcuts(self):
        """Hallucinated shortcuts should be rejected."""
        hallucinations = [
            "User creates Order",  # Skips Cart, Checkout, Payment
            "Cart triggers Delivery",  # Skips Checkout, Payment, Order
            "Item requires Payment",  # No direct relation
            "Delivery has Cart",  # Reverse direction
        ]
        
        for claim in hallucinations:
            proof = self.strict_engine.prove(claim)
            self.assertFalse(proof.is_valid, f"Should be hallucination: {claim}")
    
    def test_audit_response(self):
        """Response auditing should catch hallucinations."""
        response = {
            "claims": [
                "User has Cart",  # Valid
                "Cart contains Item",  # Valid
                "User creates Order",  # HALLUCINATION
            ]
        }
        
        audit = self.strict_engine.audit_response(response)
        
        self.assertEqual(audit["total_claims"], 3)
        self.assertEqual(audit["proven"], 2)
        self.assertEqual(audit["failed"], 1)
        self.assertEqual(len(audit["hallucinations"]), 1)
        self.assertIn("User creates Order", audit["hallucinations"][0]["claim"])


def run_benchmark():
    """Run full benchmark and print results."""
    print("=" * 70)
    print("  HALLUCINATION DETECTION INTEGRATION TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestProofModes))
    suite.addTests(loader.loadTestsFromTestCase(TestOntologicalAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestRelationAwareEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndHallucinationDetection))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun:.1%}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
