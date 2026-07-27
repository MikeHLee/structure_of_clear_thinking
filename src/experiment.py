"""
Experiment Runner for Ontological Induction & Sequence Modeling

Demonstrates:
1. Hybrid Encoder Pipeline (AMR + LLM)
2. Ontological Tokenization
3. Obstruction Detection (H¹ cohomology analog)
4. Penal Method for inconsistent generations
"""

import argparse
import json
from olog_core import OlogGraph, SchemaInducer, CommutativeFact
from hybrid_encoder import HybridOlogEncoder, OntologicalTokenizer, AnthropicBackend


def run_legacy_simulation():
    """Original mock-based simulation for comparison."""
    print("=" * 60)
    print("   LEGACY SIMULATION (Mock SchemaInducer)")
    print("=" * 60)
    
    raw_text = "A customer places an order. The order generates an invoice."
    inducer = SchemaInducer()
    olog = inducer.induce(raw_text)
    
    # Inject obstruction
    olog.add_type("Inventory")
    olog.add_aspect("Order", "Inventory", "reduces")
    olog.add_aspect("Invoice", "Inventory", "increases")
    
    try:
        olog.add_fact(CommutativeFact(
            source_node="Customer",
            target_node="Inventory",
            path_a_labels=["places", "reduces"],
            path_b_labels=["places", "generates", "increases"]
        ))
    except Exception as e:
        print(f">> Fact Injection Error: {e}")

    report = olog.generate_health_report()
    print_report(report)


def run_hybrid_simulation(use_live_llm: bool = False):
    """New hybrid encoder simulation."""
    print("=" * 60)
    print("   HYBRID ENCODER SIMULATION (AMR + LLM Pipeline)")
    print("=" * 60)
    
    # Test cases with varying complexity
    test_cases = [
        {
            "name": "Simple Business Flow",
            "text": "A customer places an order. The order generates an invoice.",
            "inject_obstruction": False
        },
        {
            "name": "E-Commerce with Inventory",
            "text": "A customer places an order. The order reduces inventory. The order generates an invoice.",
            "inject_obstruction": False
        },
        {
            "name": "Contradictory Inventory (Obstruction Test)",
            "text": "A customer places an order. The order reduces inventory. The invoice increases inventory.",
            "inject_obstruction": True,
            "obstruction_fact": {
                "source": "Customer",
                "target": "Inventory",
                "path_a": ["places", "reduces"],
                "path_b": ["places", "generates", "increases"]
            }
        }
    ]
    
    # Create encoder
    if use_live_llm:
        llm_backend = AnthropicBackend()
        encoder = HybridOlogEncoder(llm_backend=llm_backend, use_mock=False)
    else:
        encoder = HybridOlogEncoder(use_mock=True)
    
    tokenizer = OntologicalTokenizer(encoder)
    
    for i, case in enumerate(test_cases):
        print(f"\n{'─' * 60}")
        print(f"TEST CASE {i+1}: {case['name']}")
        print(f"{'─' * 60}")
        print(f"Input: \"{case['text']}\"")
        
        # Encode
        olog, metadata = encoder.encode(case["text"], f"Olog_{i+1}")
        
        # Inject obstruction if specified
        if case.get("inject_obstruction") and case.get("obstruction_fact"):
            fact = case["obstruction_fact"]
            # Ensure required types exist
            for t in [fact["source"], fact["target"]]:
                if t not in olog.graph:
                    olog.add_type(t)
            
            # Ensure required aspects exist for path_b
            # In this case, we need Order -> Invoice -> Inventory
            if "Invoice" not in olog.graph:
                olog.add_type("Invoice")
            if not olog._validate_path("Order", ["generates"]):
                olog.add_aspect("Order", "Invoice", "generates")
            if not olog._validate_path("Invoice", ["increases"]):
                olog.add_aspect("Invoice", "Inventory", "increases")
            
            try:
                olog.add_fact(CommutativeFact(
                    source_node=fact["source"],
                    target_node=fact["target"],
                    path_a_labels=fact["path_a"],
                    path_b_labels=fact["path_b"]
                ))
                print(f"\n[INJECTED COMMUTATIVE FACT]")
                print(f"  Asserting: {fact['source']} --{fact['path_a']}--> {fact['target']}")
                print(f"           = {fact['source']} --{fact['path_b']}--> {fact['target']}")
            except Exception as e:
                print(f"  >> Fact injection failed: {e}")
        
        # Display results
        print(f"\n[PIPELINE STAGES]")
        print(f"  AMR Concepts: {metadata['stages']['amr']['concept_count']}")
        print(f"  AMR Relations: {metadata['stages']['amr']['relation_count']}")
        print(f"  Olog Types: {olog.graph.number_of_nodes()}")
        print(f"  Olog Aspects: {olog.graph.number_of_edges()}")
        
        # Tokenization
        tokens, _ = tokenizer.tokenize(case["text"])
        print(f"\n[ONTOLOGICAL TOKENS]")
        for token in tokens:
            print(f"  {token}")
        
        # Health report
        report = olog.generate_health_report()
        print_report(report)


def run_obstruction_demo():
    """Demonstrates obstruction detection (H¹ analog)."""
    print("=" * 60)
    print("   OBSTRUCTION DETECTION DEMO (H¹ Cohomology Analog)")
    print("=" * 60)
    
    print("""
This demo shows how non-commuting paths in an Olog create
"obstructions" - analogous to non-trivial H¹ cohomology classes.

Scenario: An e-commerce system where:
- Orders reduce inventory (direct path)
- Orders generate invoices, invoices update inventory (indirect path)

If these paths don't converge to the same state, we have an obstruction.
""")
    
    # Build a manual Olog with known obstruction
    olog = OlogGraph("ECommerceWithObstruction")
    
    # Types
    for t in ["Customer", "Order", "Invoice", "Inventory", "Payment"]:
        olog.add_type(t)
    
    # Aspects (morphisms)
    olog.add_aspect("Customer", "Order", "places")
    olog.add_aspect("Order", "Invoice", "generates")
    olog.add_aspect("Order", "Inventory", "reduces")
    olog.add_aspect("Invoice", "Payment", "requires")
    olog.add_aspect("Invoice", "Inventory", "increases")  # CONTRADICTION!
    olog.add_aspect("Payment", "Customer", "confirms")
    
    print("[OLOG STRUCTURE]")
    print("  Types:", list(olog.graph.nodes()))
    print("  Aspects:")
    for u, v, k in olog.graph.edges(keys=True):
        print(f"    {u} --{k}--> {v}")
    
    # Add the contradictory fact
    print("\n[ADDING COMMUTATIVE FACT]")
    print("  Asserting: Customer->places->reduces == Customer->places->generates->increases")
    
    try:
        olog.add_fact(CommutativeFact(
            source_node="Customer",
            target_node="Inventory",
            path_a_labels=["places", "reduces"],
            path_b_labels=["places", "generates", "increases"]
        ))
    except Exception as e:
        print(f"  >> Error: {e}")
    
    report = olog.generate_health_report()
    print_report(report)
    
    print("\n[INTERPRETATION]")
    if report["obstruction_count"] > 0:
        print("  ✗ Non-trivial H¹ detected!")
        print("  → The Olog has inconsistent path semantics")
        print("  → An LLM generating this structure should be penalized")
        print("  → Recommended: Reject generation, request clarification")
    else:
        print("  ✓ H¹ = 0 (trivial)")
        print("  → All paths commute as expected")
        print("  → Generation is semantically consistent")


def print_report(report: dict):
    """Pretty-print health report."""
    status_symbols = {"VALID": "✓", "DEGRADED": "⚠", "INVALID": "✗"}
    symbol = status_symbols.get(report["status"], "?")
    
    print(f"\n[HEALTH REPORT]")
    print(f"  Status: {symbol} {report['status']}")
    print(f"  Consistency Score: {report['semantic_consistency_score']:.2f}")
    print(f"  Obstructions: {report['obstruction_count']}")
    
    if report["obstructions"]:
        print(f"\n[OBSTRUCTIONS DETECTED]")
        for i, issue in enumerate(report["obstructions"]):
            print(f"  {i+1}. {issue}")
        
        print("\n>> PENAL METHOD TRIGGERED: Non-trivial cohomology detected.")


def main():
    parser = argparse.ArgumentParser(description="Ontological Induction Experiments")
    parser.add_argument("--mode", choices=["legacy", "hybrid", "obstruction", "all"],
                       default="all", help="Which simulation to run")
    parser.add_argument("--live-llm", action="store_true",
                       help="Use live LLM (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()
    
    if args.mode == "legacy" or args.mode == "all":
        run_legacy_simulation()
        print("\n")
    
    if args.mode == "hybrid" or args.mode == "all":
        run_hybrid_simulation(use_live_llm=args.live_llm)
        print("\n")
    
    if args.mode == "obstruction" or args.mode == "all":
        run_obstruction_demo()


if __name__ == "__main__":
    main()
