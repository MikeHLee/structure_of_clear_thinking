import logging
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from olog_core import OlogGraph, OlogNode, OlogMorphism, CommutativeFact
from olog_ops import OlogPushout
import numpy as np

logger = logging.getLogger(__name__)

def run_pushout_experiment():
    print("=" * 60)
    print("  CATEGORICAL PUSHOUT EXPERIMENT")
    print("  Gluing Clinical and Billing Domains")
    print("=" * 60)
    
    # 1. Domain A: Clinical (Healthcare)
    clinical = OlogGraph("Clinical")
    clinical.add_type("Patient")
    clinical.add_type("Doctor")
    clinical.add_type("Treatment")
    clinical.add_type("Diagnosis")
    clinical.add_aspect("Patient", "Doctor", "sees")
    clinical.add_aspect("Doctor", "Diagnosis", "makes")
    clinical.add_aspect("Diagnosis", "Treatment", "requires")
    
    # Add a commutative fact: Patient -> Diagnosis (via Doctor)
    # Actually, let's add two paths to treatment
    clinical.add_aspect("Patient", "Treatment", "directly_requests")
    # Patient --sees--> Doctor --makes--> Diagnosis --requires--> Treatment
    fact = CommutativeFact(
        source_node="Patient",
        target_node="Treatment",
        path_a_labels=["sees", "makes", "requires"],
        path_b_labels=["directly_requests"]
    )
    clinical.add_fact(fact)
    
    # 2. Domain B: Billing (Finance)
    billing = OlogGraph("Billing")
    billing.add_type("Customer")
    billing.add_type("Service")
    billing.add_type("Invoice")
    billing.add_type("Payment")
    billing.add_aspect("Customer", "Service", "receives")
    billing.add_aspect("Service", "Invoice", "generates")
    billing.add_aspect("Invoice", "Payment", "requires")
    
    # 3. Define the Interface (The Span)
    # We map 'Patient' -> 'Customer' and 'Treatment' -> 'Service'
    mapping = {
        "Patient": "Customer",
        "Treatment": "Service"
    }
    
    print("\n[STEP 1] Computing Pushout...")
    unified = OlogPushout.compute(clinical, billing, mapping, name="Healthcare_LMS_Pushout")
    
    print(f"\n  Unified Olog: {unified.name}")
    print(f"  Types: {list(unified.graph.nodes())}")
    print(f"  Aspects: {unified.graph.number_of_edges()}")
    print(f"  Facts: {len(unified.facts)}")
    
    # 4. Verify Transitive Reachability across Glue-points
    # Clinical: Patient -> Doctor -> Diagnosis -> Treatment
    # Glue: Treatment is Service
    # Billing: Service -> Invoice -> Payment
    # Therefore: Patient -> Payment should exist.
    
    print("\n[STEP 2] Verifying Cross-Domain Compositionality...")
    has_path = nx.has_path(unified.graph, "Patient", "Payment")
    print(f"  Can a Clinical 'Patient' reach a Billing 'Payment'? {has_path}")
    
    if has_path:
        path = nx.shortest_path(unified.graph, "Patient", "Payment")
        print(f"  Sample Path: {' -> '.join(path)}")
        
    # 5. Check Consistency Score of the Pushout
    # If the glued graph introduces cycles or contradictory facts, score will drop.
    report = unified.generate_health_report(include_semantic=False)
    print(f"\n[STEP 3] Topological Health Check:")
    print(f"  Consistency Score: {report['semantic_consistency_score']:.4f}")
    print(f"  Status: {report['status']}")
    if report['obstructions']:
        print(f"  Obstructions Found: {len(report['obstructions'])}")
        for obs in report['obstructions']:
            print(f"    - {obs}")
            
    # 6. Verify that the clinical fact was remapped correctly
    print("\n[STEP 4] Fact Remapping Validation:")
    for i, f in enumerate(unified.facts):
        print(f"  Fact {i}: {f.source_node} --({f.path_a_labels})--> {f.target_node} == {f.source_node} --({f.path_b_labels})--> {f.target_node}")

    print("\n" + "=" * 60)
    print("  Experiment Complete.")

if __name__ == "__main__":
    run_pushout_experiment()
