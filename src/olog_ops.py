import logging
from typing import Dict, List, Tuple, Set, Optional
from olog_core import OlogGraph, OlogNode, OlogMorphism, CommutativeFact

logger = logging.getLogger(__name__)

class OlogCategoricalOps:
    """
    Implements formal categorical operations for Ologs.
    """
    
    @staticmethod
    def compute_pushout(
        olog_b: OlogGraph,
        olog_c: OlogGraph,
        node_mapping: Dict[str, str], # Maps node in B to corresponding node in C
        name: str = "Pushout_Olog"
    ) -> OlogGraph:
        """Computes the pushout (Unification)."""
        pushout = OlogGraph(name)
        
        # 1. Add all types from B
        for node_name, node_data in olog_b.graph.nodes(data='data'):
            pushout.add_type(node_name, node_data.description if node_data else "")
            
        rev_mapping = {v: k for k, v in node_mapping.items()}
        for node_name, node_data in olog_c.graph.nodes(data='data'):
            if node_name in rev_mapping:
                existing_node = pushout.graph.nodes[rev_mapping[node_name]].get('data')
                if existing_node and node_data:
                    existing_node.description += f" | In {olog_c.name}: {node_data.description}"
                continue
            pushout.add_type(node_name, node_data.description if node_data else "")

        def get_canonical(node_name, source_is_c=False):
            if source_is_c:
                return rev_mapping.get(node_name, node_name)
            return node_name

        for u, v, key, data in olog_b.graph.edges(keys=True, data='data'):
            pushout.add_aspect(u, v, key, data.description if data else "")

        for u, v, key, data in olog_c.graph.edges(keys=True, data='data'):
            src = get_canonical(u, source_is_c=True)
            tgt = get_canonical(v, source_is_c=True)
            if pushout.graph.has_edge(src, tgt, key=key): continue
            pushout.add_aspect(src, tgt, key, data.description if data else "")

        for fact in olog_b.facts: pushout.add_fact(fact)
        for fact in olog_c.facts:
            try:
                remapped_fact = CommutativeFact(
                    source_node=get_canonical(fact.source_node, source_is_c=True),
                    target_node=get_canonical(fact.target_node, source_is_c=True),
                    path_a_labels=fact.path_a_labels,
                    path_b_labels=fact.path_b_labels
                )
                pushout.add_fact(remapped_fact)
            except: pass
        return pushout

    @staticmethod
    def compute_pullback(
        olog_b: OlogGraph,
        olog_c: OlogGraph,
        name: str = "Pullback_Olog"
    ) -> OlogGraph:
        """
        Computes the pullback (The common interface / intersection).
        This identifies nodes and morphisms that exist in both domains.
        """
        pullback = OlogGraph(name)
        
        b_nodes = set(olog_b.graph.nodes())
        c_nodes = set(olog_c.graph.nodes())
        shared_nodes = b_nodes.intersection(c_nodes)
        
        for node in shared_nodes:
            # Merge descriptions
            d_b = olog_b.graph.nodes[node].get('data').description or ""
            d_c = olog_c.graph.nodes[node].get('data').description or ""
            pullback.add_type(node, f"Interface node: {d_b} / {d_c}")
            
        # Shared aspects
        for u, v, key in olog_b.graph.edges(keys=True):
            if u in shared_nodes and v in shared_nodes:
                if olog_c.graph.has_edge(u, v, key=key):
                    pullback.add_aspect(u, v, key, "Shared Morphism")
                    
        return pullback

class OlogPushout(OlogCategoricalOps): # Backwards compatibility
    @staticmethod
    def compute(*args, **kwargs):
        return OlogCategoricalOps.compute_pushout(*args, **kwargs)

def demo_pushout():
    # Domain 1: Healthcare
    healthcare = OlogGraph("Healthcare")
    healthcare.add_type("Patient", "A person receiving care")
    healthcare.add_type("Doctor", "A medical professional")
    healthcare.add_type("Diagnosis", "A medical determination")
    healthcare.add_aspect("Patient", "Doctor", "sees")
    healthcare.add_aspect("Doctor", "Diagnosis", "makes")
    
    # Domain 2: Business
    business = OlogGraph("Business")
    business.add_type("Customer", "A person who pays")
    business.add_type("Order", "A request for goods")
    business.add_type("Invoice", "A request for payment")
    business.add_aspect("Customer", "Order", "places")
    business.add_aspect("Order", "Invoice", "generates")
    
    # The Pushout: Glue 'Patient' to 'Customer'
    # This asserts that Patients are Customers in this context
    mapping = {"Patient": "Customer"}
    
    unified = OlogPushout.compute(healthcare, business, mapping, name="Healthcare_Business_Pushout")
    
    print(f"Pushout Olog: {unified.name}")
    print(f"Types: {list(unified.graph.nodes())}")
    print(f"Aspects: {unified.graph.number_of_edges()}")
    
    # Verify connectivity across domains
    # Can a Patient now reach an Invoice?
    # Path: Patient (is Customer) -> Order -> Invoice
    try:
        # Check reachability in networkx
        import networkx as nx
        has_path = nx.has_path(unified.graph, "Patient", "Invoice")
        print(f"Can Patient reach Invoice? {has_path}")
    except Exception as e:
        print(f"Path error: {e}")

if __name__ == "__main__":
    demo_pushout()
