"""
OWL/RDF Export Module for Olog Graphs

Serializes OlogGraph structures to standard semantic web formats:
- Turtle (.ttl)
- RDF/XML (.rdf)
- JSON-LD (.jsonld)
- N-Triples (.nt)

The export maps Olog concepts to OWL:
- Types → owl:Class
- Aspects → owl:ObjectProperty (functional)
- Instances → owl:NamedIndividual
- CommutativeFacts → owl:equivalentProperty chains (approximation)
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default namespace for generated ontologies
DEFAULT_BASE = "http://example.org/olog/"


class OlogRDFExporter:
    """
    Exports OlogGraph to RDF/OWL formats.
    
    Mapping:
    - OlogNode (Type) → owl:Class
    - OlogMorphism (Aspect) → owl:ObjectProperty with domain/range
    - CommutativeFact → rdfs:comment annotation (OWL2 property chains for full support)
    """
    
    def __init__(self, base_uri: str = DEFAULT_BASE):
        self.base_uri = base_uri.rstrip("/") + "/"
        self.ns = Namespace(self.base_uri)
        
    def export(
        self, 
        olog_graph, 
        output_path: Optional[str] = None,
        format: str = "turtle"
    ) -> str:
        """
        Export OlogGraph to RDF.
        
        Args:
            olog_graph: The OlogGraph to export
            output_path: Optional file path to write to
            format: Output format - "turtle", "xml", "json-ld", "nt"
        
        Returns:
            Serialized RDF string
        """
        g = Graph()
        
        # Bind namespaces
        g.bind("olog", self.ns)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        
        # Create ontology header
        ontology_uri = URIRef(self.base_uri[:-1])
        g.add((ontology_uri, RDF.type, OWL.Ontology))
        g.add((ontology_uri, RDFS.label, Literal(olog_graph.name)))
        g.add((ontology_uri, RDFS.comment, Literal(
            f"Ontology generated from Olog: {olog_graph.name}"
        )))
        
        # Export types as OWL classes
        for node_name in olog_graph.graph.nodes():
            node_data = olog_graph.graph.nodes[node_name].get("data")
            self._add_class(g, node_name, node_data)
        
        # Export aspects as OWL object properties
        for source, target, key in olog_graph.graph.edges(keys=True):
            edge_data = olog_graph.graph.edges[source, target, key].get("data")
            self._add_property(g, key, source, target, edge_data)
        
        # Export facts as annotations
        for i, fact in enumerate(olog_graph.facts):
            self._add_fact_annotation(g, i, fact)
        
        # Serialize
        rdf_str = g.serialize(format=format)
        
        if output_path:
            Path(output_path).write_text(rdf_str)
            logger.info(f"Exported to {output_path}")
        
        return rdf_str
    
    def _add_class(self, g: Graph, name: str, node_data):
        """Add an OWL class for an Olog type."""
        class_uri = self.ns[self._sanitize_uri(name)]
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(name)))
        
        if node_data and hasattr(node_data, 'description') and node_data.description:
            g.add((class_uri, RDFS.comment, Literal(node_data.description)))
    
    def _add_property(self, g: Graph, label: str, domain: str, range_: str, edge_data):
        """Add an OWL object property for an Olog aspect."""
        prop_uri = self.ns[self._sanitize_uri(label)]
        domain_uri = self.ns[self._sanitize_uri(domain)]
        range_uri = self.ns[self._sanitize_uri(range_)]
        
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDF.type, OWL.FunctionalProperty))  # Aspects are functional
        g.add((prop_uri, RDFS.label, Literal(label)))
        g.add((prop_uri, RDFS.domain, domain_uri))
        g.add((prop_uri, RDFS.range, range_uri))
        
        if edge_data and hasattr(edge_data, 'description') and edge_data.description:
            g.add((prop_uri, RDFS.comment, Literal(edge_data.description)))
    
    def _add_fact_annotation(self, g: Graph, index: int, fact):
        """
        Add a commutative fact as an annotation.
        
        Full OWL2 support for property chains would use owl:propertyChainAxiom,
        but this requires OWL2 reasoning. For simplicity, we add as annotation.
        """
        fact_uri = self.ns[f"CommutativeFact_{index}"]
        
        g.add((fact_uri, RDF.type, self.ns["CommutativeFact"]))
        g.add((fact_uri, self.ns["sourceNode"], Literal(fact.source_node)))
        g.add((fact_uri, self.ns["targetNode"], Literal(fact.target_node)))
        g.add((fact_uri, self.ns["pathA"], Literal(" → ".join(fact.path_a_labels))))
        g.add((fact_uri, self.ns["pathB"], Literal(" → ".join(fact.path_b_labels))))
        g.add((fact_uri, RDFS.comment, Literal(
            f"Asserts path equivalence: {fact.source_node} via [{', '.join(fact.path_a_labels)}] "
            f"equals [{', '.join(fact.path_b_labels)}] to {fact.target_node}"
        )))
    
    def _sanitize_uri(self, name: str) -> str:
        """Sanitize a name for use in URI."""
        # Replace spaces and special chars
        return name.replace(" ", "_").replace("-", "_").replace(".", "_")


class OlogRDFImporter:
    """
    Imports OlogGraph from RDF/OWL formats.
    """
    
    def __init__(self, base_uri: str = DEFAULT_BASE):
        self.base_uri = base_uri.rstrip("/") + "/"
        self.ns = Namespace(self.base_uri)
    
    def import_from_file(self, file_path: str, format: str = None):
        """
        Import an OlogGraph from an RDF file.
        
        Args:
            file_path: Path to RDF file
            format: Format hint (auto-detected if None)
        
        Returns:
            OlogGraph
        """
        from olog_core import OlogGraph, CommutativeFact
        
        g = Graph()
        g.parse(file_path, format=format)
        
        # Extract ontology name
        ontology_name = "ImportedOlog"
        for s, p, o in g.triples((None, RDF.type, OWL.Ontology)):
            for _, _, label in g.triples((s, RDFS.label, None)):
                ontology_name = str(label)
                break
        
        olog = OlogGraph(ontology_name)
        
        # Import classes as types
        for s, p, o in g.triples((None, RDF.type, OWL.Class)):
            if isinstance(s, URIRef):
                name = self._uri_to_name(str(s))
                description = ""
                for _, _, comment in g.triples((s, RDFS.comment, None)):
                    description = str(comment)
                    break
                olog.add_type(name, description)
        
        # Import object properties as aspects
        for s, p, o in g.triples((None, RDF.type, OWL.ObjectProperty)):
            if isinstance(s, URIRef):
                label = self._uri_to_name(str(s))
                
                domain = None
                range_ = None
                description = ""
                
                for _, _, d in g.triples((s, RDFS.domain, None)):
                    domain = self._uri_to_name(str(d))
                for _, _, r in g.triples((s, RDFS.range, None)):
                    range_ = self._uri_to_name(str(r))
                for _, _, c in g.triples((s, RDFS.comment, None)):
                    description = str(c)
                
                if domain and range_:
                    try:
                        olog.add_aspect(domain, range_, label, description)
                    except ValueError as e:
                        logger.warning(f"Skipping aspect {label}: {e}")
        
        return olog
    
    def _uri_to_name(self, uri: str) -> str:
        """Extract name from URI."""
        # Get the fragment or last path component
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.rstrip("/").split("/")[-1]


def add_export_methods_to_olog():
    """
    Monkey-patch export methods onto OlogGraph class.
    Call this at module import to add .to_turtle(), .to_rdf(), etc.
    """
    from olog_core import OlogGraph
    
    def to_turtle(self, output_path: Optional[str] = None, base_uri: str = DEFAULT_BASE) -> str:
        """Export to Turtle format."""
        exporter = OlogRDFExporter(base_uri)
        return exporter.export(self, output_path, format="turtle")
    
    def to_rdf_xml(self, output_path: Optional[str] = None, base_uri: str = DEFAULT_BASE) -> str:
        """Export to RDF/XML format."""
        exporter = OlogRDFExporter(base_uri)
        return exporter.export(self, output_path, format="xml")
    
    def to_jsonld(self, output_path: Optional[str] = None, base_uri: str = DEFAULT_BASE) -> str:
        """Export to JSON-LD format."""
        exporter = OlogRDFExporter(base_uri)
        return exporter.export(self, output_path, format="json-ld")
    
    OlogGraph.to_turtle = to_turtle
    OlogGraph.to_rdf_xml = to_rdf_xml
    OlogGraph.to_jsonld = to_jsonld


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate RDF export."""
    print("=" * 60)
    print("  OWL/RDF EXPORT DEMO")
    print("=" * 60)
    
    from olog_core import OlogGraph, CommutativeFact
    
    # Create sample Olog
    olog = OlogGraph("ECommerceOntology")
    
    # Add types with descriptions
    olog.add_type("Customer", "A person who purchases products")
    olog.add_type("Order", "A request to purchase one or more products")
    olog.add_type("Invoice", "A document requesting payment for an order")
    olog.add_type("Inventory", "Stock of available products")
    olog.add_type("Product", "An item available for purchase")
    
    # Add aspects
    olog.add_aspect("Customer", "Order", "places", "A customer places an order")
    olog.add_aspect("Order", "Invoice", "generates", "An order generates an invoice")
    olog.add_aspect("Order", "Inventory", "reduces", "An order reduces inventory")
    olog.add_aspect("Order", "Product", "contains", "An order contains products")
    olog.add_aspect("Invoice", "Customer", "billed_to", "An invoice is billed to a customer")
    
    # Add a commutative fact
    olog.add_fact(CommutativeFact(
        source_node="Customer",
        target_node="Invoice",
        path_a_labels=["places", "generates"],
        path_b_labels=["places", "generates"]  # Trivial fact for demo
    ))
    
    # Export to Turtle
    exporter = OlogRDFExporter("http://example.org/ecommerce/")
    turtle_str = exporter.export(olog)
    
    print("\n[TURTLE OUTPUT]")
    print("-" * 40)
    print(turtle_str)
    
    # Save to file
    output_path = "/tmp/ecommerce_olog.ttl"
    exporter.export(olog, output_path)
    print(f"\n[SAVED TO] {output_path}")
    
    # Show JSON-LD
    print("\n[JSON-LD OUTPUT]")
    print("-" * 40)
    jsonld_str = exporter.export(olog, format="json-ld")
    print(jsonld_str[:500] + "..." if len(jsonld_str) > 500 else jsonld_str)
    
    # Test import
    print("\n[ROUND-TRIP TEST]")
    print("-" * 40)
    importer = OlogRDFImporter("http://example.org/ecommerce/")
    imported_olog = importer.import_from_file(output_path)
    print(f"Imported Olog: {imported_olog.name}")
    print(f"  Types: {list(imported_olog.graph.nodes())}")
    print(f"  Aspects: {imported_olog.graph.number_of_edges()}")


if __name__ == "__main__":
    demo()
