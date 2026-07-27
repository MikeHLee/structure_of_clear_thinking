import os
import logging
import json
from typing import List, Dict, Optional, Any, Tuple
from docling.document_converter import DocumentConverter
from olog_core import OlogGraph
from hybrid_encoder import HybridOlogEncoder, MockAMRParser, MockLLMBackend, OllamaBackend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DoclingParser:
    """Uses Docling to extract high-fidelity text from PDFs."""
    
    def __init__(self):
        self.converter = DocumentConverter()
    
    def extract_markdown(self, pdf_path: str) -> str:
        """Convert PDF to Markdown string."""
        logger.info(f"Converting {pdf_path} to Markdown via Docling...")
        conv_result = self.converter.convert(pdf_path)
        return conv_result.document.export_to_markdown()

class PDFOntologyInducer:
    """
    Prototype Pipeline:
    PDF -> Docling (MD) -> Semantic Chunking -> Hybrid Encoder -> OlogGraph
    """
    
    def __init__(self, backend_type: str = "mock", model_name: Optional[str] = None):
        self.docling = DoclingParser()
        
        if backend_type == "ollama":
            backend = OllamaBackend(model_name=model_name)
        elif backend_type == "triplex":
            from hybrid_encoder import TriplexBackend
            backend = TriplexBackend(model_name=model_name or "sciphi/triplex:latest")
        elif backend_type == "anthropic":
            from hybrid_encoder import AnthropicBackend
            backend = AnthropicBackend()
        else:
            backend = MockLLMBackend()
            
        self.encoder = HybridOlogEncoder(llm_backend=backend, use_mock=(backend_type == "mock"))
        
    def induce_from_pdf(
        self, 
        pdf_path: str, 
        olog_name: str = "PDF_Induced_Olog",
        start_char: int = 1000,
        end_char: int = 4000,
        iterations: int = 2
    ) -> Tuple[OlogGraph, Dict]:
        # 1. Extract text structure
        markdown = self.docling.extract_markdown(pdf_path)
        
        # 2. Slice based on parameters
        sample_text = markdown[start_char:end_char]
        
        logger.info(f"Inducing Olog from extracted text (sample size: {len(sample_text)} chars)")
        
        # 3. Encode to Olog
        olog, metadata = self.encoder.encode(sample_text, olog_name, iterations=iterations)
        
        # Add the raw markdown to metadata
        metadata["raw_markdown_length"] = len(markdown)
        metadata["pdf_source"] = pdf_path
        
        return olog, metadata

def run_prototype(backend_type="mock", model_name=None):
    print("=" * 60)
    print(f"  PDF -> OLOG PROTOTYPE (Docling + Hybrid Encoder | Backend: {backend_type})")
    print("=" * 60)
    
    # Use one of the research papers in the directory
    pdf_path = "memgpt_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return
        
    inducer = PDFOntologyInducer(backend_type=backend_type, model_name=model_name)
    
    try:
        olog, metadata = inducer.induce_from_pdf(pdf_path, olog_name="MemGPT_Ontology")
        
        print("\n[PROTOTYPE SUCCESS]")
        print(f"  Source: {metadata['pdf_source']}")
        print(f"  Markdown Extracted: {metadata['raw_markdown_length']} chars")
        print(f"  Induced Types: {olog.graph.number_of_nodes()}")
        print(f"  Induced Aspects: {olog.graph.number_of_edges()}")
        
        print("\n[INDUCED TYPES]")
        for node in olog.graph.nodes():
            print(f"  • {node}")
            
        print("\n[INDUCED MORPHISMS]")
        for u, v, key in olog.graph.edges(keys=True):
            print(f"  • {u} --({key})--> {v}")
            
        print("\n[TOPOLOGICAL HEALTH]")
        health = metadata["health_report"]
        print(f"  Score: {health['semantic_consistency_score']:.2f}")
        print(f"  Status: {health['status']}")
        
    except Exception as e:
        logger.error(f"Prototype failed: {e}", exc_info=True)

import sys
if __name__ == "__main__":
    backend = "mock"
    model = None
    if len(sys.argv) > 1:
        backend = sys.argv[1]
    if len(sys.argv) > 2:
        model = sys.argv[2]
    run_prototype(backend, model)
