import os
import logging
import json
from typing import Dict, List, Tuple, Optional
from olog_core import OlogGraph
from olog_ops import OlogPushout
from pdf_to_olog_prototype import PDFOntologyInducer
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from ontological_embeddings import OlogEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_pushout_visualization(olog: OlogGraph, output_path: str):
    """Generates a 3D visualization for the merged Olog."""
    print(f"\n[STEP 3] Generating 3D Visualization: {output_path}")
    
    embedder = OlogEmbedder(olog, embedding_dim=32)
    type_names = list(olog.graph.nodes())
    
    if not type_names:
        print("No types found to visualize.")
        return

    X = np.array([embedder._type_embeddings[t] for t in type_names])
    
    # Normalize
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    pca = PCA(n_components=3)
    coords = pca.fit_transform(X)
    
    df = pd.DataFrame({
        'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2],
        'label': type_names,
        'domain': ["Merged" for _ in type_names]
    })
    
    fig = px.scatter_3d(
        df, x='x', y='y', z='z', text='label', color='domain',
        title=f'3D Categorical Pushout: {olog.name}',
        template='plotly_dark'
    )
    
    # Add Edges
    edge_x, edge_y, edge_z, edge_text = [], [], [], []
    for u, v, key in olog.graph.edges(keys=True):
        idx_u = type_names.index(u)
        idx_v = type_names.index(v)
        p1, p2 = coords[idx_u], coords[idx_v]
        edge_x.extend([p1[0], p2[0], None])
        edge_y.extend([p1[1], p2[1], None])
        edge_z.extend([p1[2], p2[2], None])
        edge_text.extend([f"{u} --({key})--> {v}", f"{u} --({key})--> {v}", None])

    fig.add_scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', 
                      line=dict(color='gold', width=4),
                      hoverinfo='text', text=edge_text, name='Morphisms')
    
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")

def main():
    print("=" * 60)
    print("  RESEARCH DOMAIN PUSHOUT: MemGPT + Transformers (Phi-4 Extraction)")
    print("=" * 60)
    
    # Use Phi-4 for robust triplet extraction
    inducer = PDFOntologyInducer(backend_type="ollama", model_name="zac/phi4-tools:latest")
    
    # [STEP 1] Extract MemGPT Olog (Balanced for completeness/timeout)
    print("\n[STEP 1.1] Inducing Olog from MemGPT Paper...")
    olog_memgpt, _ = inducer.induce_from_pdf("memgpt_paper.pdf", olog_name="MemGPT", start_char=1000, end_char=7000)
    
    # [STEP 2] Extract Transformer Olog (Balanced for completeness/timeout)
    transformer_pdf = "1706.00526v2.pdf"
    print(f"\n[STEP 1.2] Inducing Olog from Transformer Paper ({transformer_pdf})...")
    # Taking a middle chunk for better concept coverage
    olog_transformer, _ = inducer.induce_from_pdf(transformer_pdf, olog_name="Transformer", start_char=2000, end_char=8000)
    
    # [STEP 3] Compute Pushout
    print("\n[STEP 2] Computing Categorical Pushout...")
    
    memgpt_types = set(olog_memgpt.graph.nodes())
    trans_types = set(olog_transformer.graph.nodes())
    
    print(f"  MemGPT Types: {memgpt_types}")
    print(f"  Transformer Types: {trans_types}")
    
    mapping = {}
    
    # Look for common entities for the bridge
    # Intersection logic
    common = memgpt_types.intersection(trans_types)
    if common:
        for c in common:
            mapping[c] = c
            print(f"  Mapping bridge (direct match): {c} <-> {c}")
    
    # Heuristic fallback if no direct matches
    if not mapping:
        # Check for semantically close terms (manual prototype logic)
        # LLM in MemGPT is often synonymous with Transformer architecture
        m_terms = {"LLM", "Large Language Model", "Model"}
        t_terms = {"Transformer", "Model", "Architecture"}
        
        m_hit = next((t for t in memgpt_types if any(term in t for term in m_terms)), None)
        t_hit = next((t for t in trans_types if any(term in t for term in t_terms)), None)
        
        if m_hit and t_hit:
            mapping[m_hit] = t_hit
            print(f"  Mapping bridge (heuristic): {m_hit} <-> {t_hit}")

    merged_olog = OlogPushout.compute(olog_memgpt, olog_transformer, mapping, name="Merged_AI_Research")
    
    print(f"\n  Pushout Result: {merged_olog.name}")
    print(f"  Total Types: {merged_olog.graph.number_of_nodes()}")
    print(f"  Total Morphisms: {merged_olog.graph.number_of_edges()}")
    
    # [STEP 4] New Visualization
    vis_path = "results/merged_research_pushout.html"
    generate_pushout_visualization(merged_olog, vis_path)
    
    # Open the new visualization
    import subprocess
    try:
        subprocess.run(["open", vis_path])
    except:
        pass

if __name__ == "__main__":
    main()
