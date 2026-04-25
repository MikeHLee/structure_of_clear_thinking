import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Set, Optional
from olog_core import OlogGraph
from olog_ops import OlogPushout
from hybrid_encoder import HybridOlogEncoder, OllamaBackend
from pdf_to_olog_prototype import PDFOntologyInducer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContrastiveOlogEmbedder(nn.Module):
    def __init__(self, n_types: int, embed_dim: int = 32):
        super().__init__()
        self.embeddings = nn.Embedding(n_types, embed_dim)
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.1)
        
    def forward(self, indices):
        return self.embeddings(indices)

def train_enhanced_embeddings(
    olog: OlogGraph, 
    shadows: List[Dict], 
    invalids: List[Tuple[str, str]],
    epochs: int = 250
):
    type_names = sorted(list(set(olog.graph.nodes())))
    type_to_idx = {t: i for i, t in enumerate(type_names)}
    n_types = len(type_names)
    
    if n_types == 0:
        return np.array([]), []
        
    model = ContrastiveOlogEmbedder(n_types)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    valid_pairs = [(type_to_idx[u], type_to_idx[v]) for u, v in olog.graph.edges() if u in type_to_idx and v in type_to_idx]
    shadow_pairs = [(type_to_idx[s['source']], type_to_idx[s['target']]) for s in shadows if s['source'] in type_to_idx and s['target'] in type_to_idx]
    invalid_pairs = [(type_to_idx[u], type_to_idx[v]) for u, v in invalids if u in type_to_idx and v in type_to_idx]

    print(f"  Refining geometry: {len(valid_pairs)} valid, {len(shadow_pairs)} shadow, {len(invalid_pairs)} invalid pairs.")

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        
        # 1. Pull Valid/Shadow (Structure)
        if valid_pairs:
            src_v = torch.tensor([p[0] for p in valid_pairs]); tgt_v = torch.tensor([p[1] for p in valid_pairs])
            loss += torch.mean(torch.norm(model(src_v) - model(tgt_v), dim=1))
        
        if shadow_pairs:
            src_s = torch.tensor([p[0] for p in shadow_pairs]); tgt_s = torch.tensor([p[1] for p in shadow_pairs])
            loss += 0.5 * torch.mean(torch.norm(model(src_s) - model(tgt_s), dim=1))
            
        # 2. Push Invalids (Categorical Safety)
        if invalid_pairs:
            src_i = torch.tensor([p[0] for p in invalid_pairs]); tgt_i = torch.tensor([p[1] for p in invalid_pairs])
            dist = torch.norm(model(src_i) - model(tgt_i), dim=1)
            loss += torch.mean(torch.relu(5.0 - dist)) # High margin for clear visibility

        if loss > 0:
            loss.backward(); optimizer.step()
        
    return model.embeddings.weight.detach().numpy(), type_names

def visualize_enhanced(weights, type_names, olog, shadows, invalids, output_path: str):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(weights)
    
    node_descriptions = []
    for t in type_names:
        data = olog.graph.nodes[t].get('data')
        node_descriptions.append(data.description if data and data.description else "No description")

    df = pd.DataFrame({
        'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2],
        'Type': type_names,
        'Description': node_descriptions
    })
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', text='Type', 
                        hover_data=['Description'],
                        title="Comprehensive Research Geometry (MemGPT + Categorical Logic)",
                        template='plotly_dark')
    
    def add_edges(pairs, color, name, width, dash=None, labels=None):
        e_x, e_y, e_z, e_text = [], [], [], []
        for i, (u, v) in enumerate(pairs):
            if u not in type_names or v not in type_names: continue
            idx_u, idx_v = type_names.index(u), type_names.index(v)
            p1, p2 = coords[idx_u], coords[idx_v]
            e_x.extend([p1[0], p2[0], None]); e_y.extend([p1[1], p2[1], None]); e_z.extend([p1[2], p2[2], None])
            label = labels[i] if labels else name
            e_text.extend([label, label, None])
        
        fig.add_scatter3d(x=e_x, y=e_y, z=e_z, mode='lines', 
                          line=dict(color=color, width=width, dash=dash), 
                          text=e_text, hoverinfo='text', name=name)

    # 1. Valid Gold Morphisms
    v_pairs = [(u, v) for u, v, k in olog.graph.edges(keys=True)]
    v_labels = [f"Morphism: {k}" for u, v, k in olog.graph.edges(keys=True)]
    add_edges(v_pairs, 'gold', 'Valid Morphisms', 6, labels=v_labels)

    # 2. Shadow Cyan Morphisms
    s_pairs = [(s['source'], s['target']) for s in shadows]
    s_labels = [f"Shadow (AMR: {s['amr_label']})" for s in shadows]
    add_edges(s_pairs, 'cyan', 'Shadow (Potential)', 2, dash='dash', labels=s_labels)

    # 3. Invalid Red Morphisms
    add_edges(invalids, 'red', 'Invalid Hallucination', 4)

    fig.update_layout(scene=dict(xaxis_title='Latent Dim 1', yaxis_title='Latent Dim 2', zaxis_title='Latent Dim 3'))
    fig.write_html(output_path)
    print(f"Final visualization saved to {output_path}")

def main():
    print("=" * 60)
    print("  RESEARCH GEOMETRY: MemGPT + Bicategories (Pushout + Contrastive)")
    print("=" * 60)
    
    inducer = PDFOntologyInducer(backend_type="ollama", model_name="zac/phi4-tools:latest")
    
    # 1. Induce Small Graphs (Single pass for maximum stability)
    print("\n[STEP 1] Inducing Ologs from Research Papers...")
    olog_memgpt, meta_m = inducer.induce_from_pdf("memgpt_paper.pdf", olog_name="MemGPT", start_char=1000, end_char=3000, iterations=1)
    olog_trans, meta_t = inducer.induce_from_pdf("1706.00526v2.pdf", olog_name="Transformer", start_char=2000, end_char=4000, iterations=1)
    
    # 2. Formal Pushout
    print("\n[STEP 2] Computing Categorical Pushout...")
    # Intersection check
    m_types = set(olog_memgpt.graph.nodes())
    t_types = set(olog_trans.graph.nodes())
    print(f"  MemGPT Types: {len(m_types)} | Transformer Types: {len(t_types)}")
    
    mapping = {}
    # Heuristic bridge: LLM <-> Transformer
    m_bridge = next((t for t in m_types if "LLM" in t), None)
    t_bridge = next((t for t in t_types if "Transformer" in t), None)
    if m_bridge and t_bridge:
        mapping[m_bridge] = t_bridge
        print(f"  Bridge identified: {m_bridge} <-> {t_bridge}")

    merged_olog = OlogPushout.compute(olog_memgpt, olog_trans, mapping, name="Merged_AI_Logic")
    
    # 3. Collect Shadows
    shadows = meta_m["stages"].get("shadow_morphisms", []) + meta_t["stages"].get("shadow_morphisms", [])
    
    # 4. Inject Invalids
    invalids = []
    nodes = list(merged_olog.graph.nodes())
    if len(nodes) >= 2:
        # Cross-domain hallucination: LLM --(is_a)--> Description Logic
        dl_node = next((n for n in nodes if "Logic" in n), nodes[0])
        llm_node = next((n for n in nodes if "LLM" in n), nodes[1])
        invalids.append((llm_node, dl_node))
        # Inverse: Memory --(manages)--> MemGPT
        mem_node = next((n for n in nodes if "Memory" in n), nodes[0])
        mg_node = next((n for n in nodes if "MemGPT" in n), nodes[1])
        invalids.append((mem_node, mg_node))

    # 5. Training
    print("\n[STEP 3] Re-embedding into 32D Categorical Space...")
    weights, type_names = train_enhanced_embeddings(merged_olog, shadows, invalids)
    
    # 6. Visualize
    vis_path = "results/enhanced_olog_geometry.html"
    visualize_enhanced(weights, type_names, merged_olog, shadows, invalids, vis_path)
    
    import subprocess
    try: subprocess.run(["open", vis_path])
    except: pass

if __name__ == "__main__":
    main()
