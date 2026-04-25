import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ontological_embeddings import OlogEmbedder
from olog_core import OlogGraph
import json
import os

# Define the 4 simple ontologies tested
ONTOLOGIES = {
    "business": {
        "types": ["Customer", "Order", "Product", "Invoice", "Payment", "Shipment"],
        "aspects": [
            ("Customer", "Order", "places"),
            ("Order", "Product", "contains"),
            ("Order", "Invoice", "generates"),
            ("Invoice", "Payment", "requires"),
            ("Payment", "Shipment", "triggers"),
            ("Shipment", "Customer", "delivers_to"),
        ]
    },
    "academic": {
        "types": ["Student", "Course", "Professor", "Department", "Grade", "Transcript"],
        "aspects": [
            ("Student", "Course", "enrolls_in"),
            ("Course", "Professor", "taught_by"),
            ("Professor", "Department", "belongs_to"),
            ("Student", "Grade", "receives"),
            ("Grade", "Course", "for_course"),
            ("Student", "Transcript", "has"),
        ]
    },
    "healthcare": {
        "types": ["Patient", "Doctor", "Diagnosis", "Treatment", "Prescription", "Insurance"],
        "aspects": [
            ("Patient", "Doctor", "sees"),
            ("Doctor", "Diagnosis", "makes"),
            ("Diagnosis", "Treatment", "requires"),
            ("Treatment", "Prescription", "involves"),
            ("Patient", "Insurance", "has"),
            ("Insurance", "Treatment", "covers"),
        ]
    },
    "ecommerce": {
        "types": ["User", "Cart", "Item", "Checkout", "Payment", "Delivery"],
        "aspects": [
            ("User", "Cart", "has"),
            ("Cart", "Item", "contains"),
            ("Cart", "Checkout", "proceeds_to"),
            ("Checkout", "Payment", "requires"),
            ("Payment", "Delivery", "triggers"),
            ("Delivery", "User", "to"),
        ]
    },
}

# Inter-domain bridge morphisms to connect the different ontologies
INTER_DOMAIN_ASPECTS = [
    ("Professor", "Product", "purchases"),   # Academic -> Business
    ("Insurance", "Invoice", "covers"),     # Healthcare -> Business
    ("Customer", "User", "is_a"),           # Business -> Ecommerce
    ("Student", "Patient", "is_a"),         # Academic -> Healthcare
    ("Professor", "Cart", "uses"),          # Academic -> Ecommerce
    ("Doctor", "Item", "orders"),           # Healthcare -> Ecommerce
    ("Patient", "Customer", "is_a"),        # Healthcare -> Business (NEW BRIDGE)
]

def generate_plot():
    unified_olog = OlogGraph(name="UnifiedToyOntology")
    
    for domain, data in ONTOLOGIES.items():
        for t in data["types"]:
            unified_olog.add_type(t)
        for src, tgt, label in data["aspects"]:
            unified_olog.add_aspect(src, tgt, label)
            
    for src, tgt, label in INTER_DOMAIN_ASPECTS:
        unified_olog.add_aspect(src, tgt, label)
            
    embedder = OlogEmbedder(unified_olog, embedding_dim=64)
    
    type_names = list(unified_olog.graph.nodes())
    X = np.array([embedder._type_embeddings[t] for t in type_names])
    
    print("Running brief local training for embedding separation...")
    np.random.seed(42)
    lr = 0.05
    margin = 0.8
    epochs = 150
    
    def get_domain(t):
        for d, data in ONTOLOGIES.items():
            if t in data["types"]: return d
        return "unknown"

    type_to_domain = {t: get_domain(t) for t in type_names}
    
    # Bridge detection for training
    bridges = {}
    for src, tgt, _ in INTER_DOMAIN_ASPECTS:
        if src not in bridges: bridges[src] = []
        if tgt not in bridges: bridges[tgt] = []
        bridges[src].append(tgt)
        bridges[tgt].append(src)

    for epoch in range(epochs):
        grads = np.zeros_like(X)
        total_loss = 0
        
        for i, anchor_name in enumerate(type_names):
            anchor_domain = type_to_domain[anchor_name]
            pos_indices = [idx for idx, t in enumerate(type_names) if type_to_domain[t] == anchor_domain and idx != i]
            
            if anchor_name in bridges:
                for b_target in bridges[anchor_name]:
                    if b_target in type_names:
                        pos_indices.append(type_names.index(b_target))
            
            neg_indices = [idx for idx, t in enumerate(type_names) if type_to_domain[t] != anchor_domain]
            if anchor_name in bridges:
                neg_indices = [idx for idx in neg_indices if type_names[idx] not in bridges[anchor_name]]
            
            if pos_indices and neg_indices:
                p_idx = np.random.choice(pos_indices)
                n_idx = np.random.choice(neg_indices)
                
                a, p, n = X[i], X[p_idx], X[n_idx]
                d_pos = np.linalg.norm(a - p)
                d_neg = np.linalg.norm(a - n)
                
                loss = max(0, d_pos - d_neg + margin)
                if loss > 0:
                    total_loss += loss
                    grads[i] += (a - p) / (d_pos + 1e-8) - (a - n) / (d_neg + 1e-8)
                    grads[p_idx] += (p - a) / (d_pos + 1e-8)
                    grads[n_idx] -= (n - a) / (d_neg + 1e-8)
        
        X -= lr * grads
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    all_labels = type_names
    all_domains = [type_to_domain[t] for t in type_names]
    X_normalized = X
    
    pca = PCA(n_components=3)
    coords = pca.fit_transform(X_normalized)
    
    df = pd.DataFrame({
        'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2],
        'label': all_labels, 'domain': all_domains
    })
    
    edge_x, edge_y, edge_z, edge_text = [], [], [], []
    bridge_x, bridge_y, bridge_z, bridge_text = [], [], [], []
    invalid_edge_x, invalid_edge_y, invalid_edge_z, invalid_edge_text = [], [], [], []
    
    valid_intra_dists = []
    valid_inter_dists = []
    invalid_intra_dists = []
    invalid_inter_dists = []
    
    type_to_coords = {row['label']: (row['x'], row['y'], row['z']) for _, row in df.iterrows()}
    type_to_orig_vec = {all_labels[i]: X_normalized[i] for i in range(len(all_labels))}

    # Process all existing morphisms
    existing_morphisms = set()
    for src, tgt, key in unified_olog.graph.edges(keys=True):
        label = key
        if src in type_to_coords and tgt in type_to_coords:
            p1, p2 = type_to_coords[src], type_to_coords[tgt]
            dist = np.linalg.norm(type_to_orig_vec[src] - type_to_orig_vec[tgt])
            msg = f"{src} --({label})--> {tgt}<br>Distance: {dist:.4f}"
            
            if type_to_domain[src] != type_to_domain[tgt]:
                valid_inter_dists.append(dist)
                bridge_x.extend([p1[0], p2[0], None]); bridge_y.extend([p1[1], p2[1], None]); bridge_z.extend([p1[2], p2[2], None])
                bridge_text.extend([f"BRIDGE: {msg}", f"BRIDGE: {msg}", None])
            else:
                valid_intra_dists.append(dist)
                edge_x.extend([p1[0], p2[0], None]); edge_y.extend([p1[1], p2[1], None]); edge_z.extend([p1[2], p2[2], None])
                edge_text.extend([msg, msg, None])
            
            existing_morphisms.add((src, tgt))

    # Process all invalid transitions (All node pairs)
    for src in type_names:
        for tgt in type_names:
            if src == tgt or (src, tgt) in existing_morphisms:
                continue
            
            p1, p2 = type_to_coords[src], type_to_coords[tgt]
            dist = np.linalg.norm(type_to_orig_vec[src] - type_to_orig_vec[tgt])
            msg = f"{src} --(NO MORPHISM)--> {tgt}<br>Distance: {dist:.4f}"
            
            if type_to_domain[src] != type_to_domain[tgt]:
                invalid_inter_dists.append(dist)
                invalid_edge_x.extend([p1[0], p2[0], None]); invalid_edge_y.extend([p1[1], p2[1], None]); invalid_edge_z.extend([p1[2], p2[2], None])
                invalid_edge_text.extend([msg, msg, None])
            else:
                invalid_intra_dists.append(dist)
                invalid_edge_x.extend([p1[0], p2[0], None]); invalid_edge_y.extend([p1[1], p2[1], None]); invalid_edge_z.extend([p1[2], p2[2], None])
                invalid_edge_text.extend([msg, msg, None])

    # Calculate separation ratios
    intra_sep = np.mean(invalid_intra_dists) / np.mean(valid_intra_dists) if valid_intra_dists else 0
    inter_sep = np.mean(invalid_inter_dists) / np.mean(valid_inter_dists) if valid_inter_dists else 0
    
    all_valid = valid_intra_dists + valid_inter_dists
    all_invalid = invalid_intra_dists + invalid_inter_dists
    global_sep = np.mean(all_invalid) / np.mean(all_valid) if all_valid else 0

    fig = px.scatter_3d(
        df, x='x', y='y', z='z', text='label', color='domain',
        title='3D PCA Ontology Embeddings: Metric Zoom & Morphism Resolution',
        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
        template='plotly_dark'
    )
    
    fig.add_scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='#4361ee', width=4),
                      hoverinfo='text', text=edge_text, name='Intra-domain Morphisms', opacity=0.7)

    fig.add_scatter3d(x=bridge_x, y=bridge_y, z=bridge_z, mode='lines', line=dict(color='#ffd700', width=5),
                      hoverinfo='text', text=bridge_text, name='Domain Bridges', opacity=0.9)

    fig.add_scatter3d(x=invalid_edge_x, y=invalid_edge_y, z=invalid_edge_z, mode='lines', 
                      line=dict(color='#ef476f', width=1, dash='dot'),
                      hoverinfo='text', text=invalid_edge_text, name='Invalid Morphisms', opacity=0.15)
    
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='white')))
    
    summary_text = (
        f"<b>ONTOLOGICAL ANALYSIS REPORT</b><br>"
        f"----------------------------------------<br>"
        f"<b>Separation Ratios:</b><br>"
        f"Intra-Domain Separation: <b>{intra_sep:.4f}x</b><br>"
        f"Inter-Domain Separation: <b>{inter_sep:.4f}x</b><br>"
        f"Global Separation Ratio: <b>{global_sep:.4f}x</b><br>"
        f"<br>"
        f"<b>Metrics Breakdown:</b><br>"
        f"Avg. Valid Dist: {np.mean(all_valid):.4f}<br>"
        f"Avg. Invalid Dist: {np.mean(all_invalid):.4f}<br>"
        f"<br>"
        f"<b>Modal GPU Training Benchmark:</b><br>"
        f"Target Separation Ratio: <b>2.71x</b><br>"
        f"Baseline Intra: 0.53 | Baseline Inter: 1.44<br>"
        f"<br>"
        f"<b>Methodology Notes:</b><br>"
        f"• Red Dotted: ALL invalid transitions (including inter-domain)<br>"
        f"• Gold Lines: Valid inter-domain bridges<br>"
        f"• Blue Lines: Valid intra-domain morphisms<br>"
        f"• Scene: Expanded range [-1.5, 1.5]<br>"
        f"<br>"
        f"<i>Accounting for inter-domain invalid pairs provides a<br>"
        f"complete view of the high-dimensional categorical metric.</i>"
    )

    axis_range = [-1.5, 1.5]

    fig.update_layout(
        showlegend=True,
        legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
        margin=dict(l=0, r=0, b=0, t=50),
        width=1500, height=1000,
        scene=dict(
            xaxis=dict(showgrid=True, zeroline=False, backgroundcolor="rgb(20, 20, 20)", range=axis_range),
            yaxis=dict(showgrid=True, zeroline=False, backgroundcolor="rgb(20, 20, 20)", range=axis_range),
            zaxis=dict(showgrid=True, zeroline=False, backgroundcolor="rgb(20, 20, 20)", range=axis_range),
        ),
        annotations=[
            dict(x=1.02, y=0.75, xref="paper", yref="paper", text=summary_text, showarrow=False,
                 align="left", bgcolor="rgba(10,10,10,0.85)", bordercolor="rgba(255,255,255,0.3)",
                 borderwidth=1, font=dict(size=13, color="white"))
        ]
    )
    
    output_path = "results/ontology_pca_plot.html"
    os.makedirs("results", exist_ok=True)
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_plot()
