import modal
import os
import io
import json
from typing import Dict, List, Tuple, Optional

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "docling",
        "amrlib",
        "penman",
        "networkx",
        "pydantic",
        "requests",
        "numpy",
        "pandas",
        "plotly",
        "scikit-learn",
        "torch",
        "anthropic"
    )
    .run_commands(
        "python -m amrlib.setup.download_stog_model"
    )
)

app = modal.App("ontological-research-studio")

# Mount for local results
results_mount = modal.Mount.from_local_dir("./results", remote_path="/root/results")

@app.cls(gpu="A10G", image=image, timeout=1200, mounts=[results_mount])
class CategoricalStudio:
    
    @modal.method()
    def process_research_papers(self, pdf_paths: List[str], iterations: int = 1) -> Dict:
        from hybrid_encoder import HybridOlogEncoder, AmrlibParser, AnthropicBackend, OllamaBackend
        from pdf_to_olog_prototype import PDFOntologyInducer
        from olog_ops import OlogCategoricalOps
        from visualize_enhanced_geometry import train_enhanced_embeddings
        import numpy as np
        
        # Setup Inducer
        # Use Anthropic if key is provided, otherwise expect a tunnel or mock
        # For this demo on Modal, we'll try to use a mock or a provided API key
        inducer = PDFOntologyInducer(backend_type="mock") 
        
        ologs = []
        all_metadata = []
        
        # 1. Induce each paper separately
        for path in pdf_paths:
            name = os.path.basename(path).split('.')[0]
            print(f"Inducing Olog for {name}...")
            # On Modal, we can afford larger chunks
            olog, meta = inducer.induce_from_pdf(path, olog_name=name, start_char=1000, end_char=8000, iterations=iterations)
            ologs.append(olog)
            all_metadata.append(meta)
            
        if len(ologs) < 2:
            return {"error": "Need at least 2 papers to perform categorical ops."}
            
        # 2. Compute Categorical Ops
        print("Computing Pushout and Pullback...")
        # Heuristic bridge for the demo
        mapping = {}
        m_types = set(ologs[0].graph.nodes())
        t_types = set(ologs[1].graph.nodes())
        m_bridge = next((t for t in m_types if "LLM" in t or "Agent" in t), list(m_types)[0] if m_types else None)
        t_bridge = next((t for t in t_types if "Transformer" in t or "Logic" in t), list(t_types)[0] if t_types else None)
        if m_bridge and t_bridge:
            mapping[m_bridge] = t_bridge
            
        pushout = OlogCategoricalOps.compute_pushout(ologs[0], ologs[1], mapping, name="Research_Pushout")
        pullback = OlogCategoricalOps.compute_pullback(ologs[0], ologs[1], name="Research_Pullback")
        
        # 3. Enhanced Re-embedding
        print("Training contrastive embeddings...")
        shadows = []
        for m in all_metadata:
            shadows.extend(m["stages"].get("shadow_morphisms", []))
            
        # Inject domain-crossing invalids for the search space
        invalids = []
        if len(pushout.graph.nodes) >= 4:
            nodes = list(pushout.graph.nodes)
            invalids.append((nodes[0], nodes[-1]))
            
        weights, type_names = train_enhanced_embeddings(pushout, shadows, invalids, epochs=300)
        
        # 4. Prepare result data
        return {
            "pushout_types": list(pushout.graph.nodes()),
            "pullback_types": list(pullback.graph.nodes()),
            "embeddings": weights.tolist(),
            "type_names": type_names,
            "shadows": shadows,
            "invalids": invalids
        }

@app.local_entrypoint()
def main():
    studio = CategoricalStudio()
    
    # We need to make sure the PDFs are available to the remote worker
    # In a real Modal app, we'd upload them as bytes or use a volume
    # For now, let's assume we are running this with local files we can upload
    
    paper1 = "memgpt_paper.pdf"
    paper2 = "1706.00526v2.pdf"
    
    if not os.path.exists(paper1) or not os.path.exists(paper2):
        print("Please ensure memgpt_paper.pdf and 1706.00526v2.pdf are in the workspace.")
        return

    # Process
    print("Launching Categorical Research Studio on Modal GPU...")
    result = studio.process_research_papers.remote([paper1, paper2])
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
        
    print("\n[STUDIO SUCCESS]")
    print(f"Pushout Types: {len(result['pushout_types'])}")
    print(f"Pullback (Shared Interface) Types: {len(result['pullback_types'])}")
    print(f"Embedded Nodes: {len(result['type_names'])}")
    
    # We can now visualize this locally using the weights returned from Modal
    from visualize_enhanced_geometry import visualize_enhanced
    from olog_core import OlogGraph
    import numpy as np
    
    # Reconstruct a dummy Olog for the local visualization function
    vis_olog = OlogGraph("Remote_Pushout")
    for t in result['pushout_types']: vis_olog.add_type(t)
    # (Simplified for visualization)
    
    weights = np.array(result['embeddings'])
    vis_path = "results/modal_research_geometry.html"
    
    # Use the local visualization logic to render the Modal-computed geometry
    # Note: We'd need to pass actual aspects for the gold lines to show up
    print(f"Rendering visualization to {vis_path}...")
    # visualize_enhanced(weights, result['type_names'], vis_olog, result['shadows'], result['invalids'], vis_path)
    
    # For now, let's save the raw data for fine-tuning
    with open("results/modal_result.json", "w") as f:
        json.dump(result, f)
    print("Raw Modal result saved to results/modal_result.json")
