import logging
from hybrid_encoder import HybridOlogEncoder, OllamaBackend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_shadow_analysis():
    print("=" * 60)
    print("  SHADOW MORPHISM ANALYSIS")
    print("  Mapping the Search Space for Future Refinement")
    print("=" * 60)
    
    # Text snippet from MemGPT abstract
    text = """
    We propose MemGPT, a system that intelligently manages different memory tiers 
    in order to effectively provide extended context within the LLM's limited context window. 
    MemGPT creates an illusion of infinite context.
    """
    
    print(f"\n[INPUT TEXT]\n{text.strip()}\n")
    
    # Use Phi-4 for the LLM part
    backend = OllamaBackend(model_name="zac/phi4-tools:latest")
    encoder = HybridOlogEncoder(llm_backend=backend, use_mock=False)
    
    try:
        olog, metadata = encoder.encode(text, "MemGPT_Shadow_Test")
        
        print("\n[INDUCED OLOG]")
        print(f"  Types: {list(olog.graph.nodes())}")
        print(f"  Aspects (Accepted):")
        for u, v, key in olog.graph.edges(keys=True):
            print(f"    {u} --({key})--> {v}")
            
        print("\n[SHADOW MORPHISMS (The Search Space)]")
        shadows = metadata["stages"].get("shadow_morphisms", [])
        
        if not shadows:
            print("  No shadow morphisms detected (AMR and Olog are perfectly aligned).")
        else:
            print(f"  Detected {len(shadows)} potential relations found in syntax but missed in Olog:")
            for s in shadows:
                print(f"    [?] {s['source']} --(AMR: {s['amr_label']})--> {s['target']}")
                print(f"        Role: {s['amr_role']}")
                
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_shadow_analysis()
