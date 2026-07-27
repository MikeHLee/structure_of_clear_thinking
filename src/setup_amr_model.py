"""
AMR Model Setup Script

Downloads and installs the AMR parsing model for amrlib.
Run this once before using live AMR parsing.

Models available:
- parse_xfm_bart_large: Best accuracy, ~2GB (recommended)
- parse_xfm_bart_base: Smaller, ~500MB
- parse_spring: Alternative parser

Usage:
    python setup_amr_model.py [--model MODEL_NAME]
"""

import argparse
import sys
import os
from pathlib import Path


def check_model_installed():
    """Check if any AMR model is installed."""
    try:
        import amrlib
        model_dir = Path(amrlib.__file__).parent / "data"
        
        # Check for stog (sentence-to-graph) models
        stog_models = list(model_dir.glob("model_stog*"))
        parse_models = list(model_dir.glob("model_parse*"))
        
        if stog_models or parse_models:
            print(f"[✓] Found installed models:")
            for m in stog_models + parse_models:
                print(f"    - {m.name}")
            return True
        else:
            print("[✗] No AMR models found")
            return False
    except Exception as e:
        print(f"[✗] Error checking models: {e}")
        return False


MODEL_URLS = {
    "parse_xfm_bart_large": "https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz",
    "parse_xfm_bart_base": "https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_base-v0_1_0/model_parse_xfm_bart_base-v0_1_0.tar.gz",
    "parse_spring": "https://github.com/bjascob/amrlib-models/releases/download/parse_spring-v0_1_0/model_parse_spring-v0_1_0.tar.gz",
}


def download_model(model_name: str = "parse_xfm_bart_base"):
    """Download and install AMR model."""
    print(f"\n[*] Downloading model: {model_name}")
    print("    This may take a few minutes...\n")
    
    if model_name not in MODEL_URLS:
        print(f"[✗] Unknown model: {model_name}")
        print(f"    Available: {', '.join(MODEL_URLS.keys())}")
        return False
    
    try:
        import amrlib
        
        url = MODEL_URLS[model_name]
        # Use the amrlib.download function
        # model_name for symlink should be "model_stog" for parse models
        amrlib.download("model_stog", url)
        
        print(f"\n[✓] Model {model_name} installed successfully!")
        return True
        
    except ImportError as e:
        print(f"[✗] Import error: {e}")
        print("    Make sure amrlib is installed: pip install amrlib")
        return False
    except Exception as e:
        print(f"[✗] Download error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test the installed model."""
    print("\n[*] Testing AMR parsing...")
    
    try:
        import amrlib
        
        # Load model
        stog = amrlib.load_stog_model()
        
        # Test sentence
        test_sentence = "A customer places an order."
        graphs = stog.parse_sents([test_sentence])
        
        if graphs and graphs[0]:
            print(f"\n[✓] Model working!")
            print(f"\nInput: {test_sentence}")
            print(f"AMR:\n{graphs[0]}")
            return True
        else:
            print("[✗] Model returned empty result")
            return False
            
    except Exception as e:
        print(f"[✗] Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup AMR parsing model")
    parser.add_argument("--model", default="parse_xfm_bart_base",
                       choices=["parse_xfm_bart_large", "parse_xfm_bart_base", "parse_spring"],
                       help="Model to download (default: parse_xfm_bart_base)")
    parser.add_argument("--check", action="store_true",
                       help="Only check if model is installed")
    parser.add_argument("--test", action="store_true",
                       help="Test the installed model")
    args = parser.parse_args()
    
    print("=" * 50)
    print("  AMR MODEL SETUP")
    print("=" * 50)
    
    if args.check:
        check_model_installed()
        return
    
    if args.test:
        test_model()
        return
    
    # Check current status
    if check_model_installed():
        response = input("\nModel already installed. Re-download? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping download.")
            test_model()
            return
    
    # Download
    if download_model(args.model):
        test_model()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
