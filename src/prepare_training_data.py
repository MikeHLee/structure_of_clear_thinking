"""
Training Data Preparation for Olog Fine-Tuning

Downloads Text2KGBench and WebNLG datasets, converts them to Olog format
for fine-tuning an LLM on ontology-constrained knowledge graph generation.

Usage:
    python prepare_training_data.py --dataset text2kg
    python prepare_training_data.py --dataset webnlg
    python prepare_training_data.py --dataset all
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "training_data"


@dataclass
class OlogTrainingSample:
    """A single training sample for Olog generation."""
    id: str
    text: str
    ontology_name: str
    ontology_types: List[str]
    ontology_relations: List[Dict[str, str]]  # {name, domain, range}
    olog_types: List[Dict[str, str]]  # {name, description}
    olog_aspects: List[Dict[str, str]]  # {source, label, target}
    source_triples: List[Dict[str, str]]  # Original KG triples


def download_text2kg():
    """Clone Text2KGBench repository."""
    repo_dir = DATA_DIR / "Text2KGBench"
    
    if repo_dir.exists():
        logger.info("Text2KGBench already downloaded")
        return repo_dir
    
    logger.info("Cloning Text2KGBench repository...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        ["git", "clone", "--depth", "1", 
         "https://github.com/cenguix/Text2KGBench.git",
         str(repo_dir)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to clone: {result.stderr}")
        return None
    
    logger.info(f"Downloaded to {repo_dir}")
    return repo_dir


def download_webnlg():
    """Download WebNLG via HuggingFace datasets."""
    cache_dir = DATA_DIR / "webnlg_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        logger.info("Loading WebNLG from HuggingFace...")
        dataset = load_dataset("GEM/web_nlg", cache_dir=str(cache_dir))
        logger.info(f"Loaded WebNLG: {dataset}")
        return dataset
    except ImportError:
        logger.error("Install datasets: pip install datasets")
        return None
    except Exception as e:
        logger.error(f"Failed to load WebNLG: {e}")
        return None


def parse_text2kg_ontology(ttl_path: Path) -> Dict[str, Any]:
    """Parse a TTL ontology file to extract types and relations."""
    try:
        from rdflib import Graph, RDF, RDFS, OWL
        
        g = Graph()
        g.parse(ttl_path, format="turtle")
        
        # Extract classes
        types = []
        for s in g.subjects(RDF.type, OWL.Class):
            label = g.value(s, RDFS.label)
            types.append(str(label) if label else str(s).split("/")[-1])
        
        # Extract properties
        relations = []
        for s in g.subjects(RDF.type, OWL.ObjectProperty):
            label = g.value(s, RDFS.label)
            domain = g.value(s, RDFS.domain)
            range_ = g.value(s, RDFS.range)
            relations.append({
                "name": str(label) if label else str(s).split("/")[-1],
                "domain": str(domain).split("/")[-1] if domain else "Thing",
                "range": str(range_).split("/")[-1] if range_ else "Thing"
            })
        
        return {"types": types, "relations": relations}
    except Exception as e:
        logger.warning(f"Failed to parse {ttl_path}: {e}")
        return {"types": [], "relations": []}


def convert_text2kg_sample(
    sample: Dict[str, Any],
    ontology: Dict[str, Any],
    ontology_name: str
) -> OlogTrainingSample:
    """Convert a Text2KGBench sample to Olog format."""
    
    # Extract unique entities as types
    entities = set()
    for triple in sample.get("triples", []):
        entities.add(triple["sub"])
        if not triple["obj"].startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            entities.add(triple["obj"])
    
    olog_types = [{"name": e, "description": f"Entity: {e}"} for e in entities]
    
    # Convert triples to aspects
    olog_aspects = []
    for triple in sample.get("triples", []):
        olog_aspects.append({
            "source": triple["sub"],
            "label": triple["rel"],
            "target": triple["obj"]
        })
    
    return OlogTrainingSample(
        id=sample.get("id", "unknown"),
        text=sample.get("sent", ""),
        ontology_name=ontology_name,
        ontology_types=ontology.get("types", []),
        ontology_relations=ontology.get("relations", []),
        olog_types=olog_types,
        olog_aspects=olog_aspects,
        source_triples=sample.get("triples", [])
    )


def convert_webnlg_sample(example: Dict[str, Any]) -> OlogTrainingSample:
    """Convert a WebNLG sample to Olog format."""
    
    # Extract triples
    triples = []
    entities = set()
    
    modified_triples = example.get("input", [])
    for triple_str in modified_triples:
        parts = triple_str.split(" | ")
        if len(parts) == 3:
            sub, rel, obj = parts
            triples.append({"sub": sub.strip(), "rel": rel.strip(), "obj": obj.strip()})
            entities.add(sub.strip())
            entities.add(obj.strip())
    
    olog_types = [{"name": e, "description": f"Entity: {e}"} for e in entities]
    olog_aspects = [
        {"source": t["sub"], "label": t["rel"], "target": t["obj"]}
        for t in triples
    ]
    
    # Get reference text
    refs = example.get("references", [])
    text = refs[0] if refs else example.get("target", "")
    
    return OlogTrainingSample(
        id=example.get("gem_id", "unknown"),
        text=text,
        ontology_name=example.get("category", "General"),
        ontology_types=list(entities),
        ontology_relations=[],
        olog_types=olog_types,
        olog_aspects=olog_aspects,
        source_triples=triples
    )


def process_text2kg(repo_dir: Path) -> List[OlogTrainingSample]:
    """Process all Text2KGBench data."""
    samples = []
    
    for dataset_name in ["wikidata_tekgen", "dbpedia_webnlg"]:
        dataset_dir = repo_dir / "data" / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"Dataset dir not found: {dataset_dir}")
            continue
        
        # Load ontologies
        ontologies = {}
        ont_dir = dataset_dir / "ontologies" / "owl"
        if ont_dir.exists():
            for ttl_file in ont_dir.glob("*.ttl"):
                ont_name = ttl_file.stem
                ontologies[ont_name] = parse_text2kg_ontology(ttl_file)
                logger.info(f"Loaded ontology: {ont_name}")
        
        # Load training data (JSONL format with individual triples)
        train_dir = dataset_dir / "train"
        if train_dir.exists():
            for jsonl_file in train_dir.glob("*.jsonl"):
                ont_name = jsonl_file.stem.replace("_train", "")
                ontology = ontologies.get(ont_name, {"types": [], "relations": []})
                
                # Group samples by sentence to combine triples
                sent_to_triples = {}
                with open(jsonl_file) as f:
                    for line in f:
                        try:
                            row = json.loads(line.strip())
                            sent = row.get("sent", "")
                            if sent not in sent_to_triples:
                                sent_to_triples[sent] = {
                                    "id": row.get("id", ""),
                                    "sent": sent,
                                    "triples": []
                                }
                            sent_to_triples[sent]["triples"].append({
                                "sub": row.get("sub_label", ""),
                                "rel": row.get("rel_label", ""),
                                "obj": row.get("obj_label", "")
                            })
                        except json.JSONDecodeError:
                            continue
                
                # Convert grouped samples
                for sample in sent_to_triples.values():
                    if sample["sent"] and sample["triples"]:
                        converted = convert_text2kg_sample(sample, ontology, ont_name)
                        samples.append(converted)
        
        logger.info(f"Processed {len(samples)} samples from {dataset_name}")
    
    return samples


def process_webnlg(dataset) -> List[OlogTrainingSample]:
    """Process WebNLG dataset."""
    samples = []
    
    for split in ["train", "validation"]:
        if split in dataset:
            for example in dataset[split]:
                converted = convert_webnlg_sample(example)
                if converted.text and converted.olog_aspects:
                    samples.append(converted)
    
    logger.info(f"Processed {len(samples)} WebNLG samples")
    return samples


def create_training_jsonl(samples: List[OlogTrainingSample], output_path: Path):
    """Create JSONL file for fine-tuning."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for sample in samples:
            # Create training format
            prompt = f"""Extract an Olog from the following text using the given ontology.

Ontology: {sample.ontology_name}
Available Types: {', '.join(sample.ontology_types[:10])}

Text: {sample.text}

Output the Olog as JSON with types and aspects."""

            response = json.dumps({
                "types": sample.olog_types,
                "aspects": sample.olog_aspects
            }, indent=2)
            
            training_example = {
                "messages": [
                    {"role": "system", "content": "You extract Ontological Logs (Ologs) from text."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }
            
            f.write(json.dumps(training_example) + "\n")
    
    logger.info(f"Wrote {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Olog training data")
    parser.add_argument("--dataset", choices=["text2kg", "webnlg", "all"], 
                       default="all", help="Dataset to process")
    parser.add_argument("--output", type=Path, 
                       default=DATA_DIR / "olog_training.jsonl",
                       help="Output JSONL path")
    args = parser.parse_args()
    
    all_samples = []
    
    if args.dataset in ["text2kg", "all"]:
        repo_dir = download_text2kg()
        if repo_dir:
            samples = process_text2kg(repo_dir)
            all_samples.extend(samples)
            logger.info(f"Text2KGBench: {len(samples)} samples")
    
    if args.dataset in ["webnlg", "all"]:
        dataset = download_webnlg()
        if dataset:
            samples = process_webnlg(dataset)
            all_samples.extend(samples)
            logger.info(f"WebNLG: {len(samples)} samples")
    
    if all_samples:
        create_training_jsonl(all_samples, args.output)
        print(f"\n✓ Created training data: {args.output}")
        print(f"  Total samples: {len(all_samples)}")
    else:
        print("No samples processed. Check errors above.")


if __name__ == "__main__":
    main()
