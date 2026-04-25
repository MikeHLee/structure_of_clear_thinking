# Fine-Tuning Strategy for Olog Generation

## Objective
Train a small LLM to generate valid Ontological Logs (Ologs) from natural language text, constrained by a given ontology schema.

## Available Datasets

### 1. Text2KGBench (Primary)
**Source:** https://github.com/cenguix/Text2KGBench  
**License:** CC BY 4.0

| Dataset | Ontologies | Sentences | Task |
|---------|-----------|-----------|------|
| Wikidata-TekGen | 10 | 13,474 | Text → KG triples |
| DBpedia-WebNLG | 19 | 4,860 | Text → KG triples |

**Format:**
```json
{
  "id": "ont_music_test_n",
  "sent": "\"The Loco-Motion\" is a 1962 pop song written by Gerry Goffin and Carole King.",
  "triples": [
    {"sub": "The Loco-Motion", "rel": "publication date", "obj": "01 January 1962"},
    {"sub": "The Loco-Motion", "rel": "lyrics by", "obj": "Gerry Goffin"},
    {"sub": "The Loco-Motion", "rel": "lyrics by", "obj": "Carole King"}
  ]
}
```

**Relevance:** Direct ontology-constrained KG generation. Closest to our Olog task.

### 2. WebNLG (GEM Benchmark)
**Source:** `datasets.load_dataset('GEM/web_nlg')`  
**License:** CC BY-NC-SA 4.0

| Split | Triple Sets | Texts |
|-------|-------------|-------|
| English | 17,000 | 45,000 |
| Russian | 7,000 | 19,000 |

**Format:** RDF triples (subject, predicate, object) ↔ Natural language  
**Properties:** ~450 DBpedia properties  
**Relevance:** Bidirectional text↔KG, useful for both parsing and generation.

### 3. AMR Banks (LDC)
**Source:** LDC2020T02 (requires license)  
**Size:** ~60,000 sentences with AMR annotations

**Relevance:** Pre-training for semantic parsing stage of hybrid encoder.

### 4. ConceptNet / ATOMIC
**Source:** https://conceptnet.io/, https://allenai.org/data/atomic  
**License:** CC BY 4.0

**Relevance:** Commonsense knowledge for semantic contradiction detection enhancement.

---

## Training Pipeline

### Phase 1: Data Preparation

```
Text2KGBench + WebNLG
        ↓
  Convert to Olog Format
        ↓
  (text, ontology_schema) → OlogGraph JSON
```

**Conversion Script Tasks:**
1. Parse ontology TTL files → extract types, relations, domain/range
2. Convert KG triples → OlogGraph structure
3. Generate commutative facts from path patterns
4. Validate with H¹ obstruction detection

### Phase 2: Model Selection

| Model | Parameters | Quantized | Notes |
|-------|-----------|-----------|-------|
| Qwen2.5-Coder-1.5B | 1.5B | 4-bit ~1GB | Best for structured output |
| Phi-3-mini | 3.8B | 4-bit ~2GB | Strong reasoning |
| Llama-3.2-1B | 1B | 4-bit ~700MB | Lightweight |
| Mistral-7B | 7B | 4-bit ~4GB | If GPU available |

**Recommended:** Qwen2.5-Coder-1.5B (excellent JSON/code generation)

### Phase 3: Training Configuration

```yaml
# LoRA Config
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training
learning_rate: 2e-4
batch_size: 4
gradient_accumulation: 4
epochs: 3
warmup_ratio: 0.1

# Data
max_seq_length: 2048
train_split: 0.9
```

### Phase 4: Prompt Template

```
<|system|>
You are an ontology extraction system. Given text and an ontology schema, 
extract an Olog (Ontological Log) that:
1. Uses only types and relations defined in the schema
2. Maintains functional aspects (each source has exactly one target per aspect)
3. Ensures commutative facts are semantically consistent
<|end|>

