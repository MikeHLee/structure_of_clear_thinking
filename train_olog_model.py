"""
Fine-Tuning Script for Olog Generation Model

Trains a small LLM (Qwen2.5-Coder-1.5B or similar) to generate valid Ologs
from natural language text using LoRA.

Requirements:
    pip install transformers peft accelerate bitsandbytes trl

Usage:
    python train_olog_model.py --model qwen2.5-coder-1.5b --data training_data/olog_training.jsonl
    python train_olog_model.py --test  # Quick test with small subset
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "qwen2.5-coder-1.5b": {
        "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "max_length": 2048,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "phi-3-mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "max_length": 4096,
        "target_modules": ["qkv_proj", "o_proj"],
    },
    "llama-3.2-1b": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "max_length": 2048,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
}


def load_training_data(data_path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load training data from JSONL file."""
    samples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} training samples")
    return samples


def format_for_training(samples: List[Dict]) -> List[Dict]:
    """Format samples for SFT training."""
    formatted = []
    for sample in samples:
        messages = sample.get("messages", [])
        if messages:
            formatted.append({"messages": messages})
    return formatted


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import peft
    except ImportError:
        missing.append("peft")
    try:
        import trl
    except ImportError:
        missing.append("trl")
    try:
        import accelerate
    except ImportError:
        missing.append("accelerate")
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install with: pip install transformers peft trl accelerate bitsandbytes")
        return False
    return True


def train(
    model_key: str,
    data_path: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Run LoRA fine-tuning."""
    
    if not check_dependencies():
        return
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    import torch
    
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        logger.error(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
        return
    
    model_name = config["name"]
    max_length = config["max_length"]
    target_modules = config["target_modules"]
    
    logger.info(f"Loading model: {model_name}")
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    samples = load_training_data(data_path, max_samples)
    formatted = format_for_training(samples)
    
    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_list(formatted)
    
    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        max_seq_length=max_length,
        packing=False,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")


def test_inference(model_dir: Path, prompt: str):
    """Test inference with trained model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    
    # Load base model and adapter
    base_config = MODEL_CONFIGS["qwen2.5-coder-1.5b"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_config["name"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, model_dir)
    
    # Generate
    messages = [
        {"role": "system", "content": "You extract Ontological Logs (Ologs) from text."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


def main():
    parser = argparse.ArgumentParser(description="Train Olog generation model")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                       default="qwen2.5-coder-1.5b", help="Base model to fine-tune")
    parser.add_argument("--data", type=Path,
                       default=Path("training_data/olog_training.jsonl"),
                       help="Training data JSONL path")
    parser.add_argument("--output", type=Path,
                       default=Path("models/olog_generator"),
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit training samples (for testing)")
    parser.add_argument("--test", action="store_true",
                       help="Quick test with 100 samples")
    parser.add_argument("--inference", type=str, default=None,
                       help="Run inference with this prompt")
    args = parser.parse_args()
    
    if args.inference:
        test_inference(args.output, args.inference)
        return
    
    max_samples = 100 if args.test else args.max_samples
    
    train(
        model_key=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_samples=max_samples,
        epochs=1 if args.test else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
