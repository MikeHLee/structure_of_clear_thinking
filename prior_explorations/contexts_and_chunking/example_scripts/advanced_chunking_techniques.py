#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Text Chunking Techniques for LLMs

This script demonstrates advanced chunking techniques for dividing large texts
into semantically meaningful chunks that can be processed by LLMs with limited context windows.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install nltk scikit-learn sentence-transformers
"""

import os
import re
import nltk
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Try to import sentence-transformers, but provide a fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Semantic chunking will be simulated.")

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Sample text for demonstration
sample_text = """
Large Language Models (LLMs) have revolutionized natural language processing. 
They can generate human-like text, translate languages, write different kinds of creative content, 
and answer questions in an informative way. However, most LLMs have a limited context window, 
which means they can only process a certain number of tokens at once. This limitation 
necessitates effective chunking strategies when dealing with long documents.

Chunking is the process of breaking down large texts into smaller, manageable pieces that 
fit within an LLM's context window. Effective chunking preserves the semantic meaning and 
context of the original text, ensuring that the model can still understand and process 
the information correctly.

There are several approaches to chunking, each with its own advantages and disadvantages. 
Simple approaches include character-based chunking, word-based chunking, sentence-based chunking, 
and paragraph-based chunking. More advanced approaches include semantic chunking, which attempts 
to keep related content together, and recursive chunking, which applies chunking at multiple levels.

The choice of chunking strategy depends on the specific task, the nature of the text, and the 
requirements of the application. For example, for question answering tasks, it might be important 
to keep related sentences together, while for summarization tasks, paragraph-based chunking 
might be more appropriate.

Semantic chunking is particularly useful for maintaining the coherence of the text. By grouping 
sentences or paragraphs based on their semantic similarity, we can ensure that related information 
stays together, even if it's not physically adjacent in the original text.

Recursive chunking involves applying chunking at multiple levels. For example, we might first 
chunk a document into paragraphs, and then further chunk each paragraph into sentences if they're 
still too large for the model's context window.

Sliding window approaches with overlap can also be effective, especially for tasks where context 
from adjacent chunks is important. By including some overlap between chunks, we can ensure that 
the model has access to the necessary context when processing each chunk.
"""


def semantic_chunking(text: str, num_chunks: int = 3) -> List[str]:
    """
    Chunk text based on semantic similarity using sentence embeddings.
    
    Args:
        text (str): The input text to chunk
        num_chunks (int): Number of semantic chunks to create
        
    Returns:
        List[str]: List of text chunks grouped by semantic similarity
    """
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    if SENTENCE_TRANSFORMERS_AVAILABLE and len(sentences) > num_chunks:
        # Load a pre-trained sentence transformer model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Generate embeddings for each sentence
        embeddings = model.encode(sentences)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_chunks, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group sentences by cluster
        grouped_sentences = [[] for _ in range(num_chunks)]
        for i, cluster_id in enumerate(clusters):
            grouped_sentences[cluster_id].append(sentences[i])
        
        # Join sentences within each cluster
        chunks = [' '.join(group) for group in grouped_sentences if group]
    else:
        # Fallback to simple chunking if sentence-transformers not available
        # or if there are fewer sentences than requested chunks
        chunk_size = max(1, len(sentences) // num_chunks)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            end = min(i + chunk_size, len(sentences))
            chunks.append(' '.join(sentences[i:end]))
    
    return chunks


def recursive_chunking(text: str, max_chars: int = 500, max_depth: int = 2) -> List[str]:
    """
    Apply chunking recursively at multiple levels.
    
    Args:
        text (str): The input text to chunk
        max_chars (int): Maximum number of characters per final chunk
        max_depth (int): Maximum recursion depth
        
    Returns:
        List[str]: List of text chunks
    """
    def _chunk_recursive(text: str, depth: int = 0) -> List[str]:
        # Base case: if text is short enough or max depth reached
        if len(text) <= max_chars or depth >= max_depth:
            return [text]
        
        chunks = []
        
        # First level: split by paragraphs
        if depth == 0:
            paragraphs = re.split(r'\n\s*\n', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            for paragraph in paragraphs:
                # Recursively chunk each paragraph
                paragraph_chunks = _chunk_recursive(paragraph, depth + 1)
                chunks.extend(paragraph_chunks)
        
        # Second level: split by sentences
        elif depth == 1:
            sentences = nltk.sent_tokenize(text)
            
            current_chunk = ""
            for sentence in sentences:
                # If adding this sentence would exceed max_chars, start a new chunk
                if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    return _chunk_recursive(text)


def sliding_window_chunking(text: str, window_size: int = 200, step_size: int = 100) -> List[str]:
    """
    Chunk text using a sliding window approach with overlap.
    
    Args:
        text (str): The input text to chunk
        window_size (int): Size of the sliding window in characters
        step_size (int): Step size for sliding the window
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    text_len = len(text)
    
    # Ensure step_size is not larger than window_size
    step_size = min(step_size, window_size)
    
    for i in range(0, text_len, step_size):
        # Extract chunk using the sliding window
        end = min(i + window_size, text_len)
        chunk = text[i:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # If we've reached the end of the text, break
        if end == text_len:
            break
    
    return chunks


def hybrid_chunking(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    A hybrid approach that combines paragraph boundaries with size constraints.
    
    Args:
        text (str): The input text to chunk
        max_chunk_size (int): Maximum size of each chunk in characters
        
    Returns:
        List[str]: List of text chunks
    """
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_chunk_size, start a new chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            # Add paragraph separator if needed
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def print_chunks(chunks: List[str], strategy_name: str) -> None:
    """
    Print chunks with their indices and character counts.
    
    Args:
        chunks (List[str]): List of text chunks
        strategy_name (str): Name of the chunking strategy
    """
    print(f"\n--- {strategy_name} ---")
    print(f"Total chunks: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({len(chunk)} chars):\n{chunk}\n{'-'*50}")


def main():
    print("\n=== Advanced Text Chunking Techniques for LLMs ===\n")
    
    # Example 1: Semantic chunking
    semantic_chunks = semantic_chunking(sample_text, num_chunks=3)
    print_chunks(semantic_chunks, "Semantic chunking (3 clusters)")
    
    # Example 2: Recursive chunking
    recursive_chunks = recursive_chunking(sample_text, max_chars=300, max_depth=2)
    print_chunks(recursive_chunks, "Recursive chunking (max 300 chars, depth 2)")
    
    # Example 3: Sliding window chunking
    sliding_chunks = sliding_window_chunking(sample_text, window_size=250, step_size=150)
    print_chunks(sliding_chunks, "Sliding window chunking (250 char window, 150 char step)")
    
    # Example 4: Hybrid chunking
    hybrid_chunks = hybrid_chunking(sample_text, max_chunk_size=400)
    print_chunks(hybrid_chunks, "Hybrid chunking (max 400 chars per chunk)")


if __name__ == "__main__":
    main()
