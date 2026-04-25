#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Text Chunking Strategies for LLMs

This script demonstrates various basic chunking strategies for dividing large texts
into smaller chunks that can be processed by LLMs with limited context windows.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install nltk
"""

import os
import re
import nltk
from typing import List, Dict, Any, Tuple

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
"""


def character_chunking(text: str, chunk_size: int = 100, overlap: int = 0) -> List[str]:
    """
    Chunk text by character count with optional overlap.
    
    Args:
        text (str): The input text to chunk
        chunk_size (int): Maximum number of characters per chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # Calculate end position for this chunk
        end = min(start + chunk_size, text_len)
        
        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position for next chunk, considering overlap
        start = end - overlap if overlap < chunk_size else start + 1
    
    return chunks


def word_chunking(text: str, chunk_size: int = 20, overlap: int = 0) -> List[str]:
    """
    Chunk text by word count with optional overlap.
    
    Args:
        text (str): The input text to chunk
        chunk_size (int): Maximum number of words per chunk
        overlap (int): Number of words to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Split text into words
    words = text.split()
    total_words = len(words)
    chunks = []
    start = 0
    
    while start < total_words:
        # Calculate end position for this chunk
        end = min(start + chunk_size, total_words)
        
        # Extract the chunk
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start position for next chunk, considering overlap
        start = end - overlap if overlap < chunk_size else start + 1
    
    return chunks


def sentence_chunking(text: str, chunk_size: int = 5, overlap: int = 0) -> List[str]:
    """
    Chunk text by sentence count with optional overlap.
    
    Args:
        text (str): The input text to chunk
        chunk_size (int): Maximum number of sentences per chunk
        overlap (int): Number of sentences to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences)
    chunks = []
    start = 0
    
    while start < total_sentences:
        # Calculate end position for this chunk
        end = min(start + chunk_size, total_sentences)
        
        # Extract the chunk
        chunk = ' '.join(sentences[start:end])
        chunks.append(chunk)
        
        # Move start position for next chunk, considering overlap
        start = end - overlap if overlap < chunk_size else start + 1
    
    return chunks


def paragraph_chunking(text: str, max_paragraphs: int = 2) -> List[str]:
    """
    Chunk text by paragraph with a maximum number of paragraphs per chunk.
    
    Args:
        text (str): The input text to chunk
        max_paragraphs (int): Maximum number of paragraphs per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Split text into paragraphs (assuming paragraphs are separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    total_paragraphs = len(paragraphs)
    chunks = []
    
    for i in range(0, total_paragraphs, max_paragraphs):
        end = min(i + max_paragraphs, total_paragraphs)
        chunk = '\n\n'.join(paragraphs[i:end])
        chunks.append(chunk)
    
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
    print("\n=== Basic Text Chunking Strategies for LLMs ===\n")
    print("Sample text:\n", sample_text)
    
    # Example 1: Character-based chunking
    char_chunks = character_chunking(sample_text, chunk_size=200, overlap=50)
    print_chunks(char_chunks, "Character-based chunking (200 chars, 50 char overlap)")
    
    # Example 2: Word-based chunking
    word_chunks = word_chunking(sample_text, chunk_size=30, overlap=5)
    print_chunks(word_chunks, "Word-based chunking (30 words, 5 word overlap)")
    
    # Example 3: Sentence-based chunking
    sentence_chunks = sentence_chunking(sample_text, chunk_size=3, overlap=1)
    print_chunks(sentence_chunks, "Sentence-based chunking (3 sentences, 1 sentence overlap)")
    
    # Example 4: Paragraph-based chunking
    paragraph_chunks = paragraph_chunking(sample_text, max_paragraphs=1)
    print_chunks(paragraph_chunks, "Paragraph-based chunking (1 paragraph per chunk)")


if __name__ == "__main__":
    main()
