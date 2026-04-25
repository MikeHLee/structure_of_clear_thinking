#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Structured Output with Outlines

This script demonstrates how to use the Outlines library to generate structured output
from language models using various constraint methods:
1. Multiple choice generation
2. Regex-based generation
3. JSON schema-based generation using Pydantic

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install outlines transformers pydantic
"""

import outlines
from pydantic import BaseModel, Field
from typing import List, Optional

# Sample text to analyze
sample_text = """
Apple Inc. reported a total market value of approximately $2,628,553,000,000 held by 
non-affiliates as of March 29, 2024. The company has 15,115,823,000 shares of common 
stock issued and outstanding as of October 18, 2024. Apple continues to innovate in 
the technology sector with new product launches planned for the upcoming fiscal year.
"""

# Define a Pydantic model for structured extraction
class EntityExtraction(BaseModel):
    mentioned_entities: List[str] = Field(description="Names of companies, organizations, or products mentioned in the text")
    mentioned_places: List[str] = Field(description="Names of locations or places mentioned in the text")
    financial_metrics: Optional[List[dict]] = Field(description="Financial metrics mentioned in the text with their values")


def multiple_choice_example(model, text):
    """
    Demonstrate multiple choice generation with Outlines.
    
    Args:
        model: The Outlines model to use
        text (str): The input text to analyze
        
    Returns:
        str: The selected choice
    """
    prompt = f"""
    You are a sentiment-labelling assistant specialized in Financial Statements.
    Is the following document positive, negative, or neutral?

    Document: {text}
    """

    generator = outlines.generate.choice(model, ["Positive", "Negative", "Neutral"])
    answer = generator(prompt)
    return answer


def regex_example(model, text):
    """
    Demonstrate regex-based generation with Outlines.
    
    Args:
        model: The Outlines model to use
        text (str): The input text to analyze
        
    Returns:
        str: The generated text matching the regex pattern
    """
    # Define a regex pattern for extracting a dollar amount
    dollar_pattern = r"\$[0-9,]+(?:\.[0-9]{2})?"
    
    prompt = f"""
    Extract a dollar amount from the following text. Format it as a dollar amount with a $ sign.
    
    Text: {text}
    
    Dollar amount: 
    """
    
    generator = outlines.generate.regex(model, dollar_pattern)
    amount = generator(prompt)
    return amount


def json_schema_example(model, text):
    """
    Demonstrate JSON schema-based generation with Outlines using Pydantic.
    
    Args:
        model: The Outlines model to use
        text (str): The input text to analyze
        
    Returns:
        EntityExtraction: The structured extraction result
    """
    prompt = f"""
    You are an expert at structured data extraction. Extract entities, places, and financial metrics 
    from the following text and format the response according to the provided schema.

    Text: {text}
    """
    
    generator = outlines.generate.json(model, EntityExtraction)
    extraction = generator(prompt)
    return extraction


def main():
    print("\n=== Structured Output with Outlines ===\n")
    print("Sample text:\n", sample_text)
    
    # Load a small model for demonstration
    print("\nLoading model... (this may take a moment)")
    try:
        # Try to load a small model that works well with Outlines
        model = outlines.models.transformers("Qwen/Qwen2.5-0.5B-Instruct")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNote: If you encounter issues loading the model, you may need to:")
        print("1. Install additional dependencies: pip install torch accelerate")
        print("2. Try a different model, such as 'gpt2' or 'distilgpt2'")
        print("3. Ensure you have enough disk space and RAM")
        return
    
    # Example 1: Multiple choice generation
    print("\n--- Example 1: Multiple choice generation ---")
    try:
        sentiment = multiple_choice_example(model, sample_text)
        print(f"Sentiment: {sentiment}")
    except Exception as e:
        print(f"Error with multiple choice example: {e}")
    
    # Example 2: Regex-based generation
    print("\n--- Example 2: Regex-based generation ---")
    try:
        dollar_amount = regex_example(model, sample_text)
        print(f"Extracted dollar amount: {dollar_amount}")
    except Exception as e:
        print(f"Error with regex example: {e}")
    
    # Example 3: JSON schema-based generation
    print("\n--- Example 3: JSON schema-based generation ---")
    try:
        extraction = json_schema_example(model, sample_text)
        print("Extracted entities:", extraction.mentioned_entities)
        print("Extracted places:", extraction.mentioned_places)
        if extraction.financial_metrics:
            print("Financial metrics:")
            for metric in extraction.financial_metrics:
                print(f"  - {metric}")
    except Exception as e:
        print(f"Error with JSON schema example: {e}")


if __name__ == "__main__":
    main()
