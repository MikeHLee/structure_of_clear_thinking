#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt Engineering for Structured Output

This script demonstrates how to use prompt engineering techniques to guide an LLM
to produce structured output in JSON format.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install openai python-dotenv
3. Set up your OpenAI API key in a .env file
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize OpenAI client
client = OpenAI()

# Sample text to analyze (in a real scenario, this would be loaded from a file)
sample_text = """
Apple Inc. reported a total market value of approximately $2,628,553,000,000 held by 
non-affiliates as of March 29, 2024. The company has 15,115,823,000 shares of common 
stock issued and outstanding as of October 18, 2024. Apple continues to innovate in 
the technology sector with new product launches planned for the upcoming fiscal year.
"""


def get_structured_output_with_one_shot_prompting(text):
    """
    Use one-shot prompting to guide the LLM to produce structured output.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: The parsed JSON response, or None if parsing fails
    """
    prompt = """
    Generate a two-person discussion about the key financial data from the following text in JSON format.

    <JSON_FORMAT>
    {
       "Person1": {
         "name": "Alice",
         "statement": "The revenue for Q1 has increased by 20% compared to last year."
       },
       "Person2": {
         "name": "Bob",
         "statement": "That's great news! What about the net profit margin?"
       }
    }
    </JSON_FORMAT>
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can change to a different model if needed
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    
    response_content = response.choices[0].message.content
    print("\nRaw response:\n", response_content)
    
    # Try to extract JSON from the response
    try:
        # First, check if the response is already valid JSON
        parsed_json = json.loads(response_content)
        return parsed_json
    except json.JSONDecodeError:
        # If not, try to extract JSON from markdown code blocks
        if "```json" in response_content and "```" in response_content.split("```json", 1)[1]:
            json_str = response_content.split("```json", 1)[1].split("```", 1)[0].strip()
            try:
                parsed_json = json.loads(json_str)
                return parsed_json
            except json.JSONDecodeError:
                print("Failed to parse JSON from code block")
                return None
        else:
            print("No valid JSON found in the response")
            return None


def get_structured_output_with_json_mode(text):
    """
    Use the JSON mode feature to guide the LLM to produce structured output.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: The parsed JSON response
    """
    prompt = f"""
    Generate a two-person discussion about the key financial data from the following text in JSON format.
    TEXT: {text}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Make sure to use a model that supports JSON mode
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    response_content = response.choices[0].message.content
    print("\nJSON mode response:\n", response_content)
    
    # Parse the JSON response
    return json.loads(response_content)


def main():
    print("\n=== Prompt Engineering for Structured Output ===\n")
    print("Sample text:\n", sample_text)
    
    # Example 1: One-shot prompting
    print("\n--- Example 1: One-shot prompting ---")
    result1 = get_structured_output_with_one_shot_prompting(sample_text)
    if result1:
        print("\nParsed JSON:")
        print(json.dumps(result1, indent=2))
    
    # Example 2: JSON mode
    print("\n--- Example 2: JSON mode ---")
    try:
        result2 = get_structured_output_with_json_mode(sample_text)
        print("\nParsed JSON:")
        print(json.dumps(result2, indent=2))
    except Exception as e:
        print(f"Error with JSON mode: {e}")
        print("Note: JSON mode requires a model that supports this feature.")


if __name__ == "__main__":
    main()
