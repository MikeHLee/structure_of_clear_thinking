#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Structured Output with LangChain

This script demonstrates how to use LangChain to generate structured output
from language models using the with_structured_output method.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install langchain langchain-openai pydantic python-dotenv
3. Set up your OpenAI API key in a .env file
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv(override=True)

# Sample text to analyze
sample_text = """
Apple Inc. reported a total market value of approximately $2,628,553,000,000 held by 
non-affiliates as of March 29, 2024. The company has 15,115,823,000 shares of common 
stock issued and outstanding as of October 18, 2024. Apple continues to innovate in 
the technology sector with new product launches planned for the upcoming fiscal year.
"""

# Define Pydantic models for structured output
class FinancialMetric(BaseModel):
    metric_name: str = Field(description="Name of the financial metric")
    value: str = Field(description="Value of the metric, including any units or currency symbols")
    date: Optional[str] = Field(description="Date associated with the metric, if available")

class EntityExtraction(BaseModel):
    mentioned_entities: List[str] = Field(description="Names of companies, organizations, or products mentioned in the text")
    mentioned_places: List[str] = Field(description="Names of locations or places mentioned in the text")
    financial_metrics: List[FinancialMetric] = Field(description="Financial metrics mentioned in the text with their values")

class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment of the text (positive, negative, or neutral)")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation for the sentiment classification")


def extract_entities(text):
    """
    Extract structured entity information from text using LangChain.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        EntityExtraction: The structured extraction result
    """
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # You can change to a different model if needed
    
    # Create a structured output wrapper
    structured_llm = llm.with_structured_output(EntityExtraction)
    
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at structured data extraction. Extract entities, places, and financial metrics from the following text."),
        ("human", "{text}")
    ])
    
    # Create and run the chain
    chain = prompt_template | structured_llm
    result = chain.invoke({"text": text})
    
    return result


def analyze_sentiment(text):
    """
    Analyze the sentiment of text using LangChain with structured output.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        SentimentAnalysis: The structured sentiment analysis result
    """
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # You can change to a different model if needed
    
    # Create a structured output wrapper
    structured_llm = llm.with_structured_output(SentimentAnalysis)
    
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a financial sentiment analysis expert. Analyze the sentiment of the following text and provide your reasoning."),
        ("human", "{text}")
    ])
    
    # Create and run the chain
    chain = prompt_template | structured_llm
    result = chain.invoke({"text": text})
    
    return result


def main():
    print("\n=== Structured Output with LangChain ===\n")
    print("Sample text:\n", sample_text)
    
    # Example 1: Entity extraction
    print("\n--- Example 1: Entity extraction ---")
    try:
        entities = extract_entities(sample_text)
        print("\nExtracted entities:", entities.mentioned_entities)
        print("Extracted places:", entities.mentioned_places)
        print("\nFinancial metrics:")
        for metric in entities.financial_metrics:
            print(f"  - {metric.metric_name}: {metric.value}" + 
                  (f" ({metric.date})" if metric.date else ""))
    except Exception as e:
        print(f"Error with entity extraction: {e}")
    
    # Example 2: Sentiment analysis
    print("\n--- Example 2: Sentiment analysis ---")
    try:
        sentiment = analyze_sentiment(sample_text)
        print(f"\nSentiment: {sentiment.sentiment}")
        print(f"Confidence: {sentiment.confidence:.2f}")
        print(f"Reasoning: {sentiment.reasoning}")
    except Exception as e:
        print(f"Error with sentiment analysis: {e}")


if __name__ == "__main__":
    main()
