import requests
import json

def test_triplex():
    url = "http://localhost:11434/api/generate"
    text = "MemGPT uses a virtual context management system to extend the context window of LLMs. It resembles the memory hierarchy of operating systems."
    
    # Try a more explicit prompt
    prompt = f"Extract all entities and relations from the following text as a list of triplets [subject, predicate, object].\n\nText: {text}"
    
    payload = {
        "model": "sciphi/triplex:latest",
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Calling Triplex with text: {text}...")
    response = requests.post(url, json=payload)
    print("\n[RAW RESPONSE]")
    print(response.json().get("response", ""))

if __name__ == "__main__":
    test_triplex()
