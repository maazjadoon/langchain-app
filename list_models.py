"""Quick script to list all embedding-capable Gemini models available to your API key."""
import os
from dotenv import load_dotenv

load_dotenv()

import google.genai as genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Embedding-capable models:")
print("-" * 50)
for model in client.models.list():
    # Check if model supports embedContent
    supported = getattr(model, 'supported_actions', None) or []
    name = model.name
    if 'embed' in name.lower() or 'embedding' in name.lower():
        print(f"  {name}")

print()
print("All models (full list):")
print("-" * 50)
for model in client.models.list():
    print(f"  {model.name}")
