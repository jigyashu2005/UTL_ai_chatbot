from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print(f"Key found: {api_key[:10]}...")

base_url = None
if api_key.startswith("sk-or-"):
    base_url = "https://openrouter.ai/api/v1"
    print(f"Using OpenRouter Base URL: {base_url}")

client = OpenAI(api_key=api_key, base_url=base_url)

print("\n--- Sending Request ---")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, are you working?"}],
    )
    print("Response received:")
    print(response.choices[0].message.content)
    print("\n[SUCCESS] Connection verified.")
except Exception as e:
    print(f"\n[FAIL] Error: {e}")
