import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')

# 2. Load Env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not set in .env file.")
else:
    # 3. Call API
    try:
        base_url = None
        if api_key.startswith("sk-or-"):
            base_url = "https://openrouter.ai/api/v1"
            logging.info("OpenRouter key detected. Using OpenRouter API.")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        logging.info("Sending simple request...")
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # OpenRouter supports this mapping
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Give me a fun fact about Python."}
            ]
        )
        
        print("\nResponse:")
        print(completion.choices[0].message.content)
        logging.info("Success!")

    except Exception as e:
        print(f"Failed: {e}")
        logging.error(f"API Error: {e}")
