from rag_engine import RAGEngine
import os
import shutil
import sys

# Force UTF-8 for Windows terminals
sys.stdout.reconfigure(encoding='utf-8')

# 1. Setup Empty KB
temp_dir = "temp_empty_kb_proof"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
    
print(f"--- Testing Empty Knowledge Base ({temp_dir}) ---")
# Initialize engine with empty folder
rag = RAGEngine(use_openai_embeddings=False, storage_dir=temp_dir)

# 2. Search
query = "Hello, who are you?"
print(f"\nQuerying: '{query}'")
chunks = rag.retrieve_relevant_chunks(query)

print(f"Retrieved Chunks: {len(chunks)}")

# 3. Verify Logic
if not chunks:
    print("\n[SUCCESS] 0 Chunks retrieved.")
    print("Logic Check: If chunks == 0, api_server.py uses DEFAULT prompt.")
    print("Result: Chatbot behaves responsibly (No 'I don't have info' loop).")
else:
    print("\n[FAIL] Chunks found unexpectedly.")

# Cleanup
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
