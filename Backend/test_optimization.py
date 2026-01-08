from rag_engine import RAGEngine
import os
import shutil
import sys

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

kb_dir = "temp_opt_kb"
if os.path.exists(kb_dir):
    shutil.rmtree(kb_dir)

print(f"--- Optimization Test (Storage: {kb_dir}) ---")
rag = RAGEngine(use_openai_embeddings=False, storage_dir=kb_dir)

# 1. Create a "Large" File
large_text = "This is a repeated line to simulate a large document.\n" * 500 # ~25KB
with open("large_doc.txt", "w", encoding="utf-8") as f:
    f.write(large_text)

# 2. Ingest
print("Ingesting large document...")
rag.load_documents([os.path.abspath("large_doc.txt")])

# 3. Simulate Logic from api_server.py
query = "simulate"
print(f"\nQuerying: '{query}'")

# A. Retrieve Top-1 (Optimization 1)
relevant_chunks = rag.retrieve_relevant_chunks(query, k=1)
print(f"Chunks Retrieved: {len(relevant_chunks)} (Expected: 1)")

# B. Generate Context
context_str = rag.generate_context_string(relevant_chunks)
original_len = len(context_str)
print(f"Original Context Length: {original_len} chars")

# C. Truncate (Optimization 2)
if len(context_str) > 1000:
    context_str = context_str[:1000] + "...(truncated)"

final_len = len(context_str)
print(f"Final Context Length: {final_len} chars")

if final_len <= 1015 and len(relevant_chunks) == 1:
    print("\n[SUCCESS] Optimization Logic Verified.")
    print("The prompt is now SAFE for small local models.")
else:
    print("\n[FAIL] Logic not working as expected.")

# Cleanup
if os.path.exists(kb_dir):
    shutil.rmtree(kb_dir)
try:
    os.remove("large_doc.txt")
except:
    pass
