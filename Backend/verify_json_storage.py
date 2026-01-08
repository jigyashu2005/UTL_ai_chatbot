from rag_engine import RAGEngine
import os
import shutil
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

kb_dir = "verify_json_kb"
if os.path.exists(kb_dir):
    shutil.rmtree(kb_dir)

print(f"--- JSON Storage Verification ({kb_dir}) ---")
rag = RAGEngine(use_openai_embeddings=False, storage_dir=kb_dir)

# 1. Create File
with open("json_test.txt", "w", encoding="utf-8") as f:
    f.write("This specific text must appear in the JSON file.")

# 2. Ingest
rag.load_documents([os.path.abspath("json_test.txt")])

# 3. Check JSON
json_path = os.path.join(kb_dir, "knowledge_base.json")
if os.path.exists(json_path):
    print(f"\n[SUCCESS] File found: {json_path}")
    print("Content:")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(json.dumps(data, indent=2))
else:
    print(f"\n[FAIL] File not found: {json_path}")

# Cleanup
if os.path.exists(kb_dir):
    shutil.rmtree(kb_dir)
try:
    os.remove("json_test.txt")
except:
    pass
