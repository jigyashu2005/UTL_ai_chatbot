from rag_engine import RAGEngine
import os
import sys

# Ensure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = os.path.abspath("test_rag_doc.pdf")

if not os.path.exists(pdf_path):
    print("PDF missing, run create_dummy_pdf.py first")
    exit(1)

print("Initializing Engine (Local Mode)...")
rag = RAGEngine(use_openai_embeddings=False)

print(f"Loading PDF: {pdf_path}")
pages = rag.load_pdf(pdf_path)

print("Chunking...")
rag.chunk_text(pages)

print("Embedding & Storing...")
rag.embed_and_store()

# --- Persistence Test ---
print("\n--- Testing Persistence ---")
db_dir = "test_knowledge_base"
rag.save_index(db_dir)

print("Clearing memory...")
rag.index = None
rag.chunks = []

print(f"Loading from {db_dir}...")
rag.load_index(db_dir)
# ------------------------

query = "What do solar inverters do?"
print(f"\nQuerying: '{query}'")
results = rag.search(query)

print("\n--- Search Results ---")
for res in results:
    print(f"[Score: {res['score']:.4f}] [Page {res['metadata']['page_number']}] {res['text'][:100]}...")
    
print("\n--- Context Format ---")
context = rag.format_context(results)
print(context)

print("\nDONE")
