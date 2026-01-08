import os
import fitz  # PyMuPDF
import json
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Optional

# Dependencies
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

# Optional imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

class RAGEngine:
    def __init__(self, 
                 use_openai_embeddings: bool = False, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50,
                 storage_dir: str = "knowledge_base"):
        """
        Advanced RAG Engine supporting PDF, DOCX, TXT.
        """
        self.use_openai_embeddings = use_openai_embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage_dir = storage_dir
        
        # State
        self.chunks: List[Dict] = []
        self.index = None
        self.embed_model = None
        self.client = None

        # Ensure storage directory exists
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        # Initialize Models
        if self.use_openai_embeddings:
            if not OpenAI:
                raise ImportError("OpenAI library not found.")
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        # Attempt to load existing data
        self.load_from_disk()

    # ==========================
    # 1. FILE UPLOAD & INGESTION
    # ==========================
    def load_documents(self, file_paths: List[str]):
        """
        Loads multiple files and adds them to the knowledge base.
        """
        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"File not found: {fp}")
                continue
            
            ext = os.path.splitext(fp)[1].lower()
            text_data = [] # List of {"text": str, "page_number": int}
            
            try:
                if ext == ".pdf":
                    text_data = self._extract_pdf(fp)
                elif ext in [".docx", ".doc"]:
                    text_data = self._extract_docx(fp)
                elif ext == ".txt":
                    text_data = self._extract_txt(fp)
                else:
                    print(f"Unsupported format: {ext}")
                    continue
                
                if text_data:
                    self.chunk_documents(text_data, filename=os.path.basename(fp))
                    print(f"Successfully processed: {fp}")
                    
            except Exception as e:
                print(f"Error processing {fp}: {e}")

        # After processing all, re-build index
        self.create_vector_store()
        self.save_to_disk()

    def _extract_pdf(self, fp: str) -> List[Dict]:
        doc = fitz.open(fp)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({"text": text, "page_number": i + 1})
        return pages

    def _extract_docx(self, fp: str) -> List[Dict]:
        if not DocxDocument:
            print("python-docx not installed. Skipping .docx")
            return []
        
        doc = DocxDocument(fp)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        # DOCX doesn't have reliable pagination, so we return as Page 1
        return [{"text": full_text, "page_number": 1}]

    def _extract_txt(self, fp: str) -> List[Dict]:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [{"text": text, "page_number": 1}]

    # ==========================
    # 2. CHUNKING
    # ==========================
    def chunk_documents(self, pages: List[Dict], filename: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        current_chunk_id = len(self.chunks)
        
        for page in pages:
            texts = splitter.split_text(page["text"])
            for t in texts:
                chunk = {
                    "text": t,
                    "metadata": {
                        "chunk_id": current_chunk_id,
                        "file_name": filename,
                        "file_type": os.path.splitext(filename)[1],
                        "page_number": page["page_number"]
                    }
                }
                self.chunks.append(chunk)
                current_chunk_id += 1

    # ==========================
    # 3. EMBEDDINGS & 4. VECTOR DB
    # ==========================
    def create_vector_store(self):
        """
        Regenerates the FAISS index from current chunks.
        """
        texts = [c["text"] for c in self.chunks]
        if not texts:
            return

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.generate_embeddings(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.use_openai_embeddings:
            embeds = []
            # Batching could be added here for large datasets
            for t in texts:
                t = t.replace("\n", " ")
                resp = self.client.embeddings.create(input=t, model="text-embedding-3-small")
                embeds.append(resp.data[0].embedding)
            return np.array(embeds).astype('float32')
        else:
            return self.embed_model.encode(texts).astype('float32')

    # ==========================
    # PERSISTENCE
    # ==========================
    def save_to_disk(self):
        """
        Saves chunks to JSON and index to FAISS file.
        """
        # Save JSON (Traceability)
        json_path = os.path.join(self.storage_dir, "knowledge_base.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            
        # Save FAISS
        if self.index:
            faiss.write_index(self.index, os.path.join(self.storage_dir, "vector_store.index"))
        
        print(f"Saved {len(self.chunks)} chunks to {self.storage_dir}")

    def load_from_disk(self):
        """
        Loads data from disk if exists.
        """
        json_path = os.path.join(self.storage_dir, "knowledge_base.json")
        index_path = os.path.join(self.storage_dir, "vector_store.index")
        
        if os.path.exists(json_path) and os.path.exists(index_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.index = faiss.read_index(index_path)
            print(f"Loaded {len(self.chunks)} chunks from disk.")
        else:
            print("No existing knowledge base found. Initialized empty.")

    # ==========================
    # 5. RETRIEVAL & LOGGING
    # ==========================
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Dict]:
        if not self.index or not self.chunks:
            return []
            
        # Embed query
        q_vec = self.generate_embeddings([query])
        
        # Search
        distances, indices = self.index.search(q_vec, k)
        
        results = []
        self.log_sources_to_terminal(query, distances[0], indices[0])
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx != -1:
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(distances[0][i])
                results.append(chunk)
                
        return results

    def log_sources_to_terminal(self, query: str, distances: np.ndarray, indices: np.ndarray):
        print(f"\n[RAG Retrieval] Query: '{query}'")
        print("-" * 60)
        print(f"{'SCORE':<10} | {'SOURCE':<30} | {'PAGE':<5}")
        print("-" * 60)
        
        for i, idx in enumerate(indices):
            if idx < len(self.chunks) and idx != -1:
                chunk = self.chunks[idx]
                meta = chunk["metadata"]
                score = distances[i]
                print(f"{score:.4f}     | {meta['file_name']:<30} | {meta['page_number']}")
        print("-" * 60 + "\n")

    def generate_context_string(self, chunks: List[Dict]) -> str:
        if not chunks:
            return ""
            
        context = []
        for c in chunks:
            m = c["metadata"]
            citation = f"SOURCE: {m['file_name']} (Page {m['page_number']})"
            context.append(f"[{citation}]\n{c['text']}")
            
        return "\n\n".join(context)
