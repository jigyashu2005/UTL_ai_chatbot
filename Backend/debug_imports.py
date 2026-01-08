print("Start...")
import sys
print("Importing os...")
import os
print("Importing numpy...")
import numpy
print("Importing pymupdf...")
import fitz
print("Importing faiss...")
import faiss
print("Importing langchain...")
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
print("Importing sentence_transformers...")
from sentence_transformers import SentenceTransformer
print("All imports done.")
