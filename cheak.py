import sys
print("Python:", sys.version)

# Try the key libs
import pypdf, pandas, sklearn
from sentence_transformers import SentenceTransformer
import chromadb

print("pypdf OK")
print("pandas OK")
print("scikit-learn OK")
print("chromadb OK")

# Tiny embedding test
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = m.encode(["hello finance"])
print("Embeddings shape:", emb.shape)
print(" Environment is ready")
